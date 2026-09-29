import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.scripts import index_corpus


class SmallEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text)), 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, float(len(text)), 0.0]


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embeddings unavailable")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Embeddings unavailable")


def sample_chunks() -> list[Document]:
    return [
        Document(
            id="d2fb48f6-b211-4f29-8367-b8b11afc93e0",
            page_content="Ridge applies an L2 penalty.",
            metadata={"source": "https://scikit-learn.org/1.9/modules/linear_model.html#ridge-regression"},
        ),
        Document(
            id="dbeee9de-b5fa-4a75-9f1e-72ef57d7d1dc",
            page_content="Lasso applies an L1 penalty.",
            metadata={"source": "https://scikit-learn.org/1.9/modules/linear_model.html#lasso"},
        ),
    ]


def create_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )


def alias_target(client: QdrantClient) -> str | None:
    return next(
        (alias.collection_name for alias in client.get_aliases().aliases if alias.alias_name == "sklearn_docs"),
        None,
    )


def test_e5_factory_uses_passage_and_query_prompts() -> None:
    from app.rag.embeddings import get_embeddings

    with patch("app.rag.embeddings.HuggingFaceEmbeddings") as embedding_class:
        get_embeddings()

    kwargs = embedding_class.call_args.kwargs
    assert kwargs["encode_kwargs"] == {"normalize_embeddings": True, "prompt": "passage: "}
    assert kwargs["query_encode_kwargs"] == {"normalize_embeddings": True, "prompt": "query: "}


def test_load_chunks_preserves_stable_ids(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps({
            "id": doc.id, "content": doc.page_content, "metadata": doc.metadata,
        }) for doc in sample_chunks()) + "\n",
        encoding="utf-8",
    )

    loaded = index_corpus.load_chunks(path)

    assert [doc.id for doc in loaded] == [doc.id for doc in sample_chunks()]
    assert [doc.metadata["source"] for doc in loaded] == [
        doc.metadata["source"] for doc in sample_chunks()
    ]


@pytest.mark.parametrize("point_id", [None, "not-a-uuid"])
def test_load_chunks_rejects_invalid_point_ids(tmp_path: Path, point_id: str | None) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps({
            "id": point_id,
            "content": "Ridge uses L2.",
            "metadata": {"source": "https://example.org/ridge"},
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UUID"):
        index_corpus.load_chunks(path)


def test_collection_name_changes_with_corpus_bytes(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("first\n", encoding="utf-8")
    first = index_corpus.corpus_collection_name(path, "sklearn_docs")
    assert first == index_corpus.corpus_collection_name(path, "sklearn_docs")

    path.write_text("second\n", encoding="utf-8")
    assert index_corpus.corpus_collection_name(path, "sklearn_docs") != first
    assert first.startswith("sklearn_docs_")


def test_index_chunks_switches_alias_after_verification_and_keeps_old_collection() -> None:
    client = QdrantClient(":memory:")
    create_collection(client, "old_collection")
    client.update_collection_aliases([
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name="old_collection", alias_name="sklearn_docs",
        )),
    ])
    chunks = sample_chunks()

    index_corpus.index_chunks(client, chunks, SmallEmbeddings(), "new_collection", "sklearn_docs", 3)

    assert alias_target(client) == "new_collection"
    assert client.collection_exists("old_collection")
    points, _ = client.scroll("new_collection", limit=10, with_payload=True)
    assert {str(point.id) for point in points} == {str(UUID(doc.id)) for doc in chunks}
    assert {point.payload["metadata"]["source"] for point in points} == {
        doc.metadata["source"] for doc in chunks
    }
    assert {point.payload["page_content"] for point in points} == {
        doc.page_content for doc in chunks
    }


def test_index_chunks_does_not_rewrite_an_already_active_verified_collection() -> None:
    client = QdrantClient(":memory:")
    chunks = sample_chunks()
    index_corpus.index_chunks(client, chunks, SmallEmbeddings(), "new_collection", "sklearn_docs", 3)

    index_corpus.index_chunks(
        client, chunks, FailingEmbeddings(), "new_collection", "sklearn_docs", 3,
    )

    assert alias_target(client) == "new_collection"
    assert client.count("new_collection", exact=True).count == len(chunks)


def test_index_chunks_keeps_alias_when_verification_fails() -> None:
    client = QdrantClient(":memory:")
    create_collection(client, "old_collection")
    create_collection(client, "new_collection")
    client.upsert("new_collection", points=[
        models.PointStruct(
            id="ac5575c0-032b-4cd1-b71c-6eb8849d9113",
            vector=[1.0, 0.0, 0.0],
            payload={"page_content": "stray", "metadata": {"source": "wrong"}},
        ),
    ])
    client.update_collection_aliases([
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name="old_collection", alias_name="sklearn_docs",
        )),
    ])

    with pytest.raises(RuntimeError, match="count|points"):
        index_corpus.index_chunks(
            client, sample_chunks(), SmallEmbeddings(), "new_collection", "sklearn_docs", 3,
        )

    assert alias_target(client) == "old_collection"
    assert client.collection_exists("old_collection")


def test_index_chunks_refuses_physical_collection_named_like_alias() -> None:
    client = QdrantClient(":memory:")
    create_collection(client, "sklearn_docs")

    with pytest.raises(RuntimeError, match="physical collection"):
        index_corpus.index_chunks(
            client, sample_chunks(), SmallEmbeddings(), "new_collection", "sklearn_docs", 3,
        )

    assert not client.collection_exists("new_collection")
    assert client.collection_exists("sklearn_docs")
