import hashlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient, models

from app.rag import chain, retrieval
from app.scripts import index_corpus
from tests.test_index_corpus import SmallEmbeddings, sample_chunks


def hit(point_id: str, score: float = 0.8):
    return SimpleNamespace(id=point_id, score=score, payload={
        "page_content": f"Text {point_id}",
        "metadata": {"source": f"https://example.org/page#{point_id}", "title": "Section"},
    })


def test_bm25_document_has_explicit_shared_options() -> None:
    document = retrieval.bm25_document("Как работает Ridge?")
    assert document.text == "Как работает Ridge?"
    assert document.model == "qdrant/bm25"
    assert document.options == {
        "tokenizer": "word", "lowercase": True, "ascii_folding": False,
        "stopwords": "english", "stemmer": {"type": "none"},
        "k": 1.2, "b": 0.75, "avg_len": 256,
    }


def test_hybrid_sends_two_prefetches_and_server_rrf_preserving_order() -> None:
    client = Mock()
    client.query_points.return_value.points = [hit(str(i)) for i in range(6)]
    search = retrieval.CorpusRetriever(client, "physical", SmallEmbeddings(), "hybrid", 4)

    docs = search.invoke("Как работает Ridge?")

    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "physical"
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert kwargs["limit"] == 40
    assert kwargs["with_payload"] is True
    dense, sparse = kwargs["prefetch"]
    assert dense.using == "dense" and dense.limit == 20
    assert dense.query == SmallEmbeddings().embed_query("Как работает Ridge?")
    assert sparse.using == "bm25" and sparse.limit == 20
    assert sparse.query == retrieval.bm25_document("Как работает Ridge?")
    assert [doc.id for doc in docs] == [str(i) for i in range(4)]
    assert docs[0].metadata == {"source": "https://example.org/page#0", "title": "Section"}
    assert docs[0].page_content == "Text 0"
    client.query_points.assert_called_once()


def test_candidates_are_unique_by_point_not_section_and_keep_scores() -> None:
    client = Mock()
    first, second = hit("one", 0.9), hit("two", 0.7)
    second.payload["metadata"]["source"] = first.payload["metadata"]["source"]
    client.query_points.return_value.points = [first, first, second]
    search = retrieval.CorpusRetriever(client, "physical", SmallEmbeddings(), "hybrid", 4)

    results = search.search_with_score("Ridge?", limit=40)

    assert [(doc.id, score) for doc, score in results] == [("one", 0.9), ("two", 0.7)]


def test_dense_uses_only_dense_vector_and_top_k() -> None:
    client = Mock()
    client.query_points.return_value.points = [hit("one")]
    search = retrieval.CorpusRetriever(client, "physical", SmallEmbeddings(), "dense", 4)
    assert [doc.id for doc in search.invoke("Ridge?")] == ["one"]
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["using"] == "dense"
    assert kwargs["limit"] == 4
    assert "prefetch" not in kwargs


@pytest.mark.parametrize("points", [[], [hit("one")]])
def test_hybrid_accepts_empty_or_short_results(points) -> None:
    client = Mock()
    client.query_points.return_value.points = points
    search = retrieval.CorpusRetriever(client, "physical", SmallEmbeddings(), "hybrid", 4)
    assert len(search.invoke("unknown")) == len(points)


def test_rrf_returns_dense_candidates_when_sparse_has_no_hits() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "hybrid", vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upsert("hybrid", [models.PointStruct(
        id=doc.id, vector={"dense": [1.0, 1.0, 0.0], "bm25": models.SparseVector(indices=[1], values=[1.0])},
        payload={"page_content": doc.page_content, "metadata": doc.metadata},
    ) for doc in sample_chunks()])
    search = retrieval.CorpusRetriever(client, "hybrid", SmallEmbeddings(), "hybrid", 4)
    with patch("app.rag.retrieval.bm25_document", return_value=models.SparseVector(indices=[99], values=[1.0])):
        docs = search.invoke("unknown")
    assert {doc.id for doc in docs} == {doc.id for doc in sample_chunks()}
    client.close()


def test_collection_name_includes_index_parameters(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("corpus\n", encoding="utf-8")
    original = index_corpus.corpus_collection_name(corpus, "docs")
    assert original != "docs_" + hashlib.sha256(corpus.read_bytes()).hexdigest()[:12]
    monkeypatch.setattr(retrieval.settings, "embedding_model", "another-model")
    assert index_corpus.corpus_collection_name(corpus, "docs") != original


def test_schema_rejects_legacy_dense_and_different_bm25_options() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("legacy", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    with pytest.raises(RuntimeError, match="schema"):
        retrieval.verify_index_schema(client, "legacy", 3)
    metadata = {"index_signature": retrieval.index_signature(3)}
    metadata["index_signature"]["bm25_options"]["b"] = 0.5
    client.create_collection(
        "wrong", vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata=metadata,
    )
    with pytest.raises(RuntimeError, match="parameters"):
        retrieval.verify_index_schema(client, "wrong", 3)
    client.close()


def test_ready_rejects_incompatible_index() -> None:
    with patch("app.rag.chain.QdrantClient") as factory, patch(
        "app.rag.chain.verify_index_schema", side_effect=RuntimeError("Incompatible index schema")
    ):
        factory.return_value.collection_exists.return_value = True
        assert chain.index_available() is False
    factory.return_value.close.assert_called_once()


def test_ready_endpoint_returns_503_for_legacy_dense_schema() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    with patch("app.rag.chain.QdrantClient") as factory, patch("app.main.build_rag_chain") as build:
        client = factory.return_value
        client.collection_exists.return_value = True
        client.get_collection.return_value.config.params.vectors = models.VectorParams(
            size=384, distance=models.Distance.COSINE,
        )
        with TestClient(app) as http:
            assert http.get("/health").status_code == 200
            assert http.get("/ready").status_code == 503
        build.assert_not_called()


def test_search_failure_is_not_silently_replaced_by_dense() -> None:
    client = Mock()
    client.query_points.side_effect = ConnectionError("Qdrant down")
    search = retrieval.CorpusRetriever(client, "physical", SmallEmbeddings(), "hybrid", 4)
    with pytest.raises(ConnectionError):
        search.invoke("Ridge?")
    client.query_points.assert_called_once()
