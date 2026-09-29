"""Build and verify a new Qdrant collection before activating its alias."""

import hashlib
import json
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    VectorParams,
)

from app.config import settings
from app.rag.embeddings import get_embeddings

CHUNKS_PATH = Path("data/corpus_chunks.jsonl")


def load_chunks(path: Path) -> list[Document]:
    """Read JSONL chunks with their stable point IDs."""
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            point_id = row.get("id")
            if not isinstance(point_id, str):
                raise ValueError("Chunk IDs must be UUID strings")
            try:
                point_id = str(UUID(point_id))
            except ValueError as exc:
                raise ValueError("Chunk IDs must be UUID strings") from exc
            docs.append(Document(id=point_id, page_content=row["content"], metadata=row["metadata"]))
    if not docs or len({doc.id for doc in docs}) != len(docs):
        raise ValueError("Corpus must contain chunks with unique IDs")
    print(f"Loaded {len(docs)} chunks from {path}")
    return docs


def corpus_collection_name(path: Path, alias_name: str) -> str:
    """Name a physical collection by the exact input corpus bytes."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{alias_name}_{digest}"


def verify_collection(client: QdrantClient, chunks: list[Document], collection_name: str) -> None:
    """Check every point ID, source and text before allowing alias activation."""
    expected = {str(doc.id): doc for doc in chunks}
    count = client.count(collection_name, exact=True).count
    if count != len(expected):
        raise RuntimeError(f"Unexpected point count: {count} != {len(expected)}")

    seen = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name, limit=256, offset=offset, with_payload=True,
        )
        for point in points:
            point_id = str(point.id)
            doc = expected.get(point_id)
            if doc is None or point.payload.get("page_content") != doc.page_content or (
                point.payload.get("metadata", {}).get("source") != doc.metadata.get("source")
            ):
                raise RuntimeError(f"Unexpected point: {point_id}")
            seen.add(point_id)
        if offset is None:
            break
    if seen != expected.keys():
        raise RuntimeError("Collection points differ from corpus chunks")


def index_chunks(
    client: QdrantClient,
    chunks: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    alias_name: str,
    embedding_dim: int,
) -> None:
    """Upsert, verify, then atomically point the active alias at the new collection."""
    aliases = {alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases}
    if client.collection_exists(alias_name) and alias_name not in aliases:
        raise RuntimeError(f"{alias_name} is a physical collection; use the new Qdrant data directory")
    if aliases.get(alias_name) == collection_name:
        verify_collection(client, chunks, collection_name)
        print(f"Already active: {alias_name} -> {collection_name}")
        return
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    verify_collection(client, chunks, collection_name)
    hits = vectorstore.similarity_search("How does Ridge regression work?", k=1)
    if not hits or hits[0].metadata.get("source") not in {
        doc.metadata.get("source") for doc in chunks
    }:
        raise RuntimeError("Sanity search did not return a corpus source")

    if aliases.get(alias_name) != collection_name:
        operations = []
        if alias_name in aliases:
            operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name)))
        operations.append(CreateAliasOperation(create_alias=CreateAlias(
            collection_name=collection_name, alias_name=alias_name,
        )))
        if not client.update_collection_aliases(operations):
            raise RuntimeError("Qdrant did not update the active alias")
    print(f"Verified {len(chunks)} chunks; {alias_name} -> {collection_name}")


def main() -> None:
    """Build the corpus collection and activate it only after verification."""
    chunks = load_chunks(CHUNKS_PATH)
    collection_name = corpus_collection_name(CHUNKS_PATH, settings.collection_name)
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)
    try:
        index_chunks(
            client, chunks, get_embeddings(), collection_name,
            settings.collection_name, settings.embedding_dim,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
