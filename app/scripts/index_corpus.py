"""Build and verify a new Qdrant collection before activating its alias."""

import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.retrieval import (
    DENSE_VECTOR, SPARSE_VECTOR, CorpusRetriever, bm25_document,
    index_signature, verify_index_schema,
)

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
    """Name an index by corpus bytes and the complete vector configuration."""
    corpus_digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    signature = json.dumps(index_signature(), sort_keys=True, separators=(",", ":"))
    index_digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return f"{alias_name}_{corpus_digest}_hybrid_{index_digest}"


def verify_collection(
    client: QdrantClient, chunks: list[Document], collection_name: str,
    embedding_dim: int | None = None,
) -> None:
    """Check schema, both vectors and every payload before alias activation."""
    verify_index_schema(client, collection_name, embedding_dim)
    dimension = settings.embedding_dim if embedding_dim is None else embedding_dim
    expected = {str(doc.id): doc for doc in chunks}
    count = client.count(collection_name, exact=True).count
    if count != len(expected):
        raise RuntimeError(f"Unexpected point count: {count} != {len(expected)}")

    seen = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name, limit=256, offset=offset, with_payload=True, with_vectors=True,
        )
        for point in points:
            point_id = str(point.id)
            doc = expected.get(point_id)
            if doc is None or point.payload.get("page_content") != doc.page_content or (
                point.payload.get("metadata") != doc.metadata
            ):
                raise RuntimeError(f"Unexpected point: {point_id}")
            vectors = point.vector
            if not isinstance(vectors, dict) or set(vectors) != {DENSE_VECTOR, SPARSE_VECTOR}:
                raise RuntimeError(f"Missing point vectors: {point_id}")
            dense, sparse = vectors[DENSE_VECTOR], vectors[SPARSE_VECTOR]
            if not isinstance(dense, list) or len(dense) != dimension or not all(
                isinstance(value, (float, int)) and math.isfinite(value) for value in dense
            ) or not isinstance(sparse, SparseVector) or not sparse.indices or (
                len(sparse.indices) != len(sparse.values)
                or len(set(sparse.indices)) != len(sparse.indices)
                or not all(math.isfinite(value) for value in sparse.values)
            ):
                raise RuntimeError(f"Invalid point vectors: {point_id}")
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
        verify_collection(client, chunks, collection_name, embedding_dim)
        print(f"Already active: {alias_name} -> {collection_name}")
        return
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE_VECTOR: VectorParams(size=embedding_dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
            metadata={"index_signature": index_signature(embedding_dim)},
        )
    verify_index_schema(client, collection_name, embedding_dim)
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        vectors = embeddings.embed_documents([doc.page_content for doc in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count differs from chunk count")
        client.upsert(collection_name=collection_name, points=[
            PointStruct(
                id=doc.id, vector={DENSE_VECTOR: vector, SPARSE_VECTOR: bm25_document(doc.page_content)},
                payload={"page_content": doc.page_content, "metadata": doc.metadata},
            ) for doc, vector in zip(batch, vectors)
        ], wait=True)
    verify_collection(client, chunks, collection_name, embedding_dim)
    for mode in ("dense", "hybrid"):
        search = CorpusRetriever(client, collection_name, embeddings, mode, 1)
        hits = search.invoke("How does Ridge regression work?")
        if not hits or hits[0].metadata.get("source") not in {
            doc.metadata.get("source") for doc in chunks
        }:
            raise RuntimeError(f"Sanity {mode} search did not return a corpus source")

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
    client = QdrantClient(url=settings.qdrant_url, trust_env=False, cloud_inference=True)
    try:
        index_chunks(
            client, chunks, get_embeddings(), collection_name,
            settings.collection_name, settings.embedding_dim,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
