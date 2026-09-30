"""Dense and BM25 retrieval with reciprocal rank fusion in Qdrant."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.config import settings
from app.rag.reranker import CrossEncoderReranker

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
CANDIDATES_PER_SEARCH = 20


def bm25_document(text: str) -> models.Document:
    """Use identical server-side text processing for documents and queries."""
    return models.Document(text=text, model="qdrant/bm25", options={
        "tokenizer": "word", "lowercase": True, "ascii_folding": False,
        "stopwords": "english", "stemmer": {"type": "none"},
        "k": 1.2, "b": 0.75, "avg_len": 256,
    })


def index_signature(embedding_dim: int | None = None) -> dict:
    """Parameters that require a separate physical index when changed."""
    return {
        "schema_version": 1,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim if embedding_dim is None else embedding_dim,
        "normalize_embeddings": settings.normalize_embeddings,
        "passage_prompt": "passage: ",
        "query_prompt": "query: ",
        "dense_vector": DENSE_VECTOR,
        "sparse_vector": SPARSE_VECTOR,
        "distance": "Cosine",
        "bm25_model": "qdrant/bm25",
        "bm25_options": bm25_document("").options,
        "sparse_modifier": "idf",
    }


def verify_index_schema(
    client: QdrantClient, collection_name: str, embedding_dim: int | None = None,
) -> None:
    """Reject legacy or incompatible indices before retrieval or activation."""
    signature = index_signature(embedding_dim)
    config = client.get_collection(collection_name).config
    vectors = config.params.vectors
    sparse = config.params.sparse_vectors or {}
    if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors or (
        vectors[DENSE_VECTOR].size != signature["embedding_dim"]
        or vectors[DENSE_VECTOR].distance != models.Distance.COSINE
        or SPARSE_VECTOR not in sparse
        or sparse[SPARSE_VECTOR].modifier != models.Modifier.IDF
    ):
        raise RuntimeError("Incompatible index schema; reindex the corpus")
    if (config.metadata or {}).get("index_signature") != signature:
        raise RuntimeError("Incompatible index parameters; reindex the corpus")


class CorpusRetriever:
    """Return one ordered document list for generation and citations."""

    def __init__(
        self, client: QdrantClient, collection_name: str, embeddings: Embeddings,
        mode: str, top_k: int, reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if mode not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if mode == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank requires a loaded reranker")
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.mode = mode
        self.top_k = top_k
        self.reranker = reranker

    def search_with_score(self, question: str, limit: int) -> list[tuple[Document, float]]:
        """Retrieve and optionally rerank, cutting to limit after scoring."""
        dense = self.embeddings.embed_query(question)
        if self.mode == "dense":
            result = self.client.query_points(
                collection_name=self.collection_name, query=dense,
                using=DENSE_VECTOR, limit=limit, with_payload=True,
            )
        else:
            result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE_VECTOR, limit=CANDIDATES_PER_SEARCH),
                    models.Prefetch(
                        query=bm25_document(question), using=SPARSE_VECTOR,
                        limit=CANDIDATES_PER_SEARCH,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=2 * CANDIDATES_PER_SEARCH if self.mode == "hybrid_rerank" else limit,
                with_payload=True,
            )
        hits = []
        seen = set()
        for point in result.points:
            point_id = str(point.id)
            if point_id in seen:
                continue
            seen.add(point_id)
            hits.append((Document(
                id=point_id, page_content=point.payload["page_content"],
                metadata=point.payload["metadata"],
            ), float(point.score)))
        if self.mode == "hybrid_rerank":
            return self.reranker.rerank(question, [doc for doc, _ in hits])[:limit]
        return hits

    def invoke(self, question: str) -> list[Document]:
        limit = 2 * CANDIDATES_PER_SEARCH if self.mode == "hybrid" else self.top_k
        return [doc for doc, _ in self.search_with_score(question, limit)[:self.top_k]]
