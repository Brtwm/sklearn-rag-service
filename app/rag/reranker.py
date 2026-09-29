"""One CPU cross-encoder per process, reused across RAG initialization."""

from threading import Lock

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings

MAX_LENGTH = 512
BATCH_SIZE = 8


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name, device="cpu", max_length=MAX_LENGTH)
        self._predict_lock = Lock()

    def rerank(self, question: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """Score original question/text pairs; keep RRF order for equal scores."""
        if not documents:
            return []
        pairs = [(question, doc.page_content) for doc in documents]
        with self._predict_lock:
            scores = self.model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
        results = [(doc, float(score)) for doc, score in zip(documents, scores, strict=True)]
        return sorted(results, key=lambda item: item[1], reverse=True)


_reranker: CrossEncoderReranker | None = None
_model_lock = Lock()


def get_reranker() -> CrossEncoderReranker:
    """Cache a successfully loaded model without caching initialization errors."""
    global _reranker
    with _model_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker(settings.reranker_model)
        return _reranker
