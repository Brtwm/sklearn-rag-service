"""E5 embeddings shared by corpus indexing and retrieval."""

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


def get_embeddings(*, device: str | None = None) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        **({"model_kwargs": {"device": device}} if device is not None else {}),
        encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings,
            "prompt": "passage: ",
        },
        query_encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings,
            "prompt": "query: ",
        },
    )
