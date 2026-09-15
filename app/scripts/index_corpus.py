"""Индексация чанков из `data/corpus_chunks.jsonl` в Qdarnt.

Пересоздаем коллекцию, считаем эмбеддинги через `multilingual-e5-small` и сразу проверяем,
что retriever ходит: три sanity-запроса (EN / RU / meta-вопрос).
Все параметры - из `app.config.settings`"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

CHUNKS_PATH = Path("data/corpus_chunks.jsonl")


def load_chunks(path: Path) -> list[Document]:
    """Прочитать jsonl и собрать список Document'ов с метаданными."""
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs.append(Document(page_content=row["content"], metadata=row["metadata"]))
    print(f"Loaded {len(docs)} chunks from {path}")
    return docs


def main() -> None:
    """Пересоздать коллекцию Qdrant, проиндексировать чанки, прогнать sanity-чек."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)

    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    print(f"Collection {settings.collection_name} (re)created at {settings.qdrant_url}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings}
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )

    chunks = load_chunks(CHUNKS_PATH)
    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks")

    # Sanity: один запрос на английском, один на русском, один meta-вопрос
    for query in [
        "How does Ridge regression work?",
        "Что такое переобучение?",
        "what do you know about?",
    ]:
        hits = vectorstore.similarity_search(query, k=3)
        print(f"\n--- Sanity: {query!r} ---")
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. {hit.metadata.get('source', '?')[:80]}")
            print(f"     {hit.page_content[:100].strip()!r}")


if __name__ == "__main__":
    main()
