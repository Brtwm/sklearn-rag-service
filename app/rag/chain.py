"""RAG retrieval and LCEL answer generation."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings
from app.llm import get_llm


SYSTEM_PROMPT = """You are a study assistant for the Classic ML cycle of an ML/DS course.

Rules:
- Use ONLY the provided context. If the answer is not in the context, say so honestly.
- Cite sources using [1], [2], ... — the numbers correspond to the source list in the context block.
- Reply in the SAME LANGUAGE as the user's question.
- Translate the relevant facts; keep code identifiers
  (function names, parameter names, classes) in English.
- If the user asks meta-questions ("what do you know about?", "что ты умеешь?") —
  answer based on the internal "About this RAG assistant" context.
"""

HUMAN_PROMPT = """Context:
{context}

Question:
{question}

Answer (with citations):"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def get_vectorstore() -> QdrantVectorStore:
    """Поднять клиент Qdrant + эмбеддер и завернуть в LangChain-VectorStore."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings
        },
    )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )


def index_available() -> bool:
    """Check whether the active Qdrant collection is reachable."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)
    try:
        return client.collection_exists(settings.collection_name)
    finally:
        client.close()


def format_docs_with_sources(docs: list[Document]) -> str:
    """Склеить топ-k чанков в нумерованный context-блок для prompt'а LLM."""
    lines = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(lines)


def build_rag_chain():
    """Собрать LCEL-цепочку и вернуть пару (chain, retriever)."""
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k}
    )

    llm = get_llm()

    chain = PROMPT | llm | StrOutputParser()

    return chain, retriever
