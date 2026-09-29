"""RAG retrieval and LCEL answer generation."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings
from app.llm import get_llm
from app.rag.embeddings import get_embeddings
from app.rag.retrieval import DENSE_VECTOR, CorpusRetriever, verify_index_schema


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
    """Expose the named dense vector for the dense evaluation script."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)

    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
        vector_name=DENSE_VECTOR,
    )


def index_available() -> bool:
    """Check that the active collection has compatible dense and BM25 vectors."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)
    try:
        if not client.collection_exists(settings.collection_name):
            return False
        try:
            verify_index_schema(client, settings.collection_name)
        except RuntimeError:
            return False
        return True
    finally:
        client.close()


def get_retriever() -> CorpusRetriever:
    """Build retrieval independently of generation and LLM calls."""
    client = QdrantClient(url=settings.qdrant_url, trust_env=False, cloud_inference=True)
    try:
        verify_index_schema(client, settings.collection_name)
        return CorpusRetriever(
            client, settings.collection_name, get_embeddings(),
            settings.retrieval_mode, settings.top_k,
        )
    except Exception:
        client.close()
        raise


def format_docs_with_sources(docs: list[Document]) -> str:
    """Склеить топ-k чанков в нумерованный context-блок для prompt'а LLM."""
    lines = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(lines)


def build_rag_chain():
    """Собрать LCEL-цепочку и вернуть пару (chain, retriever)."""
    retriever = get_retriever()

    llm = get_llm()

    chain = PROMPT | llm | StrOutputParser()

    return chain, retriever
