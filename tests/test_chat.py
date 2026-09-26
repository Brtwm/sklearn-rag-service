from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.main import app, respond
from app.rag import chain as rag_chain
from app.rag.chain import build_rag_chain


@pytest.fixture(autouse=True)
def mock_rag_chain():
    with patch("app.main.build_rag_chain") as mock_build, patch(
        "app.main.index_available", return_value=True
    ):
        mock_chain = MagicMock()
        mock_retriever = MagicMock()
        mock_build.return_value = (mock_chain, mock_retriever)
        yield mock_chain, mock_retriever


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_validates_empty_question() -> None:
    """Empty question must be rejected by Pydantic before reaching the chain."""
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_returns_answer_with_sources(mock_rag_chain) -> None:
    """Smoke test with fully mocked chain - no LLM call, no Qdrant call"""
    mock_chain, mock_retriever = mock_rag_chain
    mock_chain.invoke.return_value = "Ridge uses L2 penalty [1]."

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "https://scikit-learn.org/stable/linear.html"}
    mock_doc.page_content = "Ridge regression addresses..."
    mock_retriever.invoke.return_value = [mock_doc]

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 200
    body = response.json()
    assert "Ridge uses L2" in body["answer"]
    assert len(body["sources"]) == 1
    assert "scikit-learn.org" in body["sources"][0]["url"]


def test_chat_passes_retrieved_sources_to_llm(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    docs = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"}),
        Document(page_content="Lasso uses L1.", metadata={"source": "https://example.org/lasso"}),
    ]
    mock_retriever.invoke.return_value = docs
    mock_chain.invoke.return_value = "Ridge uses L2 [1]."

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 200
    assert [source["url"] for source in response.json()["sources"]] == [
        "https://example.org/ridge",
        "https://example.org/lasso",
    ]
    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    assert mock_chain.invoke.call_args.args[0] == {
        "question": "What is Ridge?",
        "context": "[1] Source: https://example.org/ridge\nRidge uses L2.\n\n---\n\n"
        "[2] Source: https://example.org/lasso\nLasso uses L1.",
    }


def test_ready_recovers_when_index_appears(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    with patch("app.main.index_available", side_effect=[False, False, False, True, True]):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 503
            assert client.post("/chat", json={"question": "Hello"}).status_code == 503
            mock_retriever.invoke.return_value = []
            mock_chain.invoke.return_value = "No context."
            assert client.post("/chat", json={"question": "Hello"}).status_code == 200
            assert client.get("/ready").status_code == 200


def test_gradio_stream_uses_retrieved_context(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"})
    ]
    mock_chain.stream.return_value = iter(["Ridge", " [1]"])

    with TestClient(app):
        events = list(respond("What is Ridge?", []))

    assert "https://example.org/ridge" in events[0][3]
    assert events[-1][0][-1]["content"] == "Ridge [1]"
    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    assert mock_chain.stream.call_args.args[0] == {
        "question": "What is Ridge?",
        "context": "[1] Source: https://example.org/ridge\nRidge uses L2.",
    }


def test_generation_chain_does_not_repeat_retrieval() -> None:
    retrieval_calls = []
    retriever = RunnableLambda(lambda question: retrieval_calls.append(question) or [])
    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value = retriever
    with patch("app.rag.chain.get_vectorstore", return_value=vectorstore), patch(
        "app.rag.chain.get_llm", return_value=RunnableLambda(lambda prompt: "Answer [1].")
    ):
        chain, returned_retriever = build_rag_chain()
        answer = chain.invoke({"question": "What is Ridge?", "context": "[1] Source: x\nRidge uses L2."})

    assert answer == "Answer [1]."
    assert returned_retriever is retriever
    assert retrieval_calls == []


def test_index_availability_checks_active_collection() -> None:
    with patch.object(rag_chain.settings, "collection_name", "active_test_collection"), patch(
        "app.rag.chain.QdrantClient"
    ) as client_class:
        client_class.return_value.collection_exists.return_value = True
        assert rag_chain.index_available() is True

    client_class.return_value.collection_exists.assert_called_once_with("active_test_collection")
    client_class.return_value.close.assert_called_once_with()
