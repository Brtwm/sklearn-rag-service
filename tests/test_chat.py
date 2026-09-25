from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_rag_chain():
    with patch("app.main.build_rag_chain") as mock_build:
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
