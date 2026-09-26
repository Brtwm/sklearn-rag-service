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


@pytest.mark.parametrize("question", [
    pytest.param("", id="empty"),
    pytest.param("   ", id="spaces"),
    pytest.param("x" * 501, id="too_long"),
])
def test_chat_rejects_invalid_question(question: str, mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_chain.invoke.return_value = "Answer"
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": question})
    assert response.status_code == 422
    mock_retriever.invoke.assert_not_called()
    mock_chain.invoke.assert_not_called()


def test_chat_accepts_question_at_length_limit(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = []
    mock_chain.invoke.return_value = "Answer"

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "x" * 500})

    assert response.status_code == 200
    mock_retriever.invoke.assert_called_once_with("x" * 500)


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


def test_startup_survives_qdrant_connection_failure(mock_rag_chain) -> None:
    with patch("app.main.index_available", side_effect=ConnectionError("Qdrant down")):
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/").status_code == 200
            assert client.get("/ready").status_code == 503
            assert client.post("/chat", json={"question": "Ridge?"}).status_code == 503


def test_startup_survives_rag_initialization_failure(mock_rag_chain) -> None:
    with patch("app.main.build_rag_chain", side_effect=RuntimeError("model unavailable")):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 503


def test_ready_reports_index_lost_after_startup(mock_rag_chain) -> None:
    with patch("app.main.index_available", side_effect=[True, False]):
        with TestClient(app) as client:
            assert client.get("/ready").status_code == 503
            assert client.get("/health").status_code == 200


def test_four_citations_follow_rest_source_order(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    docs = [
        Document(page_content=f"Passage {number}", metadata={"source": f"https://example.org/{number}"})
        for number in range(1, 5)
    ]
    mock_retriever.invoke.return_value = docs
    mock_chain.invoke.return_value = "See [1] and [4]."

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "Compare methods"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "See [1] and [4].",
        "sources": [
            {"url": f"https://example.org/{number}", "snippet": f"Passage {number}"}
            for number in range(1, 5)
        ],
    }
    context = mock_chain.invoke.call_args.args[0]["context"]
    assert context.split("\n\n---\n\n") == [
        f"[{number}] Source: https://example.org/{number}\nPassage {number}"
        for number in range(1, 5)
    ]


def test_chat_returns_503_when_qdrant_search_fails(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "Ridge?"})

    assert response.status_code == 503
    assert "Retrieval temporarily unavailable" in response.json()["detail"]
    mock_chain.invoke.assert_not_called()


def test_chat_returns_503_when_llm_fails(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"})
    ]
    mock_chain.invoke.side_effect = TimeoutError("provider timeout")

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "Ridge?"})

    assert response.status_code == 503
    assert "LLM provider temporarily unavailable" in response.json()["detail"]
    mock_retriever.invoke.assert_called_once_with("Ridge?")


def test_gradio_stream_uses_retrieved_context(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"})
    ]
    mock_chain.stream.return_value = iter(["Ridge", " [1]"])

    with TestClient(app):
        stream = respond("What is Ridge?", [])
        first = next(stream)
        assert first[0][-1]["content"] == ""
        assert "_streaming…_" in first[2]
        assert "https://example.org/ridge" in first[3]
        second = next(stream)
        assert second[0][-1]["content"] == "Ridge"
        events = list(stream)

    assert events[-1][0][-1]["content"] == "Ridge [1]"
    assert "**LLM stream (full):**" in events[-1][2]
    assert events[-1][3] == first[3]
    mock_retriever.invoke.assert_called_once_with("What is Ridge?")
    assert mock_chain.stream.call_args.args[0] == {
        "question": "What is Ridge?",
        "context": "[1] Source: https://example.org/ridge\nRidge uses L2.",
    }


def test_gradio_reports_qdrant_search_failure(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")

    with TestClient(app):
        events = list(respond("Ridge?", []))

    assert len(events) == 1
    assert "Поиск сейчас недоступен" in events[0][0][-1]["content"]
    assert "_—_" in events[0][3]
    mock_chain.stream.assert_not_called()


def test_gradio_reports_missing_index(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    with patch("app.main.index_available", return_value=False):
        with TestClient(app):
            events = list(respond("Ridge?", []))

    assert len(events) == 1
    assert "Индекс пока недоступен" in events[0][0][-1]["content"]
    mock_retriever.invoke.assert_not_called()
    mock_chain.stream.assert_not_called()


def test_gradio_reports_llm_failure_after_partial_answer(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"})
    ]

    def partial_stream():
        yield "Ridge"
        raise TimeoutError("provider timeout")

    mock_chain.stream.return_value = partial_stream()
    with TestClient(app):
        events = list(respond("Ridge?", []))

    assert "LLM-провайдер сейчас недоступен" in events[-1][0][-1]["content"]
    assert "LLM stream (full)" not in events[-1][2]
    assert "https://example.org/ridge" in events[-1][3]


def test_gradio_reports_empty_llm_stream(mock_rag_chain) -> None:
    mock_chain, mock_retriever = mock_rag_chain
    mock_retriever.invoke.return_value = [
        Document(page_content="Ridge uses L2.", metadata={"source": "https://example.org/ridge"})
    ]
    mock_chain.stream.return_value = iter([])

    with TestClient(app):
        events = list(respond("Ridge?", []))

    assert len(events) == 2
    assert "LLM не вернула ответ" in events[-1][0][-1]["content"]
    assert "https://example.org/ridge" in events[-1][3]


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
