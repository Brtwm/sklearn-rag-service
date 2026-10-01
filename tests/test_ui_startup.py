from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as service
from app.main import app


def test_gradio_page_loads_from_mounted_app() -> None:
    with patch("app.main.index_available", return_value=True), patch(
        "app.main.build_rag_chain", return_value=(MagicMock(), MagicMock())
    ):
        with TestClient(app) as client:
            response = client.get("/")

    assert response.status_code == 200
    assert "scikit-learn docs RAG" in response.text


def test_send_and_enter_share_a_bounded_queue() -> None:
    events = [event for event in service.demo.config["dependencies"]
              if event["backend_fn"] and len(event["targets"]) == 2]
    assert len(events) == 1
    event = events[0]
    assert {target[1] for target in event["targets"]} == {"submit", "click"}
    assert event["trigger_mode"] == "once"
    fn = service.demo.fns[event["id"]]
    assert fn.concurrency_limit == 1
    assert service.demo._queue.max_size == 8


@pytest.mark.parametrize("failure", [None, "readiness", "retrieval", "llm", "unexpected"])
def test_ui_restores_controls_after_request(failure) -> None:
    chain, retriever = MagicMock(), MagicMock()
    chain.stream.return_value = iter(["Answer"])
    retriever.invoke.return_value = []
    if failure == "retrieval":
        retriever.invoke.side_effect = ConnectionError()
    if failure == "llm":
        chain.stream.side_effect = TimeoutError()
    with patch("app.main._ensure_ready", return_value=failure != "readiness") as ready, patch(
        "app.main._chain", chain
    ), patch("app.main._retriever", retriever):
        if failure == "unexpected":
            ready.side_effect = RuntimeError()
        events = list(service._respond_ui("Ridge?", []))
    assert events[0][1]["interactive"] is False
    assert events[0][4]["interactive"] is False
    assert events[-1][1]["interactive"] is True
    assert events[-1][4]["interactive"] is True
    assert all(event[1].get("interactive") is False for event in events[:-1])
