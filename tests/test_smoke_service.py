import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from app.scripts import smoke_service as smoke


QUESTION = {"id": "en-01", "language": "en", "question": "Ridge?"}
CONTEXT = "[1] Source: https://example.org/ridge#regression\nRidge uses L2."
CORPUS = {("https://example.org/ridge#regression", "Ridge uses L2."): "chunk-1"}


def response_body(question="Ridge?", context=CONTEXT):
    return {"answer": json.dumps({"question": question, "context": context}), "sources": [
        {"url": "https://example.org/ridge#regression", "snippet": "Ridge uses L2."},
    ]}


def test_stub_receives_the_generation_prompt_without_calling_a_provider():
    from app.rag.chain import PROMPT
    prompt = PROMPT.invoke({"question": "Ridge?", "context": CONTEXT})
    assert json.loads(smoke.stub_answer(prompt)) == {"question": "Ridge?", "context": CONTEXT}


def test_validate_response_preserves_context_order_and_identifies_chunks():
    assert smoke.validate_response(response_body(), QUESTION, CORPUS) == ["chunk-1"]


def test_validation_rejects_another_requests_valid_corpus_context():
    corpus = {**CORPUS, ("https://example.org/lasso", "Lasso uses L1."): "chunk-2"}
    body = {"answer": json.dumps({"question": "Ridge?",
                                  "context": "[1] Source: https://example.org/lasso\nLasso uses L1."}),
            "sources": [{"url": "https://example.org/lasso", "snippet": "Lasso uses L1."}]}
    with pytest.raises(ValueError, match="serial reference"):
        smoke.validate_response(body, QUESTION, corpus, expected_ids=["chunk-1"])


@pytest.mark.parametrize("change", ["question", "sources", "context", "duplicate"])
def test_validate_response_rejects_mixing_or_invalid_sources(change):
    body = response_body()
    if change == "question":
        body["answer"] = json.dumps({"question": "Different question", "context": CONTEXT})
    elif change == "sources":
        body["sources"][0]["url"] = "https://example.org/other"
    elif change == "context":
        body["answer"] = json.dumps({"question": "Ridge?", "context": CONTEXT + " invented"})
    else:
        body["answer"] = json.dumps({"question": "Ridge?", "context": CONTEXT + "\n\n---\n\n" + CONTEXT})
        body["sources"] *= 2
    with pytest.raises(ValueError):
        smoke.validate_response(body, QUESTION, CORPUS)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_http_batch_uses_bounded_concurrency_and_checks_every_response(concurrency):
    active = peak = 0
    lock = threading.Lock()

    def handler(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return httpx.Response(200, json=response_body())

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        rows = smoke.run_batch(client, [QUESTION] * 8, CORPUS, concurrency, 1)
    assert len(rows) == 8
    assert all(row["success"] and row["chunk_ids"] == ["chunk-1"] for row in rows)
    assert peak == concurrency


@pytest.mark.parametrize("failure", ["http", "timeout", "mixed"])
def test_failed_requests_are_recorded_instead_of_counted_as_success(failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(503 if failure == "http" else 200,
                              json=response_body(question="wrong"))

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        rows = smoke.run_batch(client, [QUESTION], CORPUS, 1, 1)
    assert not rows[0]["success"]
    assert rows[0]["error"]
    assert smoke.summarize(rows)["errors"] == 1


def test_summary_uses_all_latencies_and_keeps_error_counts():
    rows = [{"latency_ms": n, "success": n != 40} for n in (10, 20, 30, 40)]
    result = smoke.summarize(rows)
    assert result == {"requests": 4, "successful": 3, "errors": 1, "p50_ms": 25, "p95_ms": 38.5}


def test_ui_observations_reject_overlapping_generation():
    observations = {"peak_active": 2, "started": 2, "finished": 2}
    assert not smoke.ui_queue_passed(observations, 2)
    assert smoke.ui_queue_passed({**observations, "peak_active": 1}, 2)


@pytest.mark.parametrize("controls_enabled", [True, False])
def test_ui_check_reads_button_updates_and_checks_restored_controls(controls_enabled):
    history = [{"role": "assistant", "content": [{"type": "text", "text": response_body()["answer"]}]}]
    panel = "**[1]** [Ridge](<https://example.org/ridge#regression>)"
    completed = (history, {"interactive": False}, "Завершено", panel, {"interactive": False})
    restored = ({"__type__": "update"}, {"interactive": controls_enabled}, {}, {},
                {"interactive": controls_enabled})

    def client_factory(*args, **kwargs):
        # The installed client drops Button outputs by default. Mirror that boundary.
        outputs = [completed, restored]
        if kwargs.get("_skip_components", True):
            outputs = [event[:4] for event in outputs]
        job = SimpleNamespace(result=lambda **kw: outputs[-1], outputs=lambda: outputs)
        return SimpleNamespace(submit=lambda *a, **kw: job, close=lambda: None)

    with patch("gradio_client.Client", side_effect=client_factory):
        result = smoke.check_ui("http://test", [QUESTION], CORPUS)
    assert result["passed"] is controls_enabled
    if controls_enabled:
        assert result["requests"][0]["chunk_ids"] == ["chunk-1"]


def test_worker_refuses_changed_inputs_before_loading_models(tmp_path):
    corpus, questions = tmp_path / "corpus.jsonl", tmp_path / "questions.json"
    corpus.write_text("changed")
    questions.write_text("[]")
    args = SimpleNamespace(corpus=corpus, questions=questions, corpus_sha256="old", questions_sha256="old")
    with pytest.raises(ValueError, match="Inputs changed"):
        smoke.serve_worker(args)


def test_main_saves_failed_smoke_and_returns_nonzero(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(smoke.sys, "argv", ["smoke", "--output", str(output)])
    with patch.object(smoke, "run_smoke", return_value={"passed": False, "reason": "failed request"}):
        with pytest.raises(SystemExit) as exc:
            smoke.main()
    assert exc.value.code == 1
    assert json.loads(output.read_text())["passed"] is False
