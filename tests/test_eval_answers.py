import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.scripts import eval_answers as evaluation


QUESTION = {"id": "en-01", "question": "Ridge?", "language": "en", "category": "single",
            "expected_facts": ["L2 penalty"], "expected_refusal": False}
DOC = Document(page_content="Ridge uses an L2 penalty.", metadata={"source": "https://example.org/ridge"})


def verdict(refusal=None):
    return json.dumps({**{name: {"score": 2, "explanation": "Supported by [1]."}
                         for name in evaluation.METRICS}, "refusal": refusal})


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply, response_metadata={}, usage_metadata=None)


class FakeRetriever:
    def __init__(self):
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return [DOC]


def execute(tmp_path, replies, questions=None):
    llm, retriever = FakeLLM(replies), FakeRetriever()
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    report = evaluation.run_evaluation(questions or [QUESTION], retriever, llm, tmp_path / "report.json",
                                       clock=lambda: now[0], sleep=sleep)
    return report, llm, retriever, sleeps


def test_exact_budget_one_search_and_two_calls_per_question_with_spacing(tmp_path):
    questions = [{**QUESTION, "id": str(i)} for i in range(6)]
    report, llm, retriever, sleeps = execute(tmp_path, ["L2 [1]", verdict()] * 6, questions)
    assert report["complete"] and report["calls_attempted"] == 12
    assert len(llm.prompts) == 12 and len(retriever.questions) == 6
    assert len(sleeps) == 11 and all(seconds >= 20 for seconds in sleeps)
    assert report["summary"]["grounding"] == {"mean": 2.0, "n": 6}
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["complete"]


def test_budget_rejects_more_than_six_questions_before_any_calls(tmp_path):
    llm, retriever = FakeLLM([]), FakeRetriever()
    with pytest.raises(ValueError, match="12"):
        evaluation.run_evaluation([QUESTION] * 7, retriever, llm, tmp_path / "report.json")
    assert not llm.prompts and not retriever.questions


@pytest.mark.parametrize("stage", ["generation", "judge"])
def test_rate_limit_stops_immediately_and_saves_partial_report(tmp_path, stage):
    limit = RuntimeError("Do not save this sensitive exception")
    limit.status_code = 429
    replies = [limit] if stage == "generation" else ["L2 [1]", limit]
    report, llm, retriever, _ = execute(tmp_path, replies, [QUESTION, {**QUESTION, "id": "next"}])
    assert not report["complete"] and report["stop_reason"] == "provider_limit"
    assert len(llm.prompts) == len(replies) and len(retriever.questions) == 1
    assert report["summary"]["grounding"] == {"mean": None, "n": 0}
    saved = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "sensitive" not in saved
    if stage == "judge":
        assert report["questions"][0]["answer"] == "L2 [1]"


@pytest.mark.parametrize("raw", ["not JSON", '{}', '{"grounding": {"score": true}}'])
def test_invalid_judge_has_no_repair_call_or_zero_substitution(tmp_path, raw):
    report, llm, _, _ = execute(tmp_path, ["L2 [1]", raw])
    assert len(llm.prompts) == 2 and not report["complete"]
    assert report["questions"][0]["judge_raw"] == raw
    assert report["questions"][0]["scores"] is None
    assert report["summary"]["grounding"]["mean"] is None


def test_citation_range_does_not_claim_support(tmp_path):
    report, _, _, _ = execute(tmp_path, ["Invented [1], invalid [0] [2]", verdict()])
    check = report["questions"][0]["citations"]
    assert check == {"numbers": [0, 1, 2], "invalid": [0, 2], "in_range": False}
    assert report["questions"][0]["scores"]["citation_support"]["score"] == 2
    # Range validation remains distinct from the judge's semantic assessment.


def test_judge_receives_exact_generation_context_and_expected_facts(tmp_path):
    report, llm, _, _ = execute(tmp_path, ["L2 [1]", verdict()])
    generation = llm.prompts[0].to_messages()[-1].content
    judge = json.loads(llm.prompts[1][-1].content)
    context = report["questions"][0]["context"]
    assert context in generation and judge["context"] == context
    assert judge["expected_facts"] == QUESTION["expected_facts"]


def test_response_saved_before_judge_starts(tmp_path):
    class InspectLLM(FakeLLM):
        def invoke(self, prompt, **kwargs):
            if self.prompts:
                saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
                assert saved["questions"][0]["answer"] == "L2 [1]"
                assert saved["calls"][0]["status"] == "completed"
            return super().invoke(prompt, **kwargs)

    llm = InspectLLM(["L2 [1]", verdict()])
    evaluation.run_evaluation([QUESTION], FakeRetriever(), llm, tmp_path / "report.json", sleep=lambda _: None)


def test_refusal_is_evaluated_separately(tmp_path):
    q = {**QUESTION, "category": "out_of_corpus", "expected_facts": [], "expected_refusal": True}
    report, _, _, _ = execute(tmp_path, ["Not in the context.", verdict({"score": 2, "explanation": "Honest refusal."})], [q])
    assert report["summary"]["refusal"] == {"mean": 2.0, "n": 1}


def test_empty_generation_is_saved_and_does_not_waste_judge_call(tmp_path):
    report, llm, _, _ = execute(tmp_path, [" "])
    assert len(llm.prompts) == 1 and not report["complete"]
    assert report["questions"][0]["status"] == "empty_answer"


def test_fixed_questions_come_from_existing_set():
    questions = evaluation.load_questions()
    assert [q["id"] for q in questions] == ["en-01", "en-07", "en-11", "ru-01", "ru-06", "ooc-ru-01"]
    assert all(q["expected_facts"] or q["expected_refusal"] for q in questions)


def test_real_sdk_does_not_retry_rate_limits(tmp_path, monkeypatch):
    import httpx
    from langchain_openai import ChatOpenAI
    import langchain_openai

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"message": "Rate limited", "type": "rate_limit", "code": "rate_limit"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory(**kwargs):
            return ChatOpenAI(**{**kwargs, "api_key": "test-only"}, http_client=client)

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", factory)
        llm = evaluation.create_llm()
        report = evaluation.run_evaluation([QUESTION], FakeRetriever(), llm, tmp_path / "report.json")
    assert len(requests) == 1
    assert report["calls_attempted"] == 1 and report["stop_reason"] == "provider_limit"


def test_evaluation_uses_streaming_transport_with_one_provider_call():
    llm = evaluation.create_llm()
    assert llm.streaming is True
    assert llm.max_retries == 0


@pytest.mark.parametrize("bad", [True, 3, -1, 2.0, "2"])
def test_judge_rejects_non_integer_or_out_of_range_scores(bad):
    raw = json.loads(verdict())
    raw["grounding"]["score"] = bad
    with pytest.raises(ValueError):
        evaluation.parse_judge(json.dumps(raw), False)


def test_retrieval_failure_uses_no_provider_budget(tmp_path):
    class BrokenRetriever:
        def invoke(self, question):
            raise ConnectionError("secret details")

    llm = FakeLLM([])
    report = evaluation.run_evaluation([QUESTION], BrokenRetriever(), llm, tmp_path / "report.json")
    assert report["calls_attempted"] == 0 and not report["complete"]
    assert report["questions"][0]["status"] == "retrieval_error"


def test_resume_skips_completed_question_and_keeps_provenance(tmp_path):
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", verdict()])
    previous["questions"][0].pop("judge_origin")  # Reports written before resume support.
    previous["questions"].append({**QUESTION, "id": "next", "status": "not_started", "answer": None,
                                  "context": None, "sources": [], "citations": None,
                                  "judge_raw": None, "scores": None})
    llm, retriever = FakeLLM(["L2 [1]", verdict()]), FakeRetriever()
    report = evaluation.run_evaluation([QUESTION, {**QUESTION, "id": "next"}], retriever, llm,
        tmp_path / "continued.json", previous=previous, parent_info={"path": "original.json", "sha256": "abc"},
        call_budget=2, sleep=lambda _: None)
    assert len(retriever.questions) == 1 and len(llm.prompts) == 2
    assert report["complete"] and report["calls_attempted"] == 2
    assert report["total_calls_attempted"] == 4
    assert report["questions"][0]["answer"] == previous["questions"][0]["answer"]
    assert report["questions"][0]["judge_origin"] == "original.json"
    assert report["previous_runs"][-1]["sha256"] == "abc"


def test_resume_reuses_answer_when_only_judge_is_missing(tmp_path):
    limit = RuntimeError()
    limit.status_code = 429
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", limit])
    llm, retriever = FakeLLM([verdict()]), FakeRetriever()
    report = evaluation.run_evaluation([QUESTION], retriever, llm, tmp_path / "continued.json",
        previous=previous, parent_info={"path": "original.json", "sha256": "abc"}, call_budget=1,
        sleep=lambda _: None)
    assert report["complete"] and len(llm.prompts) == 1 and not retriever.questions
    assert report["questions"][0]["generation_origin"] == previous["questions"][0]["generation_origin"]


@pytest.mark.parametrize("change", ["model", "corpus", "rubric"])
def test_resume_rejects_incompatible_configuration(tmp_path, change):
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", verdict()])
    previous["parameters"] = {"model": "same", "input_sha256": {"data/corpus_chunks.jsonl": "abc"}}
    parameters = {"model": "same", "input_sha256": {"data/corpus_chunks.jsonl": "abc"}}
    if change == "model":
        parameters["model"] = "different"
    elif change == "corpus":
        parameters["input_sha256"]["data/corpus_chunks.jsonl"] = "changed"
    else:
        previous["rubric"] = "different"
    with pytest.raises(ValueError, match="changed"):
        evaluation.run_evaluation([QUESTION], FakeRetriever(), FakeLLM([]), tmp_path / "continued.json",
                                  previous=previous, parameters=parameters)


def test_resumed_limit_retains_all_previous_calls_and_results(tmp_path):
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", verdict()])
    previous["questions"].append({**QUESTION, "id": "next", "status": "not_started", "answer": None,
                                  "context": None, "sources": [], "citations": None,
                                  "judge_raw": None, "scores": None})
    limit = RuntimeError()
    limit.status_code = 429
    report = evaluation.run_evaluation([QUESTION, {**QUESTION, "id": "next"}], FakeRetriever(), FakeLLM([limit]),
        tmp_path / "continued.json", previous=previous, parent_info={"path": "original.json"}, call_budget=2,
        sleep=lambda _: None)
    assert report["total_calls_attempted"] == 3 and not report["complete"]
    assert report["summary"]["grounding"] == {"mean": 2.0, "n": 1}
    assert len(report["prior_calls"]) == 2 and report["stop_reason"] == "provider_limit"


def test_resume_rejects_changed_facts_before_any_call(tmp_path):
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", verdict()])
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="changed"):
        evaluation.run_evaluation([{**QUESTION, "expected_facts": ["Different"]}], FakeRetriever(), llm,
                                  tmp_path / "continued.json", previous=previous, call_budget=1)
    assert not llm.prompts


def test_resume_requires_enough_budget_before_any_call(tmp_path):
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="budget"):
        evaluation.run_evaluation([QUESTION], FakeRetriever(), llm, tmp_path / "report.json", call_budget=1)
    assert not llm.prompts


def test_limit_diagnostics_does_not_copy_sensitive_message():
    exc = RuntimeError("gsk-secret")
    exc.body = {"error": {"code": "rate_limit_exceeded", "message":
        "Rate limit reached for model; tokens per minute (TPM). Try again in 12.5s. gsk-secret"}}
    exc.response = SimpleNamespace(headers={"retry-after": "13", "authorization": "gsk-secret"})
    details = evaluation.limit_details(exc)
    assert details == {"code": "rate_limit_exceeded", "scope": "tokens_per_minute", "retry_after_s": 13.0}
    assert "gsk-secret" not in json.dumps(details)


def test_continuation_can_use_slower_pacing_without_shortening_minimum(tmp_path):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    evaluation.run_evaluation([QUESTION], FakeRetriever(), FakeLLM(["L2 [1]", verdict()]),
                              tmp_path / "slow.json", interval_s=60, clock=lambda: now[0], sleep=sleep)
    assert sleeps == [60]
    with pytest.raises(ValueError, match="20"):
        evaluation.run_evaluation([QUESTION], FakeRetriever(), FakeLLM([]), tmp_path / "fast.json", interval_s=19)


@pytest.mark.parametrize("body,headers,scope", [
    ({"detail": "session limit reached — cooldown 10 min"}, {"retry-after": "600"}, "session"),
    ({"detail": "Rate limit exceeded ... Limit type: max_parallel_requests"}, {}, "parallel"),
    ({"detail": "Rate limit exceeded ... Limit type: rpm"}, {}, "requests_per_minute"),
    ({"detail": "week limit reached — reset in 1h"}, {"x-window": "week"}, "week"),
])
def test_neuraldeep_limit_details_are_safe_and_recognizable(body, headers, scope):
    exc = RuntimeError()
    exc.body = body
    exc.response = SimpleNamespace(headers=headers)
    assert evaluation.limit_details(exc)["scope"] == scope


def test_resume_does_not_reuse_an_empty_answer(tmp_path):
    previous, _, _, _ = execute(tmp_path, [" "])
    llm, retriever = FakeLLM(["L2 [1]", verdict()]), FakeRetriever()
    report = evaluation.run_evaluation([QUESTION], retriever, llm, tmp_path / "continued.json",
                                      previous=previous, call_budget=2, sleep=lambda _: None)
    assert report["complete"] and len(llm.prompts) == 2 and len(retriever.questions) == 1


def test_resume_waits_after_the_previous_runs_last_call(tmp_path):
    limit = RuntimeError()
    limit.status_code = 429
    previous, _, _, _ = execute(tmp_path, ["L2 [1]", limit])
    previous["calls"][-1]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    sleeps = []
    evaluation.run_evaluation([QUESTION], FakeRetriever(), FakeLLM([verdict()]), tmp_path / "continued.json",
                              previous=previous, call_budget=1, sleep=sleeps.append)
    assert len(sleeps) == 1 and 19 <= sleeps[0] <= 20


def test_index_is_verified_and_pinned_before_alias_can_change(tmp_path):
    from unittest.mock import Mock, patch
    alias = SimpleNamespace(alias_name="sklearn_docs", collection_name="physical")
    client = Mock()
    client.get_aliases.return_value.aliases = [alias]
    retriever = SimpleNamespace(client=client, collection_name="sklearn_docs")
    with patch("app.scripts.index_corpus.corpus_collection_name", return_value="physical"), patch(
        "app.scripts.index_corpus.load_chunks", return_value=[DOC]
    ), patch("app.scripts.index_corpus.verify_collection") as verify:
        assert evaluation.pin_collection(retriever, tmp_path / "corpus.jsonl", "sklearn_docs") == "physical"
    alias.collection_name = "another-index"
    assert retriever.collection_name == "physical"
    verify.assert_called_once_with(client, [DOC], "physical")


def test_wrong_corpus_is_rejected_before_retrieval(tmp_path):
    from unittest.mock import Mock, patch
    client = Mock()
    client.get_aliases.return_value.aliases = [SimpleNamespace(alias_name="sklearn_docs", collection_name="wrong")]
    retriever = SimpleNamespace(client=client, collection_name="sklearn_docs")
    with patch("app.scripts.index_corpus.corpus_collection_name", return_value="expected"), pytest.raises(
        ValueError, match="corpus"
    ):
        evaluation.pin_collection(retriever, tmp_path / "corpus.jsonl", "sklearn_docs")
    assert retriever.collection_name == "sklearn_docs"


def test_judge_gets_strict_output_schema_in_the_same_single_call(tmp_path):
    options = []

    class InspectLLM(FakeLLM):
        def invoke(self, prompt, **kwargs):
            options.append(kwargs)
            return super().invoke(prompt, **kwargs)

    llm = InspectLLM(["L2 [1]", verdict()])
    report = evaluation.run_evaluation([QUESTION], FakeRetriever(), llm, tmp_path / "report.json", sleep=lambda _: None)
    assert report["calls_attempted"] == 2 and options[0] == {}
    response_format = options[1]["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"]
    schema = response_format["json_schema"]["schema"]
    assert set(schema["required"]) == {*evaluation.METRICS, "refusal"}
    assert schema["properties"]["grounding"]["required"] == ["score", "explanation"]
