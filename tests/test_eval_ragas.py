import asyncio
import json
from types import SimpleNamespace

import pytest

from app.scripts import eval_ragas as evaluation


ROW = {"id": "en-01", "question": "What penalty does Ridge use?", "answer": "Ridge uses L2 [1].",
       "expected_refusal": False, "sources": [{"number": 1, "content": "Ridge uses L2.", "metadata": {}}]}


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)

    def invoke(self, messages, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply, response_metadata={"finish_reason": "stop"}, usage_metadata=None)


class Embeddings:
    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def execute(tmp_path, replies, rows=None, budget=18):
    now, pauses = [0.0], []

    def sleep(seconds):
        pauses.append(seconds)
        now[0] += seconds

    output = tmp_path / "ragas.json"
    report = evaluation.run_evaluation(rows or [ROW], Replies(replies), Embeddings(), output,
                                       call_budget=budget, clock=lambda: now[0], sleep=sleep)
    return report, json.loads(output.read_text(encoding="utf-8")), pauses


def test_real_ragas_metrics_use_saved_answers_and_three_bounded_calls(tmp_path):
    report, saved, pauses = execute(tmp_path, [
        '{"statements": ["Ridge uses L2.", "Ridge is sparse."]}',
        '{"statements": [{"statement": "Ridge uses L2.", "reason": "Supported", "verdict": 1},'
        '{"statement": "Ridge is sparse.", "reason": "Absent", "verdict": 0}]}',
        '{"question": "What penalty does Ridge use?", "noncommittal": 0}',
    ])
    assert report["complete"] and saved["calls_attempted"] == 3
    assert saved["questions"][0]["metrics"]["faithfulness"]["value"] == 0.5
    assert saved["summary"]["in_corpus"]["answer_relevancy"] == {"mean": 1.0, "n": 1}
    assert pauses == [20.0, 20.0]
    assert saved["questions"][0]["answer"] == ROW["answer"]
    assert "Ridge uses L2." in saved["calls"][1]["prompt"]


def test_invalid_ragas_json_is_saved_without_repair_or_fake_zero(tmp_path):
    report, saved, _ = execute(tmp_path, ["not JSON", '{"question": "Ridge?", "noncommittal": 0}'])
    assert not report["complete"] and saved["calls_attempted"] == 2
    assert saved["questions"][0]["metrics"]["faithfulness"]["value"] is None
    assert saved["summary"]["in_corpus"]["faithfulness"] == {"mean": None, "n": 0}
    assert saved["calls"][0]["response"] == "not JSON"


def test_provider_limit_stops_all_metrics_and_preserves_partial_report(tmp_path):
    error = RuntimeError("secret exception detail must not be saved")
    error.status_code = 429
    report, saved, _ = execute(tmp_path, [error], [ROW, {**ROW, "id": "ru-01"}])
    assert not report["complete"] and saved["stop_reason"] == "provider_limit"
    assert saved["calls_attempted"] == 1
    assert saved["questions"][1]["metrics"]["faithfulness"]["status"] == "not_started"
    assert "secret exception" not in json.dumps(saved)


def test_budget_stops_before_an_extra_provider_call(tmp_path):
    report, saved, _ = execute(tmp_path, ['{"statements": ["Ridge uses L2."]}'], budget=1)
    assert not report["complete"] and saved["stop_reason"] == "call_budget"
    assert saved["calls_attempted"] == 1


def test_nonfinite_score_stays_unscored_and_refusal_is_separate(tmp_path):
    row = {**ROW, "id": "ooc-ru-01", "expected_refusal": True}
    report, saved, _ = execute(tmp_path, ['{"statements": []}', '{"question": "Ridge?", "noncommittal": 1}'], [row])
    assert not report["complete"]
    assert saved["questions"][0]["metrics"]["faithfulness"]["status"] == "undefined"
    assert saved["summary"]["out_of_corpus"]["answer_relevancy"] == {"mean": 0.0, "n": 1}
    assert saved["summary"]["in_corpus"]["answer_relevancy"] == {"mean": None, "n": 0}


@pytest.mark.parametrize("seconds", [19, float("nan"), float("inf")])
def test_invalid_interval_rejected_before_calls(tmp_path, seconds):
    with pytest.raises(ValueError, match="interval"):
        evaluation.run_evaluation([ROW], Replies([]), Embeddings(), tmp_path / "report.json", interval_s=seconds)


def test_question_embeddings_share_query_prefix():
    class LocalModel:
        def embed_query(self, text):
            return [float(len(text))]

        def embed_documents(self, texts):
            pytest.fail("RAGAS reconstructed questions must use the E5 query prefix")

    embeddings = evaluation.QuestionEmbeddings(LocalModel())
    assert embeddings.embed_documents(["abc", "de"]) == [[3.0], [2.0]]
    assert asyncio.run(embeddings.aembed_query("abcd")) == [4.0]
