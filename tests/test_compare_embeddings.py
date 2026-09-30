import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.scripts import compare_embeddings as comparison


def variant(model, dim, mrr=0.8):
    return {
        "corpus_sha256": "corpus", "questions_sha256": "questions", "top_k": 4,
        "mode": "hybrid_rerank", "reranker_model": comparison.MINILM,
        "reranker_revision": "fixed", "embedding_model": model,
        "index_signature": {"embedding_model": model, "embedding_dim": dim, "bm25": "same"},
        "collection": model, "runtime": {"device": "cpu", "torch_threads": 4},
        "benchmark": {"repeats": 3, "warmup_queries": 2, "concurrency": 1,
                      "batch_size": 8, "max_length": 512, "p95_ms": 100,
                      "peak_memory_bytes": 2_000_000_000},
        "summary": {"covered": {"recall_at_4": 0.9, "mrr_at_4": mrr}},
        "questions": [{"id": "one", "question": "Ridge?", "language": "en",
                       "category": "single", "expected_urls": ["url"],
                       "metrics": {"recall_at_4": 1, "mrr_at_4": 1, "complete_at_4": True}},
                      {"id": "ooc", "question": "outside?", "language": "ru",
                       "category": "out_of_corpus", "expected_urls": [], "metrics": None}],
    }


def reports():
    return variant(comparison.SMALL, 384), variant(comparison.LARGE, 1024, 0.85)


def test_gate_accepts_exact_mrr_and_latency_boundaries_and_excludes_ooc():
    small, large = reports()
    large["benchmark"]["p95_ms"] = 200
    result = comparison.build_comparison(small, large)
    assert result["decision"]["passed"]
    assert result["decision"]["recommended_model"] == comparison.LARGE
    assert len(result["changes"]["questions"]) == 1


@pytest.mark.parametrize("section,key,value,check", [
    ("summary", "mrr_at_4", 0.849, "mrr_gain_at_least_0_05"),
    ("summary", "recall_at_4", 0.899, "recall_not_lower"),
    ("benchmark", "p95_ms", 200.1, "p95_at_most_twice_small"),
    ("benchmark", "peak_memory_bytes", 6_000_000_000, "service_peak_below_6gb"),
])
def test_each_failed_criterion_keeps_small(section, key, value, check):
    small, large = reports()
    target = large[section]["covered"] if section == "summary" else large[section]
    target[key] = value
    decision = comparison.build_comparison(small, large)["decision"]
    assert not decision["passed"]
    assert not decision["checks"][check]
    assert decision["recommended_model"] == comparison.SMALL


@pytest.mark.parametrize("key", ["corpus_sha256", "questions_sha256", "top_k", "mode", "reranker_model", "reranker_revision"])
def test_comparison_rejects_mixed_pipeline(key):
    small, large = reports()
    large[key] = "different"
    with pytest.raises(ValueError, match="different"):
        comparison.build_comparison(small, large)


@pytest.mark.parametrize("section,key", [("index_signature", "bm25"), ("runtime", "device"),
                                         ("benchmark", "repeats")])
def test_comparison_rejects_different_index_and_measurement_settings(section, key):
    small, large = reports()
    large[section][key] = "different"
    with pytest.raises(ValueError, match="different"):
        comparison.build_comparison(small, large)


def test_comparison_rejects_changed_question_annotations():
    small, large = reports()
    large["questions"][0]["expected_urls"] = ["other"]
    with pytest.raises(ValueError, match="different"):
        comparison.build_comparison(small, large)


def test_worker_rejects_changed_inputs_before_loading_service(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    questions = tmp_path / "questions"
    corpus.write_text("changed")
    questions.write_text("[]")
    monkeypatch.setattr(comparison, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(comparison, "QUESTIONS_PATH", questions)
    with pytest.raises(ValueError, match="Inputs changed"):
        comparison.benchmark(SimpleNamespace(corpus_sha256="old", questions_sha256="old"))


def test_main_preserves_report_on_experiment_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("previous")
    monkeypatch.setattr(comparison.sys, "argv", ["compare", "--output", str(output)])
    with patch.object(comparison, "run_experiment", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            comparison.main()
    assert output.read_text() == "previous"


def experiment_fixture(tmp_path, monkeypatch):
    corpus, questions, baseline = (tmp_path / name for name in ("corpus", "questions", "baseline"))
    corpus.write_text("corpus")
    questions.write_text("[]")
    baseline.write_text(json.dumps({"corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(), "questions": []}))
    for name, path in (("CHUNKS_PATH", corpus), ("QUESTIONS_PATH", questions), ("BASELINE_PATH", baseline)):
        monkeypatch.setattr(comparison, name, path)
    monkeypatch.setattr(comparison, "load_chunks", lambda _: [])
    monkeypatch.setattr(comparison, "verify_collection", lambda *args: None)
    monkeypatch.setattr(comparison, "corpus_collection_name", lambda *args: "small")
    return corpus


def test_experiment_uses_isolated_cpu_workers_and_does_not_activate_alias(tmp_path, monkeypatch):
    experiment_fixture(tmp_path, monkeypatch)
    calls = []

    def subprocess_run(command, *, check, env):
        calls.append((command, env))
        if command[2] == "app.scripts.index_corpus":
            assert "--no-activate" in command and "--device" in command
        else:
            output = command[command.index("--output") + 1]
            result = {"collection": "large"} if "resolve" in command else (
                reports()[0] if env["EMBEDDING_MODEL"] == comparison.SMALL else reports()[1]
            )
            from pathlib import Path
            Path(output).write_text(json.dumps(result))

    monkeypatch.setattr(comparison.subprocess, "run", subprocess_run)
    with patch.object(comparison, "QdrantClient") as cls:
        cls.return_value.get_aliases.return_value.aliases = [SimpleNamespace(alias_name="sklearn_docs", collection_name="small")]
        result = comparison.run_experiment()
        cls.return_value.update_collection_aliases.assert_not_called()
    assert result["active_alias"]["unchanged"]
    workers = [(command, env) for command, env in calls if "benchmark" in command]
    assert [env["COLLECTION_NAME"] for _, env in workers] == ["small", "large"]
    assert all(env["RETRIEVAL_MODE"] == "hybrid_rerank" for _, env in calls)


def test_experiment_detects_alias_change_on_worker_failure(tmp_path, monkeypatch):
    experiment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(comparison.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")))
    with patch.object(comparison, "QdrantClient") as cls:
        cls.return_value.get_aliases.side_effect = [
            SimpleNamespace(aliases=[SimpleNamespace(alias_name="sklearn_docs", collection_name=name)])
            for name in ("small", "other")
        ]
        with pytest.raises(RuntimeError, match="aliases changed"):
            comparison.run_experiment()
        cls.return_value.close.assert_called_once()


def test_full_service_worker_measures_real_lifespan_without_calling_llm(tmp_path, monkeypatch):
    from langchain_core.documents import Document
    import app.main as service
    from app.rag import chain

    docs = [Document(id=str(i), page_content=f"text {i}", metadata={"source": f"https://docs/#s{i}"}) for i in (1, 2)]
    questions = [
        {"id": "en", "question": "Ridge?", "language": "en", "category": "single", "expected_urls": [docs[0].metadata["source"]]},
        {"id": "ru", "question": "Что такое Ridge?", "language": "ru", "category": "single", "expected_urls": [docs[0].metadata["source"]]},
        {"id": "two", "question": "Ridge and Lasso?", "language": "en", "category": "two_part", "expected_urls": [d.metadata["source"] for d in docs]},
        {"id": "ooc", "question": "Other?", "language": "ru", "category": "out_of_corpus", "expected_urls": []},
    ]
    corpus_path, questions_path = tmp_path / "corpus", tmp_path / "questions"
    corpus_path.write_text("corpus")
    questions_path.write_text(json.dumps(questions))
    monkeypatch.setattr(comparison, "CHUNKS_PATH", corpus_path)
    monkeypatch.setattr(comparison, "QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(comparison, "load_chunks", lambda _: docs)
    monkeypatch.setattr(service, "_chain", None)
    monkeypatch.setattr(service, "_retriever", None)
    monkeypatch.setattr(service, "index_available", lambda: True)
    from unittest.mock import MagicMock
    retriever = MagicMock(collection_name="physical")
    retriever.search_with_score.return_value = [(d, 0.5) for d in docs]
    retriever.embeddings._client.device = "cpu"
    retriever.embeddings._client._first_module.return_value.auto_model.config._commit_hash = "embedding-revision"
    retriever.reranker.model.model.config._commit_hash = "reranker-revision"
    monkeypatch.setattr(chain, "get_retriever", lambda: retriever)
    with patch.object(chain, "get_llm", side_effect=AssertionError("Real LLM must not be initialized")) as provider:
        report = comparison.benchmark(SimpleNamespace(
            corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            questions_sha256=hashlib.sha256(questions_path.read_bytes()).hexdigest(), collection="physical",
        ))
    provider.assert_not_called()
    assert retriever.search_with_score.call_count == 14  # two warmups + three runs of four questions
    assert report["benchmark"]["n_timed_queries"] == 12
    assert report["benchmark"]["peak_memory_bytes"] > 0
    assert report["summary"]["covered"]["n_questions"] == 3
    assert report["questions"][-1]["metrics"] is None
    assert len(report["benchmark"]["runs"]) == 3
    assert service._retriever is None  # lifespan shutdown ran
    json.dumps(report, allow_nan=False)
