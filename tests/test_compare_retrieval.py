from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.scripts import compare_retrieval as comparison


def report(mode, recall=0.8, mrr=0.7, ru_recall=0.7, ru_mrr=0.6):
    return {
        "mode": mode, "corpus_sha256": "corpus", "questions_sha256": "questions",
        "collection": "physical", "index_signature": {"schema_version": 1},
        "embedding_model": "e5", "top_k": 4,
        "reranker_model": comparison.MINILM if mode == "hybrid_rerank" else None,
        "summary": {
            "covered": {"recall_at_4": recall, "mrr_at_4": mrr},
            "by_language": {"ru": {"recall_at_4": ru_recall, "mrr_at_4": ru_mrr}},
        },
        "questions": [{"id": "ru-hard", "language": "ru", "category": "two_part",
                       "metrics": {"recall_at_4": recall, "mrr_at_4": mrr, "complete_at_4": recall == 1}},
                      {"id": "ooc", "language": "ru", "category": "out_of_corpus", "metrics": None}],
    }


def test_percentiles_use_linear_interpolation() -> None:
    assert comparison.percentile([10, 20, 30, 40], 50) == 25
    assert comparison.percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)


@pytest.mark.parametrize("key", ["corpus_sha256", "questions_sha256", "collection", "index_signature", "embedding_model", "top_k"])
def test_comparison_rejects_mixed_inputs(key) -> None:
    dense, hybrid = report("dense"), report("hybrid")
    hybrid[key] = "different"
    with pytest.raises(ValueError, match="different"):
        comparison.build_comparison([dense, hybrid])


def test_comparison_lists_wins_and_losses_and_excludes_out_of_corpus() -> None:
    dense = report("dense")
    hybrid = report("hybrid", recall=0.9, mrr=0.65)
    result = comparison.build_comparison([dense, hybrid])
    changes = result["comparisons"]["hybrid_vs_dense"]
    assert changes["recall_wins"] == ["ru-hard"]
    assert changes["mrr_losses"] == ["ru-hard"]
    assert len(changes["questions"]) == 1
    assert changes["questions"][0]["id"] == "ru-hard"


def test_gate_requires_recall_ru_and_mrr_and_a_hard_case_improvement() -> None:
    dense = report("dense")
    good = report("hybrid_rerank", 0.9, 0.8, 0.8, 0.7)
    assert comparison.acceptance(dense, good)["passed"]
    bad = deepcopy(good)
    bad["summary"]["by_language"]["ru"]["mrr_at_4"] = 0.5
    assert not comparison.acceptance(dense, bad)["passed"]
    good["summary"]["covered"]["mrr_at_4"] = 0.7
    assert not comparison.acceptance(dense, good)["passed"]


def test_bge_is_required_when_minilm_regresses_against_hybrid_on_ru() -> None:
    dense = report("dense")
    hybrid = report("hybrid", 0.9, 0.8, 0.9, 0.8)
    mini = report("hybrid_rerank", 0.9, 0.8, 0.8, 0.7)
    assert comparison.needs_bge(dense, hybrid, mini)
    mini["summary"] = deepcopy(hybrid["summary"])
    assert not comparison.needs_bge(dense, hybrid, mini)


def test_hit_validation_checks_chunk_ids_payload_links_and_duplicates() -> None:
    doc = Document(id="one", page_content="Ridge", metadata={"source": "https://example.org/#ridge"})
    corpus = {doc.id: doc}
    comparison.validate_hits([(doc, 0.5)], corpus)
    with pytest.raises(ValueError, match="Duplicate"):
        comparison.validate_hits([(doc, 0.5), (doc, 0.4)], corpus)
    wrong = doc.model_copy(update={"metadata": {"source": "https://example.org/#lasso"}})
    with pytest.raises(ValueError, match="corpus"):
        comparison.validate_hits([(wrong, 0.5)], corpus)


def test_selection_keeps_minilm_if_it_passes_and_uses_bge_if_only_it_passes() -> None:
    dense = report("dense")
    hybrid = report("hybrid", 0.9, 0.8, 0.8, 0.7)
    mini = report("hybrid_rerank", 0.9, 0.85, 0.8, 0.7)
    bge = report("hybrid_rerank", 1.0, 0.95, 1.0, 0.9)
    bge["reranker_model"] = comparison.BGE
    assert comparison.choose_model(dense, hybrid, mini, bge)["model"] == comparison.MINILM
    mini["summary"]["by_language"]["ru"]["mrr_at_4"] = 0.5
    assert comparison.choose_model(dense, hybrid, mini, bge)["model"] == comparison.BGE
    bge["summary"]["covered"]["recall_at_4"] = 0.1
    assert comparison.choose_model(dense, hybrid, mini, bge)["passed"] is False


def test_worker_rejects_changed_inputs_before_connecting(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("changed", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(comparison, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(comparison, "QUESTIONS_PATH", questions)
    with patch("app.scripts.compare_retrieval.QdrantClient") as client:
        with pytest.raises(ValueError, match="Inputs changed"):
            comparison.benchmark(SimpleNamespace(corpus_sha256="old", questions_sha256="old"))
    client.assert_not_called()


def test_main_preserves_previous_report_on_baseline_mismatch(tmp_path, monkeypatch) -> None:
    import json
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("changed", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text("[]", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"corpus_sha256": "old", "questions": []}), encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(comparison, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(comparison, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(comparison, "BASELINE_PATH", baseline)
    monkeypatch.setattr(comparison.sys, "argv", ["compare", "--output", str(output)])
    with pytest.raises(ValueError, match="baseline"):
        comparison.main()
    assert output.read_text(encoding="utf-8") == "previous report"
