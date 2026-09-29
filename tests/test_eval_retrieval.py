import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.scripts import eval_retrieval


RIDGE = "https://scikit-learn.org/1.9/modules/linear_model.html#regression"
LASSO = "https://scikit-learn.org/1.9/modules/linear_model.html#lasso"
TREE = "https://scikit-learn.org/1.9/modules/tree.html#classification"


def test_metrics_use_distinct_sections_and_original_ranks() -> None:
    hits = [RIDGE, RIDGE, TREE, LASSO]

    result = eval_retrieval.section_metrics([TREE, LASSO], hits)

    assert result == {"recall_at_4": 1.0, "mrr_at_4": 1 / 3, "complete_at_4": True}
    assert eval_retrieval.section_metrics([LASSO], hits) == {
        "recall_at_4": 1.0, "mrr_at_4": 0.25, "complete_at_4": True,
    }
    assert eval_retrieval.section_metrics([TREE, LASSO], hits[:3]) == {
        "recall_at_4": 0.5, "mrr_at_4": 1 / 3, "complete_at_4": False,
    }
    assert eval_retrieval.section_metrics([RIDGE], [RIDGE, RIDGE]) == {
        "recall_at_4": 1.0, "mrr_at_4": 1.0, "complete_at_4": True,
    }
    assert eval_retrieval.section_metrics([TREE], [RIDGE] * 4 + [TREE]) == {
        "recall_at_4": 0.0, "mrr_at_4": 0.0, "complete_at_4": False,
    }


def test_report_excludes_out_of_corpus_from_metrics_and_keeps_top_four_order() -> None:
    questions = [
        {"id": "en-01", "language": "en", "category": "single", "question": "Ridge?", "expected_urls": [RIDGE]},
        {"id": "ru-01", "language": "ru", "category": "two_part", "question": "Два раздела?", "expected_urls": [TREE, LASSO]},
        {"id": "ooc-01", "language": "en", "category": "out_of_corpus", "question": "Weather?", "expected_urls": []},
    ]
    results = {
        "Ridge?": [RIDGE, TREE, LASSO, RIDGE, TREE],
        "Два раздела?": [RIDGE, TREE, RIDGE, LASSO],
        "Weather?": [TREE, RIDGE, LASSO, TREE],
    }

    def search(question: str):
        return [(Document(page_content="text", metadata={"source": url}), 0.9 - i / 10)
                for i, url in enumerate(results[question])]

    report = eval_retrieval.build_report(
        questions, search, "abc123", "sklearn_docs_abc123", "intfloat/multilingual-e5-small",
    )

    assert report["corpus_sha256"] == "abc123"
    assert report["collection"] == "sklearn_docs_abc123"
    assert report["top_k"] == 4
    assert report["summary"]["covered"]["n_questions"] == 2
    assert report["summary"]["covered"]["recall_at_4"] == 1.0
    assert report["summary"]["covered"]["mrr_at_4"] == 0.75
    assert report["summary"]["by_category"]["two_part"]["complete_at_4"] == 1.0
    assert report["summary"]["out_of_corpus"]["n_questions"] == 1
    assert "recall_at_4" not in report["summary"]["out_of_corpus"]
    assert [hit["url"] for hit in report["questions"][0]["top_4"]] == results["Ridge?"][:4]
    assert [hit["rank"] for hit in report["questions"][0]["top_4"]] == [1, 2, 3, 4]
    assert report["questions"][2]["metrics"] is None
    reordered = eval_retrieval.build_report(
        [questions[1], questions[0], questions[2]], search, "abc123", "sklearn_docs_abc123", "model",
    )
    assert reordered["summary"] == report["summary"]


def test_question_set_has_required_coverage() -> None:
    path = Path("data/eval/retrieval_questions_v1.json")
    sources = {url for q in json.loads(path.read_text(encoding="utf-8")) for url in q["expected_urls"]}
    questions = eval_retrieval.load_questions(path, sources)
    covered = [q for q in questions if q["category"] != "out_of_corpus"]
    ooc = [q for q in questions if q["category"] == "out_of_corpus"]

    assert len(questions) == 30
    assert len(covered) == 24
    assert len(ooc) == 6
    assert {lang: sum(q["language"] == lang for q in covered) for lang in ("en", "ru")} == {"en": 12, "ru": 12}
    assert sum(q["category"] == "two_part" for q in covered) >= 4
    assert all(len(q["expected_urls"]) == 2 for q in covered if q["category"] == "two_part")
    assert {url.split("/modules/")[1].split(".html")[0]
            for q in covered for url in q["expected_urls"]} == {
                "linear_model", "tree", "model_evaluation", "ensemble", "cross_validation",
                "preprocessing", "compose", "grid_search", "impute", "feature_selection",
            }


def test_expected_sections_exist_in_generated_corpus() -> None:
    corpus = Path("data/corpus_chunks.jsonl")
    if not corpus.exists():
        pytest.skip("Generated corpus is unavailable")
    with corpus.open(encoding="utf-8") as fh:
        sources = {json.loads(line)["metadata"]["source"] for line in fh}
    eval_retrieval.load_questions(Path("data/eval/retrieval_questions_v1.json"), sources)


def test_question_loader_rejects_missing_or_duplicate_expected_sections(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "one", "language": "en", "category": "two_part", "question": "A and B?", "expected_urls": [RIDGE, RIDGE]},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        eval_retrieval.load_questions(path, {RIDGE})

    path.write_text(json.dumps([
        {"id": "one", "language": "en", "category": "single", "question": "A?", "expected_urls": [LASSO]},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus"):
        eval_retrieval.load_questions(path, {RIDGE})


def test_main_does_not_replace_report_when_active_alias_has_another_corpus(
    tmp_path: Path, monkeypatch,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"metadata": {"source": RIDGE}}) + "\n", encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"id": "one", "language": "en", "category": "single", "question": "Ridge?", "expected_urls": [RIDGE]},
    ]), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text("previous baseline", encoding="utf-8")
    client = Mock()
    client.get_aliases.return_value = SimpleNamespace(aliases=[
        SimpleNamespace(alias_name=eval_retrieval.settings.collection_name, collection_name="old_corpus"),
    ])
    store = Mock(client=client)
    monkeypatch.setattr(eval_retrieval, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(eval_retrieval, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(eval_retrieval, "REPORT_PATH", report)
    monkeypatch.setattr(eval_retrieval, "get_vectorstore", lambda: store)

    with pytest.raises(RuntimeError, match="reindex"):
        eval_retrieval.main()

    store.similarity_search_with_score.assert_not_called()
    client.close.assert_called_once()
    assert report.read_text(encoding="utf-8") == "previous baseline"


def test_main_keeps_physical_collection_when_alias_switches_during_search(
    tmp_path: Path, monkeypatch,
) -> None:
    questions = json.loads(eval_retrieval.QUESTIONS_PATH.read_text(encoding="utf-8"))
    sources = {url for question in questions for url in question["expected_urls"]}
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(
        json.dumps({"metadata": {"source": source}}) + "\n" for source in sorted(sources)
    ), encoding="utf-8")
    alias = eval_retrieval.settings.collection_name
    physical = eval_retrieval.corpus_collection_name(corpus, alias)
    client = Mock()
    client.get_aliases.return_value = SimpleNamespace(aliases=[
        SimpleNamespace(alias_name=alias, collection_name=physical),
    ])
    store = SimpleNamespace(client=client, collection_name=alias)
    active_target = physical
    searched_collections = []

    def search(text: str, k: int):
        nonlocal active_target
        searched_collections.append(store.collection_name)
        target = active_target if store.collection_name == alias else store.collection_name
        active_target = "another_corpus"
        source = RIDGE if target == physical else TREE
        return [(Document(page_content="text", metadata={"source": source}), 0.9)]

    store.similarity_search_with_score = search
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(eval_retrieval, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(eval_retrieval, "REPORT_PATH", report_path)
    monkeypatch.setattr(eval_retrieval, "get_vectorstore", lambda: store)

    eval_retrieval.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(searched_collections) == {physical}
    assert report["collection"] == physical
    assert all(row["top_4"][0]["url"] == RIDGE for row in report["questions"])
    client.close.assert_called_once()
