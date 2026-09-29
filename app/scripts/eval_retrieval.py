"""Evaluate section-level dense retrieval without calling the LLM."""

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Callable

from langchain_core.documents import Document

from app.config import settings
from app.rag.chain import get_vectorstore
from app.scripts.index_corpus import corpus_collection_name

CHUNKS_PATH = Path("data/corpus_chunks.jsonl")
QUESTIONS_PATH = Path("data/eval/retrieval_questions_v1.json")
REPORT_PATH = Path("notebooks/retrieval_baseline_v1.json")
TOP_K = 4


def load_questions(path: Path, corpus_sources: set[str]) -> list[dict]:
    """Validate the hand-labelled questions against the exact local corpus."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question set must be a non-empty list")
    ids = set()
    for question in questions:
        category = question["category"]
        expected = question["expected_urls"]
        if question["id"] in ids:
            raise ValueError("Question IDs must be distinct")
        ids.add(question["id"])
        if question["language"] not in {"en", "ru"} or category not in {
            "single", "two_part", "out_of_corpus",
        } or not question["question"].strip():
            raise ValueError(f"Invalid question: {question['id']}")
        if not isinstance(expected, list) or len(expected) != len(set(expected)):
            raise ValueError("Expected sections must be distinct")
        if category == "out_of_corpus":
            if expected:
                raise ValueError("Out-of-corpus questions cannot have expected sections")
        elif not expected or (category == "two_part" and len(expected) != 2):
            raise ValueError("Covered questions need their expected sections")
        if any(url not in corpus_sources or "#" not in url for url in expected):
            raise ValueError("Expected section is absent from the corpus")
    return questions


def section_metrics(expected_urls: list[str], retrieved_urls: list[str]) -> dict:
    """Recall is the fraction of required sections found; MRR uses the first hit."""
    required = set(expected_urls)
    top_urls = retrieved_urls[:TOP_K]
    found = required.intersection(top_urls)
    first_rank = next((rank for rank, url in enumerate(top_urls, 1) if url in required), None)
    return {
        "recall_at_4": len(found) / len(required),
        "mrr_at_4": 1 / first_rank if first_rank else 0.0,
        "complete_at_4": len(found) == len(required),
    }


def _aggregate(rows: list[dict], include_complete: bool = False) -> dict:
    result = {
        "n_questions": len(rows),
        "recall_at_4": fmean(row["metrics"]["recall_at_4"] for row in rows),
        "mrr_at_4": fmean(row["metrics"]["mrr_at_4"] for row in rows),
    }
    if include_complete:
        result["complete_at_4"] = fmean(row["metrics"]["complete_at_4"] for row in rows)
    return result


def build_report(
    questions: list[dict],
    search: Callable[[str], list[tuple[Document, float]]],
    corpus_sha256: str,
    collection: str,
    embedding_model: str,
) -> dict:
    """Run top-4 retrieval in question order and calculate section metrics."""
    rows = []
    for question in questions:
        hits = [
            {"rank": rank, "url": doc.metadata.get("source"), "score": float(score)}
            for rank, (doc, score) in enumerate(search(question["question"])[:TOP_K], 1)
        ]
        metrics = None
        if question["category"] != "out_of_corpus":
            metrics = section_metrics(question["expected_urls"], [hit["url"] for hit in hits])
        rows.append({**question, "top_4": hits, "metrics": metrics})

    covered = [row for row in rows if row["metrics"] is not None]
    summary = {
        "covered": _aggregate(covered),
        "by_language": {
            language: _aggregate([row for row in covered if row["language"] == language])
            for language in ("en", "ru")
        },
        "by_category": {
            category: _aggregate(
                [row for row in covered if row["category"] == category],
                include_complete=category == "two_part",
            )
            for category in ("single", "two_part")
        },
        "out_of_corpus": {"n_questions": len(rows) - len(covered)},
    }
    return {
        "corpus_sha256": corpus_sha256,
        "collection": collection,
        "embedding_model": embedding_model,
        "top_k": TOP_K,
        "summary": summary,
        "questions": rows,
    }


def main() -> None:
    """Save a reproducible baseline for the active corpus and alias."""
    corpus_bytes = CHUNKS_PATH.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    corpus_sources = {
        json.loads(line)["metadata"]["source"]
        for line in corpus_bytes.decode("utf-8").splitlines()
    }
    questions = load_questions(QUESTIONS_PATH, corpus_sources)
    expected_collection = corpus_collection_name(CHUNKS_PATH, settings.collection_name)
    vectorstore = get_vectorstore()
    try:
        aliases = {alias.alias_name: alias.collection_name for alias in vectorstore.client.get_aliases().aliases}
        collection = aliases.get(settings.collection_name)
        if collection != expected_collection:
            raise RuntimeError(
                f"Active alias points to {collection}, expected {expected_collection}; reindex the corpus"
            )
        vectorstore.collection_name = collection
        report = build_report(
            questions,
            lambda text: vectorstore.similarity_search_with_score(text, k=TOP_K),
            corpus_sha256,
            collection,
            settings.embedding_model,
        )
    finally:
        vectorstore.client.close()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(report['questions'])} questions to {REPORT_PATH}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
