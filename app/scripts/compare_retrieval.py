"""Compare dense, RRF and cross-encoder search in isolated processes."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from qdrant_client import QdrantClient

from app.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.reranker import BATCH_SIZE, MAX_LENGTH, CrossEncoderReranker
from app.rag.retrieval import CorpusRetriever, index_signature, verify_index_schema
from app.scripts.eval_retrieval import CHUNKS_PATH, QUESTIONS_PATH, build_report, load_questions
from app.scripts.index_corpus import corpus_collection_name, load_chunks

MINILM = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BGE = "BAAI/bge-reranker-v2-m3"
REPEATS = 3
REPORT_PATH = Path("notebooks/retrieval_comparison_v1.json")
BASELINE_PATH = Path("notebooks/retrieval_baseline_v1.json")


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def peak_memory_mib() -> float:
    """OS peak RSS/working set for this worker, including model initialization."""
    if os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / (1024 * 1024)


def validate_hits(hits: list, corpus: dict) -> None:
    ids = [doc.id for doc, _ in hits]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs in retrieval")
    for doc, score in hits:
        original = corpus.get(doc.id)
        if original is None or original.page_content != doc.page_content or (
            original.metadata != doc.metadata or not math.isfinite(score)
        ):
            raise ValueError("Retrieved ID, text, metadata or score differs from corpus")


def changes(before: dict, after: dict) -> dict:
    previous = {row["id"]: row for row in before["questions"]}
    if set(previous) != {row["id"] for row in after["questions"]}:
        raise ValueError("Cannot compare different question IDs")
    rows = []
    for row in after["questions"]:
        old = previous[row["id"]]
        if row["metrics"] is None:
            continue
        rows.append({
            "id": row["id"], "language": row["language"], "category": row["category"],
            "recall_delta": row["metrics"]["recall_at_4"] - old["metrics"]["recall_at_4"],
            "mrr_delta": row["metrics"]["mrr_at_4"] - old["metrics"]["mrr_at_4"],
            "complete_delta": int(row["metrics"]["complete_at_4"]) - int(old["metrics"]["complete_at_4"]),
            "before_top_4": old.get("top_4", []), "after_top_4": row.get("top_4", []),
        })
    result = {"questions": rows}
    for metric in ("recall", "mrr"):
        for label, sign in (("wins", 1), ("losses", -1)):
            result[f"{metric}_{label}"] = [row["id"] for row in rows if sign * row[f"{metric}_delta"] > 0]
    return result


def acceptance(dense: dict, candidate: dict) -> dict:
    old, new = dense["summary"], candidate["summary"]
    previous = {row["id"]: row for row in dense["questions"]}
    difficult_wins = [row["id"] for row in changes(dense, candidate)["questions"] if (
        row["category"] == "two_part" or previous[row["id"]]["metrics"]["recall_at_4"] < 1
    ) and (row["recall_delta"] > 0 or row["mrr_delta"] > 0)]
    checks = {
        "recall_not_lower": new["covered"]["recall_at_4"] >= old["covered"]["recall_at_4"],
        "mrr_improved": new["covered"]["mrr_at_4"] > old["covered"]["mrr_at_4"],
        "ru_recall_not_lower": new["by_language"]["ru"]["recall_at_4"] >= old["by_language"]["ru"]["recall_at_4"],
        "ru_mrr_not_lower": new["by_language"]["ru"]["mrr_at_4"] >= old["by_language"]["ru"]["mrr_at_4"],
        "hard_cases_improved": bool(difficult_wins),
    }
    return {"passed": all(checks.values()), "checks": checks, "difficult_wins": difficult_wins}


def needs_bge(dense: dict, hybrid: dict, mini: dict) -> bool:
    for baseline in (dense, hybrid):
        for group in ("covered", "ru"):
            old = baseline["summary"]["covered"] if group == "covered" else baseline["summary"]["by_language"]["ru"]
            new = mini["summary"]["covered"] if group == "covered" else mini["summary"]["by_language"]["ru"]
            if any(new[key] < old[key] for key in ("recall_at_4", "mrr_at_4")):
                return True
    return not acceptance(dense, mini)["passed"]


def choose_model(dense: dict, hybrid: dict, mini: dict, bge: dict | None) -> dict:
    for report in (mini, bge):
        if report is not None and acceptance(dense, report)["passed"]:
            return {"passed": True, "mode": "hybrid_rerank", "model": report["reranker_model"],
                    "acceptance": acceptance(dense, report)}
    return {"passed": False, "mode": "hybrid", "model": None,
            "reason": "Neither reranker passed; stage 3 requires further tuning"}


def build_comparison(reports: list[dict]) -> dict:
    dense = reports[0]
    shared = ("corpus_sha256", "questions_sha256", "collection", "index_signature", "embedding_model", "top_k")
    for report in reports[1:]:
        for key in shared:
            if report[key] != dense[key]:
                raise ValueError(f"Cannot compare different {key}")
    variants = {}
    for report in reports:
        key = report["mode"]
        if key == "hybrid_rerank":
            key += "_bge" if report["reranker_model"] == BGE else "_minilm"
        variants[key] = report
    comparisons = {f"{key}_vs_dense": changes(dense, report)
                   for key, report in variants.items() if key != "dense"}
    for key, report in variants.items():
        if key.startswith("hybrid_rerank") and "hybrid" in variants:
            comparisons[f"{key}_vs_hybrid"] = changes(variants["hybrid"], report)
    result = {key: dense[key] for key in shared}
    result.update({"variants": variants, "comparisons": comparisons})
    if "hybrid_rerank_minilm" in variants:
        result["decision"] = choose_model(dense, variants["hybrid"], variants["hybrid_rerank_minilm"], variants.get("hybrid_rerank_bge"))
    return result


def benchmark(args) -> dict:
    import torch
    corpus_bytes = CHUNKS_PATH.read_bytes()
    question_bytes = QUESTIONS_PATH.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != args.corpus_sha256 or (
        hashlib.sha256(question_bytes).hexdigest() != args.questions_sha256
    ):
        raise ValueError("Inputs changed between benchmark processes")
    corpus = {doc.id: doc for doc in load_chunks(CHUNKS_PATH)}
    questions = load_questions(QUESTIONS_PATH, {doc.metadata["source"] for doc in corpus.values()})
    client = QdrantClient(url=settings.qdrant_url, trust_env=False, cloud_inference=True)
    try:
        verify_index_schema(client, args.collection)
        started = time.perf_counter()
        embeddings = get_embeddings()
        embedding_load_s = time.perf_counter() - started
        started = time.perf_counter()
        ranker = CrossEncoderReranker(args.reranker) if args.worker == "hybrid_rerank" else None
        reranker_load_s = time.perf_counter() - started if ranker else 0.0
        search = CorpusRetriever(client, args.collection, embeddings, args.worker, 4, reranker=ranker)
        limit = 40 if args.worker == "hybrid" else 4
        for language in ("en", "ru"):
            search.search_with_score(next(q["question"] for q in questions if q["language"] == language), limit)
        runs, latencies = [], {q["id"]: [] for q in questions}
        for repeat in range(REPEATS):
            measured_hits = {}

            def measured(text):
                started = time.perf_counter()
                hits = search.search_with_score(text, limit)
                duration_ms = (time.perf_counter() - started) * 1000
                validate_hits(hits, corpus)
                measured_hits[text] = (hits[:4], duration_ms)
                return hits[:4]

            report = build_report(questions, measured, args.corpus_sha256, args.collection, settings.embedding_model)
            for row in report["questions"]:
                hits, duration = measured_hits[row["question"]]
                latencies[row["id"]].append(duration)
                for hit, (doc, _) in zip(row["top_4"], hits, strict=True):
                    hit["id"] = doc.id
            runs.append(report)
            print(f"{args.worker} {args.reranker if ranker else ''}: pass {repeat + 1}/{REPEATS} complete", flush=True)
        report = runs[0]
        for row in report["questions"]:
            row["latency_ms_runs"] = latencies[row["id"]]
        all_latencies = [value for values in latencies.values() for value in values]
        report.update({
            "questions_sha256": args.questions_sha256, "index_signature": index_signature(),
            "mode": args.worker, "reranker_model": args.reranker if ranker else None,
            "reranker_revision": getattr(ranker.model.model.config, "_commit_hash", None) if ranker else None,
            "embedding_revision": getattr(embeddings._client._first_module().auto_model.config, "_commit_hash", None),
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(),
                        "device": str(embeddings._client.device),
                        "versions": {name: importlib.metadata.version(name) for name in (
                            "torch", "sentence-transformers", "qdrant-client", "langchain-huggingface")}},
            "benchmark": {"repeats": REPEATS, "warmup_queries": 2, "concurrency": 1,
                          "batch_size": BATCH_SIZE if ranker else None, "max_length": MAX_LENGTH if ranker else None,
                          "embedding_load_s": embedding_load_s, "reranker_load_s": reranker_load_s,
                          "load_includes_download_if_not_cached": True,
                          "n_timed_queries": len(all_latencies), "p50_ms": percentile(all_latencies, 50),
                          "p95_ms": percentile(all_latencies, 95), "peak_memory_mib": peak_memory_mib(),
                          "memory_method": "OS peak worker RSS/working set, includes loading; excludes Qdrant",
                          "summaries_by_run": [run["summary"] for run in runs],
                          "ranking_stable": all(
                              [[hit["id"] for hit in row["top_4"]] for row in run["questions"]]
                              == [[hit["id"] for hit in row["top_4"]] for row in report["questions"]]
                              for run in runs)},
        })
        return report
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--worker", choices=("dense", "hybrid", "hybrid_rerank"), help=argparse.SUPPRESS)
    parser.add_argument("--collection", help=argparse.SUPPRESS)
    parser.add_argument("--corpus-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--questions-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--reranker", default=MINILM, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = benchmark(args)
    else:
        corpus_hash = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
        questions_hash = hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest()
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        if baseline["corpus_sha256"] != corpus_hash or [
            {key: row[key] for key in ("id", "question", "language", "category", "expected_urls")}
            for row in baseline["questions"]
        ] != questions:
            raise ValueError("Corpus/questions differ from stage-2 baseline")
        client = QdrantClient(url=settings.qdrant_url, trust_env=False)
        try:
            aliases = {alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases}
            physical = aliases.get(settings.collection_name)
            if physical != corpus_collection_name(CHUNKS_PATH, settings.collection_name):
                raise ValueError("Active alias differs from the expected corpus/index; reindex")
            verify_index_schema(client, physical)
        finally:
            client.close()
        with tempfile.TemporaryDirectory(prefix="sklearn-rag-comparison-") as directory:
            def worker(mode, model=MINILM):
                output = Path(directory) / f"{mode}_{'bge' if model == BGE else 'mini'}.json"
                command = [sys.executable, "-m", "app.scripts.compare_retrieval", "--worker", mode,
                           "--collection", physical, "--corpus-sha256", corpus_hash,
                           "--questions-sha256", questions_hash, "--reranker", model, "--output", str(output)]
                subprocess.run(command, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                return json.loads(output.read_text(encoding="utf-8"))

            dense, hybrid, mini = worker("dense"), worker("hybrid"), worker("hybrid_rerank")
            reports = [dense, hybrid, mini]
            if needs_bge(dense, hybrid, mini):
                print("MiniLM requires BGE comparison", flush=True)
                reports.append(worker("hybrid_rerank", BGE))
            result = build_comparison(reports)
            result["historical_dense_baseline"] = {"path": str(BASELINE_PATH), "summary": baseline["summary"]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.worker:
        print(f"Saved comparison to {args.output}")
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
