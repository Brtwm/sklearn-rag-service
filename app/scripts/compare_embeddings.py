"""Compare E5 small/large in the full local service without changing its alias."""

import argparse
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from qdrant_client import QdrantClient

from app.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.retrieval import index_signature
from app.scripts.compare_retrieval import (
    BASELINE_PATH, MINILM, REPEATS, changes, peak_memory_mib, percentile, validate_hits,
)
from app.scripts.eval_retrieval import CHUNKS_PATH, QUESTIONS_PATH, build_report, load_questions
from app.scripts.index_corpus import corpus_collection_name, load_chunks, verify_collection

SMALL = "intfloat/multilingual-e5-small"
LARGE = "intfloat/multilingual-e5-large"
REPORT_PATH = Path("notebooks/embedding_comparison_v1.json")
QUESTION_FIELDS = ("id", "question", "language", "category", "expected_urls")


def build_comparison(small: dict, large: dict) -> dict:
    """Require identical inputs/pipeline, allowing only model and vector size to differ."""
    for key in ("corpus_sha256", "questions_sha256", "top_k", "mode", "reranker_model", "reranker_revision"):
        if small[key] != large[key]:
            raise ValueError(f"Cannot compare different {key}")
    for group in ("index_signature", "runtime", "benchmark"):
        if group == "index_signature":
            keys = (small[group].keys() | large[group].keys()) - {"embedding_model", "embedding_dim"}
        elif group == "runtime":
            keys = small[group].keys() | large[group].keys()
        else:
            keys = ("repeats", "warmup_queries", "concurrency", "batch_size", "max_length")
        for key in keys:
            if small[group].get(key) != large[group].get(key):
                raise ValueError(f"Cannot compare different {group}.{key}")
    if [[row[key] for key in QUESTION_FIELDS] for row in small["questions"]] != [
        [row[key] for key in QUESTION_FIELDS] for row in large["questions"]
    ]:
        raise ValueError("Cannot compare different question annotations")
    for report, model, dimension in ((small, SMALL, 384), (large, LARGE, 1024)):
        if report["embedding_model"] != model or report["index_signature"]["embedding_model"] != model or (
            report["index_signature"]["embedding_dim"] != dimension
        ):
            raise ValueError("Unexpected embedding model/dimension")
    old, new = small["summary"]["covered"], large["summary"]["covered"]
    gain = Decimal(str(new["mrr_at_4"])) - Decimal(str(old["mrr_at_4"]))
    checks = {
        "mrr_gain_at_least_0_05": gain >= Decimal("0.05"),
        "recall_not_lower": new["recall_at_4"] >= old["recall_at_4"],
        "p95_at_most_twice_small": large["benchmark"]["p95_ms"] <= 2 * small["benchmark"]["p95_ms"],
        "service_peak_below_6gb": large["benchmark"]["peak_memory_bytes"] < 6_000_000_000,
    }
    return {
        "corpus_sha256": small["corpus_sha256"], "questions_sha256": small["questions_sha256"],
        "variants": {"small": small, "large": large}, "changes": changes(small, large),
        "decision": {"passed": all(checks.values()), "checks": checks,
                     "mrr_gain": float(gain), "recommended_model": LARGE if all(checks.values()) else SMALL,
                     "thresholds": {"mrr_gain_min": 0.05, "p95_ratio_max": 2,
                                    "service_peak_bytes_exclusive_max": 6_000_000_000},
                     "applied": False},
    }


def check_inputs(corpus_hash: str, questions_hash: str) -> None:
    if hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest() != corpus_hash or (
        hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest() != questions_hash
    ):
        raise ValueError("Inputs changed between experiment processes")


def benchmark(args) -> dict:
    """Use the real app lifespan and retriever, with an entirely local LLM stub."""
    check_inputs(args.corpus_sha256, args.questions_sha256)
    import torch
    from fastapi.testclient import TestClient
    from langchain_core.runnables import RunnableLambda
    from app.rag.reranker import BATCH_SIZE, MAX_LENGTH

    corpus = {doc.id: doc for doc in load_chunks(CHUNKS_PATH)}
    questions = load_questions(QUESTIONS_PATH, {doc.metadata["source"] for doc in corpus.values()})
    runs, latencies = [], {q["id"]: [] for q in questions}
    with patch("app.rag.chain.get_llm", return_value=RunnableLambda(lambda _: "local evaluation stub")), patch(
        "app.rag.chain.get_embeddings", side_effect=lambda: get_embeddings(device="cpu"),
    ):
        import app.main as service
        started = time.perf_counter()
        with TestClient(service.app) as api:
            startup_s = time.perf_counter() - started
            if api.get("/ready").status_code != 200:
                raise RuntimeError("Full service failed readiness check")
            search = service._retriever
            if search.collection_name != args.collection:
                raise RuntimeError("Service uses a different collection")
            for language in ("en", "ru"):
                search.search_with_score(next(q["question"] for q in questions if q["language"] == language), 4)
            for repeat in range(REPEATS):
                measured_hits = {}

                def measured(text):
                    started = time.perf_counter()
                    hits = search.search_with_score(text, 4)
                    duration_ms = (time.perf_counter() - started) * 1000
                    validate_hits(hits, corpus)
                    measured_hits[text] = (hits, duration_ms)
                    return hits

                report = build_report(questions, measured, args.corpus_sha256, args.collection, settings.embedding_model)
                for row in report["questions"]:
                    hits, duration = measured_hits[row["question"]]
                    latencies[row["id"]].append(duration)
                    for hit, (doc, _) in zip(row["top_4"], hits, strict=True):
                        hit["id"] = doc.id
                runs.append(report)
                print(f"{settings.embedding_model}: pass {repeat + 1}/{REPEATS} complete", flush=True)
            embeddings, ranker = search.embeddings, search.reranker
            peak_mib = peak_memory_mib()
            report = runs[0]
            for row in report["questions"]:
                row["latency_ms_runs"] = latencies[row["id"]]
            durations = [v for values in latencies.values() for v in values]
            report.update({
                "questions_sha256": args.questions_sha256, "index_signature": index_signature(),
                "mode": "hybrid_rerank", "reranker_model": settings.reranker_model,
                "reranker_revision": getattr(ranker.model.model.config, "_commit_hash", None),
                "embedding_revision": getattr(embeddings._client._first_module().auto_model.config, "_commit_hash", None),
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count(),
                            "torch_threads": torch.get_num_threads(), "device": str(embeddings._client.device),
                            "versions": {name: importlib.metadata.version(name) for name in (
                                "torch", "sentence-transformers", "qdrant-client", "langchain-huggingface",
                                "fastapi", "gradio", "langchain-core")}},
                "benchmark": {"repeats": REPEATS, "warmup_queries": 2, "concurrency": 1,
                              "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
                              "service_startup_s": startup_s, "load_includes_download_if_not_cached": True,
                              "n_timed_queries": len(durations), "p50_ms": percentile(durations, 50),
                              "p95_ms": percentile(durations, 95), "peak_memory_mib": peak_mib,
                              "peak_memory_bytes": round(peak_mib * 1024 * 1024),
                              "memory_method": "OS peak full service RSS/working set through measured queries; "
                                               "includes FastAPI/Gradio, TestClient, E5, MiniLM and startup; "
                                               "excludes Qdrant/indexing/other workers; local LLM stub",
                              "summaries_by_run": [run["summary"] for run in runs],
                              "ranking_stable": all(
                                  [[hit["id"] for hit in row["top_4"]] for row in run["questions"]] ==
                                  [[hit["id"] for hit in row["top_4"]] for row in report["questions"]]
                                  for run in runs),
                              "runs": [{"summary": run["summary"], "questions": run["questions"]} for run in runs]},
            })
            search.client.close()
    return report


def run_experiment() -> dict:
    corpus_hash = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    questions_hash = hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if baseline["corpus_sha256"] != corpus_hash or [
        {key: row[key] for key in QUESTION_FIELDS} for row in baseline["questions"]
    ] != questions:
        raise ValueError("Corpus/questions differ from stage-2 baseline")
    if settings.embedding_model != SMALL or settings.embedding_dim != 384:
        raise ValueError("Run the experiment with the existing small/384 configuration")
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)
    try:
        aliases_before = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
        small_collection = aliases_before.get(settings.collection_name)
        if small_collection != corpus_collection_name(CHUNKS_PATH, settings.collection_name):
            raise ValueError("Active alias differs from expected small index")
        verify_collection(client, load_chunks(CHUNKS_PATH), small_collection)
        with tempfile.TemporaryDirectory(prefix="sklearn-rag-embeddings-") as directory:
            reports = []
            for label, model, dimension in (("small", SMALL, 384), ("large", LARGE, 1024)):
                check_inputs(corpus_hash, questions_hash)
                env = {**os.environ, "EMBEDDING_MODEL": model, "EMBEDDING_DIM": str(dimension),
                       "RETRIEVAL_MODE": "hybrid_rerank", "RERANKER_MODEL": MINILM, "TOP_K": "4",
                       "PYTHONIOENCODING": "utf-8"}
                output = Path(directory) / f"{label}.json"
                if label == "large":
                    subprocess.run([sys.executable, "-m", "app.scripts.index_corpus", "--no-activate", "--device", "cpu"],
                                   check=True, env=env)
                    # Resolve the deterministic name in a child with large's settings.
                    name_output = Path(directory) / "collection.json"
                    subprocess.run([sys.executable, "-m", "app.scripts.compare_embeddings", "--worker", "resolve",
                                    "--output", str(name_output)], check=True, env=env)
                    collection = json.loads(name_output.read_text(encoding="utf-8"))["collection"]
                else:
                    collection = small_collection
                subprocess.run([sys.executable, "-m", "app.scripts.compare_embeddings", "--worker", "benchmark",
                                "--collection", collection, "--corpus-sha256", corpus_hash,
                                "--questions-sha256", questions_hash, "--output", str(output)],
                               check=True, env={**env, "COLLECTION_NAME": collection})
                reports.append(json.loads(output.read_text(encoding="utf-8")))
            check_inputs(corpus_hash, questions_hash)
            result = build_comparison(*reports)
        return {**result, "active_alias": {"name": settings.collection_name, "collection": small_collection,
                                            "unchanged": True}}
    finally:
        try:
            aliases_after = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
            if "aliases_before" in locals() and aliases_after != aliases_before:
                raise RuntimeError("Qdrant aliases changed during experiment; report not saved")
        finally:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--worker", choices=("resolve", "benchmark"), help=argparse.SUPPRESS)
    parser.add_argument("--collection", help=argparse.SUPPRESS)
    parser.add_argument("--corpus-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--questions-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == "resolve":
        result = {"collection": corpus_collection_name(CHUNKS_PATH, settings.collection_name)}
    elif args.worker == "benchmark":
        result = benchmark(args)
    else:
        result = run_experiment()
    # Replace only after the complete experiment and successful JSON serialization.
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)
    if not args.worker:
        print(f"Saved comparison to {args.output}")
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
