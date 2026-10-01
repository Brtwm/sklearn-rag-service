"""Exercise the real local service over HTTP with a provider-free LLM stub."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
from threading import Lock
import time
from unittest.mock import patch
from urllib.parse import unquote

import httpx

QUESTION_IDS = ("en-01", "en-03", "en-07", "en-11", "ru-01", "ru-06", "ru-07", "ru-11")
REPORT_PATH = Path("notebooks/service_smoke_v1.json")
REPEATS = 3


def stub_answer(prompt) -> str:
    """Echo precisely the prompt's question and context, without network access."""
    human = prompt.to_messages()[-1].content
    context, separator, question = human.removeprefix("Context:\n").rpartition("\n\nQuestion:\n")
    if not separator or not question.endswith("\n\nAnswer (with citations):"):
        raise ValueError("Unexpected generation prompt")
    return json.dumps({"question": question.removesuffix("\n\nAnswer (with citations):"),
                       "context": context}, ensure_ascii=False)


def validate_response(body: dict, question: dict, corpus: dict, expected_ids: list[str] | None = None) -> list[str]:
    answer = json.loads(body["answer"])
    if answer["question"] != question["question"]:
        raise ValueError("Response belongs to another question")
    sources, ids = [], []
    for number, passage in enumerate(answer["context"].split("\n\n---\n\n"), 1):
        heading, _, content = passage.partition("\n")
        prefix = f"[{number}] Source: "
        if not heading.startswith(prefix):
            raise ValueError("Invalid context citation order")
        url = heading.removeprefix(prefix)
        chunk_id = corpus.get((url, content))
        if chunk_id is None:
            raise ValueError("Context differs from the corpus")
        ids.append(chunk_id)
        sources.append({"url": url, "snippet": content[:200].strip()})
    if not 1 <= len(ids) <= 4 or len(ids) != len(set(ids)) or sources != body["sources"]:
        raise ValueError("Response sources differ from generation context or contain duplicates")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("Retrieved context differs from this question's serial reference")
    return ids


def run_batch(client, questions: list[dict], corpus: dict, concurrency: int, repeat: int,
              reference: dict | None = None) -> list[dict]:
    def request(question):
        started = time.perf_counter()
        row = {"id": question["id"], "language": question["language"], "repeat": repeat,
               "success": False, "status_code": None}
        try:
            response = client.post("/chat", json={"question": question["question"]})
            row["status_code"] = response.status_code
            response.raise_for_status()
            row["chunk_ids"] = validate_response(response.json(), question, corpus,
                                                (reference or {}).get(question["id"]))
            row["success"] = True
        except Exception as exc:
            # Never save raw exceptions, which may contain provider/configuration data.
            row["error"] = type(exc).__name__
        row["latency_ms"] = (time.perf_counter() - started) * 1000
        return row

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(request, questions))


def summarize(rows: list[dict]) -> dict:
    values = sorted(row["latency_ms"] for row in rows)

    def percentile(percent):
        if not values:
            return None
        position = (len(values) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    successful = sum(row["success"] for row in rows)
    return {"requests": len(rows), "successful": successful, "errors": len(rows) - successful,
            "p50_ms": percentile(50), "p95_ms": percentile(95)}


def ui_queue_passed(observations: dict, count: int) -> bool:
    return observations["peak_active"] == 1 and observations["started"] == observations["finished"] == count


def check_inputs(args) -> None:
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in (
        (args.corpus, args.corpus_sha256), (args.questions, args.questions_sha256),
    )):
        raise ValueError("Inputs changed between smoke processes")


def serve_worker(args) -> None:
    check_inputs(args)
    import torch
    import uvicorn
    from fastapi.routing import APIRoute
    from langchain_core.runnables import RunnableLambda
    from app.config import settings
    from app.rag.embeddings import get_embeddings
    from app.rag.retrieval import index_signature
    from app.rag.reranker import BATCH_SIZE, MAX_LENGTH
    from app.scripts.compare_retrieval import peak_memory_mib

    observations = {"active": 0, "peak_active": 0, "started": 0, "finished": 0}
    lock = Lock()
    with patch("app.rag.chain.get_llm", return_value=RunnableLambda(stub_answer)), patch(
        "app.rag.chain.get_embeddings", side_effect=lambda: get_embeddings(device="cpu"),
    ):
        import app.main as service
        original = service._respond_ui

        def observed_ui(message, history):
            with lock:
                observations["active"] += 1
                observations["started"] += 1
                observations["peak_active"] = max(observations["peak_active"], observations["active"])
            try:
                yield from original(message, history)
            finally:
                with lock:
                    observations["active"] -= 1
                    observations["finished"] += 1

        for fn in service.demo.fns.values():
            if fn.fn is original:
                fn.fn = observed_ui

        def runtime():
            search = service._retriever
            peak = peak_memory_mib()
            with lock:
                ui = dict(observations)
            return {
                "peak_memory_mib": peak, "peak_memory_bytes": round(peak * 1024 * 1024), "ui": ui,
                "python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(), "cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(), "device": str(search.embeddings._client.device),
                "collection": search.collection_name, "embedding_model": settings.embedding_model,
                "index_signature": index_signature(),
                "reranker_batch_size": BATCH_SIZE, "reranker_max_length": MAX_LENGTH,
                "embedding_revision": getattr(search.embeddings._client._first_module().auto_model.config,
                                              "_commit_hash", None),
                "reranker_model": settings.reranker_model,
                "reranker_revision": getattr(search.reranker.model.model.config, "_commit_hash", None),
                "versions": {name: importlib.metadata.version(name) for name in (
                    "fastapi", "gradio", "gradio-client", "torch", "sentence-transformers", "qdrant-client",
                )},
            }

        # This endpoint exists only in the disposable smoke server, before Gradio's root mount.
        service.app.router.routes.insert(0, APIRoute("/__smoke__/runtime", runtime, methods=["GET"]))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            args.address_file.write_text(json.dumps({"url": url}), encoding="utf-8")
            uvicorn.Server(uvicorn.Config(service.app, log_level="warning")).run(sockets=[listener])


@contextmanager
def worker_client(args, collection: str):
    with tempfile.TemporaryDirectory(prefix="sklearn-rag-smoke-") as directory:
        address = Path(directory) / "address.json"
        env = {**os.environ, "COLLECTION_NAME": collection, "LLM_API_KEY": "local-smoke-only",
               "LLM_BASE_URL": "http://127.0.0.1:1", "PYTHONIOENCODING": "utf-8"}
        command = [sys.executable, "-m", "app.scripts.smoke_service", "--worker",
                   "--address-file", str(address), "--corpus", str(args.corpus.resolve()),
                   "--questions", str(args.questions.resolve()), "--corpus-sha256", args.corpus_sha256,
                   "--questions-sha256", args.questions_sha256]
        with (Path(directory) / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 600
                while not address.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Smoke worker failed before HTTP startup")
                    time.sleep(0.2)
                url = json.loads(address.read_text(encoding="utf-8"))["url"]
                with httpx.Client(base_url=url, timeout=120, trust_env=False) as client:
                    while True:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError("Smoke worker did not become live")
                        try:
                            if client.get("/health", timeout=1).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.2)
                    yield client, url
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def check_ui(url: str, questions: list[dict], corpus: dict, reference: dict | None = None) -> dict:
    from gradio_client import Client

    sessions = [Client(url, verbose=False, httpx_kwargs={"trust_env": False},
                       _skip_components=False, analytics_enabled=False) for _ in questions]
    try:
        jobs = [session.submit(question["question"], [], api_name="/respond")
                for session, question in zip(sessions, questions, strict=True)]
        rows = []
        for question, job in zip(questions, jobs, strict=True):
            row = {"id": question["id"], "success": False}
            try:
                job.result(timeout=120)
                events = [event for event in job.outputs() if isinstance(event[0], list) and event[0]]
                last = events[-1]
                content = last[0][-1]["content"]
                if isinstance(content, list):
                    content = "".join(part["text"] for part in content if part["type"] == "text")
                answer = json.loads(content)
                passages = answer["context"].split("\n\n---\n\n")
                sources = []
                for passage in passages:
                    heading, _, text = passage.partition("\n")
                    sources.append({"url": heading.split("Source: ", 1)[1], "snippet": text[:200].strip()})
                row["chunk_ids"] = validate_response({"answer": content, "sources": sources}, question, corpus,
                                                    (reference or {}).get(question["id"]))
                headers = re.findall(r"\*\*\[(\d+)\]\*\* (.+)", last[3])
                if [int(n) for n, _ in headers] != list(range(1, len(sources) + 1)) or any(
                    f"(<{source['url']}>)" not in unquote(header)
                    for (_, header), source in zip(headers, sources, strict=True)
                ):
                    raise ValueError("UI sources differ from generation context")
                if "Завершено" not in last[2]:
                    raise ValueError("UI did not reach the completed state")
                final = job.outputs()[-1]
                if not final[1].get("interactive") or not final[4].get("interactive"):
                    raise ValueError("UI controls were not restored")
                row["success"] = True
            except Exception as exc:
                row["error"] = type(exc).__name__
            rows.append(row)
        return {"sessions": len(sessions), "requests": rows, "passed": all(row["success"] for row in rows)}
    finally:
        for session in sessions:
            session.close()


def sample_qdrant(container: str) -> dict:
    try:
        output = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                                check=True, capture_output=True, text=True, timeout=15)
        return {"container": container, "memory_usage": output.stdout.strip(),
                "method": "Docker memory usage snapshot; not a peak; separate from application RSS"}
    except (OSError, subprocess.SubprocessError):
        return {"container": container, "memory_usage": None, "method": "Docker snapshot unavailable"}


def run_smoke(args) -> dict:
    from app.config import settings

    args.corpus_sha256 = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    args.questions_sha256 = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    chunks = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = {(row["metadata"]["source"], row["content"]): row["id"] for row in chunks}
    available = {q["id"]: q for q in json.loads(args.questions.read_text(encoding="utf-8"))}
    questions = [available[key] for key in QUESTION_IDS]
    if settings.retrieval_mode != "hybrid_rerank" or settings.top_k != 4:
        raise ValueError("Smoke requires hybrid_rerank and TOP_K=4")
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "scenario_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "corpus_sha256": args.corpus_sha256, "questions_sha256": args.questions_sha256,
              "questions": questions, "corpus_chunks": len(chunks), "mode": settings.retrieval_mode,
              "llm": {"method": "local prompt echo; no provider client", "external_calls": 0},
              "measurement": {"repeats": REPEATS, "warmup_requests": 2, "requests_per_repeat": 8,
                              "validation": "ordered chunk IDs from concurrency-1 pass-1 are the per-question "
                                            "reference for later REST passes and both Gradio sessions",
                              "latency": "HTTP /chat round trip, including readiness, retrieval and local stub",
                              "percentiles": "linear interpolation over all 24 measured requests per concurrency",
                              "memory": "OS peak full server RSS/working set from import/startup through REST and UI; "
                                        "excludes driver, Qdrant and indexing; CPU models; startup includes "
                                        "model downloads if the cache is cold"},
              "variants": {}}
    with httpx.Client(base_url=settings.qdrant_url, trust_env=False, timeout=30) as qdrant:
        aliases = qdrant.get("/aliases")
        aliases.raise_for_status()
        before = aliases.json()["result"]["aliases"]
        collection = next(item["collection_name"] for item in before if item["alias_name"] == settings.collection_name)
        report["qdrant_version"] = qdrant.get("/").json()["version"]
        report["active_alias"] = {"name": settings.collection_name, "collection": collection}
        reference = {}
        for concurrency in (1, 4):
            check_inputs(args)
            started = time.perf_counter()
            with worker_client(args, collection) as (client, url):
                startup_s = time.perf_counter() - started
                endpoints = {path: client.get(path).status_code for path in ("/health", "/ready", "/")}
                if any(status != 200 for status in endpoints.values()):
                    raise RuntimeError("Service failed readiness/page checks")
                warmup = run_batch(client, [questions[0], questions[4]], corpus, 1, 0, reference)
                rows, runs = [], []
                for repeat in range(1, REPEATS + 1):
                    started = time.perf_counter()
                    measured = run_batch(client, questions, corpus, concurrency, repeat, reference)
                    duration_s = time.perf_counter() - started
                    if concurrency == 1 and repeat == 1:
                        reference = {row["id"]: row["chunk_ids"] for row in measured if row["success"]}
                        report["reference_chunk_ids"] = reference
                    rows.extend(measured)
                    runs.append({"repeat": repeat, "duration_s": duration_s, **summarize(measured)})
                    print(f"Concurrency {concurrency}: pass {repeat}/{REPEATS}, "
                          f"{sum(row['success'] for row in measured)}/8 successful", flush=True)
                ui = check_ui(url, [questions[0], questions[4]], corpus, reference)
                runtime = client.get("/__smoke__/runtime").json()
                if runtime["collection"] != collection:
                    raise RuntimeError("Smoke server uses a different collection")
                ui["observations"] = runtime.pop("ui")
                ui["passed"] = ui["passed"] and ui_queue_passed(ui["observations"], ui["sessions"])
                report["variants"][str(concurrency)] = {
                    "concurrency": concurrency, "service_startup_s": startup_s, "endpoints": endpoints,
                    "warmup": warmup, "summary": summarize(rows), "runs": runs, "requests": rows,
                    "runtime": runtime, "ui_queue": ui, "qdrant_memory": sample_qdrant(args.qdrant_container),
                }
        after = qdrant.get("/aliases").json()["result"]["aliases"]
        report["active_alias"]["unchanged"] = before == after
    check_inputs(args)
    report["passed"] = report["active_alias"]["unchanged"] and all(
        variant["summary"]["errors"] == 0 and all(row["success"] for row in variant["warmup"])
        and variant["ui_queue"]["passed"] for variant in report["variants"].values()
    )
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus_chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/retrieval_questions_v1.json"))
    parser.add_argument("--qdrant-container", default="qdrant", help="Docker container for a separate memory snapshot")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--address-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--corpus-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--questions-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Neither a provider secret nor a reachable provider endpoint is needed for this scenario.
    os.environ["LLM_API_KEY"] = "local-smoke-only"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1"
    if args.worker:
        serve_worker(args)
        return
    result = run_smoke(args)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved smoke report to {args.output}; passed={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
