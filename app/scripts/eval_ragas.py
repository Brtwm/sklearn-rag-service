"""Evaluate saved answers with RAGAS; no retrieval or answer generation."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean
import threading
import time
from urllib.parse import urlsplit

# Evaluation data stays local except for the explicitly requested LLM calls.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langchain_core.outputs import Generation, LLMResult
from ragas import SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.prompt import PydanticPrompt

from app.scripts.eval_answers import QUESTION_IDS, limit_details, provider_limit

METRICS = ("faithfulness", "answer_relevancy")


class BudgetExhausted(Exception):
    pass


def save_report(report, output):
    report["summary"] = {}
    for group, refusal in (("in_corpus", False), ("out_of_corpus", True)):
        report["summary"][group] = {}
        for name in METRICS:
            values = [row["metrics"][name]["value"] for row in report["questions"]
                      if row["expected_refusal"] == refusal and row["metrics"][name]["value"] is not None]
            report["summary"][group][name] = {"mean": fmean(values) if values else None, "n": len(values)}
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)


class NoRepairPrompt(PydanticPrompt):
    def __init__(self, prompt):
        super().__init__(name=prompt.name, language=prompt.language)
        self.name = prompt.name
        self.instruction = prompt.instruction
        self.examples = deepcopy(prompt.examples)
        self.input_model = prompt.input_model
        self.output_model = prompt.output_model

    async def generate_multiple(self, *args, **kwargs):
        kwargs["retries_left"] = 0
        return await super().generate_multiple(*args, **kwargs)


class BudgetLLM(BaseRagasLLM):
    def __init__(self, llm, report, output, clock, sleep):
        super().__init__()
        self.llm, self.report, self.output = llm, report, output
        self.clock, self.sleep = clock, sleep
        self.last_finished = None
        self.lock = threading.Lock()
        self.question_id = self.metric = None

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if n != 1:
            raise ValueError("Only one completion per request is permitted")
        with self.lock:
            if self.report["calls_attempted"] >= self.report["call_budget"]:
                raise BudgetExhausted()
            if self.last_finished is not None:
                delay = max(0, self.report["interval_s"] - (self.clock() - self.last_finished))
                if delay:
                    self.sleep(delay)
            self.report["calls_attempted"] += 1
            record = {"number": self.report["calls_attempted"], "question_id": self.question_id,
                      "metric": self.metric, "status": "started", "prompt": prompt.to_string(),
                      "started_at_utc": datetime.now(timezone.utc).isoformat()}
            self.report["calls"].append(record)
            save_report(self.report, self.output)
            started = self.clock()
            try:
                response = self.llm.invoke(prompt.to_messages(), **({"stop": stop} if stop else {}))
                record.update(status="completed", response=response.content,
                              finish_reason=response.response_metadata.get("finish_reason"), usage=response.usage_metadata)
                if record["finish_reason"] == "length":
                    raise ValueError("Truncated RAGAS output")
                return LLMResult(generations=[[Generation(text=response.content)]])
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
                if provider_limit(exc):
                    self.report["stop_reason"] = "provider_limit"
                    record["limit"] = limit_details(exc)
                raise
            finally:
                self.last_finished = self.clock()
                record["duration_s"] = self.last_finished - started
                record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                save_report(self.report, self.output)

    async def agenerate_text(self, *args, **kwargs):
        return await asyncio.to_thread(self.generate_text, *args, **kwargs)

    async def generate(self, *args, **kwargs):
        # Bypass BaseRagasLLM's automatic retry wrapper entirely.
        return await self.agenerate_text(*args, **kwargs)


class QuestionEmbeddings(BaseRagasEmbeddings):
    """Both sides of relevancy are questions, so both use E5's query prefix."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_query(self, text):
        return self.model.embed_query(text)

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text):
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)


def run_evaluation(rows, llm, embeddings, output, *, parameters=None, call_budget=18,
                   interval_s=20, clock=time.monotonic, sleep=time.sleep):
    if not rows or len(rows) > 6 or any(not row["answer"] or not row["sources"] for row in rows):
        raise ValueError("Expected 1–6 saved answers with their original contexts")
    if not math.isfinite(interval_s) or interval_s < 20:
        raise ValueError("Call interval must be finite and at least 20 seconds")
    if type(call_budget) is not int or not 1 <= call_budget <= 18:
        raise ValueError("Call budget must be 1–18")
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
              "stop_reason": None, "call_budget": call_budget, "calls_attempted": 0,
              "interval_s": interval_s, "parameters": parameters or {}, "calls": [],
              "questions": [{**deepcopy(row), "metrics": {name: {"value": None, "status": "not_started"}
                             for name in METRICS}} for row in rows]}
    save_report(report, output)
    adapter = BudgetLLM(llm, report, output, clock, sleep)
    faithfulness = Faithfulness(llm=adapter)
    faithfulness.statement_generator_prompt = NoRepairPrompt(faithfulness.statement_generator_prompt)
    faithfulness.nli_statements_prompt = NoRepairPrompt(faithfulness.nli_statements_prompt)
    relevancy = ResponseRelevancy(llm=adapter, embeddings=QuestionEmbeddings(embeddings), strictness=1)
    relevancy.question_generation = NoRepairPrompt(relevancy.question_generation)

    async def score():
        for row in report["questions"]:
            sample = SingleTurnSample(user_input=row["question"], response=row["answer"],
                                      retrieved_contexts=[source["content"] for source in row["sources"]])
            for metric in (faithfulness, relevancy):
                print(f"RAGAS {row['id']}: {metric.name}", flush=True)
                adapter.question_id, adapter.metric = row["id"], metric.name
                result = row["metrics"][metric.name]
                result["status"] = "running"
                save_report(report, output)
                try:
                    value = float(await metric.single_turn_ascore(sample))
                    result.update(value=value if math.isfinite(value) else None,
                                  status="completed" if math.isfinite(value) else "undefined")
                except BudgetExhausted:
                    result["status"] = "budget_exhausted"
                    report["stop_reason"] = "call_budget"
                except Exception as exc:
                    result.update(status="error", error_type=type(exc).__name__)
                save_report(report, output)
                if report["stop_reason"]:
                    return

    try:
        asyncio.run(score())
    except KeyboardInterrupt:
        report["stop_reason"] = "interrupted"
    finally:
        report["complete"] = all(result["status"] == "completed" for row in report["questions"]
                                 for result in row["metrics"].values())
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(report, output)
    return report


def main():
    from langchain_openai import ChatOpenAI
    from app.config import settings
    from app.rag.embeddings import get_embeddings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("notebooks/answer_evaluation_neuraldeep_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("notebooks/answer_ragas_neuraldeep_v1.json"))
    parser.add_argument("--interval-s", type=float, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose another path to preserve the previous run")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if [row["id"] for row in source["questions"]] != list(QUESTION_IDS):
        parser.error("Expected the fixed six-question answer report")
    rows = [{key: row[key] for key in ("id", "question", "language", "category", "answer", "sources", "citations",
                                     "expected_facts", "expected_refusal")} for row in source["questions"]]
    llm = ChatOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                    model=settings.llm_model, temperature=settings.llm_temperature, max_retries=0,
                    timeout=120, max_tokens=4096, streaming=False).bind(response_format={"type": "json_object"})
    parameters = {"provider_host": urlsplit(settings.llm_base_url).hostname, "judge_model": settings.llm_model,
                  "temperature": settings.llm_temperature, "max_output_tokens": 4096, "timeout_s": 120,
                  "sdk_retries": 0, "ragas_retries": 0, "output_repair_calls": 0,
                  "strictness": 1, "prompt_language": "english", "response_format": "json_object",
                  "embedding_model": settings.embedding_model, "embedding_prefix_both_sides": "query: ",
                  "normalize_embeddings": settings.normalize_embeddings,
                  "input_report": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "source_parameters": source["parameters"], "python": platform.python_version(), "platform": platform.platform(),
                  "versions": {name: importlib.metadata.version(name) for name in (
                      "ragas", "langchain-openai", "openai", "sentence-transformers")},
                  "limitations": ["Six-question smoke-test, not an overall quality estimate.",
                                  "strictness=1: one reconstructed question per answer.",
                                  "Same model generated and evaluated answers; possible shared bias.",
                                  "Faithfulness checks context support, not numbered citation correctness.",
                                  "Out-of-corpus refusal is reported separately; low relevancy can be appropriate."]}
    report = run_evaluation(rows, llm, get_embeddings(device="cpu"), args.output,
                            parameters=parameters, interval_s=args.interval_s)
    print(f"Saved {args.output}: complete={report['complete']}, calls={report['calls_attempted']}/18")
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
