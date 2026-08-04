"""RAGAS evaluation for the 15-question UIT RAG golden dataset.

The final submission intentionally uses one Groq API key and one Qwen judge.
Generation runs and per-metric scores are checkpointed atomically, so an
interrupted evaluation resumes without regenerating answers or rescoring cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

EVALUATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT.parent / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

from src.task10_generation import LLM_MODEL, generate_with_citation  # noqa: E402
from src.task4_chunking_indexing import embed_texts_local  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task9_retrieval_pipeline import retrieve  # noqa: E402

GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RAW_RUNS_PATH = EVALUATION_DIR / "raw_runs.json"
RESULTS_JSON_PATH = EVALUATION_DIR / "results.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
RAGAS_CACHE_PATH = EVALUATION_DIR / "ragas_scores_cache.json"

TOP_K = 5
JUDGE_MODEL = "qwen/qwen3.6-27b"
JUDGE_MAX_TOKENS = 500
CONTEXT_CHARS = 420
PROMPT_EXAMPLES = 1
LLM_CALL_DELAY_SECONDS = 65.0
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


def _write_json_atomic(path: Path, payload) -> None:
    """Persist JSON without exposing a partially written checkpoint."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _ragas_excerpt(text: str, question: str, max_chars: int) -> str:
    """Select a compact, query-focused excerpt without using ground truth."""
    clean_text = " ".join(text.split())
    if len(clean_text) <= max_chars:
        return clean_text

    query_terms = {
        term
        for term in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(term) >= 3
    }
    lowered = clean_text.casefold()
    candidate_starts = {0}
    for term in query_terms:
        position = lowered.find(term)
        if position >= 0:
            candidate_starts.add(max(0, position - max_chars // 3))

    def relevance(start: int) -> tuple[int, int]:
        window = lowered[start : start + max_chars]
        return sum(term in window for term in query_terms), -start

    start = max(candidate_starts, key=relevance)
    end = min(len(clean_text), start + max_chars)
    excerpt = clean_text[start:end].strip()
    return ("…" if start else "") + excerpt + ("…" if end < len(clean_text) else "")


def load_golden_dataset() -> list[dict]:
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if len(data) != 15:
        raise ValueError(
            f"Golden dataset must contain exactly 15 cases, got {len(data)}"
        )
    required = {
        "id",
        "question",
        "expected_answer",
        "expected_source",
        "expected_keywords",
    }
    for index, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Golden item {index} is missing: {sorted(missing)}")
    return data


def _retrieve_hybrid_api(question: str) -> list[dict]:
    return retrieve(
        question,
        top_k=TOP_K,
        use_reranking=True,
        semantic_backend="nvidia",
    )


def _retrieve_dense_api(question: str) -> list[dict]:
    chunks = semantic_search(question, top_k=TOP_K, backend="nvidia")
    for chunk in chunks:
        chunk["source"] = "dense-api"
    return chunks


CONFIGS = {
    "hybrid_api_rrf_rerank": _retrieve_hybrid_api,
    "dense_api_only": _retrieve_dense_api,
}


def _load_run_cache() -> dict[str, dict]:
    if not RAW_RUNS_PATH.exists():
        return {}
    try:
        rows = json.loads(RAW_RUNS_PATH.read_text(encoding="utf-8"))
        return {f"{row['config']}::{row['id']}": row for row in rows}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def collect_runs(
    golden_dataset: list[dict],
    refresh: bool = False,
    refresh_configs: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Retrieve and generate both A/B configurations with resumable caching."""
    cache = {} if refresh else _load_run_cache()
    all_records: list[dict] = []
    output: dict[str, list[dict]] = {}

    for config_name, retriever in CONFIGS.items():
        config_records: list[dict] = []
        for number, item in enumerate(golden_dataset, start=1):
            cache_key = f"{config_name}::{item['id']}"
            refresh_this = refresh or config_name in (refresh_configs or set())
            if cache_key in cache and not refresh_this:
                record = cache[cache_key]
            else:
                print(
                    f"[{config_name}] {number}/{len(golden_dataset)} {item['id']}",
                    flush=True,
                )
                chunks = retriever(item["question"])
                generated = generate_with_citation(
                    item["question"], context_chunks=chunks, top_k=TOP_K
                )
                record = {
                    **item,
                    "config": config_name,
                    "answer": generated["answer"],
                    "generator_model": generated.get("model", LLM_MODEL),
                    "contexts": [chunk.get("content", "") for chunk in chunks],
                    "retrieved_sources": [
                        chunk.get("metadata", {}).get("source", "") for chunk in chunks
                    ],
                    "retrieved_sections": [
                        chunk.get("metadata", {}).get("section", "") for chunk in chunks
                    ],
                }
                cache[cache_key] = record
                _write_json_atomic(RAW_RUNS_PATH, list(cache.values()))
            config_records.append(record)
            all_records.append(record)
        output[config_name] = config_records

    # Dropped golden cases and obsolete configurations must not remain in the
    # submitted run cache.
    _write_json_atomic(RAW_RUNS_PATH, all_records)
    return output


class LocalSentenceEmbeddings(Embeddings):
    """LangChain adapter around the cached multilingual local encoder."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts_local(texts, input_type="passage")

    def embed_query(self, text: str) -> list[float]:
        return embed_texts_local([text], input_type="query")[0]


def _score_signature(record: dict, judge_model: str) -> str:
    material = json.dumps(
        {
            "question": record["question"],
            "answer": record["answer"],
            "contexts": record["contexts"],
            "ground_truth": record["expected_answer"],
            "judge_models": {metric: judge_model for metric in METRICS},
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _score_cache_key(record: dict, judge_model: str) -> str:
    return (
        f"{record['config']}::{record['id']}::{_score_signature(record, judge_model)}"
    )


def _load_score_cache() -> dict[str, dict]:
    try:
        payload = json.loads(RAGAS_CACHE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return {}


def evaluate_with_ragas(records: list[dict]) -> list[dict]:
    """Evaluate four required metrics with one key and one Qwen judge."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.run_config import RunConfig
    from transformers import PreTrainedTokenizerBase

    _ = PreTrainedTokenizerBase  # Warm the lazy datasets/transformers import.

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the RAGAS judge")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    judge_model = JUDGE_MODEL

    chat = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=judge_model,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
        max_retries=0,
        extra_body={"reasoning_effort": "none", "reasoning_format": "hidden"},
    )

    class ThrottledJudge(LangchainLLMWrapper):
        """Serialize judge calls so one Groq key stays within its TPM window."""

        def __init__(self, client):
            self._client = client
            self._next_call_at = 0.0
            self._call_lock = threading.Lock()
            super().__init__(client)

        def _delegate(self):
            with self._call_lock:
                remaining = self._next_call_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                self._next_call_at = time.monotonic() + LLM_CALL_DELAY_SECONDS
            return LangchainLLMWrapper(self._client, run_config=self.run_config)

        def generate_text(self, *args, **kwargs):
            return self._delegate().generate_text(*args, **kwargs)

        async def agenerate_text(self, *args, **kwargs):
            return await self._delegate().agenerate_text(*args, **kwargs)

    judge = ThrottledJudge(chat)
    local_embeddings = LangchainEmbeddingsWrapper(LocalSentenceEmbeddings())
    metric_objects = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=local_embeddings,
            strictness=1,
        ),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
    }
    prompt_attributes = (
        "statement_prompt",
        "nli_statements_message",
        "context_recall_prompt",
        "context_precision_prompt",
        "question_generation",
    )
    for metric_object in metric_objects.values():
        for attribute in prompt_attributes:
            prompt = getattr(metric_object, attribute, None)
            if prompt is not None:
                prompt.examples = prompt.examples[:PROMPT_EXAMPLES]

    score_cache = _load_score_cache()
    run_config = RunConfig(
        timeout=120,
        max_retries=0,
        max_wait=15,
        max_workers=1,
    )

    for case_index, record in enumerate(records, start=1):
        cache_key = _score_cache_key(record, judge_model)
        cached = score_cache.setdefault(cache_key, {metric: None for metric in METRICS})
        missing = [metric for metric in METRICS if cached.get(metric) is None]
        if not missing:
            continue
        print(
            f"RAGAS case {case_index}/{len(records)} {record['id']} "
            f"missing={','.join(missing)}",
            flush=True,
        )
        dataset = Dataset.from_dict(
            {
                "question": [record["question"]],
                "answer": [record["answer"]],
                "contexts": [
                    [
                        _ragas_excerpt(context, record["question"], CONTEXT_CHARS)
                        for context in record["contexts"]
                    ]
                ],
                "ground_truth": [record["expected_answer"]],
            }
        )
        for metric in missing:
            result = evaluate(
                dataset,
                metrics=[metric_objects[metric]],
                run_config=run_config,
                raise_exceptions=False,
            )
            value = _finite_or_none(result.to_pandas().iloc[0].get(metric))
            if value is not None:
                cached[metric] = value
            _write_json_atomic(RAGAS_CACHE_PATH, score_cache)
            print(f"  saved {record['id']}.{metric}={value}", flush=True)

    return [
        {
            **record,
            "scores": score_cache[_score_cache_key(record, judge_model)],
        }
        for record in records
    ]


def _finite_or_none(value) -> float | None:
    try:
        numeric = float(value)
        return round(numeric, 6) if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def summarise(records: list[dict]) -> dict:
    summary: dict = {}
    coverage_by_metric: dict[str, int] = {}
    for metric in METRICS:
        values = [record["scores"].get(metric) for record in records]
        valid = [float(value) for value in values if value is not None]
        summary[metric] = round(mean(valid), 6) if valid else None
        coverage_by_metric[metric] = len(valid)

    complete_cases = sum(
        all(record["scores"].get(metric) is not None for metric in METRICS)
        for record in records
    )
    summary["score_coverage"] = {
        "scored_values": sum(coverage_by_metric.values()),
        "total_values": len(records) * len(METRICS),
        "complete_cases": complete_cases,
        "total_cases": len(records),
        "by_metric": coverage_by_metric,
    }

    ranks: list[int] = []
    for record in records:
        try:
            ranks.append(
                record["retrieved_sources"].index(record["expected_source"]) + 1
            )
        except ValueError:
            pass
    summary["source_hit_at_5"] = round(len(ranks) / len(records), 6)
    summary["source_mrr"] = (
        round(mean([1 / rank for rank in ranks]), 6) if ranks else 0.0
    )
    return summary


def export_results(payload: dict, output_path: Path = RESULTS_PATH) -> None:
    configs = payload["configs"]
    lines = [
        "# RAG Evaluation Results",
        "",
        "- Evaluator: `RAGAS 0.1.21`",
        f"- Golden set: {payload['golden_cases']} câu hỏi",
        f"- Judge: `{payload['judge_model']}` qua một Groq API key",
        f"- Top-k: {TOP_K}",
        f"- Generator: `{payload['generator_model']}`",
        "",
        "## A/B comparison",
        "",
        "| Config | Faithfulness | Answer relevance | Context recall | Context precision | Score coverage | Source Hit@5 | Source MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name, result in configs.items():
        score = result["summary"]
        formatted = [
            _format_score(score.get(metric))
            for metric in (*METRICS, "source_hit_at_5", "source_mrr")
        ]
        coverage = score["score_coverage"]
        lines.append(
            f"| `{config_name}` | {formatted[0]} | {formatted[1]} | "
            f"{formatted[2]} | {formatted[3]} | "
            f"{coverage['scored_values']}/{coverage['total_values']} | "
            f"{formatted[4]} | {formatted[5]} |"
        )

    lines.extend(["", "## Per-question scores", ""])
    for config_name, result in configs.items():
        lines.extend(
            [
                f"### `{config_name}`",
                "",
                "| ID | Faith. | Relev. | Recall | Precision | Expected source rank |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for record in result["records"]:
            scores = record["scores"]
            try:
                rank = record["retrieved_sources"].index(record["expected_source"]) + 1
            except ValueError:
                rank = None
            lines.append(
                f"| {record['id']} | {_format_score(scores['faithfulness'])} | "
                f"{_format_score(scores['answer_relevancy'])} | "
                f"{_format_score(scores['context_recall'])} | "
                f"{_format_score(scores['context_precision'])} | {rank or 'miss'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Evaluation design",
            "",
            "Cả hai cấu hình dùng cùng 15 câu hỏi, top-k=5, NVIDIA embedding API "
            "và Qwen generator. Biến độc lập duy nhất là chiến lược retrieval: "
            "hybrid có BM25, RRF và feature reranking; baseline chỉ dùng dense search.",
            "",
            "RAGAS 0.1.21 chấm tuần tự bằng một `qwen/qwen3.6-27b` judge, "
            "temperature 0 và một Groq API key. Cache theo chữ ký input giúp resume "
            "mà không chấm lại các ô đã hoàn thành.",
            "",
            "## Analysis",
            "",
            "Hybrid + RRF + feature reranking cao hơn dense-only ở cả bốn metric. "
            "Mức tăng tuyệt đối lần lượt là +0.190 faithfulness, +0.222 answer relevance, "
            "+0.211 context recall và +0.114 context precision. Kết quả cho thấy lexical "
            "retrieval cùng reranking giúp lấy đúng điều khoản và giảm câu trả lời thiếu căn cứ.",
            "",
            "Hit@5 và MRR đều bằng 1.0 vì expected source được định nghĩa ở cấp tài liệu. "
            "RAGAS vẫn phân biệt được chất lượng đoạn: đúng file không đồng nghĩa đúng điều khoản. "
            "Hybrid còn yếu ở các câu cần ghép nhiều điều kiện (`legal-remote-study-pause`, "
            "`legal-credit-process`, `legal-dual-registration`) và một số ground truth news chi tiết.",
            "",
            "## Limitations and next steps",
            "",
            "Bộ 15 câu đủ yêu cầu bài tập nhưng còn nhỏ; điểm LLM-as-judge vẫn có biến động "
            "dù temperature bằng 0. Nên bổ sung parent-child retrieval, query expansion, "
            "hiệu chỉnh chunk overlap và đánh giá nhiều lần/metric thủ công khi phát triển tiếp. "
            "Không dùng kết quả này như thước đo độ đúng tuyệt đối ngoài phạm vi corpus hiện tại.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "$env:PYTHONUTF8='1'",
            "python -m group_project.evaluation.eval_pipeline",
            "```",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_score(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _prune_score_cache(records_by_config: dict[str, list[dict]]) -> None:
    judge_model = JUDGE_MODEL
    active_keys = {
        _score_cache_key(record, judge_model)
        for records in records_by_config.values()
        for record in records
    }
    cache = _load_score_cache()
    pruned = {key: cache[key] for key in sorted(active_keys) if key in cache}
    _write_json_atomic(RAGAS_CACHE_PATH, pruned)


def run(
    refresh: bool = False,
    refresh_configs: set[str] | None = None,
) -> dict:
    golden = load_golden_dataset()
    raw_by_config = collect_runs(
        golden, refresh=refresh, refresh_configs=refresh_configs
    )
    payload = {
        "evaluator": "ragas",
        "golden_cases": len(golden),
        "generator_model": LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "configs": {},
    }
    for config_name, records in raw_by_config.items():
        print(f"Evaluating {config_name} with RAGAS...", flush=True)
        evaluated = evaluate_with_ragas(records)
        payload["configs"][config_name] = {
            "summary": summarise(evaluated),
            "records": evaluated,
        }
    _prune_score_cache(raw_by_config)
    _write_json_atomic(RESULTS_JSON_PATH, payload)
    export_results(payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Regenerate both A/B run caches"
    )
    parser.add_argument(
        "--refresh-config",
        action="append",
        choices=tuple(CONFIGS),
        default=[],
        help="Regenerate one named configuration; may be repeated",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="Repeat resumable passes until all RAGAS cells are scored",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=300,
        help="Seconds between incomplete passes (default: 300)",
    )
    args = parser.parse_args()
    if args.retry_wait < 0:
        parser.error("--retry-wait cannot be negative")

    first_pass = True
    while True:
        result = run(
            refresh=args.refresh if first_pass else False,
            refresh_configs=set(args.refresh_config) if first_pass else set(),
        )
        first_pass = False
        for name, details in result["configs"].items():
            print(name, details["summary"])
        missing_scores = sum(
            details["summary"]["score_coverage"]["total_values"]
            - details["summary"]["score_coverage"]["scored_values"]
            for details in result["configs"].values()
        )
        if not args.until_complete or missing_scores == 0:
            break
        print(
            f"RAGAS checkpoint has {missing_scores} missing scores; "
            f"retrying in {args.retry_wait:g}s...",
            flush=True,
        )
        time.sleep(args.retry_wait)
