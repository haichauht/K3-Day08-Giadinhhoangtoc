"""RAGAS evaluation and A/B comparison for the UIT chatbot.

Usage from the repository root::

    python -m group_project.evaluation.eval_pipeline
    python -m group_project.evaluation.eval_pipeline --limit 3  # smoke test

Generated answers are cached in ``.runtime/evaluation_records.json`` so a
network retry does not regenerate completed cases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
CACHE_PATH = PROJECT_ROOT / ".runtime" / "evaluation_records.json"
EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def load_golden_dataset() -> list[dict[str, str]]:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 Q&A pairs")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Golden item {index} is missing required fields")
    return dataset


def _load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        value = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_records(
    rag_pipeline: Callable[..., dict[str, Any]],
    golden_dataset: list[dict[str, str]],
    *,
    config_name: str,
    use_reranking: bool,
) -> list[dict[str, Any]]:
    """Generate answers and contexts, resuming safely from the local cache."""
    cache = _load_cache()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(golden_dataset, start=1):
        cache_version = "rrf_local_v2" if use_reranking else "dense_v1"
        cache_key = f"{cache_version}::{config_name}::{item['question']}"
        legacy_key = f"{config_name}::{item['question']}"
        if not use_reranking and cache_key not in cache and legacy_key in cache:
            cache[cache_key] = cache[legacy_key]
        cached = cache.get(cache_key)
        if cached:
            record = cached
            print(f"  [{index}/{len(golden_dataset)}] cache: {item['question'][:58]}")
        else:
            print(f"  [{index}/{len(golden_dataset)}] generate: {item['question'][:58]}")
            generated = rag_pipeline(
                item["question"],
                top_k=5,
                use_reranking=use_reranking,
            )
            contexts = [
                str(source.get("content", ""))
                for source in generated.get("sources", [])
                if source.get("content")
            ]
            record = {
                "question": item["question"],
                "answer": generated.get("answer", ""),
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
                "expected_context": item["expected_context"],
                "retrieval_source": generated.get("retrieval_source", "none"),
            }
            cache[cache_key] = record
            _save_cache(cache)
        records.append(record)
    return records


def _clean_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else max(0.0, min(1.0, score))


def evaluate_with_ragas(
    rag_pipeline: Callable[..., dict[str, Any]],
    golden_dataset: list[dict[str, str]],
    *,
    config_name: str = "hybrid_rrf",
    use_reranking: bool = True,
) -> dict[str, Any]:
    """Evaluate one configuration with four RAGAS metrics."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS evaluation")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    records = collect_records(
        rag_pipeline,
        golden_dataset,
        config_name=config_name,
        use_reranking=use_reranking,
    )
    dataset = Dataset.from_dict(
        {
            "question": [record["question"] for record in records],
            "answer": [record["answer"] for record in records],
            "contexts": [record["contexts"] for record in records],
            "ground_truth": [record["ground_truth"] for record in records],
        }
    )
    evaluator_llm = ChatOpenAI(model=EVAL_MODEL, temperature=0, timeout=120, max_retries=3)
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024,
        request_timeout=120,
        max_retries=3,
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=180, max_retries=3, max_wait=20, max_workers=4),
        raise_exceptions=False,
    )
    frame = result.to_pandas()

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = {
            "question": record["question"],
            "answer": record["answer"],
            "retrieval_source": record["retrieval_source"],
        }
        for metric in METRIC_NAMES:
            row[metric] = _clean_score(frame.iloc[index].get(metric, 0.0))
        row["average"] = mean(row[metric] for metric in METRIC_NAMES)
        rows.append(row)

    aggregate = {
        metric: mean(row[metric] for row in rows) if rows else 0.0
        for metric in METRIC_NAMES
    }
    aggregate["average"] = mean(aggregate.values()) if aggregate else 0.0
    return {
        "config": config_name,
        "model": EVAL_MODEL,
        "count": len(rows),
        "metrics": aggregate,
        "rows": rows,
    }


def compare_configs(
    rag_pipeline: Callable[..., dict[str, Any]],
    golden_dataset: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compare hybrid + RRF against the dense-only ablation."""
    configs = {
        "hybrid_rrf": True,
        "dense_only": False,
    }
    comparison: dict[str, dict[str, Any]] = {}
    for config_name, use_reranking in configs.items():
        print(f"\nEvaluating config: {config_name}")
        comparison[config_name] = evaluate_with_ragas(
            rag_pipeline,
            golden_dataset,
            config_name=config_name,
            use_reranking=use_reranking,
        )
    return comparison


def _failure_stage(row: dict[str, Any]) -> str:
    if row["context_recall"] < 0.5 or row["context_precision"] < 0.5:
        return "Retrieval"
    if row["faithfulness"] < 0.5:
        return "Generation grounding"
    if row["answer_relevancy"] < 0.5:
        return "Answer formulation"
    return "Mixed"


def export_results(comparison: dict[str, dict[str, Any]]) -> Path:
    """Write aggregate metrics, worst cases, and recommendations to Markdown."""
    hybrid = comparison["hybrid_rrf"]
    dense = comparison["dense_only"]
    lines = [
        "# Kết quả đánh giá RAG",
        "",
        "## Thiết lập",
        "",
        f"- Framework: RAGAS 0.1.21",
        f"- Evaluator LLM: `{hybrid['model']}`",
        "- Embedding đánh giá: `text-embedding-3-small` (1024 chiều)",
        f"- Golden dataset: {hybrid['count']} câu hỏi UIT/ĐHQG-HCM",
        "- Config A: semantic + BM25, hợp nhất RRF và relevance rerank cục bộ",
        "- Config B: semantic dense-only (ablation, không BM25/RRF)",
        "",
        "## Điểm tổng hợp",
        "",
        "| Metric | Config A: Hybrid + RRF | Config B: Dense-only | Δ (A-B) |",
        "|---|---:|---:|---:|",
    ]
    display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "**Trung bình**",
    }
    for metric in (*METRIC_NAMES, "average"):
        a_score = hybrid["metrics"][metric]
        b_score = dense["metrics"][metric]
        lines.append(f"| {display_names[metric]} | {a_score:.3f} | {b_score:.3f} | {a_score - b_score:+.3f} |")

    winner = "Hybrid + RRF" if hybrid["metrics"]["average"] >= dense["metrics"]["average"] else "Dense-only"
    lines.extend(
        [
            "",
            "## Phân tích A/B",
            "",
            f"Cấu hình có điểm trung bình cao hơn là **{winner}**. Hybrid bổ sung khả năng bắt đúng thuật ngữ pháp lý/tên thông báo bằng BM25, còn dense-only là đường cơ sở để đo đóng góp của lexical retrieval và RRF.",
            "",
            "## Ba trường hợp kém nhất của Config A",
            "",
            "| # | Câu hỏi | Faithfulness | Relevance | Recall | Precision | Công đoạn lỗi chính |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    worst = sorted(hybrid["rows"], key=lambda row: row["average"])[:3]
    for index, row in enumerate(worst, start=1):
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {index} | {question} | {row['faithfulness']:.3f} | "
            f"{row['answer_relevancy']:.3f} | {row['context_recall']:.3f} | "
            f"{row['context_precision']:.3f} | {_failure_stage(row)} |"
        )

    lines.extend(
        [
            "",
            "## Đề xuất cải tiến",
            "",
            "1. Loại bỏ menu, CAPTCHA và danh sách bài liên quan khỏi news trước khi chunk để tăng Context Precision.",
            "2. Dùng MarkdownHeaderTextSplitter hoặc gắn heading cha vào từng chunk pháp lý để tăng Context Recall cho câu hỏi theo Điều/Khoản.",
            "3. Mở rộng golden dataset bằng câu hỏi nối tiếp và câu hỏi không có bằng chứng để kiểm tra memory và cơ chế từ chối.",
            "",
            "> Điểm trong bảng được sinh từ lần chạy thật của script; file cache chỉ lưu câu trả lời/ngữ cảnh để có thể retry RAGAS mà không gọi lại generation.",
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UIT RAG with RAGAS")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases for a smoke test")
    args = parser.parse_args()

    from src.task10_generation import generate_with_citation

    golden_dataset = load_golden_dataset()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        golden_dataset = golden_dataset[: args.limit]
    comparison = compare_configs(generate_with_citation, golden_dataset)
    output = export_results(comparison)
    print(f"\nEvaluation complete: {output}")


if __name__ == "__main__":
    main()
