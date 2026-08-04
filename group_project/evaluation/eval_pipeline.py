"""
RAG Evaluation Pipeline — dùng RAGAS.

CHẠY VỚI VENV RIÊNG (venv_eval/), KHÔNG PHẢI venv chính — ragas==0.1.21 cần
langchain<0.3 / openai<2.0, xung đột với các gói của pipeline chính (chromadb,
sentence-transformers...) khi cài chung. Xem requirements-eval.txt.

Setup (1 lần):
    python -m venv venv_eval
    source venv_eval/Scripts/activate   (venv_eval\\Scripts\\activate trên PowerShell)
    pip install -r requirements-eval.txt

Quy trình chạy đầy đủ:
    1. (venv chính)  python -m group_project.evaluation.generate_rag_outputs
       -> sinh rag_outputs_hybrid_rerank.json và rag_outputs_dense_only.json
    2. (venv_eval)   python -m group_project.evaluation.eval_pipeline
       -> đọc 2 file trên, chấm RAGAS, so sánh A/B, xuất results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS gọi LLM RẤT NHIỀU LẦN
(nhiều lần/metric/câu hỏi, không phải 1 lần/câu). Model free của OpenRouter giới
hạn 50 request/ngày CHO CẢ TÀI KHOẢN. Dùng OPENAI_API_KEY trả phí (như bài lab
này) tránh được giới hạn đó, nhưng vẫn nên theo dõi chi phí nếu golden dataset
lớn — 18 câu x 4 metric x nhiều lần gọi/metric là con số đáng kể.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

CONFIGS = {
    "hybrid_rerank": {
        "label": "Config A — Hybrid (Semantic + BM25) + RRF + Rerank",
        "outputs_path": Path(__file__).parent / "rag_outputs_hybrid_rerank.json",
    },
    "dense_only": {
        "label": "Config B — Dense-only (chỉ Semantic Search)",
        "outputs_path": Path(__file__).parent / "rag_outputs_dense_only.json",
    },
}

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_rag_outputs(config_name: str) -> list[dict]:
    path = CONFIGS[config_name]["outputs_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path.name} — chạy trước bằng venv chính:\n"
            f"  python -m group_project.evaluation.generate_rag_outputs"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Option 2: RAGAS (được chọn cho bài lab — không cần API riêng ngoài OpenAI)
# =============================================================================

def evaluate_with_ragas(rag_outputs: list[dict]):
    """
    Evaluate 1 config bằng RAGAS.

    Args:
        rag_outputs: list of {'question', 'answer', 'contexts', 'ground_truth'}
                     (đã được sinh sẵn bởi generate_rag_outputs.py)

    Returns:
        pandas.DataFrame — điểm từng câu hỏi theo 4 metric.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_data = {
        "question": [r["question"] for r in rag_outputs],
        "answer": [r["answer"] for r in rag_outputs],
        "contexts": [r["contexts"] for r in rag_outputs],
        "ground_truth": [r["ground_truth"] for r in rag_outputs],
    }
    dataset = Dataset.from_dict(eval_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs() -> dict:
    """
    So sánh A/B giữa 2 config: hybrid_rerank vs dense_only.

    Returns:
        {
            'hybrid_rerank': {'df': DataFrame, 'avg': {metric: score}},
            'dense_only':    {'df': DataFrame, 'avg': {metric: score}},
        }
    """
    comparison = {}
    for config_name in CONFIGS:
        print(f"Evaluating {config_name} ...")
        rag_outputs = _load_rag_outputs(config_name)
        df = evaluate_with_ragas(rag_outputs)
        avg = {m: round(float(df[m].mean()), 4) for m in METRIC_NAMES if m in df.columns}
        comparison[config_name] = {"df": df, "avg": avg}
    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _worst_performers(df, top_n: int = 3) -> list[dict]:
    """Lấy top_n câu hỏi có faithfulness thấp nhất (dễ lộ lỗi hallucination nhất)."""
    ranked = df.sort_values("faithfulness", ascending=True).head(top_n)
    rows = []
    for _, row in ranked.iterrows():
        rows.append({
            "question": row["question"],
            "faithfulness": round(float(row.get("faithfulness", 0)), 3),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 3),
            "context_recall": round(float(row.get("context_recall", 0)), 3),
        })
    return rows


def export_results(comparison: dict):
    """Export evaluation results to results.md"""
    hybrid = comparison["hybrid_rerank"]
    dense = comparison["dense_only"]

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    lines = ["# RAG Evaluation Results", "", "## Framework sử dụng", "", "> RAGAS 0.1.21", "", "---", ""]

    lines += ["## Overall Scores", "", "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |",
              "|--------|---------------------------|------------------------|-------|"]
    deltas = []
    for m in METRIC_NAMES:
        a = hybrid["avg"].get(m, 0.0)
        b = dense["avg"].get(m, 0.0)
        d = round(a - b, 4)
        deltas.append(d)
        lines.append(f"| {metric_labels[m]} | {a:.4f} | {b:.4f} | {d:+.4f} |")
    avg_a = round(sum(hybrid["avg"].values()) / len(hybrid["avg"]), 4)
    avg_b = round(sum(dense["avg"].values()) / len(dense["avg"]), 4)
    lines.append(f"| **Average** | **{avg_a:.4f}** | **{avg_b:.4f}** | **{avg_a - avg_b:+.4f}** |")
    lines += ["", "---", ""]

    lines += [
        "## A/B Comparison Analysis", "",
        "**Config A (hybrid_rerank):**",
        "> Semantic search (BAAI/bge-m3) + BM25 lexical search, gộp bằng Reciprocal Rank Fusion, "
        "sau đó rerank lại (RRF), có fallback PageIndex nếu điểm cosine gốc dưới ngưỡng.",
        "",
        "**Config B (dense_only):**",
        "> Chỉ dùng semantic_search() — bỏ qua lexical search, RRF merge và reranking.",
        "",
        f"**Kết luận:** Config A đạt điểm trung bình {avg_a:.4f} so với {avg_b:.4f} của Config B "
        f"({'cao hơn' if avg_a >= avg_b else 'thấp hơn'} {abs(avg_a - avg_b):.4f}). "
        + ("Hybrid retrieval + rerank giúp cải thiện chất lượng context/answer so với chỉ dùng dense search đơn thuần, "
           "đúng như kỳ vọng khi kết hợp thêm tín hiệu từ khóa (BM25) cho các câu hỏi có thuật ngữ pháp lý/tên riêng cụ thể."
           if avg_a >= avg_b else
           "Trên corpus nhỏ và câu hỏi tương đối rõ ràng hiện tại, dense-only đã đủ tốt; lợi ích của hybrid+rerank "
           "sẽ rõ hơn khi corpus lớn hơn hoặc câu hỏi chứa nhiều từ khóa/tên riêng đặc thù."),
        "", "---", "",
    ]

    lines += ["## Worst Performers (Config A, Bottom 3 theo Faithfulness)", "",
              "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
              "|---|----------|-------------|-----------|--------|---------------|------------|"]
    for i, row in enumerate(_worst_performers(hybrid["df"]), 1):
        stage = "Retrieval" if row["context_recall"] < 0.5 else "Generation"
        cause = (
            "Context không chứa đủ evidence cho câu hỏi (retrieval miss)"
            if row["context_recall"] < 0.5
            else "LLM diễn giải/suy luận vượt ngoài context được cấp"
        )
        q = row["question"][:70] + ("..." if len(row["question"]) > 70 else "")
        lines.append(
            f"| {i} | {q} | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} "
            f"| {row['context_recall']:.3f} | {stage} | {cause} |"
        )
    lines += ["", "---", ""]

    lines += [
        "## Recommendations", "",
        "### Cải tiến 1",
        "**Action:** Mở rộng corpus (thêm văn bản pháp lý/tin tức) để tăng context recall cho các câu hỏi cụ thể.",
        "**Expected impact:** Giảm tỷ lệ retrieval miss, tăng Context Recall và Faithfulness.",
        "",
        "### Cải tiến 2",
        "**Action:** Hiệu chỉnh lại `SCORE_THRESHOLD` ở Task 9 dựa trên phân bố điểm cosine thực tế của corpus hiện tại.",
        "**Expected impact:** Kích hoạt fallback PageIndex đúng lúc hơn cho các câu hỏi hybrid trả về yếu.",
        "",
        "### Cải tiến 3",
        "**Action:** Bật cross-encoder reranking (Jina API) thay vì chỉ RRF khi có JINA_API_KEY.",
        "**Expected impact:** Cải thiện Context Precision nhờ rerank ngữ nghĩa chính xác hơn RRF thuần rank-based.",
        "",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases\n")

    comparison = compare_configs()
    export_results(comparison)
