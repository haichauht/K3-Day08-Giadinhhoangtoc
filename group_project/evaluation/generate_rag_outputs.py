"""
Sinh output của RAG pipeline cho golden dataset, theo 2 config (A/B), để
eval_pipeline.py (chạy trong venv_eval riêng) đánh giá bằng RAGAS.

CHẠY VỚI VENV CHÍNH (venv/), KHÔNG PHẢI venv_eval — vì cần chromadb,
sentence-transformers, rank_bm25... (dependency của pipeline thật), những
gói này xung đột phiên bản `openai` với ragas/langchain-openai nên phải
tách quy trình làm 2 bước (xem requirements-eval.txt để biết lý do):

    Bước 1 (venv chính):  python -m group_project.evaluation.generate_rag_outputs
    Bước 2 (venv_eval):   python -m group_project.evaluation.eval_pipeline

Config A — "hybrid_rerank": pipeline đầy đủ Task 9 (semantic + lexical +
RRF + rerank + PageIndex fallback nếu score thấp) — dùng generate_with_citation()
y hệt app.py.

Config B — "dense_only": chỉ semantic_search() (bỏ qua lexical/RRF/rerank),
để so sánh hybrid retrieval có thực sự cải thiện chất lượng answer/context
hay không.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.task5_semantic_search import semantic_search
from src.task10_generation import (
    TOP_K,
    call_llm,
    format_context,
    generate_with_citation,
    reorder_for_llm,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_DIR = Path(__file__).parent


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_config_hybrid_rerank(golden_dataset: list[dict]) -> list[dict]:
    """Config A: pipeline Task 9 đầy đủ (hybrid + RRF + rerank + fallback)."""
    outputs = []
    for item in golden_dataset:
        result = generate_with_citation(item["question"], top_k=TOP_K)
        outputs.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c["content"] for c in result["sources"]],
            "ground_truth": item["expected_answer"],
        })
        print(f"  [hybrid_rerank] OK: {item['question'][:60]}...")
    return outputs


def run_config_dense_only(golden_dataset: list[dict]) -> list[dict]:
    """Config B: chỉ dense retrieval (semantic_search), không rerank/RRF."""
    outputs = []
    for item in golden_dataset:
        chunks = semantic_search(item["question"], top_k=TOP_K)
        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {item['question']}"""
        answer = call_llm(user_message)
        outputs.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [c["content"] for c in chunks],
            "ground_truth": item["expected_answer"],
        })
        print(f"  [dense_only] OK: {item['question'][:60]}...")
    return outputs


def main():
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} golden Q&A pairs\n")

    print("Running Config A: hybrid_rerank ...")
    hybrid_outputs = run_config_hybrid_rerank(golden_dataset)
    (OUTPUT_DIR / "rag_outputs_hybrid_rerank.json").write_text(
        json.dumps(hybrid_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nRunning Config B: dense_only ...")
    dense_outputs = run_config_dense_only(golden_dataset)
    (OUTPUT_DIR / "rag_outputs_dense_only.json").write_text(
        json.dumps(dense_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDone. Saved rag_outputs_hybrid_rerank.json và rag_outputs_dense_only.json")
    print("Tiếp theo: chạy eval_pipeline.py trong venv_eval để tính RAGAS metrics.")


if __name__ == "__main__":
    main()
