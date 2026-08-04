"""Task 9 - hybrid retrieval with RRF and PageIndex fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_cross_encoder, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# Calibrated for cosine scores from text-embedding-3-small on the current UIT
# corpus.  RRF scores are never compared against this threshold.
SCORE_THRESHOLD = 0.30
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict[str, Any]]:
    """Run dense and BM25 retrieval, fuse rankings, and optionally fallback."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
        raise ValueError("score_threshold must be numeric")

    # Retrieve a wider candidate pool before fusion.  RRF is robust at merging
    # rankings but an exact legal clause may occur just below top_k in one
    # ranker; the final lightweight relevance pass restores that clause.
    retrieval_k = max(top_k * 4, 20)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query.strip(), retrieval_k)
        sparse_future = executor.submit(lexical_search, query.strip(), retrieval_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    best_dense_score = float(dense_results[0]["score"]) if dense_results else 0.0
    if best_dense_score < float(score_threshold):
        fallback = pageindex_search(query.strip(), top_k=top_k)
        if fallback:
            return fallback[:top_k]

    if use_reranking:
        fused = rerank_rrf([dense_results, sparse_results], top_k=retrieval_k)
        merged = rerank_cross_encoder(query.strip(), fused, top_k=top_k)
    else:
        # Dense-only is intentionally exposed for the A/B evaluation config.
        merged = [dict(item) for item in dense_results[:retrieval_k]]

    results: list[dict[str, Any]] = []
    for item in merged[:top_k]:
        result = dict(item)
        result["source"] = "hybrid"
        result["semantic_best_score"] = best_dense_score
        results.append(result)
    return results


if __name__ == "__main__":
    queries = [
        "Điều kiện đăng ký học song ngành là gì?",
        "Thông báo thu học phí học kỳ 2 năm học 2025-2026",
        "Cách xem thời khóa biểu cho tân sinh viên UIT",
        "xyzabc123nonsense",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        for rank, result in enumerate(retrieve(query, top_k=3), start=1):
            print(f"  {rank}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
