"""Task 9 - Hybrid retrieval with RRF, reranking, and vectorless fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# Calibrated on five in-domain and three out-of-domain queries: the weakest
# relevant top-1 cosine was 0.326, while noise stayed at or below 0.144.
SCORE_THRESHOLD = 0.25
DEFAULT_TOP_K = 5
RERANK_METHOD = "feature"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    semantic_backend: str = "auto",
) -> list[dict]:
    """Run dense and BM25 retrieval, fuse via RRF, then apply fallback."""
    if not query.strip() or top_k <= 0:
        return []
    candidate_k = max(top_k * 3, 10)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            semantic_search, query, candidate_k, semantic_backend
        )
        lexical_future = executor.submit(lexical_search, query, candidate_k)
        dense_results = dense_future.result()
        lexical_results = lexical_future.result()

    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    if best_dense_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for item in fallback:
                item["semantic_best_score"] = best_dense_score
            return fallback
        return []

    merged = rerank_rrf([dense_results, lexical_results], top_k=max(top_k * 2, top_k))
    final = (
        rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        if use_reranking
        else merged[:top_k]
    )
    for item in final:
        item["source"] = "hybrid"
        item["semantic_best_score"] = best_dense_score
    return final[:top_k]


if __name__ == "__main__":
    for question in (
        "Điều kiện xét tốt nghiệp đào tạo từ xa là gì?",
        "xyzabc123nonsense",
    ):
        print(f"\n{question}")
        for result in retrieve(question, top_k=3):
            print(result["score"], result["source"], result["metadata"]["source"])
