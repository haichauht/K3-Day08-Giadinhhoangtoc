"""Task 7 - Reciprocal Rank Fusion and optional MMR reranking."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .task6_lexical_search import VI_STOPWORDS


def _candidate_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    return (
        metadata.get("chunk_id")
        or hashlib.sha1(item.get("content", "").encode("utf-8")).hexdigest()
    )


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Fuse independent rankings with RRF(d)=sum(1/(k+rank))."""
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    contributions: dict[str, dict[str, float]] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        label = "dense" if list_index == 0 else f"lexical_{list_index}"
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item.copy())
            contributions.setdefault(key, {})[label] = float(item.get("score", 0.0))

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    output: list[dict] = []
    for key in ordered[:top_k]:
        result = content_map[key].copy()
        result["score"] = round(scores[key], 8)
        result["rrf_score"] = round(scores[key], 8)
        result["retrieval_scores"] = contributions[key]
        output.append(result)
    return output


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant but non-duplicative candidates using cosine MMR."""
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be in [0, 1]")
    query = np.asarray(query_embedding, dtype=float)
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator else 0.0

    for _ in range(min(top_k, len(candidates))):
        best_index = None
        best_score = float("-inf")
        for index in remaining:
            vector = np.asarray(candidates[index]["embedding"], dtype=float)
            relevance = cosine(query, vector)
            redundancy = max(
                (
                    cosine(
                        vector,
                        np.asarray(candidates[chosen]["embedding"], dtype=float),
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score, best_index = score, index
        if best_index is None:
            break
        item = candidates[best_index].copy()
        item["score"] = round(best_score, 8)
        candidates[best_index] = item
        selected.append(best_index)
        remaining.remove(best_index)
    return [candidates[index] for index in selected]


def _query_coverage(query: str, content: str) -> float:
    terms = {
        term
        for term in re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE)
        if term not in VI_STOPWORDS
    }
    if not terms:
        return 0.0
    document_terms = set(re.findall(r"[^\W_]+", content.lower(), flags=re.UNICODE))
    return len(terms & document_terms) / len(terms)


def rerank_by_features(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Lightweight deterministic reranker blending rank score and query coverage."""
    if not candidates or top_k <= 0:
        return []
    raw = np.asarray([float(item.get("score", 0.0)) for item in candidates])

    def normalise(values: np.ndarray) -> np.ndarray:
        minimum, maximum = float(values.min()), float(values.max())
        span = maximum - minimum
        return (values - minimum) / span if span else np.ones_like(values)

    rank_scores = normalise(raw)
    dense_scores = normalise(
        np.asarray(
            [item.get("retrieval_scores", {}).get("dense", 0.0) for item in candidates],
            dtype=float,
        )
    )
    lexical_scores = normalise(
        np.asarray(
            [
                max(
                    (
                        value
                        for key, value in item.get("retrieval_scores", {}).items()
                        if key.startswith("lexical")
                    ),
                    default=0.0,
                )
                for item in candidates
            ],
            dtype=float,
        )
    )
    output: list[dict] = []
    for index, item in enumerate(candidates):
        metadata = item.get("metadata", {})
        coverage_text = f"{metadata.get('section', '')}\n{item.get('content', '')}"
        coverage = _query_coverage(query, coverage_text)
        retrieval_signal = max(float(dense_scores[index]), float(lexical_scores[index]))
        result = item.copy()
        result["score"] = round(
            0.10 * float(rank_scores[index])
            + 0.45 * coverage
            + 0.45 * retrieval_signal,
            8,
        )
        result["rerank_method"] = "rrf+normalised-signals+query-coverage"
        output.append(result)
    return sorted(output, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "feature",
) -> list[dict]:
    """Apply the selected post-fusion reranker."""
    if method == "feature":
        return rerank_by_features(query, candidates, top_k)
    if method == "mmr":
        raise ValueError("MMR requires embeddings; call rerank_mmr directly")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều kiện xét tốt nghiệp", "score": 0.8, "metadata": {}},
        {"content": "Tin hoạt động sinh viên", "score": 0.6, "metadata": {}},
    ]
    print(rerank("điều kiện tốt nghiệp", dummy, top_k=2))
