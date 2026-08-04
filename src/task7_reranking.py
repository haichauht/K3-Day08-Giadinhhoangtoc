"""Task 7 - deterministic reranking algorithms.

RRF is the default because it combines dense and BM25 rankings without another
paid API.  MMR is included for candidates that already contain embeddings.  A
small lexical reranker is kept as an offline cross-encoder substitute so the
unified interface remains usable when no reranker service is configured.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .task6_lexical_search import _tokenize


def _validate_common(query: str, candidates: list[dict], top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _candidate_key(item: dict[str, Any]) -> str:
    """Return a stable identity shared by dense and sparse results."""
    metadata = item.get("metadata") or {}
    source_path = metadata.get("source_path") or metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source_path is not None and chunk_index is not None:
        return f"{source_path}::chunk-{chunk_index}"
    content = str(item.get("content", ""))
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Offline relevance reranker used when an external cross-encoder is absent.

    It blends token coverage with the candidate's normalised retrieval score.
    The production pipeline defaults to RRF; this method mainly provides a
    deterministic, dependency-free alternative for demonstrations.
    """
    _validate_common(query, candidates, top_k)
    if not candidates:
        return []

    query_tokens = set(_tokenize(query, expand_synonyms=True))
    raw_scores = [float(item.get("score", 0.0)) for item in candidates]
    minimum, maximum = min(raw_scores), max(raw_scores)
    span = maximum - minimum

    results: list[dict[str, Any]] = []
    for item, raw_score in zip(candidates, raw_scores):
        document_tokens = set(_tokenize(str(item.get("content", ""))))
        coverage = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
        normalised_retrieval = (raw_score - minimum) / span if span else 1.0
        rerank_score = 0.75 * coverage + 0.25 * normalised_retrieval
        result = dict(item)
        result["original_score"] = raw_score
        result["score"] = float(rerank_score)
        result["rerank_method"] = "lexical_offline"
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embeddings must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select candidates balancing relevance and novelty with MMR."""
    if not query_embedding:
        raise ValueError("query_embedding must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be between 0 and 1")
    if not candidates:
        return []
    if any(not isinstance(item.get("embedding"), list) for item in candidates):
        raise ValueError("every MMR candidate must include an embedding")

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))
    selected_results: list[dict[str, Any]] = []

    while remaining and len(selected_indices) < top_k:
        best_index: int | None = None
        best_score = float("-inf")
        for index_ in remaining:
            embedding = candidates[index_]["embedding"]
            relevance = _cosine_similarity(query_embedding, embedding)
            diversity_penalty = max(
                (_cosine_similarity(embedding, candidates[selected]["embedding"]) for selected in selected_indices),
                default=0.0,
            )
            score = lambda_param * relevance - (1.0 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_index = index_

        assert best_index is not None
        selected_indices.append(best_index)
        remaining.remove(best_index)
        item = dict(candidates[best_index])
        item["original_score"] = float(item.get("score", 0.0))
        item["score"] = float(best_score)
        item["rerank_method"] = "mmr"
        selected_results.append(item)
    return selected_results


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse ranked lists using Reciprocal Rank Fusion.

    ``RRF(d) = sum(1 / (k + rank_r(d)))`` where ranks start at one.  Original
    scores are retained by ranker in ``retrieval_scores`` for diagnostics and
    threshold decisions; the returned ``score`` is only the fusion score.
    """
    if not isinstance(ranked_lists, list):
        raise TypeError("ranked_lists must be a list")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    component_scores: dict[str, dict[str, float]] = {}

    for ranker_index, ranked_list in enumerate(ranked_lists):
        seen_in_ranker: set[str] = set()
        for rank, candidate in enumerate(ranked_list, start=1):
            if not isinstance(candidate, dict) or not candidate.get("content"):
                continue
            key = _candidate_key(candidate)
            if key in seen_in_ranker:
                continue
            seen_in_ranker.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, dict(candidate))
            component_scores.setdefault(key, {})[f"ranker_{ranker_index}"] = float(
                candidate.get("score", 0.0)
            )

    ordered_keys = sorted(scores, key=lambda key: (-scores[key], key))
    results: list[dict[str, Any]] = []
    for key in ordered_keys[:top_k]:
        result = dict(items[key])
        result["original_score"] = float(result.get("score", 0.0))
        result["score"] = float(scores[key])
        result["rrf_score"] = float(scores[key])
        result["retrieval_scores"] = component_scores[key]
        result["rerank_method"] = "rrf"
        results.append(result)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "rrf") -> list[dict]:
    """Unified reranking interface used by tests and the final pipeline."""
    _validate_common(query, candidates, top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        from .task4_chunking_indexing import create_embeddings

        query_embedding = create_embeddings([query])[0]
        return rerank_mmr(query_embedding, candidates, top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    for result in rerank("tuition fee payment", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
