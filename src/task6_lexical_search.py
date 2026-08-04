"""Task 6 - lexical retrieval with an in-memory BM25 index.

The index uses exactly the same chunks as Task 4 so dense and sparse results
can be fused safely in Task 9.  Tokenisation is deliberately lightweight: it
normalises Vietnamese diacritics and expands a small set of English terms used
by the course's smoke tests.  No external service is required.
"""

from __future__ import annotations

import re
import unicodedata
from threading import Lock
from typing import Any

from .task4_chunking_indexing import chunk_documents, load_documents


CORPUS: list[dict[str, Any]] = []
_BM25_INDEX = None
_INDEX_LOCK = Lock()

_QUERY_SYNONYMS = {
    "tuition": ("hoc", "phi"),
    "fee": ("hoc", "phi"),
    "fees": ("hoc", "phi"),
    "payment": ("thanh", "toan"),
    "policy": ("quy", "dinh"),
    "university": ("dai", "hoc"),
    "scholarship": ("hoc", "bong"),
    "eligibility": ("dieu", "kien"),
    "library": ("thu", "vien"),
    "study": ("hoc",),
    "room": ("phong",),
}


def _tokenize(text: str, *, expand_synonyms: bool = False) -> list[str]:
    """Return lowercase, accent-insensitive word tokens."""
    normalised = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    normalised = "".join(char for char in normalised if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", normalised)
    if not expand_synonyms:
        return tokens

    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_SYNONYMS.get(token, ()))
    return expanded


def build_bm25_index(corpus: list[dict[str, Any]]):
    """Build and return a BM25Okapi index for ``corpus``."""
    if not corpus:
        raise ValueError("corpus must contain at least one document")
    if any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in corpus):
        raise ValueError("every corpus item must contain non-empty text")

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(item["content"]) for item in corpus]
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


def _get_index():
    """Lazily create one process-wide BM25 index."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        with _INDEX_LOCK:
            if _BM25_INDEX is None:
                CORPUS.clear()
                CORPUS.extend(chunk_documents(load_documents()))
                _BM25_INDEX = build_bm25_index(CORPUS)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search the Task 4 corpus and return positive BM25 matches."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    index = _get_index()
    scores = index.get_scores(_tokenize(query, expand_synonyms=True))
    ranked_indices = sorted(range(len(scores)), key=lambda index_: float(scores[index_]), reverse=True)

    results: list[dict[str, Any]] = []
    for index_ in ranked_indices:
        score = float(scores[index_])
        if score <= 0:
            continue
        document = CORPUS[index_]
        results.append(
            {
                "content": document["content"],
                "score": score,
                "metadata": dict(document.get("metadata", {})),
            }
        )
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("học phí và thanh toán", top_k=5):
        source = result["metadata"].get("source_path", "unknown")
        print(f"[{result['score']:.3f}] {source}: {result['content'][:100]}...")
