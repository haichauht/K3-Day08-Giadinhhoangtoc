"""Task 6 - Vietnamese-aware BM25 lexical retrieval."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []
_BM25 = None

# High-frequency function words dominated several Vietnamese BM25 queries
# (for example "học phí ... được thu và tính như thế nào").  Removing them
# lets domain terms such as "học phí" and "tín chỉ" drive the score.
VI_STOPWORDS = {
    "ai",
    "bao",
    "bằng",
    "bị",
    "các",
    "cách",
    "cho",
    "có",
    "của",
    "do",
    "đã",
    "đang",
    "để",
    "đó",
    "được",
    "gì",
    "khi",
    "không",
    "là",
    "làm",
    "một",
    "nào",
    "như",
    "những",
    "ra",
    "sau",
    "sẽ",
    "thì",
    "theo",
    "thế",
    "trong",
    "trên",
    "từ",
    "và",
    "vào",
    "về",
    "với",
}

# UIT users occasionally mix English terms into Vietnamese questions.  BM25
# cannot bridge languages by itself, so expand a deliberately small set of
# domain terms before scoring.  The original tokens remain in the query.
ENGLISH_QUERY_ALIASES = {
    "eligibility": ("điều", "kiện"),
    "fee": ("học", "phí"),
    "library": ("thư", "viện"),
    "room": ("phòng",),
    "scholarship": ("học", "bổng"),
    "study": ("học",),
    "tuition": ("học", "phí"),
}


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode text and add accent-free forms plus adjacent bigrams."""
    words = [
        word
        for word in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if word not in VI_STOPWORDS
    ]
    expanded: list[str] = []
    for word in words:
        expanded.append(word)
        plain = "".join(
            char
            for char in unicodedata.normalize("NFD", word.replace("đ", "d"))
            if unicodedata.category(char) != "Mn"
        )
        if plain != word:
            expanded.append(plain)
    # Phrase tokens reward exact concepts such as "tốt nghiệp" and "song ngành".
    expanded.extend(f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1))
    return expanded


def tokenize_query(text: str) -> list[str]:
    """Tokenize a query and expand common English university-service terms."""
    tokens = tokenize(text)
    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    aliases = [term for word in words for term in ENGLISH_QUERY_ALIASES.get(word, ())]
    if aliases:
        tokens.extend(tokenize(" ".join(aliases)))
    return tokens


def build_bm25_index(corpus: list[dict]):
    """Build Okapi BM25 (k1=1.5, b=0.75) for the supplied chunks."""
    if not corpus:
        return None

    def searchable_text(item: dict) -> str:
        metadata = item.get("metadata", {})
        # Headings carry decisive intent (for example the difference between
        # remote-training tuition and dual-degree tuition). Repeat the section
        # once as a light field boost without requiring a separate BM25F index.
        section = str(metadata.get("section", ""))
        return "\n".join(
            [str(metadata.get("title", "")), section, section, item["content"]]
        )

    return BM25Okapi(
        [tokenize(searchable_text(item)) for item in corpus],
        k1=1.5,
        b=0.75,
    )


def get_bm25_index():
    global CORPUS, _BM25
    if _BM25 is None:
        CORPUS = chunk_documents(load_documents())
        _BM25 = build_bm25_index(CORPUS)
    return _BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return chunks ranked by raw BM25 score, descending."""
    if not query.strip() or top_k <= 0:
        return []
    bm25 = get_bm25_index()
    if bm25 is None:
        return []
    scores = np.asarray(bm25.get_scores(tokenize_query(query)), dtype=float)
    ranked = np.argsort(scores)[::-1]
    results: list[dict] = []
    for index in ranked:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[int(index)]["content"],
                "score": round(score, 6),
                "metadata": CORPUS[int(index)]["metadata"],
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("điều kiện xét tốt nghiệp", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")
