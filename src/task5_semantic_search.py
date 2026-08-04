"""Task 5 - Dense semantic search over the Task 4 Chroma indexes."""

from __future__ import annotations

import requests

from .task4_chunking_indexing import (
    COLLECTION_NAME,
    LOCAL_COLLECTION_NAME,
    chunk_documents,
    embed_chunks,
    embed_texts_local,
    embed_texts_nvidia,
    get_collection,
    index_to_vectorstore,
    load_documents,
    run_pipeline,
)


def _query_collection(query: str, top_k: int, backend: str) -> list[dict]:
    from chromadb.errors import NotFoundError

    collection_name = COLLECTION_NAME if backend == "nvidia" else LOCAL_COLLECTION_NAME
    try:
        collection = get_collection(collection_name)
    except NotFoundError:
        if backend == "nvidia":
            run_pipeline(with_local_fallback=False)
        else:
            chunks = chunk_documents(load_documents())
            local_chunks = embed_chunks(chunks, backend="local")
            index_to_vectorstore(local_chunks, LOCAL_COLLECTION_NAME)
        collection = get_collection(collection_name)

    embed = embed_texts_nvidia if backend == "nvidia" else embed_texts_local
    query_vector = embed([query], input_type="query")[0]
    n_results = min(max(1, top_k), collection.count())
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    output: list[dict] = []
    for content, metadata, distance in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
        strict=True,
    ):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append(
            {
                "content": content,
                "score": round(similarity, 6),
                "metadata": metadata or {},
                "embedding_backend": backend,
            }
        )
    return sorted(output, key=lambda item: item["score"], reverse=True)[:top_k]


def semantic_search(query: str, top_k: int = 10, backend: str = "auto") -> list[dict]:
    """Return cosine-ranked chunks; API first, cached MiniLM on API failure."""
    if not query.strip() or top_k <= 0:
        return []
    if backend in {"nvidia", "local"}:
        return _query_collection(query, top_k, backend)
    if backend != "auto":
        raise ValueError("backend must be auto, nvidia, or local")
    try:
        return _query_collection(query, top_k, "nvidia")
    except (requests.RequestException, RuntimeError, KeyError, ValueError):
        return _query_collection(query, top_k, "local")


if __name__ == "__main__":
    for result in semantic_search("Điều kiện xét tốt nghiệp", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")
    (chunk_documents,)
    (embed_chunks,)
