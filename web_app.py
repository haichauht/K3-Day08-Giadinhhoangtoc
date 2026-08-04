"""FastAPI web demo for the UIT policy RAG pipeline (no Streamlit)."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.task10_generation import LLM_MODEL, generate_from_chunks
from src.task4_chunking_indexing import EMBEDDING_DIM, EMBEDDING_MODEL, get_collection
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_cross_encoder, rerank_rrf
from src.task8_pageindex_vectorless import STATE_FILE, pageindex_search
from src.task9_retrieval_pipeline import SCORE_THRESHOLD


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
STANDARDIZED_DIR = ROOT / "data" / "standardized"
LANDING_DIR = ROOT / "data" / "landing"
RESULTS_PATH = ROOT / "group_project" / "evaluation" / "results.md"
load_dotenv(ROOT / ".env")

app = FastAPI(
    title="UIT Policy RAG Demo",
    description="Interactive RAG architecture and chatbot demo",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=3, le=8)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


def _elapsed_ms(start: float) -> int:
    return max(1, round((time.perf_counter() - start) * 1000))


def _source_catalog() -> tuple[list[dict[str, Any]], dict[str, str]]:
    documents: list[dict[str, Any]] = []
    source_urls: dict[str, str] = {}

    for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        url_match = re.search(r"^\*\*Source URL:\*\*\s*(\S+)", text, re.MULTILINE)
        source_file_match = re.search(r"^\*\*Source file:\*\*\s*(\S+)", text, re.MULTILINE)
        url = url_match.group(1) if url_match else ""
        doc_type = md_path.parent.name
        record = {
            "name": md_path.name,
            "title": title_match.group(1).strip() if title_match else md_path.stem,
            "type": doc_type,
            "url": url,
            "characters": len(text),
        }
        documents.append(record)
        if url:
            source_urls[md_path.name] = url
            source_urls[md_path.stem] = url
            if source_file_match:
                source_urls[source_file_match.group(1)] = url
    return documents, source_urls


def _evaluation_metrics() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    text = RESULTS_PATH.read_text(encoding="utf-8")
    metrics: list[dict[str, Any]] = []
    row_pattern = re.compile(
        r"^\|\s*(Faithfulness|Answer Relevance|Context Recall|Context Precision|\*\*Trung bình\*\*)\s*"
        r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([+\-0-9.]+)\s*\|$",
        re.MULTILINE,
    )
    for label, hybrid, dense, delta in row_pattern.findall(text):
        metrics.append(
            {
                "name": label.replace("**", ""),
                "hybrid": float(hybrid),
                "dense": float(dense),
                "delta": float(delta),
            }
        )
    return metrics


def _pageindex_count() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _collection_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def _metadata_payload() -> dict[str, Any]:
    standardized, _ = _source_catalog()
    legal_raw = sorted((LANDING_DIR / "legal").glob("*.pdf"))
    news_raw = sorted((LANDING_DIR / "news").glob("*.json"))
    return {
        "counts": {
            "legal_pdf": len(legal_raw),
            "news_json": len(news_raw),
            "standardized_md": len(standardized),
            "chunks": _collection_count(),
            "pageindex_docs": _pageindex_count(),
        },
        "documents": standardized,
        "technology": [
            {"name": "FastAPI", "role": "Web API + static app", "kind": "Interface"},
            {"name": EMBEDDING_MODEL, "role": f"Embedding đa ngôn ngữ · {EMBEDDING_DIM}D", "kind": "Dense"},
            {"name": "ChromaDB", "role": "Cosine semantic search", "kind": "Vector store"},
            {"name": "BM25", "role": "Tìm chính xác thuật ngữ pháp lý", "kind": "Sparse"},
            {"name": "RRF + relevance", "role": "Fusion và rerank", "kind": "Ranking"},
            {"name": "PageIndex", "role": "Vectorless fallback trên 3 PDF", "kind": "Fallback"},
            {"name": LLM_MODEL, "role": "Responses API + citation", "kind": "Generation"},
            {"name": "RAGAS", "role": "15 câu · 4 metrics · A/B", "kind": "Evaluation"},
        ],
        "evaluation": _evaluation_metrics(),
        "api_ready": {
            "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "pageindex": bool(os.getenv("PAGEINDEX_API_KEY", "").strip()),
        },
    }


def _timed_search(search_fn, query: str, top_k: int) -> tuple[list[dict], int]:
    started = time.perf_counter()
    return search_fn(query, top_k), _elapsed_ms(started)


def _enrich_sources(sources: list[dict]) -> list[dict]:
    _, source_urls = _source_catalog()
    enriched: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        metadata = dict(item.get("metadata") or {})
        source_name = str(metadata.get("source", ""))
        metadata["url"] = source_urls.get(source_name, source_urls.get(Path(source_name).stem, ""))
        item["metadata"] = metadata
        enriched.append(item)
    return enriched


def run_traced_pipeline(request: ChatRequest) -> dict[str, Any]:
    query = request.query.strip()
    retrieval_k = max(request.top_k * 4, 20)
    total_started = time.perf_counter()
    trace: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_timed_search, semantic_search, query, retrieval_k)
        sparse_future = executor.submit(_timed_search, lexical_search, query, retrieval_k)
        dense_results, dense_ms = dense_future.result()
        sparse_results, sparse_ms = sparse_future.result()

    best_dense = float(dense_results[0]["score"]) if dense_results else 0.0
    trace.extend(
        [
            {
                "id": "dense",
                "label": "Semantic retrieval",
                "status": "done",
                "duration_ms": dense_ms,
                "detail": f"OpenAI embedding → ChromaDB · {len(dense_results)} candidates · cosine {best_dense:.3f}",
            },
            {
                "id": "sparse",
                "label": "BM25 lexical",
                "status": "done",
                "duration_ms": sparse_ms,
                "detail": f"Token match đa ngôn ngữ · {len(sparse_results)} candidates",
            },
        ]
    )

    fusion_started = time.perf_counter()
    fused = rerank_rrf([dense_results, sparse_results], top_k=retrieval_k)
    results = rerank_cross_encoder(query, fused, top_k=request.top_k)
    for result in results:
        result["source"] = "hybrid"
        result["semantic_best_score"] = best_dense
    trace.append(
        {
            "id": "fusion",
            "label": "RRF + relevance rerank",
            "status": "done",
            "duration_ms": _elapsed_ms(fusion_started),
            "detail": f"Fuse dense/sparse → giữ {len(results)} chunks tốt nhất",
        }
    )

    fallback_started = time.perf_counter()
    fallback_used = False
    if best_dense < SCORE_THRESHOLD:
        fallback = pageindex_search(query, top_k=request.top_k)
        if fallback:
            results = fallback
            fallback_used = True
    trace.append(
        {
            "id": "fallback",
            "label": "PageIndex vectorless fallback",
            "status": "used" if fallback_used else "skipped",
            "duration_ms": _elapsed_ms(fallback_started) if best_dense < SCORE_THRESHOLD else 0,
            "detail": (
                f"Đã dùng vì cosine {best_dense:.3f} < {SCORE_THRESHOLD:.2f}"
                if fallback_used
                else f"Không cần dùng · cosine {best_dense:.3f}"
            ),
        }
    )

    generation_started = time.perf_counter()
    history = [message.model_dump() for message in request.history[-6:]]
    generated = generate_from_chunks(query, results, history)
    trace.append(
        {
            "id": "generation",
            "label": "Grounded generation",
            "status": "done",
            "duration_ms": _elapsed_ms(generation_started),
            "detail": f"{generated.get('model') or LLM_MODEL} · Responses API · citations [S#]",
        }
    )
    generated["sources"] = _enrich_sources(generated.get("sources", []))
    generated["trace"] = trace
    generated["total_duration_ms"] = _elapsed_ms(total_started)
    generated["retrieval_stats"] = {
        "dense_candidates": len(dense_results),
        "sparse_candidates": len(sparse_results),
        "fused_candidates": len(fused),
        "best_dense_score": best_dense,
        "fallback_used": fallback_used,
    }
    return generated


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    payload = _metadata_payload()
    return {
        "status": "ok",
        "service": "uit-rag-web-demo",
        "chunks": payload["counts"]["chunks"],
        "openai_ready": payload["api_ready"]["openai"],
        "pageindex_ready": payload["api_ready"]["pageindex"],
    }


@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    return _metadata_payload()


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    return await run_in_threadpool(run_traced_pipeline, request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8001")),
        reload=False,
    )
