"""Task 8 - PageIndex cloud retrieval with a local vectorless fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import REPO_ROOT, STANDARDIZED_DIR, split_front_matter
from .task6_lexical_search import tokenize

load_dotenv(REPO_ROOT.parent / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

PAGEINDEX_CACHE_DIR = REPO_ROOT / "pageindex_cache"
TREE_INDEX_PATH = PAGEINDEX_CACHE_DIR / "tree_index.json"
CLOUD_MANIFEST_PATH = PAGEINDEX_CACHE_DIR / "cloud_documents.json"
CLOUD_QUERY_CACHE_PATH = PAGEINDEX_CACHE_DIR / "cloud_query_cache.json"
LEGAL_PDF_DIR = REPO_ROOT / "data" / "landing" / "legal"

QUERY_ALIASES = {
    "credit": "tín chỉ",
    "credits": "tín chỉ",
    "fee": "học phí",
    "graduation": "tốt nghiệp",
    "tuition": "học phí",
}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.name.encode("utf-8"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _corpus_fingerprint() -> str:
    digest = hashlib.sha1()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        digest.update(path.relative_to(STANDARDIZED_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sections_from_markdown(path: Path) -> list[dict]:
    front_matter, body = split_front_matter(path.read_text(encoding="utf-8"))
    title = front_matter.get("title", path.stem.replace("-", " ").title())
    headings: list[str] = []
    current_heading = title
    current_lines: list[str] = []
    sections: list[dict] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if not content:
            return
        sections.append(
            {
                "content": content,
                "section": " > ".join(headings) or current_heading,
                "title": title,
                "source": path.name,
                "source_path": path.relative_to(REPO_ROOT).as_posix(),
                "source_url": front_matter.get("source_url", "not-stated"),
                "published_at": front_matter.get("page_published_at", "not-stated"),
                "type": "legal" if "legal" in path.parts else "news",
            }
        )

    for line in body.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            flush()
            current_lines = [line]
            level = len(match.group(1))
            heading = match.group(2).strip()
            headings = headings[: level - 1]
            while len(headings) < level - 1:
                headings.append(title)
            headings.append(heading)
            current_heading = heading
        else:
            current_lines.append(line)
    flush()
    return sections


def _build_local_index() -> dict:
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        sections = _sections_from_markdown(path)
        documents.append(
            {
                "source": path.name,
                "title": sections[0]["title"] if sections else path.stem,
                "sections": sections,
            }
        )
    payload = {
        "method": "local-markdown-heading-tree",
        "corpus_fingerprint": _corpus_fingerprint(),
        "documents": documents,
    }
    _write_json_atomic(TREE_INDEX_PATH, payload)
    return {
        "documents": len(documents),
        "sections": sum(len(document["sections"]) for document in documents),
        "path": str(TREE_INDEX_PATH),
    }


def _load_cloud_manifest() -> dict:
    try:
        payload = json.loads(CLOUD_MANIFEST_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"documents": {}}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"documents": {}}


def _expand_query(query: str) -> str:
    lowered_words = set(re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE))
    additions = [
        vietnamese
        for english, vietnamese in QUERY_ALIASES.items()
        if english in lowered_words
    ]
    return " ".join([query, *additions]).strip()


def upload_documents(use_cloud: bool | None = None) -> dict:
    """Build the local index and upload changed legal PDFs to PageIndex cloud."""
    local_summary = _build_local_index()
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if use_cloud is None:
        use_cloud = bool(api_key)
    if not use_cloud:
        return {"local": local_summary, "cloud": {"enabled": False}}
    if not api_key:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    manifest = _load_cloud_manifest()
    cached_documents = manifest.setdefault("documents", {})
    uploaded = 0
    reused = 0
    active_paths: set[str] = set()
    for path in sorted(LEGAL_PDF_DIR.glob("*.pdf")):
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        active_paths.add(relative_path)
        fingerprint = _file_fingerprint(path)
        cached = cached_documents.get(relative_path, {})
        if cached.get("fingerprint") == fingerprint and cached.get("doc_id"):
            reused += 1
            continue
        response = client.submit_document(str(path))
        cached_documents[relative_path] = {
            "doc_id": response["doc_id"],
            "fingerprint": fingerprint,
            "source": path.name,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json_atomic(CLOUD_MANIFEST_PATH, manifest)
        uploaded += 1

    manifest["documents"] = {
        path: details
        for path, details in cached_documents.items()
        if path in active_paths
    }
    manifest["api"] = "https://api.pageindex.ai"
    _write_json_atomic(CLOUD_MANIFEST_PATH, manifest)
    return {
        "local": local_summary,
        "cloud": {
            "enabled": True,
            "uploaded": uploaded,
            "reused": reused,
            "documents": len(manifest["documents"]),
            "manifest": str(CLOUD_MANIFEST_PATH),
        },
    }


def pageindex_cloud_status() -> list[dict]:
    """Return cloud processing status without exposing the API key."""
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        return []
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    output: list[dict] = []
    for details in _load_cloud_manifest().get("documents", {}).values():
        metadata = client.get_document(details["doc_id"])
        tree = client.get_tree(details["doc_id"])
        output.append(
            {
                "source": details["source"],
                "doc_id": details["doc_id"],
                "status": tree.get("status", metadata.get("status", "unknown")),
                "retrieval_ready": tree.get("retrieval_ready", False),
            }
        )
    return output


def _load_sections() -> list[dict]:
    if TREE_INDEX_PATH.exists():
        try:
            payload = json.loads(TREE_INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    else:
        payload = {}
    if payload.get("corpus_fingerprint") != _corpus_fingerprint():
        _build_local_index()
        payload = json.loads(TREE_INDEX_PATH.read_text(encoding="utf-8"))
    return [
        section
        for document in payload.get("documents", [])
        for section in document.get("sections", [])
    ]


def _local_search(query: str, top_k: int) -> list[dict]:
    sections = _load_sections()
    if not sections:
        return []
    index_texts = [
        f"{section['title']} {section['section']} {section['content']}"
        for section in sections
    ]
    bm25 = BM25Okapi([tokenize(text) for text in index_texts], k1=1.2, b=0.65)
    raw_scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
    maximum = float(raw_scores.max()) if len(raw_scores) else 0.0
    if maximum <= 0:
        return []
    output: list[dict] = []
    for index in np.argsort(raw_scores)[::-1]:
        raw_score = float(raw_scores[index])
        if raw_score <= 0:
            continue
        section = sections[int(index)]
        output.append(
            {
                "content": section["content"],
                "score": round(raw_score / maximum, 6),
                "metadata": {
                    **{
                        key: section[key]
                        for key in (
                            "source",
                            "source_path",
                            "source_url",
                            "title",
                            "section",
                            "published_at",
                            "type",
                        )
                    },
                    "pageindex_backend": "local",
                },
                "source": "pageindex",
            }
        )
        if len(output) >= top_k:
            break
    return output


def _query_cloud_document(
    client, details: dict, query: str, timeout: float
) -> list[dict]:
    if not client.is_retrieval_ready(details["doc_id"]):
        return []
    submitted = client.submit_query(details["doc_id"], query, thinking=False)
    retrieval_id = submitted["retrieval_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get_retrieval(retrieval_id)
        status = payload.get("status")
        if status == "completed":
            output: list[dict] = []

            def content_items(value):
                if isinstance(value, dict):
                    yield value
                elif isinstance(value, list):
                    for child in value:
                        yield from content_items(child)

            for node_rank, node in enumerate(payload.get("retrieved_nodes", [])):
                for content_rank, item in enumerate(
                    content_items(node.get("relevant_contents", []))
                ):
                    content = item.get("relevant_content", "").strip()
                    if not content:
                        continue
                    physical_index = item.get("physical_index", "")
                    page_match = re.search(r"(\d+)", physical_index)
                    output.append(
                        {
                            "content": content,
                            "score": 1.0 / (1 + node_rank + content_rank),
                            "metadata": {
                                "source": details["source"],
                                "source_path": details["source_path"],
                                "title": node.get("title", details["source"]),
                                "section": node.get("title", "PageIndex result"),
                                "page": item.get("page_index")
                                or (int(page_match.group(1)) if page_match else None),
                                "type": "legal",
                                "pageindex_backend": "cloud",
                                "pageindex_doc_id": details["doc_id"],
                                "pageindex_node_id": node.get("node_id")
                                or node.get("id"),
                            },
                            "source": "pageindex",
                        }
                    )
            return output
        if status in {"failed", "error"}:
            return []
        time.sleep(1)
    return []


def _cloud_search(query: str, top_k: int) -> list[dict]:
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    documents = _load_cloud_manifest().get("documents", {})
    if not api_key or not documents:
        return []
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    timeout = float(os.getenv("PAGEINDEX_QUERY_TIMEOUT", "45"))
    details_list = [
        {**details, "source_path": source_path}
        for source_path, details in documents.items()
    ]
    # Route to likely documents with the local tree before asking PageIndex for
    # node-level retrieval. This mirrors PageIndex's document-level routing and
    # avoids launching an expensive cloud query for obvious out-of-domain text.
    routed = _local_search(query, top_k=5)
    candidate_stems = {
        Path(item.get("metadata", {}).get("source", "")).stem for item in routed
    }
    details_list = [
        details
        for details in details_list
        if Path(details["source"]).stem in candidate_stems
    ][: max(1, int(os.getenv("PAGEINDEX_CLOUD_DOCUMENTS", "2")))]
    if not details_list:
        return []

    cache_material = json.dumps(
        {
            "query": query,
            "top_k": top_k,
            "documents": sorted(details["doc_id"] for details in details_list),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = hashlib.sha1(cache_material.encode("utf-8")).hexdigest()
    try:
        query_cache = json.loads(CLOUD_QUERY_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        query_cache = {}
    if cache_key in query_cache:
        return query_cache[cache_key]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(3, len(details_list))) as executor:
        futures = {
            executor.submit(
                _query_cloud_document, client, details, query, timeout
            ): details
            for details in details_list
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except (KeyError, OSError, RuntimeError, ValueError):
                continue
    query_terms = set(tokenize(query))
    for item in results:
        metadata = item.get("metadata", {})
        result_terms = set(
            tokenize(
                f"{metadata.get('title', '')} {metadata.get('section', '')} "
                f"{item.get('content', '')}"
            )
        )
        coverage = len(query_terms & result_terms) / max(1, len(query_terms))
        item["pageindex_rank_score"] = item["score"]
        item["score"] = round(0.75 * coverage + 0.25 * item["score"], 6)
    results.sort(key=lambda item: item["score"], reverse=True)
    output = results[:top_k]
    query_cache[cache_key] = output
    _write_json_atomic(CLOUD_QUERY_CACHE_PATH, query_cache)
    return output


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Use PageIndex cloud when ready, otherwise use the local heading tree."""
    if not query.strip() or top_k <= 0:
        return []
    mode = os.getenv("PAGEINDEX_MODE", "auto").strip().lower()
    if mode not in {"auto", "cloud", "local"}:
        raise ValueError("PAGEINDEX_MODE must be auto, cloud, or local")
    expanded_query = _expand_query(query)
    if mode != "local":
        try:
            cloud_results = _cloud_search(expanded_query, top_k)
        except (ImportError, OSError, RuntimeError, ValueError):
            cloud_results = []
        if cloud_results:
            return cloud_results
    return _local_search(expanded_query, top_k)


if __name__ == "__main__":
    print(upload_documents())
    for result in pageindex_search("điều kiện tốt nghiệp", top_k=3):
        print(
            result["score"],
            result["metadata"]["pageindex_backend"],
            result["metadata"]["section"],
        )
