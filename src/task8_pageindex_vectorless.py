"""Task 8 - PageIndex vectorless retrieval.

Legal PDFs are uploaded to PageIndex and their document IDs are cached under
``.runtime/``.  If the service is temporarily unavailable, a structural local
fallback ranks Markdown sections by their headings and exact terms; this keeps
the chatbot usable without pretending that a failed API call succeeded.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from .task6_lexical_search import _tokenize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
LEGAL_PDF_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
STATE_FILE = PROJECT_ROOT / ".runtime" / "pageindex_documents.json"

load_dotenv(ENV_FILE)
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()


def _load_state() -> list[dict[str, str]]:
    if not STATE_FILE.exists():
        return []
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _save_state(documents: list[dict[str, str]]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_documents(force: bool = False) -> list[dict[str, str]]:
    """Upload the three landing legal PDFs and persist their PageIndex IDs."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured in .env")
    pdf_files = sorted(LEGAL_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No legal PDFs found in {LEGAL_PDF_DIR}")

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    existing = {item.get("source_file"): item for item in _load_state()}
    documents: list[dict[str, str]] = []

    for pdf_file in pdf_files:
        cached = existing.get(pdf_file.name)
        if cached and cached.get("doc_id") and not force:
            documents.append(cached)
            print(f"  Reusing: {pdf_file.name} -> {cached['doc_id']}")
            continue

        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}: {response}")
        record = {"source_file": pdf_file.name, "doc_id": str(doc_id)}
        documents.append(record)
        print(f"  Uploaded: {pdf_file.name} -> {doc_id}")

    _save_state(documents)
    return documents


def wait_until_ready(timeout_seconds: int = 600, poll_seconds: int = 5) -> list[dict[str, str]]:
    """Wait until cached PageIndex documents are ready for retrieval."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured in .env")
    documents = _load_state()
    if not documents:
        documents = upload_documents()

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    pending = {item["doc_id"]: item for item in documents}
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        for doc_id in list(pending):
            if client.is_retrieval_ready(doc_id):
                print(f"  Ready: {pending[doc_id]['source_file']}")
                pending.pop(doc_id)
        if pending:
            time.sleep(poll_seconds)
    if pending:
        names = ", ".join(item["source_file"] for item in pending.values())
        raise TimeoutError(f"PageIndex processing timed out for: {names}")
    return documents


def _iter_relevant_items(value: Any) -> Iterable[dict[str, Any]]:
    """Recursively yield objects containing PageIndex relevant content."""
    if isinstance(value, dict):
        if value.get("relevant_content"):
            yield value
        for child in value.values():
            yield from _iter_relevant_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_relevant_items(child)


def _remote_search(
    query: str,
    top_k: int,
    candidate_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    from pageindex.client import PageIndexClient

    documents = _load_state()
    if candidate_sources:
        documents = [item for item in documents if item.get("source_file") in candidate_sources]
    if not documents:
        return []
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    collected: list[dict[str, Any]] = []

    for document in documents:
        doc_id = document["doc_id"]
        if not client.is_retrieval_ready(doc_id):
            continue
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            continue

        deadline = time.monotonic() + 90
        response: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                response = client.get_retrieval(str(retrieval_id))
            except Exception:
                time.sleep(2)
                continue
            status = str(response.get("status", "")).casefold()
            if response.get("retrieved_nodes") or status in {
                "completed", "complete", "ready", "succeeded", "success", "done"
            }:
                break
            if status in {"failed", "error", "cancelled"}:
                response = {}
                break
            time.sleep(1)

        for item in _iter_relevant_items(response.get("retrieved_nodes", [])):
            content = str(item.get("relevant_content", "")).strip()
            if not content:
                continue
            collected.append(
                {
                    "content": content,
                    "score": 1.0,
                    "metadata": {
                        "source": document["source_file"],
                        "source_path": f"legal/{document['source_file']}",
                        "type": "legal",
                        "section": item.get("section_title", ""),
                        "pageindex_doc_id": doc_id,
                    },
                    "source": "pageindex",
                }
            )

    # PageIndex does not expose a comparable similarity score; use reciprocal
    # rank only inside this PageIndex list.
    for rank, item in enumerate(collected, start=1):
        item["score"] = 1.0 / rank
    return collected[:top_k]


def _markdown_sections() -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        heading = md_file.stem
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append(
                    {
                        "content": f"{heading}\n{content}"[:4000],
                        "metadata": {
                            "source": md_file.name,
                            "source_path": relative,
                            "type": relative.split("/", 1)[0],
                            "section": heading,
                            "pageindex_mode": "local_structural_fallback",
                        },
                    }
                )

        for line in text.splitlines():
            if line.startswith("#") and line.lstrip("#").startswith(" "):
                flush()
                body = []
                heading = line.lstrip("#").strip()
            else:
                body.append(line)
        flush()
    return sections


def _local_structural_search(query: str, top_k: int) -> list[dict[str, Any]]:
    query_tokens = set(_tokenize(query, expand_synonyms=True))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for section in _markdown_sections():
        metadata = section["metadata"]
        heading_tokens = set(_tokenize(str(metadata.get("section", ""))))
        content_tokens = _tokenize(section["content"])
        content_set = set(content_tokens)
        heading_overlap = len(query_tokens & heading_tokens)
        content_overlap = len(query_tokens & content_set)
        if content_overlap == 0:
            continue
        coverage = content_overlap / max(len(query_tokens), 1)
        density = sum(content_tokens.count(token) for token in query_tokens) / max(len(content_tokens), 1)
        score = coverage + 0.25 * heading_overlap + 0.1 * density
        ranked.append((score, section))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            **section,
            "score": float(score),
            "source": "pageindex",
        }
        for score, section in ranked[:top_k]
    ]


def pageindex_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve with PageIndex, falling back to local structural search."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    local_results = _local_structural_search(query.strip(), top_k)
    # An exact-term prefilter avoids sending obvious nonsense to a paid/remote
    # retrieval service and makes the fallback test fast and deterministic.
    if not local_results:
        return []

    candidate_sources = {
        f"{Path(str(item['metadata'].get('source', ''))).stem}.pdf"
        for item in local_results
        if item.get("metadata", {}).get("type") == "legal"
    }
    if PAGEINDEX_API_KEY and _load_state() and candidate_sources:
        try:
            remote_results = _remote_search(query.strip(), top_k, candidate_sources)
            if remote_results:
                return remote_results
        except Exception as exc:
            print(f"PageIndex API unavailable; using structural fallback: {exc}")
    return local_results


if __name__ == "__main__":
    if PAGEINDEX_API_KEY:
        print("Uploading/reusing legal documents...")
        upload_documents()
        print("Waiting for PageIndex processing...")
        wait_until_ready()
    else:
        print("PAGEINDEX_API_KEY is absent; using local structural fallback.")

    print("\nTest query:")
    for result in pageindex_search("điều kiện học song ngành", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}: {result['content'][:100]}...")
