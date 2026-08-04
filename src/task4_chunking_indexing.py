"""Task 4 - Markdown-aware chunking, embedding, and Chroma indexing.

Primary embeddings use NVIDIA's OpenAI-compatible NIM API.  A second Chroma
collection is built with a multilingual MiniLM model cached on this machine,
so retrieval can continue when the API is unavailable.  The collections are
kept separate because their vector dimensions differ (2048 vs 384).  The much
larger cached BAAI/bge-m3 remains an opt-in alternative through the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = REPO_ROOT / "data" / "standardized"
CHROMA_DIR = REPO_ROOT / "chroma_db"
MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"

load_dotenv(REPO_ROOT.parent / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

# Headings preserve legal articles/chapters; the recursive splitter only cuts a
# section when it is too long.  1,200 characters usually contains one coherent
# provision, while 180 characters of overlap keeps cross-boundary references.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
CHUNKING_METHOD = "markdown_header+recursive"

EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
EMBEDDING_DIM = 2048
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LOCAL_EMBEDDING_DIM = 384
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "uit_rag_nvidia"
LOCAL_COLLECTION_NAME = "uit_rag_local_multilingual"
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")

_local_model = None


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            pass
    return value.strip("'\"")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return simple YAML metadata and the Markdown body."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = _parse_scalar(value)
    return metadata, "\n".join(lines[end + 1 :]).strip()


def load_documents() -> list[dict]:
    """Load all standardized Markdown documents with citation metadata."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        front_matter, body = split_front_matter(raw)
        if not body.strip():
            continue
        relative = md_file.relative_to(REPO_ROOT).as_posix()
        doc_type = "legal" if "legal" in md_file.parts else "news"
        title = front_matter.get("title") or md_file.stem.replace("-", " ").title()
        documents.append(
            {
                "content": body,
                "metadata": {
                    "doc_id": front_matter.get("doc_id", md_file.stem),
                    "source": md_file.name,
                    "source_path": relative,
                    "source_url": front_matter.get("source_url", "not-stated"),
                    "title": title,
                    "type": doc_type,
                    "published_at": front_matter.get("page_published_at", "not-stated"),
                    "retrieved_at": front_matter.get("retrieved_at", "not-stated"),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents by Markdown hierarchy, then by bounded text length."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for document in documents:
        sections = header_splitter.split_text(document["content"])
        if not sections:
            sections = length_splitter.create_documents([document["content"]])

        chunk_index = 0
        for section in sections:
            section_path = " > ".join(
                str(section.metadata[key])
                for key in ("h1", "h2", "h3", "h4")
                if section.metadata.get(key)
            )
            for piece in length_splitter.split_text(section.page_content):
                content = piece.strip()
                if not content:
                    continue
                chunk_id = hashlib.sha1(
                    f"{document['metadata']['source_path']}:{chunk_index}:{content}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                chunks.append(
                    {
                        "content": content,
                        "metadata": {
                            **document["metadata"],
                            "section": section_path or document["metadata"]["title"],
                            "chunk_index": chunk_index,
                            "chunk_id": chunk_id,
                        },
                    }
                )
                chunk_index += 1
    return chunks


def _normalise(vectors: Iterable[Iterable[float]]) -> list[list[float]]:
    array = np.asarray(list(vectors), dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).tolist()


def embed_texts_nvidia(
    texts: list[str], input_type: str = "passage", batch_size: int = 32
) -> list[list[float]]:
    """Embed text through NVIDIA NIM, preserving response order."""
    if input_type not in {"query", "passage"}:
        raise ValueError("input_type must be 'query' or 'passage'")
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = session.post(
            f"{NVIDIA_BASE_URL}/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
                "input_type": input_type,
                "modality": "text",
                "embedding_type": "float",
                "encoding_format": "float",
                "truncate": "END",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return _normalise(vectors)


def get_local_embedding_model():
    """Load the configured model strictly from the Hugging Face cache."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL,
            local_files_only=True,
        )
    return _local_model


def embed_texts_local(
    texts: list[str], input_type: str = "passage", batch_size: int = 8
) -> list[list[float]]:
    """Embed locally; query and passage share the same sentence encoder."""
    del input_type
    model = get_local_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32).tolist()


def embed_chunks(chunks: list[dict], backend: str = "nvidia") -> list[dict]:
    """Return copies of chunks with an embedding from the selected backend."""
    texts = [chunk["content"] for chunk in chunks]
    if backend == "nvidia":
        embeddings = embed_texts_nvidia(texts, input_type="passage")
    elif backend == "local":
        embeddings = embed_texts_local(texts, input_type="passage")
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def get_chroma_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = COLLECTION_NAME):
    return get_chroma_client().get_collection(name=name)


def index_to_vectorstore(
    chunks: list[dict], collection_name: str = COLLECTION_NAME
) -> int:
    """Rebuild a Chroma collection from the supplied embedded chunks."""
    from chromadb.errors import NotFoundError

    if not chunks:
        return 0
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        collection.upsert(
            ids=[item["metadata"]["chunk_id"] for item in batch],
            documents=[item["content"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
    return collection.count()


def run_pipeline(with_local_fallback: bool = True) -> dict:
    """Load, chunk, index API embeddings, and optionally build local fallback."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError(f"No Markdown chunks found under {STANDARDIZED_DIR}")

    backends: dict[str, dict] = {}
    primary = embed_chunks(chunks, backend="nvidia")
    backends["nvidia"] = {
        "collection": COLLECTION_NAME,
        "model": EMBEDDING_MODEL,
        "dimension": len(primary[0]["embedding"]),
        "count": index_to_vectorstore(primary, COLLECTION_NAME),
    }

    if with_local_fallback:
        local = embed_chunks(chunks, backend="local")
        backends["local"] = {
            "collection": LOCAL_COLLECTION_NAME,
            "model": LOCAL_EMBEDDING_MODEL,
            "dimension": len(local[0]["embedding"]),
            "count": index_to_vectorstore(local, LOCAL_COLLECTION_NAME),
        }

    manifest = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunking": {
            "method": CHUNKING_METHOD,
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
        },
        "backends": backends,
    }
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk and index standardized data")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Skip the cached multilingual fallback collection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(with_local_fallback=not args.api_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
