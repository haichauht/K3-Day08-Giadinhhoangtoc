"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import os
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
ENV_FILE = Path(__file__).parent.parent / ".env"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# 800/100 là cấu hình checkpoint của LAB_GUIDE. Kích thước này đủ giữ một
# đoạn giải thích hoặc điều khoản ngắn, còn overlap 12.5% hạn chế mất ngữ cảnh
# tại ranh giới mà không làm số vector tăng quá nhiều.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 80
MAX_CHUNK_SIZE = int(CHUNK_SIZE * 1.1)
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# text-embedding-3-small hỗ trợ multilingual và chạy qua API nên không phải tải
# model nặng về máy. OpenAI cho phép giảm dimension; dùng 1024 để giữ kích thước
# vector tương đương cấu hình BGE-M3 ban đầu và giảm dung lượng ChromaDB.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 64

# ChromaDB lưu local, persistent và không yêu cầu Docker.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

# Recursive splitter ưu tiên ranh giới Markdown và điều khoản pháp lý trước
# khi phải cắt theo câu, từ hoặc ký tự.
CHUNK_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\nĐiều ",
    "\nKhoản ",
    "\n\n",
    "\n",
    ". ",
    "; ",
    " ",
    "",
]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _document_title(content: str, fallback: str) -> str:
    """Return the first level-one Markdown heading, when present."""
    for line in content.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if title:
                return title
    return fallback


def load_documents() -> list[dict[str, Any]]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Standardized data directory not found: {STANDARDIZED_DIR}")

    documents: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = relative_path.split("/", 1)[0]
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path,
                    "type": doc_type,
                    "title": _document_title(content, md_file.stem),
                },
            }
        )

    if not documents:
        raise ValueError(f"No non-empty Markdown documents found in {STANDARDIZED_DIR}")
    return documents


def _merge_short_splits(splits: list[str]) -> list[str]:
    """Attach tiny headings/page markers to adjacent substantive chunks."""
    pending = [split.strip() for split in splits if split.strip()]
    merged: list[str] = []

    for index, chunk_text in enumerate(pending):
        # PDF page markers can split one legal Article in half (for example,
        # conditions 1-2 at the end of page 3 and condition 3 on page 4).
        # Attach short page continuations backward so a single retrieved chunk
        # retains the complete rule and its heading.
        if chunk_text.startswith("## Trang ") and len(chunk_text) < 300 and merged:
            combined = f"{merged[-1]}\n\n{chunk_text}"
            if len(combined) <= MAX_CHUNK_SIZE:
                merged[-1] = combined
                continue
        if len(chunk_text) < MIN_CHUNK_SIZE:
            if index + 1 < len(pending):
                combined = f"{chunk_text}\n\n{pending[index + 1]}"
                if len(combined) <= MAX_CHUNK_SIZE:
                    pending[index + 1] = combined
                    continue
            if merged:
                combined = f"{merged[-1]}\n\n{chunk_text}"
                if len(combined) <= MAX_CHUNK_SIZE:
                    merged[-1] = combined
                    continue
        merged.append(chunk_text)

    return merged


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )

    chunks: list[dict[str, Any]] = []
    for document in documents:
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        if not isinstance(content, str) or not content.strip():
            continue
        if not isinstance(metadata, dict):
            raise TypeError("Document metadata must be a dictionary")

        splits = _merge_short_splits(splitter.split_text(content))
        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text),
                    },
                }
            )

    if not chunks:
        raise ValueError("Chunking produced no content")
    return chunks


def create_embeddings(
    texts: list[str],
    *,
    show_progress: bool = False,
) -> list[list[float]]:
    """Create OpenAI embeddings using the shared Task 4 model configuration."""
    if not texts:
        raise ValueError("Cannot embed an empty text list")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every embedding input must be a non-empty string")

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ENV_FILE)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(f"OPENAI_API_KEY is not configured in {ENV_FILE}")

    client = OpenAI(timeout=120.0, max_retries=5)
    embeddings: list[list[float]] = []
    total_tokens = 0
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(texts), EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding API returned an unexpected number of vectors")

        for chunk, vector in zip(batch, vectors):
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(vector)}; "
                    f"expected {EMBEDDING_DIM}"
                )
            embeddings.append(vector)

        total_tokens += response.usage.total_tokens
        if show_progress:
            print(f"  Embedded batch {batch_number}/{total_batches}")

    if show_progress:
        print(f"  Embedding tokens used: {total_tokens}")
    return embeddings


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        raise ValueError("Cannot embed an empty chunk list")

    vectors = create_embeddings(
        [chunk["content"] for chunk in chunks],
        show_progress=True,
    )
    embedded_chunks = [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors)
    ]
    if len(embedded_chunks) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")

    return embedded_chunks


def get_collection():
    """Open and validate the persistent ChromaDB collection."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}; run Task 4 indexing first"
        )

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME not in collection_names:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} not found; run Task 4 indexing first"
        )

    collection = client.get_collection(COLLECTION_NAME)
    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError("ChromaDB embedding model differs from the Task 4 config")
    if metadata.get("embedding_dimension") != EMBEDDING_DIM:
        raise RuntimeError("ChromaDB embedding dimension differs from the Task 4 config")
    return collection


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> int:
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        raise ValueError("Cannot index an empty chunk list")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Every chunk must contain an embedding before indexing")

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate the collection so the persistent index always mirrors the
    # current corpus and never retains chunks from deleted documents.
    existing_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )

    ids = [
        f"{chunk['metadata']['source_path']}::chunk-{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs are not unique")

    collection.add(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    indexed_count = collection.count()
    if indexed_count != len(chunks):
        raise RuntimeError(
            f"ChromaDB count mismatch: indexed {indexed_count}, expected {len(chunks)}"
        )
    return indexed_count


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    indexed_count = index_to_vectorstore(chunks)
    print(f"[OK] Indexed {indexed_count} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
