"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdf"
DOC_INDEX_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_index.json"


def _unicode_font_path() -> str | None:
    """Tìm 1 font TrueType hỗ trợ tiếng Việt có sẵn trên máy (Windows Arial)."""
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _convert_md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Convert 1 file markdown sang PDF đơn giản (plain text) bằng fpdf2."""
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = _unicode_font_path()
    if font_path:
        pdf.add_font("Body", "", font_path, uni=True)
        pdf.set_font("Body", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
        text = text.encode("latin-1", "replace").decode("latin-1")

    for line in text.splitlines() or [""]:
        pdf.multi_cell(0, 6, line)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.

    PageIndex chỉ nhận PDF nên mỗi .md trong data/standardized/ được convert
    sang PDF (data/pageindex_pdf/) trước khi upload. doc_id trả về được lưu
    lại vào data/pageindex_doc_index.json để pageindex_search() dùng lại mà
    không cần upload lại mỗi lần chạy.
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_index: dict[str, str] = {}

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        pdf_path = PDF_DIR / f"{md_file.stem}.pdf"
        _convert_md_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_index[str(md_file.relative_to(STANDARDIZED_DIR))] = doc_id
        print(f"  Uploaded: {md_file.name} -> {doc_id}")

    DOC_INDEX_PATH.write_text(json.dumps(doc_index, indent=2), encoding="utf-8")
    return doc_index


def _wait_for_retrieval_ready(client, doc_id: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.is_retrieval_ready(doc_id):
            return
        time.sleep(2)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")
    if not DOC_INDEX_PATH.exists():
        raise RuntimeError(
            "Chưa có document nào trên PageIndex — chạy upload_documents() trước"
        )

    doc_index: dict[str, str] = json.loads(DOC_INDEX_PATH.read_text(encoding="utf-8"))
    if not doc_index:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    results: list[dict] = []

    # Query từng doc đã upload (demo-scale corpus); dừng sớm khi đủ top_k.
    for source, doc_id in doc_index.items():
        if len(results) >= top_k:
            break

        _wait_for_retrieval_ready(client, doc_id)
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")

        retrieval = client.get_retrieval(retrieval_id)
        deadline = time.monotonic() + 60.0
        while retrieval.get("status") not in ("completed", "failed") and time.monotonic() < deadline:
            time.sleep(2)
            retrieval = client.get_retrieval(retrieval_id)

        for rank, node in enumerate(retrieval.get("retrieved_nodes", [])[:2], start=1):
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": round(1.0 / rank, 4),
                        "metadata": {
                            "source": source,
                            "section": item.get("section_title"),
                        },
                        "source": "pageindex",
                    })

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
