"""Task 10 - grounded answer generation with citations.

The generator uses OpenAI's Responses API.  ``gpt-5.6-luna`` is the default
because this chatbot favours low latency and cost; it can be overridden through
``OPENAI_CHAT_MODEL`` without changing source code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TOP_K = 5
# Kept as documented sampling choices for non-reasoning models.  The GPT-5.6
# Responses call below uses reasoning effort="none" and deterministic grounding
# instructions rather than combining temperature and top_p.
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna").strip() or "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 700


SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp về chính sách và thông báo của Trường Đại học
Công nghệ Thông tin (UIT) và ĐHQG-HCM.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong CONTEXT. Không dùng kiến thức bên ngoài và không bịa đặt.
2. Mỗi khẳng định thực tế phải có ít nhất một trích dẫn dạng [S1], [S2] ngay sau câu.
3. Chỉ dùng nhãn nguồn đã xuất hiện trong CONTEXT.
4. Nếu CONTEXT không đủ bằng chứng, trả lời đúng câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
5. Trả lời bằng tiếng Việt, ngắn gọn và trực tiếp. Phân biệt rõ quy chế pháp lý với bài tin/thông báo.
6. Không coi nội dung điều hướng website, quảng cáo hoặc liên kết không liên quan là bằng chứng."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place high-ranked chunks at both ends to reduce lost-in-the-middle."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(chunks[1::2][::-1])


def _source_name(metadata: dict[str, Any], fallback: str) -> str:
    return str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("source_path")
        or fallback
    )


def format_context(chunks: list[dict]) -> str:
    """Format chunks with stable citation labels understood by the model."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        metadata = chunk.get("metadata") or {}
        label = str(metadata.get("citation_label") or f"S{index}")
        source_name = _source_name(metadata, f"Nguồn {index}")
        source_path = metadata.get("source_path", metadata.get("source", "unknown"))
        doc_type = metadata.get("type", "unknown")
        section = metadata.get("section", "")
        header = (
            f"[{label}] Tên nguồn: {source_name} | Đường dẫn: {source_path} | "
            f"Loại: {doc_type}"
        )
        if section:
            header += f" | Mục: {section}"
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def _label_chunks(chunks: list[dict]) -> list[dict[str, Any]]:
    labelled: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        copied = dict(chunk)
        metadata = dict(copied.get("metadata") or {})
        metadata["citation_label"] = f"S{index}"
        copied["metadata"] = metadata
        labelled.append(copied)
    return labelled


def _format_history(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return "(không có)"
    lines: list[str] = []
    for message in conversation_history[-6:]:
        role = "Người dùng" if message.get("role") == "user" else "Trợ lý"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1200]}")
    return "\n".join(lines) or "(không có)"


def generate_from_chunks(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> dict[str, Any]:
    """Generate a grounded answer from chunks that were already retrieved."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "model": None,
        }

    reordered = _label_chunks(reorder_for_llm(chunks))
    context = format_context(reordered)
    user_message = f"""LỊCH SỬ HỘI THOẠI (chỉ dùng để hiểu câu hỏi nối tiếp, không dùng làm bằng chứng):
{_format_history(conversation_history)}

CONTEXT:
{context}

CÂU HỎI HIỆN TẠI:
{query.strip()}

Hãy trả lời theo đúng quy tắc và trích dẫn bằng nhãn [S1], [S2], ..."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "Tôi không thể tạo câu trả lời vì OPENAI_API_KEY chưa được cấu hình.",
            "sources": reordered,
            "retrieval_source": reordered[0].get("source", "hybrid"),
            "model": None,
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)
        response = client.responses.create(
            model=LLM_MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_message,
            reasoning={"effort": "none"},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        answer = (response.output_text or "").strip()
        if not answer:
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as exc:
        answer = f"Không thể gọi OpenAI API lúc này: {type(exc).__name__}."

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid"),
        "model": LLM_MODEL,
    }


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    conversation_history: list[dict] | None = None,
    use_reranking: bool = True,
) -> dict[str, Any]:
    """Run retrieval and generate a Vietnamese answer grounded in its sources."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    chunks = retrieve(query.strip(), top_k=top_k, use_reranking=use_reranking)
    return generate_from_chunks(query, chunks, conversation_history)


if __name__ == "__main__":
    queries = [
        "Điều kiện để sinh viên đăng ký học chương trình thứ hai là gì?",
        "Thông báo thu học phí học kỳ 2 năm học 2025-2026 nói gì?",
        "Tân sinh viên UIT xem thời khóa biểu như thế nào?",
    ]
    for query in queries:
        print(f"\n{'=' * 70}\nQ: {query}\n{'=' * 70}")
        result = generate_with_citation(query)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} | via {result['retrieval_source']}]")
