"""Task 10 - Evidence-grounded generation with inline citations."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .task9_retrieval_pipeline import retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT.parent / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

# Five chunks provide multiple evidence points while keeping the prompt compact.
TOP_K = 5
# Low temperature is appropriate for factual RAG; top_p still permits natural
# Vietnamese wording without encouraging unsupported creativity.
TOP_P = 0.9
TEMPERATURE = 0.1
LLM_MODEL = "qwen/qwen3.6-27b"
UNVERIFIABLE = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


SYSTEM_PROMPT = f"""Bạn là trợ lý hỏi đáp chính sách và hoạt động của UIT/ĐHQG-HCM.

Quy tắc bắt buộc:
1. Chỉ dùng dữ kiện xuất hiện rõ ràng trong CONTEXT; không dùng kiến thức ngoài.
2. Sau mỗi câu chứa dữ kiện, chèn đúng nhãn citation đã cho, dạng [Nguồn, Năm].
3. Không tự tạo tên nguồn, năm, số điều hoặc con số.
4. Nếu context không đủ bằng chứng, chỉ trả lời: \"{UNVERIFIABLE}\"
5. Trả lời bằng tiếng Việt, trực tiếp và dễ đọc.
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place high-ranked chunks at prompt edges to reduce lost-in-the-middle."""
    if len(chunks) <= 2:
        return list(chunks)
    front = list(chunks[::2])
    back = list(chunks[1::2])
    return front + back[::-1]


def citation_label(chunk: dict, index: int = 1) -> str:
    metadata = chunk.get("metadata", {})
    title = metadata.get("title") or metadata.get("source") or f"Nguồn {index}"
    title = re.sub(r"\.md$", "", str(title), flags=re.IGNORECASE)
    date = metadata.get("published_at") or metadata.get("retrieved_at") or ""
    year_match = re.search(r"\b(19|20)\d{2}\b", str(date))
    year = year_match.group(0) if year_match else "không rõ năm"
    return f"[{title}, {year}]"


def format_context(chunks: list[dict]) -> str:
    """Format evidence with exact citation labels the model is allowed to use."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        parts.append(
            "\n".join(
                [
                    f'<evidence id="{index}">',
                    f"Citation bắt buộc: {citation_label(chunk, index)}",
                    f"Mục: {metadata.get('section', 'không rõ')}",
                    f"Loại: {metadata.get('type', 'không rõ')}",
                    str(chunk.get("content", "")).strip(),
                    "</evidence>",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _extractive_fallback(chunks: list[dict]) -> str:
    statements: list[str] = []
    for index, chunk in enumerate(chunks[:2], start=1):
        content = re.sub(r"^#{1,6}\s+", "", chunk.get("content", "").strip())
        sentence = re.split(r"(?<=[.!?])\s+|\n\n+", content, maxsplit=1)[0].strip()
        if sentence:
            statements.append(f"{sentence} {citation_label(chunk, index)}")
    return "\n\n".join(statements) if statements else UNVERIFIABLE


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    context_chunks: list[dict] | None = None,
) -> dict:
    """Retrieve evidence, reorder it, call Groq, and return answer + sources."""
    chunks = (
        context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)
    )
    if not chunks:
        return {"answer": UNVERIFIABLE, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"CONTEXT:\n{context}\n\nCÂU HỎI: {query}"
    answer: str
    model_used = LLM_MODEL
    try:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=900,
            # Qwen 3.6 enables reasoning by default.  RAG only needs the final
            # grounded answer; disabling reasoning avoids spending the output
            # budget on a hidden/visible chain of thought.
            extra_body={
                "reasoning_effort": "none",
                "reasoning_format": "hidden",
            },
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = UNVERIFIABLE
    except (OpenAIError, RuntimeError, AttributeError, IndexError):
        model_used = "extractive-fallback"
        answer = _extractive_fallback(reordered)

    if answer != UNVERIFIABLE and not re.search(r"\[[^\]]+,\s*[^\]]+\]", answer):
        answer = f"{answer}\n\n{citation_label(reordered[0])}"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "model": model_used,
    }


if __name__ == "__main__":
    result = generate_with_citation("Điều kiện xét tốt nghiệp đào tạo từ xa là gì?")
    print(result["answer"])
    print(f"Sources: {len(result['sources'])}; model={result['model']}")
