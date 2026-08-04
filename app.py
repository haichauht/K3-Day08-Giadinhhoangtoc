"""Grok-inspired, single-file Streamlit interface for the UIT RAG assistant.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT.parent / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.task10_generation import generate_with_citation  # noqa: E402

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="UIT Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --accent:#a7f3d0; --muted:#9ca3af; --line:rgba(255,255,255,.10); }
    .stApp {
        background:
          radial-gradient(circle at 58% -15%, rgba(31,111,83,.23), transparent 32rem),
          linear-gradient(180deg, #101211 0%, #090a0a 100%);
        color: #f4f4f5;
    }
    [data-testid="stSidebar"] {
        background: rgba(7,8,8,.97);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
    .block-container { max-width: 1080px; padding-top: 1.5rem; padding-bottom: 6rem; }
    .brand {font-size:1.05rem;font-weight:750;letter-spacing:.02em;margin-bottom:.15rem;}
    .brand-mark {color:var(--accent);margin-right:.35rem;}
    .status-pill {
        display:inline-flex;align-items:center;gap:.45rem;padding:.3rem .62rem;
        border:1px solid rgba(167,243,208,.22);border-radius:999px;
        background:rgba(16,185,129,.08);color:#bbf7d0;font-size:.76rem;
    }
    .status-dot {width:.42rem;height:.42rem;border-radius:50%;background:#34d399;box-shadow:0 0 10px #34d399;}
    .hero {text-align:center;padding:4.4rem 1rem 2.1rem;}
    .hero-symbol {font-size:2rem;color:var(--accent);margin-bottom:.8rem;}
    .hero h1 {font-size:clamp(2rem,5vw,3.35rem);letter-spacing:-.045em;margin:.15rem 0 .65rem;}
    .hero p {max-width:650px;margin:auto;color:#a1a1aa;font-size:1.02rem;line-height:1.65;}
    .eyebrow {color:#86efac;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;}
    [data-testid="stChatMessage"] {
        border:1px solid var(--line);border-radius:18px;padding:.85rem 1rem;
        background:rgba(255,255,255,.025);margin-bottom:.75rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background:rgba(31,111,83,.10);
    }
    [data-testid="stChatInput"] {background:#141716;border:1px solid rgba(255,255,255,.16);border-radius:18px;}
    [data-testid="stExpander"] {border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.018);}
    div.stButton > button {
        border:1px solid var(--line);background:rgba(255,255,255,.025);
        border-radius:13px;min-height:2.7rem;transition:all .15s ease;text-align:left;
    }
    div.stButton > button:hover {border-color:rgba(167,243,208,.45);background:rgba(16,185,129,.08);color:#d1fae5;}
    .suggestion-label {color:#a1a1aa;font-size:.8rem;margin:1rem 0 .5rem;}
    .source-kind {font-size:.7rem;color:#86efac;text-transform:uppercase;letter-spacing:.08em;}
    .trace {color:#71717a;font-size:.74rem;margin-top:.35rem;}
    .sidebar-note {font-size:.77rem;color:#71717a;line-height:1.55;}
    hr {border-color:var(--line)!important;}
    @media (max-width: 700px) {.hero {padding-top:2.2rem}.block-container{padding-left:1rem;padding-right:1rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    ("Tốt nghiệp từ xa", "Điều kiện xét tốt nghiệp đào tạo từ xa của UIT là gì?"),
    ("Cách tính học phí", "Học phí đào tạo từ xa được tính như thế nào?"),
    ("Học song ngành", "Điều kiện đăng ký học song ngành là gì?"),
    (
        "Chuyển đổi tín chỉ",
        "Mức tương đồng tối thiểu để chuyển đổi tín chỉ là bao nhiêu?",
    ),
    ("Tin UIT mới", "Lớp tập huấn AI cho chuyên viên UIT có nội dung gì?"),
    ("Hợp tác nghiên cứu", "UIT và Giáo sư Longsheng Fu hợp tác về những hướng nào?"),
]
FOLLOW_UP_PATTERN = re.compile(
    r"\b(đó|này|trên|còn|vậy|thế|họ|nó|bao lâu|ở đâu|khi nào)\b",
    flags=re.IGNORECASE,
)


def contextualize(question: str, messages: list[dict]) -> tuple[str, bool]:
    previous = [item["content"] for item in messages if item.get("role") == "user"]
    if not previous:
        return question, False
    if len(question.split()) > 10 and not FOLLOW_UP_PATTERN.search(question):
        return question, False
    return (
        f"Câu hỏi trước: {previous[-1]}\nCâu hỏi hiện tại cần trả lời: {question}",
        True,
    )


def set_pending(question: str) -> None:
    st.session_state.pending_query = question


def render_sources(sources: list[dict], message_index: int) -> None:
    if not sources:
        st.caption("Không tìm thấy bằng chứng phù hợp trong kho dữ liệu.")
        return
    with st.expander(f"▣  {len(sources)} đoạn bằng chứng", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source", "Không rõ nguồn")
            section = metadata.get("section", "Không rõ mục")
            source_url = metadata.get("source_url", "")
            score = source.get("score")
            kind = "QUY ĐỊNH" if metadata.get("type") == "legal" else "TIN UIT"
            with st.container(border=True):
                title_col, score_col = st.columns([5, 1])
                with title_col:
                    st.markdown(
                        f'<div class="source-kind">{kind} · nguồn {index}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{title}**")
                with score_col:
                    if score is not None:
                        st.metric(
                            "score", f"{float(score):.3f}", label_visibility="visible"
                        )
                st.caption(section)
                st.text(source.get("content", "").strip()[:900])
                if source_url and source_url != "not-stated":
                    st.markdown(f"[↗ Mở trang nguồn]({source_url})")


def render_assistant_meta(message: dict) -> None:
    details = []
    if message.get("model"):
        details.append(message["model"])
    if message.get("retrieval_source"):
        details.append(message["retrieval_source"])
    if message.get("elapsed") is not None:
        details.append(f"{message['elapsed']:.1f}s")
    if details:
        st.markdown(
            f'<div class="trace">{" · ".join(details)}</div>', unsafe_allow_html=True
        )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span> UIT Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.caption("Grounded answers for UIT")
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>RAG pipeline ready</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("＋  Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.subheader("Khám phá")
    for index, (label, question) in enumerate(SUGGESTIONS[:4]):
        st.button(
            f"{label}  →",
            key=f"side_suggestion_{index}",
            use_container_width=True,
            on_click=set_pending,
            args=(question,),
        )

    st.divider()
    st.subheader("Tùy chỉnh")
    top_k = st.slider(
        "Số đoạn bằng chứng",
        3,
        8,
        5,
        help="Nhiều đoạn hơn tăng độ phủ nhưng làm prompt dài hơn.",
    )
    show_trace = st.toggle("Hiện chi tiết pipeline", value=True)

    st.divider()
    user_turns = sum(item["role"] == "user" for item in st.session_state.messages)
    st.caption(f"Phiên hiện tại · {user_turns} câu hỏi")
    st.markdown(
        '<div class="sidebar-note">8 tài liệu · 139 chunks<br>NVIDIA embeddings · BM25 · RRF<br>Qwen 3.6 27B via Groq</div>',
        unsafe_allow_html=True,
    )

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        '<div class="eyebrow">University knowledge assistant</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>Online</div>',
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-symbol">✦</div>
          <h1>Hỏi điều bạn cần biết về UIT</h1>
          <p>Tra cứu quy chế đào tạo, chuyển đổi tín chỉ, song ngành và tin tức UIT.
          Mỗi câu trả lời đều đi kèm đoạn nguồn để bạn tự kiểm chứng.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="suggestion-label">BẮT ĐẦU VỚI MỘT CÂU HỎI</div>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(SUGGESTIONS), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            suggestion_index = row_start + offset
            if suggestion_index >= len(SUGGESTIONS):
                continue
            label, question = SUGGESTIONS[suggestion_index]
            with column:
                st.button(
                    f"{label}\n\n{question}",
                    key=f"main_suggestion_{suggestion_index}",
                    use_container_width=True,
                    on_click=set_pending,
                    args=(question,),
                )
else:
    st.title("Cuộc trò chuyện")
    for message_index, message in enumerate(st.session_state.messages):
        avatar = "🧑" if message["role"] == "user" else "🎓"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if show_trace:
                    render_assistant_meta(message)
                render_sources(message.get("sources", []), message_index)
                feedback_cols = st.columns([1, 1, 8])
                feedback_cols[0].button(
                    "♡", key=f"helpful_{message_index}", help="Hữu ích"
                )
                feedback_cols[1].button(
                    "⚑", key=f"report_{message_index}", help="Báo câu trả lời chưa tốt"
                )

typed_query = st.chat_input("Hỏi về quy chế hoặc hoạt động UIT…", key="main_chat_input")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    retrieval_query, used_memory = contextualize(query, st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🎓"):
        if used_memory:
            st.caption("↳ Đã nối câu hỏi trước để hiểu lượt follow-up.")
        started = time.perf_counter()
        with st.spinner("Đọc tài liệu và đối chiếu nguồn…"):
            try:
                response = generate_with_citation(retrieval_query, top_k=top_k)
                answer = response["answer"]
                sources = response.get("sources", [])
                model = response.get("model", "")
                retrieval_source = response.get("retrieval_source", "")
            except Exception as error:
                LOGGER.exception("RAG request failed")
                answer = "Không thể xử lý câu hỏi lúc này. Vui lòng thử lại sau."
                sources, model, retrieval_source = [], "", "error"
                if show_trace:
                    st.caption(f"Chi tiết kỹ thuật: {type(error).__name__}: {error}")
        elapsed = time.perf_counter() - started
        st.markdown(answer)
        if show_trace:
            render_assistant_meta(
                {
                    "model": model,
                    "retrieval_source": retrieval_source,
                    "elapsed": elapsed,
                }
            )
        render_sources(sources, len(st.session_state.messages))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "model": model,
            "retrieval_source": retrieval_source,
            "elapsed": elapsed,
        }
    )
    st.rerun()
