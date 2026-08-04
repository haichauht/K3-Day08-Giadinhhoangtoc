"""
RAG Chatbot — University Services (UIT)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10), có
trực quan hoá pipeline (hybrid retrieval, RRF, rerank, fallback) và giải
thích lý thuyết đã áp dụng ở từng bước.

Chạy:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task4_chunking_indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)
from src.task9_retrieval_pipeline import DEFAULT_TOP_K, SCORE_THRESHOLD
from src.task10_generation import LLM_MODEL, TEMPERATURE

# =============================================================================
# PAGE CONFIG & STYLE
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-hybrid { background: #E0F2FE; color: #075985; }
    .badge-pageindex { background: #FEF3C7; color: #92400E; }
    .badge-legal { background: #DBEAFE; color: #1E40AF; }
    .badge-news { background: #DCFCE7; color: #166534; }
    .source-card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 10px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    .score-bar-bg { background: rgba(128,128,128,0.15); border-radius: 6px; height: 8px; width: 100%; }
    .score-bar-fg { background: linear-gradient(90deg,#38BDF8,#0EA5E9); border-radius: 6px; height: 8px; }
    .stage-chip {
        display: inline-flex; align-items: center; gap: 6px;
        border: 1px solid rgba(128,128,128,0.3); border-radius: 999px;
        padding: 4px 12px; font-size: 0.8rem; margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# ARCHITECTURE DIAGRAM (Graphviz — render client-side, không cần cài Graphviz)
# =============================================================================

PIPELINE_DOT = """
digraph G {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#334155", fontcolor="#0f172a"];
    edge [color="#94a3b8", fontname="Helvetica", fontsize=9, fontcolor="#64748b"];

    Query [label="Câu hỏi", fillcolor="#FDE68A"];
    Semantic [label="Semantic Search\\n(BAAI/bge-m3, cosine)", fillcolor="#A7F3D0"];
    Lexical [label="Lexical Search\\n(BM25)", fillcolor="#A7F3D0"];
    RRF [label="RRF Merge\\n1/(k+rank), k=60", fillcolor="#BAE6FD"];
    Rerank [label="Rerank\\n(RRF / MMR / Cross-Encoder)", fillcolor="#BAE6FD"];
    Threshold [label="cosine gốc\\n< ngưỡng?", shape=diamond, fillcolor="#FECACA"];
    PageIndex [label="PageIndex\\n(Vectorless Fallback)", fillcolor="#FECACA"];
    Reorder [label="Reorder\\n(chống lost-in-the-middle)", fillcolor="#DDD6FE"];
    LLM [label="LLM Generation\\n+ Citation", fillcolor="#DDD6FE"];
    Answer [label="Câu trả lời", fillcolor="#FDE68A"];

    Query -> Semantic;
    Query -> Lexical;
    Semantic -> RRF;
    Lexical -> RRF;
    RRF -> Rerank;
    Rerank -> Threshold;
    Threshold -> PageIndex [label="có"];
    Threshold -> Reorder [label="không"];
    PageIndex -> Reorder;
    Reorder -> LLM;
    LLM -> Answer;
}
"""

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp quy định học vụ & tin tức UIT (đào tạo song ngành, chuyển đổi tín chỉ, đào tạo từ xa...)")

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Sinh viên cần điều kiện gì để đăng ký học song ngành?",
        "Học phí đào tạo từ xa tại UIT được tính theo công thức nào?",
        "Nguyên tắc công nhận, chuyển đổi tín chỉ yêu cầu mức tương đồng bao nhiêu %?",
        "Sinh viên đào tạo từ xa bị cảnh báo học vụ trong trường hợp nào?",
        "UIT hợp tác với chuyên gia nào trong lĩnh vực AI và nông nghiệp thông minh?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, DEFAULT_TOP_K)
    show_pipeline_trace = st.toggle("Hiện chi tiết pipeline mỗi câu trả lời", value=True)

    st.divider()
    with st.expander("🧠 Kiến trúc & lý thuyết hệ thống", expanded=False):
        st.graphviz_chart(PIPELINE_DOT, use_container_width=True)

        st.markdown("**1. Chunking & Indexing**")
        st.caption(
            f"`RecursiveCharacterTextSplitter`, CHUNK_SIZE={CHUNK_SIZE}, "
            f"OVERLAP={CHUNK_OVERLAP} — giữ trọn đoạn văn/điều khoản, overlap "
            f"giữ ngữ cảnh ở ranh giới chunk."
        )
        st.caption(f"Embedding: `{EMBEDDING_MODEL}` ({EMBEDDING_DIM} chiều) — đa ngôn ngữ, tốt cho tiếng Việt.")
        st.caption(f"Vector store: ChromaDB, collection `{COLLECTION_NAME}`.")

        st.markdown("**2. Hybrid Retrieval**")
        st.caption("Semantic Search (dense, cosine similarity) chạy song song với Lexical Search (BM25, dựa trên tần suất từ khóa).")

        st.markdown("**3. RRF — Reciprocal Rank Fusion**")
        st.latex(r"RRF(d) = \sum_{r} \frac{1}{k + rank_r(d)}, \quad k=60")
        st.caption("Gộp 2 danh sách kết quả chỉ dựa trên **thứ hạng**, không cần chuẩn hoá thang điểm giữa cosine similarity và BM25.")

        st.markdown("**4. Vectorless Fallback**")
        st.caption(
            f"Nếu điểm cosine **gốc** (trước RRF) của kết quả tốt nhất < {SCORE_THRESHOLD} "
            f"→ chuyển sang PageIndex (đọc hiểu theo cấu trúc tài liệu, không cần embedding)."
        )

        st.markdown("**5. Chống \"Lost in the Middle\"**")
        st.caption("Sắp lại thứ tự chunks trước khi đưa vào prompt: `front = chunks[::2]`, `back = chunks[1::2]`, kết quả = `front + back[::-1]` — đặt chunk quan trọng nhất ở đầu và cuối, vì LLM nhớ tốt nhất 2 vị trí này.")

        st.markdown("**6. Generation có Citation**")
        st.caption(f"Model: `{LLM_MODEL}` (fallback OpenAI khi không có OpenRouter key), temperature={TEMPERATURE} — ưu tiên factual, bắt buộc trích dẫn `[Nguồn, Năm]`, từ chối trả lời nếu context không đủ evidence.")

    st.divider()
    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# HELPERS — RENDER SOURCES / PIPELINE TRACE
# =============================================================================

def render_sources(sources: list[dict], retrieval_source: str, elapsed: float | None = None):
    badge_class = "badge-pageindex" if retrieval_source == "pageindex" else "badge-hybrid"
    badge_text = "🗂️ PageIndex Fallback" if retrieval_source == "pageindex" else "🔀 Hybrid (Semantic + BM25 + RRF)"

    header = f'<span class="badge {badge_class}">{badge_text}</span>'
    if elapsed is not None:
        header += f'<span class="stage-chip">⏱️ {elapsed:.1f}s</span>'
    header += f'<span class="stage-chip">📦 {len(sources)} chunks</span>'
    st.markdown(header, unsafe_allow_html=True)

    if not sources:
        return

    scores = [s.get("score", 0) for s in sources]
    max_score = max(scores) if scores and max(scores) > 0 else 1.0

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0)
            bar_pct = max(4, int(100 * score / max_score))
            type_badge = "badge-legal" if doc_type == "legal" else "badge-news"

            st.markdown(
                f"""
                <div class="source-card">
                    <span class="badge {type_badge}">{doc_type}</span>
                    <b>[{i}] {source_name}</b>
                    <div class="score-bar-bg"><div class="score-bar-fg" style="width:{bar_pct}%"></div></div>
                    <small>score: {score:.4f}</small>
                    <p style="margin-top:6px; font-size:0.85rem; opacity:0.85;">{src.get("content", "")[:280]}...</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp quy định học vụ & tin tức UIT — Hybrid Retrieval + RRF + Rerank + Vectorless Fallback + Citation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") is not None:
            render_sources(msg["sources"], msg.get("retrieval_source", "hybrid"), msg.get("elapsed"))

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về quy định học vụ / tin tức UIT...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        sources: list = []
        retrieval_source = "hybrid"
        elapsed = None
        try:
            with st.spinner("🔎 Đang retrieval (semantic + BM25 + RRF)..."):
                from src.task9_retrieval_pipeline import retrieve

                start = time.time()
                sources = retrieve(query, top_k=top_k)
                retrieval_source = sources[0].get("source", "hybrid") if sources else "hybrid"

            with st.spinner("✍️ Đang tổng hợp câu trả lời có citation..."):
                from src.task10_generation import (
                    format_context,
                    reorder_for_llm,
                    call_llm,
                )

                reordered = reorder_for_llm(sources)
                context = format_context(reordered)
                user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
                answer = call_llm(user_message)
                elapsed = time.time() - start

        except Exception as e:
            answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
            sources = []

        st.markdown(answer)
        if show_pipeline_trace:
            render_sources(sources, retrieval_source, elapsed)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "elapsed": elapsed,
    })
