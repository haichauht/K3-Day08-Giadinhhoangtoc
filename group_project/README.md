# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 18 cặp Q&A (bám sát corpus thật: 3 văn bản pháp lý + 5 tin UIT)
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation (RAGAS)
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B: `hybrid_rerank` (Task 9 đầy đủ) vs `dense_only` (chỉ semantic search)

**Lưu ý khi chạy lại (2 venv, xem `requirements-eval.txt`):** `ragas==0.1.21`
xung đột phiên bản `openai` với các gói của pipeline chính (chromadb,
sentence-transformers) nên phải tách 2 bước:

```bash
# Bước 1 — venv chính (sinh câu trả lời RAG cho từng config)
source venv/Scripts/activate
python -m group_project.evaluation.generate_rag_outputs

# Bước 2 — venv_eval riêng (chấm điểm RAGAS + xuất results.md)
python -m venv venv_eval
source venv_eval/Scripts/activate
pip install -r requirements-eval.txt
python -m group_project.evaluation.eval_pipeline
```

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                                   ┌─────────────────────┐
                                   │   Câu hỏi (query)   │
                                   └──────────┬──────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
           ┌─────────────────────────┐                 ┌─────────────────────────┐
           │   Semantic Search        │                 │   Lexical Search        │
           │   (Task 5)               │                 │   (Task 6)              │
           │   BAAI/bge-m3 (1024-d)   │                 │   BM25 (rank-bm25)      │
           │   cosine similarity      │                 │   keyword matching      │
           │   → ChromaDB             │                 │   → data/standardized/  │
           └────────────┬─────────────┘                 └────────────┬────────────┘
                        │  giữ điểm cosine GỐC                        │
                        └─────────────────────┬───────────────────────┘
                                              ▼
                               ┌───────────────────────────┐
                               │   RRF Merge (Task 7)       │
                               │   RRF(d) = Σ 1/(k+rank)    │
                               │   k = 60                   │
                               └─────────────┬───────────────┘
                                              ▼
                               ┌───────────────────────────┐
                               │   Rerank (Task 7)          │
                               │   RRF / MMR / Cross-Encoder│
                               └─────────────┬───────────────┘
                                              ▼
                               ┌───────────────────────────┐
                       ┌───────┤  cosine gốc < threshold?   │
                       │ có    │  (SCORE_THRESHOLD, Task 9) │
                       ▼       └─────────────┬───────────────┘
        ┌───────────────────────────┐        │ không
        │  PageIndex Fallback        │        │
        │  (Task 8, vectorless)      │        │
        └─────────────┬───────────────┘        │
                       └───────────────┬─────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │   Reorder (Task 10)        │
                        │   chống lost-in-the-middle │
                        │   front + back[::-1]       │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Format Context + Prompt  │
                        │   (Task 10)                │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   LLM Generation            │
                        │   OpenRouter / OpenAI       │
                        │   + Citation bắt buộc       │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Câu trả lời + Sources     │
                        │   (app.py — Streamlit UI)   │
                        └───────────────────────────┘
```

**Data pipeline (thu thập → chuẩn hoá → index):**
```
data/landing/legal/*.pdf  ─┐
                            ├─► Task 3 (MarkItDown) ─► data/standardized/*.md ─► Task 4
data/landing/news/*.md    ─┘                                                    (chunk 800/100,
                                                                                   embed bge-m3,
                                                                                   index ChromaDB)
```

**Evaluation pipeline (RAGAS, so sánh A/B):**
```
golden_dataset.json (18 câu) ─► generate_rag_outputs.py (venv chính)
        │                              │
        │                    ┌─────────┴─────────┐
        │                    ▼                   ▼
        │         rag_outputs_hybrid_rerank   rag_outputs_dense_only
        │              (Config A)                (Config B)
        │                    │                   │
        └────────────────────┴─────────┬─────────┘
                                        ▼
                        eval_pipeline.py (venv_eval, RAGAS)
                        Faithfulness / Answer Relevance /
                        Context Recall / Context Precision
                                        ▼
                                  results.md
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| | | | |
| | | | |
| | | | |
| | | | |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
