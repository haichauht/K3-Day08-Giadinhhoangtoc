# Bài nhóm — UIT Policy RAG Web Demo

## Sản phẩm

Web app FastAPI + HTML/CSS/JavaScript trả lời câu hỏi từ 3 quy định legal và 5 thông báo UIT. Giao diện hiển thị chunk nguồn, citation `[S1]`, `[S2]`, thời gian từng công nghệ và sơ đồ kiến trúc. Lịch sử 6 tin nhắn gần nhất được đưa vào generation để hiểu câu hỏi nối tiếp, nhưng không được xem là bằng chứng.

## Kiến trúc

```text
Browser → FastAPI
             │
             ├→ Semantic Search (OpenAI + ChromaDB) ─┐
             └→ BM25 lexical search ─────────────────┴→ RRF → relevance rerank
                                                                  │
                                 PageIndex fallback ───────────────┤
                                                                  ▼
                                                    OpenAI Responses API
```

## Phân công

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Huỳnh Thị Hải Châu | 2A202601912 | Data, retrieval, generation, UX/UI và evaluation | Hoàn thành |

## Evaluation

- Golden dataset: 15 câu hỏi UIT/ĐHQG-HCM.
- Framework: RAGAS 0.1.21.
- Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision.
- A/B: Hybrid + RRF + local relevance rerank so với Dense-only.
- Trung bình: Hybrid `0.835`, Dense-only `0.751`.

Các file nộp: `evaluation/golden_dataset.json`, `evaluation/eval_pipeline.py`, `evaluation/results.md`.

## Chạy demo

```powershell
pip install -r requirements.txt
python -m src.task4_chunking_indexing
python web_app.py
```

Mở <http://127.0.0.1:8001>. Không cần Streamlit.

Chạy lại đánh giá:

```powershell
python -m group_project.evaluation.eval_pipeline
```
