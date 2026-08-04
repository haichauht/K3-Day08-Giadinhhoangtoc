# RAG Evaluation Results

## Framework sử dụng

> RAGAS 0.1.21

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta |
|--------|---------------------------|------------------------|-------|
| Faithfulness | 0.8968 | 0.8222 | +0.0746 |
| Answer Relevance | 0.8443 | 0.6976 | +0.1467 |
| Context Recall | 0.9630 | 0.9722 | -0.0092 |
| Context Precision | 0.9588 | 0.9329 | +0.0259 |
| **Average** | **0.9157** | **0.8562** | **+0.0595** |

---

## A/B Comparison Analysis

**Config A (hybrid_rerank):**
> Semantic search (BAAI/bge-m3) + BM25 lexical search, gộp bằng Reciprocal Rank Fusion, sau đó rerank lại (RRF), có fallback PageIndex nếu điểm cosine gốc dưới ngưỡng.

**Config B (dense_only):**
> Chỉ dùng semantic_search() — bỏ qua lexical search, RRF merge và reranking.

**Kết luận:** Config A đạt điểm trung bình 0.9157 so với 0.8562 của Config B (cao hơn 0.0595). Hybrid retrieval + rerank giúp cải thiện chất lượng context/answer so với chỉ dùng dense search đơn thuần, đúng như kỳ vọng khi kết hợp thêm tín hiệu từ khóa (BM25) cho các câu hỏi có thuật ngữ pháp lý/tên riêng cụ thể.

---

## Worst Performers (Config A, Bottom 3 theo Faithfulness)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Sinh viên cần đáp ứng điều kiện gì để được đăng ký học song ngành tại ... | 0.500 | 0.871 | 1.000 | Generation | LLM diễn giải/suy luận vượt ngoài context được cấp |
| 2 | Sinh viên đào tạo từ xa UIT bị cảnh báo học vụ trong trường hợp nào? | 0.500 | 0.833 | 1.000 | Generation | LLM diễn giải/suy luận vượt ngoài context được cấp |
| 3 | PGS.TS. Chung-Tse Michael Wu đến từ trường đại học nào và ông là chuyê... | 0.600 | 0.911 | 1.000 | Generation | LLM diễn giải/suy luận vượt ngoài context được cấp |

---

## Recommendations

### Cải tiến 1
**Action:** Mở rộng corpus (thêm văn bản pháp lý/tin tức) để tăng context recall cho các câu hỏi cụ thể.
**Expected impact:** Giảm tỷ lệ retrieval miss, tăng Context Recall và Faithfulness.

### Cải tiến 2
**Action:** Hiệu chỉnh lại `SCORE_THRESHOLD` ở Task 9 dựa trên phân bố điểm cosine thực tế của corpus hiện tại.
**Expected impact:** Kích hoạt fallback PageIndex đúng lúc hơn cho các câu hỏi hybrid trả về yếu.

### Cải tiến 3
**Action:** Bật cross-encoder reranking (Jina API) thay vì chỉ RRF khi có JINA_API_KEY.
**Expected impact:** Cải thiện Context Precision nhờ rerank ngữ nghĩa chính xác hơn RRF thuần rank-based.
