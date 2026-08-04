# Kết quả đánh giá RAG

## Thiết lập

- Framework: RAGAS 0.1.21
- Evaluator LLM: `gpt-4.1-mini`
- Embedding đánh giá: `text-embedding-3-small` (1024 chiều)
- Golden dataset: 15 câu hỏi UIT/ĐHQG-HCM
- Config A: semantic + BM25, hợp nhất RRF và relevance rerank cục bộ
- Config B: semantic dense-only (ablation, không BM25/RRF)

## Điểm tổng hợp

| Metric | Config A: Hybrid + RRF | Config B: Dense-only | Δ (A-B) |
|---|---:|---:|---:|
| Faithfulness | 0.974 | 0.788 | +0.186 |
| Answer Relevance | 0.506 | 0.406 | +0.099 |
| Context Recall | 0.967 | 0.933 | +0.033 |
| Context Precision | 0.892 | 0.875 | +0.017 |
| **Trung bình** | 0.835 | 0.751 | +0.084 |

## Phân tích A/B

Cấu hình có điểm trung bình cao hơn là **Hybrid + RRF**. Hybrid bổ sung khả năng bắt đúng thuật ngữ pháp lý/tên thông báo bằng BM25, còn dense-only là đường cơ sở để đo đóng góp của lexical retrieval và RRF.

## Ba trường hợp kém nhất của Config A

| # | Câu hỏi | Faithfulness | Relevance | Recall | Precision | Công đoạn lỗi chính |
|---:|---|---:|---:|---:|---:|---|
| 1 | Sau khi khai báo hồ sơ trực tuyến, tân sinh viên phải làm gì để dữ liệu được gửi duyệt? | 1.000 | 0.755 | 0.500 | 0.804 | Mixed |
| 2 | Sinh viên đào tạo từ xa cần đáp ứng gì để được xét tốt nghiệp? | 0.857 | 0.341 | 1.000 | 0.917 | Answer formulation |
| 3 | Sinh viên có được hưởng học bổng hoặc miễn giảm học phí cho ngành thứ hai không? | 0.750 | 0.453 | 1.000 | 0.917 | Answer formulation |

## Đề xuất cải tiến

1. Loại bỏ menu, CAPTCHA và danh sách bài liên quan khỏi news trước khi chunk để tăng Context Precision.
2. Dùng MarkdownHeaderTextSplitter hoặc gắn heading cha vào từng chunk pháp lý để tăng Context Recall cho câu hỏi theo Điều/Khoản.
3. Mở rộng golden dataset bằng câu hỏi nối tiếp và câu hỏi không có bằng chứng để kiểm tra memory và cơ chế từ chối.

> Điểm trong bảng được sinh từ lần chạy thật của script; file cache chỉ lưu câu trả lời/ngữ cảnh để có thể retry RAGAS mà không gọi lại generation.
