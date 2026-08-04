# BÁO CÁO BÀI TẬP NHÓM — UIT INTELLIGENCE

## 1. Thông tin đề tài

- **Tên sản phẩm:** UIT Intelligence — University Services RAG Chatbot
- **Chủ đề:** Hỏi đáp quy chế đào tạo và tin tức Trường Đại học Công nghệ Thông tin
- **Giao diện:** Streamlit
- **Evaluation framework:** RAGAS 0.1.21
- **Nhánh nộp bài:** `final` (tích hợp từ nhánh `01931`)

### Thành viên và phân công

| STT | Họ và tên | MSSV | Phân công | Trạng thái |
|---:|---|---|---|:---:|
| 1 | Nguyễn Chí Hiếu | 2A202601931 | Team Leader & RAG Architect; Task 9; tích hợp pipeline; tổng hợp báo cáo | Hoàn thành |
| 2 | Nguyễn Anh Trà | 2A202601735 | Data Engineering; Task 1–3; thu thập, kiểm tra provenance và chuẩn hóa dữ liệu | Hoàn thành |
| 3 | Hoàng Huyền Trang | 2A202601311 | Vector Database & Dense Search; Task 4–5; chunking, embedding và ChromaDB | Hoàn thành |
| 4 | Đinh Thị Diễm Quỳnh | 2A202601621 | Sparse Retrieval & Fallback; Task 6–8; BM25, RRF/reranking và PageIndex | Hoàn thành |
| 5 | Huỳnh Thị Hải Châu | 2A202601912 | Frontend, Generation & QA; Task 10; Streamlit, citation, golden dataset và RAGAS | Hoàn thành |

## 2. Mục tiêu và phạm vi

Đề tài xây dựng một hệ thống Retrieval-Augmented Generation (RAG) tiếng Việt có khả năng trả lời câu hỏi dựa trên tài liệu UIT/ĐHQG-HCM, kèm trích dẫn để người dùng kiểm tra nguồn. Phạm vi dữ liệu gồm ba văn bản pháp lý về đào tạo và năm bài tin UIT mới nhất tại thời điểm thu thập.

Các yêu cầu chính đã hoàn thành:

- chuẩn hóa và truy vết nguồn dữ liệu từ landing đến standardized;
- kết hợp semantic search, BM25, RRF và reranking;
- có PageIndex/vectorless fallback khi dense retrieval thiếu tự tin;
- sinh câu trả lời bằng Qwen qua Groq, kèm citation;
- hỗ trợ hội thoại nhiều lượt và hiển thị evidence trên Streamlit;
- so sánh A/B hai cấu hình retrieval trên golden dataset 15 câu.

## 3. Kiến trúc hệ thống

```text
PDF pháp lý + trang tin UIT
          │
          ▼
Landing data ──► Markdown chuẩn hóa ──► Chunk theo heading
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
              NVIDIA embedding + Chroma                   BM25 tiếng Việt
                         │                                       │
                         └──────────────► RRF ◄──────────────────┘
                                             │
                                      feature reranking
                                             │
                    dense score thấp ──► PageIndex / local tree fallback
                                             │
                                             ▼
                              Qwen qua Groq + citation
                                             │
                                             ▼
                                      Streamlit chat UI
```

Semantic search giúp tìm các đoạn tương đồng về ý nghĩa; BM25 giữ độ nhạy với từ khóa, mã điều và con số. Reciprocal Rank Fusion (RRF) hợp nhất hai bảng xếp hạng mà không phụ thuộc thang điểm gốc. Feature reranker tiếp tục điều chỉnh theo semantic score, lexical score, metadata và vị trí. Khi top dense score thấp hơn ngưỡng tin cậy, pipeline gọi PageIndex; cây heading cục bộ là fallback khi cloud không khả dụng.

## 4. Dữ liệu và chuẩn hóa

| Tầng | Nội dung | Vai trò |
|---|---|---|
| `data/landing/legal` | 3 PDF gốc và `sources.json` | Lưu bản gốc, URL và provenance |
| `data/landing/news` | 5 raw JSON và `source.csv` | Snapshot nội dung crawl, không dùng trực tiếp để retrieval |
| `data/standardized/legal` | 3 Markdown pháp lý | Text sạch, heading và front matter thống nhất |
| `data/standardized/news` | 5 Markdown bài báo | Corpus news chính thức cho retrieval |

Hai PDF có text layer rõ được chuyển bằng MarkItDown. PDF dùng font cũ có text layer lỗi được chép và đối chiếu thủ công vì tài liệu ngắn. Metadata chuẩn gồm `title`, `source_url`, `document_type`, `retrieved_at`, `document_version`, `license_or_permission` và `cleaning_version`.

## 5. Kết quả triển khai Task 1–10

| Task | Nội dung thực hiện | Kết quả |
|---:|---|:---:|
| 1 | Kiểm tra 3 PDF pháp lý và provenance | Hoàn thành |
| 2 | Crawl tin UIT, lưu raw JSON và source manifest | Hoàn thành |
| 3 | Chuyển đổi, làm sạch và chuẩn hóa Markdown | Hoàn thành |
| 4 | Heading-aware chunking, NVIDIA/local embedding và Chroma | Hoàn thành |
| 5 | Semantic search có lựa chọn backend | Hoàn thành |
| 6 | BM25 với tokenization tiếng Việt | Hoàn thành |
| 7 | RRF và feature reranking | Hoàn thành |
| 8 | PageIndex cloud và local vectorless fallback | Hoàn thành |
| 9 | Hybrid retrieval và confidence routing | Hoàn thành |
| 10 | Qwen generation có inline citation và fallback | Hoàn thành |

Bộ kiểm thử cá nhân hiện đạt **35/35 test passed**.

## 6. Giao diện sản phẩm

Ứng dụng `app.py` cung cấp giao diện chat Streamlit với các chức năng:

- trả lời có inline citation và liên kết tới nguồn gốc;
- hiển thị title, section, excerpt và retrieval score của từng evidence;
- lưu history trong session và nối câu hỏi trước cho follow-up ngắn;
- tùy chỉnh top-k và bật/tắt trace pipeline;
- hiển thị model, retrieval route và thời gian xử lý;
- dùng extractive fallback có citation nếu API sinh câu trả lời lỗi.

## 7. Thiết kế evaluation

Golden dataset có đúng 15 câu, gồm 11 câu legal và 4 câu news. A/B test giữ nguyên dataset, top-k=5, NVIDIA embedding API và Qwen generator; chỉ thay đổi chiến lược retrieval:

- **A — `hybrid_api_rrf_rerank`:** dense + BM25 + RRF + reranking;
- **B — `dense_api_only`:** chỉ dense semantic search.

RAGAS 0.1.21 chấm tuần tự bằng duy nhất model `qwen/qwen3.6-27b` và một Groq API key, temperature 0. Bốn metric gồm:

- **Faithfulness:** các nhận định trong answer có được context hỗ trợ hay không;
- **Answer relevance:** answer có trực tiếp trả lời question hay không;
- **Context recall:** context có bao phủ thông tin cần thiết trong ground truth hay không;
- **Context precision:** các đoạn hữu ích có chiếm tỷ lệ cao và đứng sớm hay không.

Answer/context đầu vào được lưu trong `raw_runs.json`. Điểm từng metric được checkpoint atomic vào `ragas_scores_cache.json`; lần chạy sau chỉ chấm ô còn thiếu. Kết quả cuối có đủ **120/120 điểm** và không còn giá trị `null`.

## 8. Kết quả và phân tích

| Cấu hình | Faithfulness | Answer relevance | Context recall | Context precision | Coverage | Hit@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid + RRF + rerank | **0.671** | **0.862** | **0.883** | **0.804** | 60/60 | 1.000 | 1.000 |
| Dense only | 0.481 | 0.641 | 0.672 | 0.690 | 60/60 | 1.000 | 1.000 |

So với dense-only, hybrid tăng tuyệt đối 0.190 faithfulness, 0.222 answer relevance, 0.211 context recall và 0.114 context precision. Kết quả cho thấy BM25 và reranking hỗ trợ tốt cho câu hỏi chứa thuật ngữ pháp lý, tên riêng và điều kiện cụ thể.

Hit@5 và MRR cùng bằng 1.0 vì expected source được đo ở cấp tài liệu: cả hai cấu hình đều lấy đúng file nguồn ở vị trí đầu. Tuy nhiên, điểm RAGAS vẫn khác rõ vì đúng tài liệu chưa đồng nghĩa đúng đoạn bằng chứng. Các case hybrid còn yếu gồm `legal-remote-study-pause`, `legal-credit-process`, `legal-dual-registration` và một số câu news có ground truth chi tiết.

### Ba trường hợp cần ưu tiên cải thiện

| ID | Faithfulness | Answer relevance | Context recall | Context precision | Điểm yếu chính | Hướng cải thiện |
|---|---:|---:|---:|---:|---|---|
| `legal-credit-process` | 0.143 | 0.711 | 1.000 | 0.500 | Context lấy đủ ý chính nhưng còn nhiễu; câu trả lời tổng hợp nhiều bước chưa bám sát từng mệnh đề | Parent-child retrieval và prompt yêu cầu đối chiếu từng bước với evidence |
| `news-ai-training` | 1.000 | 0.859 | 0.250 | 0.500 | Retriever chỉ bao phủ một phần nội dung chi tiết của lớp tập huấn | Gắn heading cha vào chunk, tăng candidate-k và rerank theo các cụm ngày/nội dung |
| `news-party-conference` | 0.889 | 0.901 | 0.000 | 1.000 | Context ngắn và chính xác nhưng chưa bao phủ đủ ground truth tham chiếu | Mở rộng chunk lân cận và hiệu chỉnh expected context theo đúng đoạn nguồn |

Bảng điểm đầy đủ theo từng câu và phân tích kết quả nằm tại [evaluation/results.md](evaluation/results.md).

## 9. Hạn chế và hướng phát triển

- Golden dataset 15 câu đáp ứng yêu cầu bài tập nhưng chưa đại diện cho mọi cách diễn đạt.
- LLM-as-judge vẫn có biến động dù temperature bằng 0.
- Retrieval đúng file nhưng đôi khi chưa lấy đủ các điều khoản phân tán ở nhiều chunk.
- PageIndex cloud phụ thuộc API và tài liệu đã upload; local tree chỉ là fallback gần đúng.

Hướng phát triển tiếp theo là parent-child retrieval, query expansion tiếng Việt, tinh chỉnh chunk overlap, thêm kiểm thử hội thoại nhiều lượt và lặp evaluation với nhiều lần chạy khi có ngân sách.

## 10. Hướng dẫn cài đặt và tái lập

Tạo `.env` trong thư mục gốc repository (khuyến nghị) hoặc thư mục cha dựa trên `.env.example`:

```dotenv
OPENAI_API_KEY=<groq-key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
NVIDIA_API_KEY=<nvidia-key>
PAGEINDEX_API_KEY=<pageindex-key>
```

```powershell
pip install -r requirements.txt

# Tạo lại vector index; cache sinh ra không được commit
python -m src.task4_chunking_indexing --with-local-fallback

# Chạy giao diện
python -m streamlit run app.py

# Chạy/resume RAGAS từ checkpoint
python -m group_project.evaluation.eval_pipeline

# Kiểm thử Task 1–10
python -m pytest tests/test_individual.py -q
```

Dùng `--refresh` khi cần sinh lại answer/context của cả hai cấu hình. Dùng `--until-complete` để tự retry nếu một metric chưa trả điểm do rate limit.

## 11. Danh sách deliverable

```text
app.py                         Giao diện Streamlit
src/task1_...task10_*.py       Pipeline Task 1–10
data/landing/                  Dữ liệu gốc và provenance
data/standardized/             Markdown chuẩn hóa
group_project/evaluation/      Golden set, raw runs, cache và báo cáo điểm
tests/test_individual.py       Bộ test Task 1–10
```

## 12. Kết luận

Sản phẩm đã hoàn thành luồng RAG từ thu thập dữ liệu đến giao diện và evaluation. Kết quả A/B cho thấy hybrid retrieval kết hợp BM25, RRF và reranking phù hợp hơn dense-only đối với corpus UIT hiện tại. Toàn bộ dữ liệu nguồn, dữ liệu chuẩn hóa, kết quả chạy thô và điểm đánh giá được lưu kèm để có thể kiểm tra và tái lập.
