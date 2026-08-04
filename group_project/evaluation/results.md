# RAG Evaluation Results

- Evaluator: `RAGAS 0.1.21`
- Golden set: 15 câu hỏi
- Judge: `qwen/qwen3.6-27b` qua một Groq API key
- Top-k: 5
- Generator: `qwen/qwen3.6-27b`

## A/B comparison

| Config | Faithfulness | Answer relevance | Context recall | Context precision | Score coverage | Source Hit@5 | Source MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_api_rrf_rerank` | 0.671 | 0.862 | 0.883 | 0.804 | 60/60 | 1.000 | 1.000 |
| `dense_api_only` | 0.481 | 0.641 | 0.672 | 0.690 | 60/60 | 1.000 | 1.000 |

## Per-question scores

### `hybrid_api_rrf_rerank`

| ID | Faith. | Relev. | Recall | Precision | Expected source rank |
|---|---:|---:|---:|---:|---:|
| legal-remote-graduation | 0.800 | 0.798 | 1.000 | 1.000 | 1 |
| legal-remote-max-duration | 0.333 | 0.966 | 1.000 | 0.500 | 1 |
| legal-remote-tuition | 0.571 | 0.646 | 1.000 | 1.000 | 1 |
| legal-remote-grade-improvement | 1.000 | 0.757 | 1.000 | 1.000 | 1 |
| legal-remote-study-pause | 0.200 | 0.850 | 1.000 | 1.000 | 1 |
| legal-remote-degree-deadline | 0.500 | 0.954 | 1.000 | 1.000 | 1 |
| legal-credit-similarity | 1.000 | 1.000 | 1.000 | 1.000 | 1 |
| legal-credit-learner-rights | 0.875 | 0.980 | 1.000 | 1.000 | 1 |
| legal-credit-process | 0.143 | 0.711 | 1.000 | 0.500 | 1 |
| legal-dual-registration | 0.250 | 1.000 | 1.000 | 0.750 | 1 |
| legal-dual-tuition | 0.500 | 0.929 | 1.000 | 1.000 | 1 |
| news-michael-wu-seminar | 1.000 | 0.878 | 1.000 | 0.000 | 1 |
| news-party-conference | 0.889 | 0.901 | 0.000 | 1.000 | 1 |
| news-summer-program | 1.000 | 0.702 | 1.000 | 0.806 | 1 |
| news-ai-training | 1.000 | 0.859 | 0.250 | 0.500 | 1 |

### `dense_api_only`

| ID | Faith. | Relev. | Recall | Precision | Expected source rank |
|---|---:|---:|---:|---:|---:|
| legal-remote-graduation | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| legal-remote-max-duration | 0.000 | 0.967 | 0.000 | 1.000 | 1 |
| legal-remote-tuition | 0.500 | 0.000 | 0.000 | 0.000 | 1 |
| legal-remote-grade-improvement | 0.500 | 0.844 | 1.000 | 0.700 | 1 |
| legal-remote-study-pause | 0.000 | 0.655 | 0.000 | 0.000 | 1 |
| legal-remote-degree-deadline | 0.500 | 0.953 | 1.000 | 0.756 | 1 |
| legal-credit-similarity | 0.500 | 0.594 | 1.000 | 1.000 | 1 |
| legal-credit-learner-rights | 0.875 | 0.822 | 1.000 | 1.000 | 1 |
| legal-credit-process | 0.000 | 0.831 | 1.000 | 1.000 | 1 |
| legal-dual-registration | 0.625 | 0.673 | 1.000 | 0.700 | 1 |
| legal-dual-tuition | 1.000 | 0.929 | 1.000 | 1.000 | 1 |
| news-michael-wu-seminar | 0.333 | 0.000 | 0.333 | 0.500 | 1 |
| news-party-conference | 0.625 | 0.849 | 0.750 | 0.887 | 1 |
| news-summer-program | 0.750 | 0.631 | 1.000 | 0.806 | 1 |
| news-ai-training | 1.000 | 0.861 | 1.000 | 1.000 | 1 |

## Evaluation design

Cả hai cấu hình dùng cùng 15 câu hỏi, top-k=5, NVIDIA embedding API và Qwen generator. Biến độc lập duy nhất là chiến lược retrieval: hybrid có BM25, RRF và feature reranking; baseline chỉ dùng dense search.

RAGAS 0.1.21 chấm tuần tự bằng một `qwen/qwen3.6-27b` judge, temperature 0 và một Groq API key. Cache theo chữ ký input giúp resume mà không chấm lại các ô đã hoàn thành.

## Analysis

Hybrid + RRF + feature reranking cao hơn dense-only ở cả bốn metric. Mức tăng tuyệt đối lần lượt là +0.190 faithfulness, +0.222 answer relevance, +0.211 context recall và +0.114 context precision. Kết quả cho thấy lexical retrieval cùng reranking giúp lấy đúng điều khoản và giảm câu trả lời thiếu căn cứ.

Hit@5 và MRR đều bằng 1.0 vì expected source được định nghĩa ở cấp tài liệu. RAGAS vẫn phân biệt được chất lượng đoạn: đúng file không đồng nghĩa đúng điều khoản. Hybrid còn yếu ở các câu cần ghép nhiều điều kiện (`legal-remote-study-pause`, `legal-credit-process`, `legal-dual-registration`) và một số ground truth news chi tiết.

## Worst performers

Ba trường hợp có điểm trung bình bốn metric thấp nhất của cấu hình `hybrid_api_rrf_rerank`:

| ID | Faith. | Relev. | Recall | Precision | Average | Chẩn đoán |
|---|---:|---:|---:|---:|---:|---|
| `legal-credit-process` | 0.143 | 0.711 | 1.000 | 0.500 | 0.588 | Retrieval bao phủ nội dung nhưng còn nhiễu; generation chưa gắn từng bước với evidence tương ứng |
| `news-ai-training` | 1.000 | 0.859 | 0.250 | 0.500 | 0.652 | Thiếu coverage cho các chi tiết ngày và nội dung tập huấn |
| `news-party-conference` | 0.889 | 0.901 | 0.000 | 1.000 | 0.697 | Context chính xác nhưng không bao phủ đủ reference answer |

Ưu tiên cải thiện bằng parent-child retrieval, gắn heading cha vào chunk, mở rộng candidate-k trước reranking và rà soát lại expected context của các câu news.

## Limitations and next steps

Bộ 15 câu đủ yêu cầu bài tập nhưng còn nhỏ; điểm LLM-as-judge vẫn có biến động dù temperature bằng 0. Nên bổ sung parent-child retrieval, query expansion, hiệu chỉnh chunk overlap và đánh giá nhiều lần/metric thủ công khi phát triển tiếp. Không dùng kết quả này như thước đo độ đúng tuyệt đối ngoài phạm vi corpus hiện tại.

## Reproduce

```powershell
$env:PYTHONUTF8='1'
python -m group_project.evaluation.eval_pipeline
```
