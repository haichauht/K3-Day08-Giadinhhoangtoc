# UIT Policy RAG — Interactive Web Demo

Hệ thống hỏi đáp chính sách và thông báo chính thức của UIT/ĐHQG-HCM. Demo web dùng **FastAPI + HTML/CSS/JavaScript**, không dùng Streamlit; giao diện cho phép quan sát trực tiếp từng bước của pipeline RAG.

## Kiến trúc

```text
3 PDF legal + 5 news JSON → Markdown → Recursive chunks (800/100)
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
            OpenAI Embedding → ChromaDB                     BM25 lexical
                         └───────────────────┬───────────────────┘
                                             ▼
                                  RRF + relevance rerank
                                             ▼
                           PageIndex fallback khi cosine < 0.30
                                             ▼
                            OpenAI Responses API + citations
```

- Embedding: `text-embedding-3-small`, 1024 chiều.
- Generation: `gpt-5.6-luna` qua Responses API.
- Vector store: ChromaDB local, cosine similarity.
- Hybrid retrieval: Semantic Search + BM25, gộp hạng bằng RRF.
- PageIndex: vectorless fallback cho 3 PDF legal.
- Evaluation: RAGAS trên 15 câu hỏi, so sánh Hybrid+RRF và Dense-only.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `OPENAI_API_KEY` và `PAGEINDEX_API_KEY` vào `.env`. Không commit file này.

## Chạy web demo

```powershell
python web_app.py
```

Mở <http://127.0.0.1:8001>. Demo gồm bốn màn hình:

1. Chat RAG thật, có nguồn và thời gian từng bước.
2. Sơ đồ luồng hệ thống có mô phỏng.
3. Data catalog và vai trò từng công nghệ.
4. Biểu đồ kết quả RAGAS A/B.

## Chạy checkpoint

```powershell
# CP1: PDF/JSON → Markdown
python -m src.task3_convert_markdown

# CP2: chunk, embedding, ChromaDB, Semantic và BM25
python -m src.task4_chunking_indexing
python -m src.task5_semantic_search
python -m src.task6_lexical_search

# CP3: RRF, PageIndex và pipeline retrieval
python -m src.task7_reranking
python -m src.task8_pageindex_vectorless
python -m src.task9_retrieval_pipeline

# CP4: bài kiểm thử cá nhân
python -m pytest tests/test_individual.py -v

# CP5: web demo và RAGAS
python web_app.py
python -m group_project.evaluation.eval_pipeline
```

## Kết quả đã xác minh

- `35 passed` trong `tests/test_individual.py`.
- ChromaDB chứa 335 chunks từ 8 tài liệu.
- PageIndex đã xử lý đủ 3 PDF legal.
- RAGAS: Hybrid+RRF `0.835`, Dense-only `0.751`.

## Cấu trúc chính

```text
data/landing/legal/          # 3 PDF gốc
data/landing/news/           # 5 JSON gốc
data/standardized/           # 8 Markdown đã chuẩn hóa
src/task3_...task10_...py    # data và RAG pipeline
group_project/evaluation/    # golden dataset, RAGAS, báo cáo
web/                         # HTML, CSS, JavaScript
web_app.py                   # FastAPI API + web server
```
