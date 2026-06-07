# 07 - Tham chiếu cấu hình

Tất cả cấu hình qua biến môi trường (file `.env`).

## AWS

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` | Fallback region |

## Bedrock Models

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Model embedding |
| `LLM_MODEL_ID` | `anthropic.claude-3-haiku-20240307-v1:0` | Model LLM |

## Retrieval Pipeline

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `CHROMA_COLLECTION` | `AI002` | Tên collection ChromaDB |
| `TOP_K_RESULTS` | `10` | Số documents trả về |
| `ENABLE_HYBRID` | `true` | Bật hybrid search (BM25 + vector) |
| `ENABLE_RERANK` | `false` | Bật reranking (cần `flashrank`) |
| `RERANK_TOP_N` | `4` | Số docs giữ lại sau rerank |
| `BM25_WEIGHT` | `0.4` | Trọng số BM25 trong ensemble |
| `VECTOR_WEIGHT` | `0.6` | Trọng số vector trong ensemble |

## Auth & Rate Limiting

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `API_KEYS` | _(rỗng)_ | Danh sách API keys (comma-separated). Rỗng = auth disabled |
| `RATE_LIMIT` | `30/minute` | Rate limit per IP |

## Memory & Cache

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `REDIS_URL` | _(rỗng)_ | Redis URL. Rỗng = in-memory cache |
| `DATABASE_URL` | `sqlite:///data/chat_history.db` | SQLAlchemy DB URL cho chat history |

## Observability

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `LANGCHAIN_TRACING_V2` | `false` | Bật LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(rỗng)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `ai002-traffic-rag` | Tên project trên LangSmith |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

## API

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `API_URL` | `http://localhost:8000` | URL backend (cho Streamlit UI) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

## Text Processing

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `CHUNK_SIZE` | `512` | Kích thước chunk (planned) |
| `CHUNK_OVERLAP` | `50` | Overlap giữa chunks (planned) |
| `MAX_QUERY_LENGTH` | `1000` | Độ dài tối đa câu hỏi |

## Data Paths (hardcoded trong `config.py`)

| Constant | Giá trị | Mô tả |
|----------|---------|-------|
| `STOPWORDS_PATH` | `data/vietnamese-stopwords-dash.txt` | File stopwords |
| `DATA_FOLDER` | `data/` | Thư mục data |
| `KEYWORD_FILE` | `data/top_keywords.txt` | File keywords cho classifier |
| `PROCESSED_JSON_FILE` | `data/output.json` | File records đã trích xuất |

## Docker Compose Overrides

Khi chạy qua Docker Compose, các biến sau được override:
```yaml
REDIS_URL: redis://redis:6379/0
DATABASE_URL: postgresql+psycopg2://ai002:ai002@db:5432/ai002
API_URL: http://api:8000
```
