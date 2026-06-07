# 06 - Hướng dẫn triển khai

## Cách 1: Chạy local (Development)

### Yêu cầu
- Python >= 3.11
- Java/JDK >= 8
- AWS credentials có quyền Bedrock

### Bước 1: Cài đặt dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Bước 2: Cấu hình .env
```bash
cp .env.template .env
# Sửa file .env: điền AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### Bước 3: Chuẩn bị dữ liệu
```bash
# Trích xuất records từ PDF → output.json
python src/data_preparation/crawl_data.py
```

### Bước 4: Chạy API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Bước 5: Chạy Streamlit UI
```bash
streamlit run src/domain/main.py
```

Truy cập:
- API: http://localhost:8000
- UI: http://localhost:8501

---

## Cách 2: Docker Compose (Production-like)

### Yêu cầu
- Docker + Docker Compose

### Chạy
```bash
# Tạo file .env từ template
cp .env.template .env
# Sửa AWS credentials trong .env

# Build và chạy tất cả services
docker compose up --build
```

### Services
| Service | Port | Mô tả |
|---------|------|--------|
| `api` | 8000 | FastAPI backend |
| `ui` | 8501 | Streamlit frontend |
| `redis` | 6379 | Response cache |
| `db` | 5432 | PostgreSQL (chat history) |

### Docker Compose tự động:
- Set `REDIS_URL=redis://redis:6379/0`
- Set `DATABASE_URL=postgresql+psycopg2://ai002:ai002@db:5432/ai002`
- Set `API_URL=http://api:8000` cho UI
- Mount `./data` vào `/app/data`

---

## Cách 3: AWS Serverless (Planned)

Kiến trúc serverless sử dụng:
- **Lambda**: RAG orchestrator, Document indexer, Health check
- **API Gateway**: REST API
- **S3**: Lưu trữ documents + vector index
- **CloudFormation**: IaC

Xem `PROJECT_STRUCTURE.md` và `deployment.yaml` cho chi tiết.

---

## Lần chạy đầu tiên

Khi khởi động lần đầu, hệ thống sẽ:
1. VnCoreNLP tự động tải model `.jar` + `.rdr` (cần internet + Java)
2. Nếu ChromaDB trống → tự động index toàn bộ documents từ `output.json`
3. Quá trình indexing mất vài phút (gọi Bedrock Titan Embed cho mỗi document)

---

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `VnCoreNLP` lỗi | Kiểm tra Java đã cài, `JAVA_HOME` đúng |
| Bedrock `AccessDenied` | Kiểm tra AWS credentials + model access trong Bedrock console |
| ChromaDB lỗi | Xóa `data/chroma_db/` và restart để re-index |
| OCR không hoạt động | Cài Tesseract + tải `tessdata/vie.traineddata` |
| Redis connection refused | Bỏ qua (fallback in-memory cache) hoặc chạy Redis |
