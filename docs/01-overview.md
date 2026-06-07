# 01 - Tổng quan dự án

## Tên dự án
**AI002 – Hệ thống ChatBot RAG Luật Giao Thông Việt Nam**

## Mục tiêu
Xây dựng chatbot hỏi-đáp tự động về luật giao thông đường bộ Việt Nam, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để trả lời chính xác dựa trên nguồn văn bản pháp luật thực tế.

## Phạm vi dữ liệu
| Văn bản | Mã số | Loại |
|---------|--------|------|
| Luật Trật tự, an toàn giao thông đường bộ | 36/2024/QH15 | Luật |
| Nghị định quy phạt vi phạm hành chính ATGT | 168/2024/NĐ-CP | Nghị định |

## Công nghệ lõi
| Thành phần | Công nghệ |
|------------|-----------|
| Framework AI | LangChain |
| LLM | Claude 3 Haiku (AWS Bedrock) |
| Embedding | Amazon Titan Embed Text v2 (AWS Bedrock) |
| Vector DB | ChromaDB (persistent local) |
| NLP tiếng Việt | VnCoreNLP (word segmentation) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Cache | Redis (fallback in-memory) |
| Chat history | SQLAlchemy (SQLite / PostgreSQL) |
| Observability | LangSmith (optional) |
| Containerization | Docker + Docker Compose |

## Yêu cầu hệ thống
- Python >= 3.11
- Java/JDK >= 8 (cho VnCoreNLP)
- AWS Account có quyền truy cập Bedrock (Claude 3 Haiku + Titan Embed)
- (Optional) Docker, Redis, PostgreSQL

## Cấu trúc thư mục chính
```
AI002/
├── api/                  # FastAPI backend
├── src/
│   ├── domain/           # Logic nghiệp vụ chính
│   │   ├── Retrieval/    # RAG pipeline (database, retrieval, chatbot, legal_graph)
│   │   ├── classification/ # Phân loại câu hỏi
│   │   ├── config.py     # Cấu hình tập trung
│   │   └── main.py       # Streamlit UI
│   ├── data_preparation/ # Trích xuất dữ liệu từ PDF
│   └── utils/            # Tiền xử lý văn bản
├── auth/                 # API key + rate limiting
├── memory/               # Cache + chat history
├── observability/        # Tracing & logging
├── data/                 # PDF nguồn, chroma_db, stopwords, keywords
├── docs/                 # Tài liệu spec (thư mục này)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
