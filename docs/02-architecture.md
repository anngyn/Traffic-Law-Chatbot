# 02 - Kiến trúc hệ thống

## Sơ đồ tổng quan

```
┌─────────────┐     HTTP      ┌──────────────┐     invoke     ┌──────────────────┐
│  Streamlit  │ ────────────► │   FastAPI    │ ──────────────► │   ChatBot        │
│  (UI:8501)  │ ◄──────────── │  (API:8000)  │ ◄────────────── │   Controller     │
└─────────────┘               └──────┬───────┘                └────────┬─────────┘
                                     │                                  │
                              ┌──────┴───────┐               ┌─────────┴──────────┐
                              │   Auth       │               │  RuleBasedClassifier│
                              │  (API Key)   │               │  (Lọc câu hỏi)     │
                              │  Rate Limit  │               └─────────┬──────────┘
                              └──────────────┘                         │
                                                              ┌────────┴──────────┐
                                                              │   Retrieval       │
                                                              │  (Hybrid Search)  │
                                                              └────────┬──────────┘
                                                                       │
                              ┌────────────────────────────────────────┼────────────────┐
                              │                                        │                │
                     ┌────────┴────────┐                 ┌─────────────┴──┐    ┌────────┴───────┐
                     │  ChromaDB       │                 │  BM25Retriever │    │  LegalGraph    │
                     │  (Vector Store) │                 │  (Keyword)     │    │  (Expansion)   │
                     └────────┬────────┘                 └────────────────┘    └────────────────┘
                              │
                     ┌────────┴────────┐
                     │  AWS Bedrock    │
                     │  Titan Embed v2 │
                     └─────────────────┘

                     ┌─────────────────┐
                     │  AWS Bedrock    │
                     │  Claude 3 Haiku │  ◄── LLM Generation
                     └─────────────────┘
```

## Luồng xử lý chính (Request Flow)

```
User Input
    │
    ▼
[1] Streamlit UI → POST /chat → FastAPI
    │
    ▼
[2] Auth check (API Key) + Rate Limit (slowapi)
    │
    ▼
[3] Load chat history (SQLAlchemy)
    │
    ▼
[4] Check cache (Redis / in-memory)
    │   ├── Cache HIT → return cached answer
    │   └── Cache MISS ↓
    ▼
[5] ChatBot.process_query()
    │
    ├── [5a] RuleBasedClassifier
    │       ├── detect_language() → nếu không phải tiếng Việt → trả lời từ chối
    │       ├── keyword matching → nếu không liên quan giao thông → trả lời từ chối
    │       └── pass → tiếp tục retrieval
    │
    ▼
[6] Retrieval.query()
    │
    ├── [6a] Hybrid Search (EnsembleRetriever)
    │       ├── Vector search (ChromaDB + Titan Embed)  weight=0.6
    │       └── BM25 keyword search                     weight=0.4
    │
    ├── [6b] (Optional) Rerank (FlashrankRerank)
    │
    ├── [6c] LegalGraph expansion (BFS 1-hop)
    │       └── Mở rộng context theo quan hệ Chương/Điều/Khoản + cross-references
    │
    └── [6d] LLM Generation (Claude 3 Haiku)
            └── Prompt: context + history + question → answer
    │
    ▼
[7] Cache response + Save history → Return answer
```

## Các thành phần hệ thống

### 1. Frontend (Streamlit)
- Thin client, chỉ gọi API backend
- Session-based chat UI
- Port: 8501

### 2. Backend API (FastAPI)
- RESTful API
- Endpoints: `/health`, `/chat`
- Auth: API Key header (optional, disabled khi không set)
- Rate limiting: 30 req/min (configurable)
- Port: 8000

### 3. RAG Engine
- **ChromaVectorStoreManager**: Quản lý vector store, embedding, indexing
- **Retrieval**: Hybrid search + rerank + graph expansion + LLM chain
- **LegalGraph**: In-memory graph liên kết Chương → Điều → Khoản + cross-references
- **ChatBot**: Controller điều phối toàn bộ pipeline

### 4. Classification
- **RuleBasedClassifier**: Lọc câu hỏi không liên quan bằng keyword matching
- Detect ngôn ngữ (chỉ chấp nhận tiếng Việt)

### 5. Infrastructure Services
- **Redis**: Response cache (TTL 1h)
- **PostgreSQL/SQLite**: Chat history persistence
- **LangSmith**: Distributed tracing (optional)

## Nguyên tắc thiết kế
1. **Modular**: Mỗi concern tách riêng module (auth, memory, observability, domain)
2. **Config-driven**: Mọi tham số đều configurable qua env vars
3. **Graceful fallback**: Redis unavailable → in-memory cache; LangSmith off → local logging
4. **Cost-aware**: Classifier lọc trước để giảm số lần gọi Bedrock API
