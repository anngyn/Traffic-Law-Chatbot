# AI002 — Vietnamese Traffic Law RAG Chatbot

Enterprise-grade retrieval-augmented generation system for Vietnamese traffic law queries, built with LangChain, AWS Bedrock, and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

AI002 delivers accurate, citation-backed answers to questions about Vietnamese road traffic regulations by combining:

- **Hybrid RAG Pipeline**: BM25 + vector semantic search with optional reranking
- **Production-Ready Backend**: FastAPI with auth, rate limiting, caching, and observability
- **Advanced NLP**: VnCoreNLP for Vietnamese word segmentation + OCR for scanned legal documents
- **Structure-Aware Knowledge Base**: Legal documents chunked by article/clause (Điều/Khoản) with field-level metadata
- **Multi-Deployment**: Docker Compose for local/team deployments + AWS CDK for serverless production

**Data Sources**: Law 36/2024/QH15 (Road Traffic Law) + Decree 168/2024/NĐ-CP (Penalty Regulations).

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │ ──HTTP──┐
└─────────────────┘         │
                            ▼
┌──────────────────────────────────────────────┐
│          FastAPI Backend (api/)              │
│  • Auth (API key + rate limit)              │
│  • Redis cache + PostgreSQL history         │
│  • LangSmith tracing (optional)             │
└────────────┬─────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│       RAG Pipeline (LangChain)               │
│  1. Hybrid Retriever (BM25 + Vector)        │
│  2. Optional FlashRank Reranker              │
│  3. Bedrock Claude 3 Haiku (LLM)            │
└────────────┬─────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│   Knowledge Base (ChromaDB + Bedrock Titan) │
│  • 600+ structured chunks (Điều/Khoản)      │
│  • Metadata: doc_type, dieu, khoan, refs    │
└──────────────────────────────────────────────┘
```

**Alternative Deployment**: AWS Lambda + API Gateway + S3 (see `infrastructure/` — CDK stack ready for `cdk deploy`).

---

## 🚀 Quick Start

### Prerequisites

- Python >= 3.11
- Java JRE >= 8 (for VnCoreNLP)
- Tesseract OCR (for PDF processing — optional if using pre-processed `output.json`)
- AWS credentials with Bedrock access (`bedrock:InvokeModel` for Claude + Titan)

### 1. Clone & Setup

```bash
git clone https://github.com/anngyn/Chat-bot-giao-thong.git
cd Chat-bot-giao-thong/AI002
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env: add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
```

### 3. Run with Docker (Recommended)

```bash
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

**Services started**: FastAPI backend, Streamlit UI, Redis cache, PostgreSQL history store.

### 4. Run Locally (Development)

```bash
# Terminal 1 — Backend
uvicorn api.main:app --reload

# Terminal 2 — UI
streamlit run src/domain/main.py
```

**First run**: ChromaDB will auto-index `data/output.json` via Bedrock Titan embeddings (~2-3 minutes).

---

## 📊 Features

### Core RAG
- **Hybrid Search**: Ensemble of BM25 (keyword-precise for legal citations) + vector semantic search
- **Conversational Memory**: Session-based chat history via SQLAlchemy (SQLite default, Postgres via env)
- **Vietnamese NLP**: VnCoreNLP word segmentation for accurate retrieval
- **Citation Tracking**: Returns source Điều/Khoản for every answer

### Production Features
- **Authentication**: API-key based (disabled by default for dev; enable via `API_KEYS` env var)
- **Rate Limiting**: Configurable per-IP via `slowapi`
- **Caching**: Redis-backed response cache (in-memory fallback)
- **Observability**: LangSmith tracing (opt-in via `LANGCHAIN_TRACING_V2=true`)
- **Evaluation**: RAGAS faithfulness/answer_relevancy/context_precision metrics (`eval/`)

### Data Pipeline
- **Structure-Aware Ingestion**: Chunks legal docs by atomic unit (Khoản — article clause)
- **OCR Support**: Tesseract Vietnamese OCR for scanned PDFs (auto-triggered by text density heuristic)
- **Metadata Extraction**: `doc_type` (luat/nghi_dinh), `dieu`, `khoan`, `references` auto-parsed

---

## 📁 Project Structure

```
AI002/
├── api/                    # FastAPI backend
├── auth/                   # API-key auth + rate limiting
├── memory/                 # Redis cache + SQLAlchemy history
├── observability/          # LangSmith tracing + usage logging
├── eval/                   # RAGAS evaluation + testset
├── src/domain/
│   ├── config.py           # Centralized configuration
│   ├── Retrieval/          # RAG pipeline (database, retrieval, chatbot)
│   ├── classification/     # Rule-based intent classifier
│   └── main.py             # Streamlit UI (thin client)
├── src/data_preparation/
│   └── crawl_data.py       # PDF → structured JSON with OCR
├── infrastructure/         # AWS CDK stack (Lambda + API Gateway)
├── lambda_functions/       # Serverless handlers
├── docs/                   # Architecture & deployment guides
├── tests/                  # Evaluation scripts + datasets
├── Dockerfile              # API + UI container
├── docker-compose.yml      # Full stack (api/ui/redis/postgres)
└── requirements.txt        # Python dependencies
```

---

## 🔧 Configuration

Key environment variables (see `.env.template`):

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS credentials for Bedrock | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for Bedrock | Required |
| `AWS_REGION` | Bedrock region | `us-east-1` |
| `LLM_MODEL_ID` | Bedrock LLM | `anthropic.claude-3-haiku-20240307-v1:0` |
| `EMBEDDING_MODEL_ID` | Bedrock embedding | `amazon.titan-embed-text-v2:0` |
| `API_KEYS` | Comma-separated API keys (auth disabled if empty) | `` |
| `REDIS_URL` | Redis cache URL | `` (in-memory fallback) |
| `DATABASE_URL` | SQLAlchemy DB URL | `sqlite:///data/chat_history.db` |
| `ENABLE_HYBRID` | Enable BM25+vector hybrid search | `true` |
| `ENABLE_RERANK` | Enable FlashRank reranking | `false` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |

---

## 📈 Evaluation

Run RAGAS metrics on the test set:

```bash
pip install -r requirements-eval.txt
python eval/evaluate.py
```

**Metrics**: Faithfulness, Answer Relevancy, Context Precision (judge LLM: Bedrock Claude).

---

## 🚢 Deployment Options

### Option 1: Docker Compose (Recommended for Teams)

```bash
docker compose up -d
```

Services: API, UI, Redis, PostgreSQL. Persistent volumes for Chroma + Postgres data.

### Option 2: AWS CDK (Serverless Production)

```bash
pip install -r requirements-cdk.txt
cdk bootstrap aws://ACCOUNT_ID/us-east-1
cdk deploy
# Upload data: aws s3 cp data/output.json s3://ai002-rag-traffic-ACCOUNT_ID/raw/
```

**Stack**: Lambda (RAG orchestrator + document indexer + health check) + API Gateway + S3 + EventBridge.  
**Cost**: ~$1-3/month (Free Tier eligible; Bedrock token cost dominant).

See [`docs/08-cdk-deployment.md`](docs/08-cdk-deployment.md) for details.

---

## 🧪 Development

### Re-index Knowledge Base

```bash
# After modifying data/output.json
rm -rf data/chroma_db
# Restart API → auto re-embeds via Bedrock
```

### Re-process PDFs (with OCR)

```bash
# Requires Tesseract + vie.traineddata
python src/data_preparation/crawl_data.py
# Outputs: data/output.json (600+ structured chunks)
```

### Run Tests

```bash
pytest tests/
```

---

## 📚 Documentation

- [Architecture Overview](docs/02-architecture.md)
- [Data Pipeline](docs/03-data-pipeline.md)
- [RAG Pipeline](docs/04-rag-pipeline.md)
- [API Specification](docs/05-api-spec.md)
- [Deployment Guide](docs/06-deployment.md)
- [CDK Deployment](docs/08-cdk-deployment.md)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, LangChain, Pydantic |
| **LLM** | AWS Bedrock (Claude 3 Haiku + Titan Embeddings v2) |
| **Vector DB** | ChromaDB (persistent local store) |
| **Cache** | Redis (optional, in-memory fallback) |
| **Database** | PostgreSQL / SQLite (SQLAlchemy) |
| **NLP** | VnCoreNLP, pytesseract (Tesseract OCR) |
| **UI** | Streamlit |
| **Infrastructure** | Docker Compose / AWS CDK (Lambda + API Gateway) |
| **Observability** | LangSmith (opt-in), structured logging |
| **Evaluation** | RAGAS |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---

## 📧 Contact

**Author**: Nguyen An  
**GitHub**: [@anngyn](https://github.com/anngyn)  
**Project**: [Chat-bot-giao-thong](https://github.com/anngyn/Chat-bot-giao-thong)

---

**Built with ❤️ for Vietnamese traffic law accessibility.**
