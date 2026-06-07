# api/main.py — FastAPI backend for the AI002 RAG chatbot.
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # for auth/, memory/, observability/
sys.path.insert(0, os.path.join(ROOT, "src", "domain"))  # for config, Retrieval, classification

load_dotenv()

import config
from Retrieval.chatbot import ChatBot
from auth.security import RATE_LIMIT, limiter, require_api_key
from memory import cache as cache_mod
from memory import history as history_mod
from observability.tracing import init_tracing

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    _state["bot"] = ChatBot(
        stopwords_path=config.STOPWORDS_PATH,
        folder_path=config.DATA_FOLDER,
        keyword_file=config.KEYWORD_FILE,
        processed_json_file=config.PROCESSED_JSON_FILE,
    )
    yield
    _state.clear()


app = FastAPI(title="AI002 - Traffic Law RAG API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    metadata_filter: dict | None = None


class ChatResponse(BaseModel):
    answer: str


class Citation(BaseModel):
    title: str
    content: str
    doc_id: str | None = None
    dieu: int | str | None = None
    chuong: str | None = None


class ChatWithCitationsResponse(BaseModel):
    answer: str
    citations: list[Citation]


class RetrieveRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None
    top_k: int | None = None


class RetrieveResponse(BaseModel):
    documents: list[Citation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.LLM_MODEL_ID}


@app.get("/metrics")
def metrics() -> dict:
    from observability.tracing import get_metrics_log
    log = get_metrics_log()
    if not log:
        return {"count": 0, "entries": []}
    avg_latency = sum(e["latency_s"] for e in log) / len(log)
    return {"count": len(log), "avg_latency_s": round(avg_latency, 3), "entries": log[-20:]}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    history = history_mod.get_history(req.session_id) if req.session_id else []
    history_str = "\n".join(f"{role}: {content}" for role, content in history)

    key = cache_mod.make_key(req.session_id or "anon", history_str, req.message)
    cached = cache_mod.cache.get(key)
    if cached is not None:
        return ChatResponse(answer=cached)

    answer = _state["bot"].process_query(req.message, history_str)
    cache_mod.cache.set(key, answer)

    if req.session_id:
        history_mod.save_message(req.session_id, "user", req.message)
        history_mod.save_message(req.session_id, "assistant", answer)

    return ChatResponse(answer=answer)


def _doc_to_citation(doc) -> Citation:
    m = doc.metadata
    return Citation(
        title=m.get("title", ""),
        content=doc.page_content,
        doc_id=m.get("doc_id"),
        dieu=m.get("dieu"),
        chuong=m.get("chuong"),
    )


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def retrieve(request: Request, req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant legal documents without generating an answer."""
    docs = _state["bot"].retrieval.retrieve(req.query, metadata_filter=req.metadata_filter)
    if req.top_k:
        docs = docs[: req.top_k]
    return RetrieveResponse(documents=[_doc_to_citation(d) for d in docs])


@app.post("/chat-with-citations", response_model=ChatWithCitationsResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def chat_with_citations(request: Request, req: ChatRequest) -> ChatWithCitationsResponse:
    """Generate an answer with source citations."""
    import time as _time
    from observability.tracing import QueryMetrics, get_last_llm_usage, record_metrics

    history = history_mod.get_history(req.session_id) if req.session_id else []
    history_str = "\n".join(f"{role}: {content}" for role, content in history)

    t0 = _time.time()
    answer, docs = _state["bot"].retrieval.query_with_context(
        req.message, history_str, metadata_filter=req.metadata_filter
    )
    latency = _time.time() - t0

    usage = get_last_llm_usage()
    record_metrics(QueryMetrics(
        question=req.message,
        latency_s=latency,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        num_docs_retrieved=len(docs),
    ))

    if req.session_id:
        history_mod.save_message(req.session_id, "user", req.message)
        history_mod.save_message(req.session_id, "assistant", answer)

    return ChatWithCitationsResponse(answer=answer, citations=[_doc_to_citation(d) for d in docs])
