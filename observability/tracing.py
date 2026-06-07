# observability/tracing.py — Structured metrics: latency, token usage, recall@k, citation accuracy.
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("observability")


@dataclass
class QueryMetrics:
    """Per-request structured metrics."""
    question: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_docs_retrieved: int = 0
    num_docs_after_graph: int = 0
    recall_at_k: float | None = None
    citation_accuracy: float | None = None


# In-memory metrics log (last N entries); production would push to CloudWatch/Prometheus.
_metrics_log: list[dict[str, Any]] = []
_MAX_LOG = int(os.getenv("METRICS_LOG_SIZE", "500"))


def record_metrics(m: QueryMetrics) -> None:
    entry = {
        "question": m.question,
        "latency_s": round(m.latency_s, 3),
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "num_docs_retrieved": m.num_docs_retrieved,
        "num_docs_after_graph": m.num_docs_after_graph,
        "recall_at_k": m.recall_at_k,
        "citation_accuracy": m.citation_accuracy,
    }
    _metrics_log.append(entry)
    if len(_metrics_log) > _MAX_LOG:
        _metrics_log.pop(0)
    logger.info("metrics: latency=%.2fs tokens_in=%d tokens_out=%d docs=%d recall@k=%s",
                m.latency_s, m.input_tokens, m.output_tokens, m.num_docs_retrieved, m.recall_at_k)


def get_metrics_log() -> list[dict[str, Any]]:
    return list(_metrics_log)


def init_tracing() -> None:
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_PROJECT", "ai002-traffic-rag")
        logger.info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])
    else:
        logger.info("LangSmith tracing disabled; using local metrics logging")


class _UsageHandler(BaseCallbackHandler):
    """Captures latency and token usage per LLM call."""

    def __init__(self):
        self.last_usage: dict[str, int] = {}

    def on_llm_start(self, *args, **kwargs):
        self._start = time.time()

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - getattr(self, "_start", time.time())
        usage: dict = {}
        try:
            usage = (response.llm_output or {}).get("usage", {})
        except Exception:
            pass
        self.last_usage = {
            "latency_s": elapsed,
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        }
        logger.info("LLM call: %.2fs tokens=%s", elapsed, usage)


_handler = _UsageHandler()


def get_callbacks() -> list:
    return [_handler]


def get_last_llm_usage() -> dict[str, int]:
    return _handler.last_usage

