"""
RAG quality evaluation: Answer Relevance, Faithfulness, Context Precision,
Context Recall — all scored by LLM (Bedrock Claude).

Metrics:
  - Answer Relevance   : does the answer address the question? (0-1)
  - Faithfulness       : are all answer claims supported by the context? (0-1)
  - Context Precision  : fraction of retrieved docs that are relevant (Precision@k)
  - Context Recall     : fraction of ground-truth claims found in context (0-1)

Run from AI002/:
    python tests/eval_rag.py [--limit N] [--k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC_DOMAIN = ROOT / "src" / "domain"
sys.path.insert(0, str(SRC_DOMAIN))
os.chdir(SRC_DOMAIN)

# ── inject AWS profile creds BEFORE any boto3/langchain import ───────────────
import boto3 as _boto3
_AWS_PROFILE = os.getenv("AWS_PROFILE", "Nova")
_sess = _boto3.Session(profile_name=_AWS_PROFILE, region_name="us-east-1")
_frozen = _sess.get_credentials().get_frozen_credentials()
os.environ["AWS_ACCESS_KEY_ID"]     = _frozen.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = _frozen.secret_key
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"
if _frozen.token:
    os.environ["AWS_SESSION_TOKEN"] = _frozen.token
print(f"AWS profile: {_AWS_PROFILE} ({_frozen.access_key[:8]}...)")

import config
from Retrieval.database import ChromaVectorStoreManager
from Retrieval.retrieval import Retrieval, _format_docs

EVAL_FILE = ROOT / "tests" / "eval_dataset.json"


# ── LLM judge helpers ─────────────────────────────────────────────────────────

def _llm_judge(llm, prompt: str) -> float:
    """Call LLM, expect JSON {'score': float 0-1, 'reason': str}."""
    from langchain_core.messages import HumanMessage
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        # strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"    [judge error] {e}  raw={text[:120] if 'text' in dir() else '?'}")
        return 0.0


ANSWER_RELEVANCE_PROMPT = """Bạn là chuyên gia đánh giá chatbot. Hãy đánh giá mức độ câu trả lời có liên quan và trả lời đúng câu hỏi.

Câu hỏi: {question}
Câu trả lời: {answer}

Trả về JSON duy nhất (không thêm gì khác):
{{"score": <float từ 0.0 đến 1.0>, "reason": "<lý do ngắn gọn>"}}

0.0 = hoàn toàn không liên quan hoặc không trả lời câu hỏi
1.0 = trả lời đầy đủ, đúng trọng tâm câu hỏi"""

FAITHFULNESS_PROMPT = """Bạn là chuyên gia đánh giá. Kiểm tra xem câu trả lời có được hỗ trợ hoàn toàn bởi ngữ cảnh không.

Ngữ cảnh:
{context}

Câu trả lời: {answer}

Trả về JSON duy nhất:
{{"score": <float từ 0.0 đến 1.0>, "reason": "<lý do>"}}

0.0 = câu trả lời chứa thông tin không có trong ngữ cảnh (hallucination)
1.0 = mọi thông tin trong câu trả lời đều có trong ngữ cảnh"""

CONTEXT_RECALL_PROMPT = """Bạn là chuyên gia đánh giá. Kiểm tra xem câu trả lời chuẩn (ground truth) có thể được suy ra từ ngữ cảnh được cung cấp không.

Ngữ cảnh:
{context}

Câu trả lời chuẩn: {ground_truth}

Trả về JSON duy nhất:
{{"score": <float từ 0.0 đến 1.0>, "reason": "<lý do>"}}

0.0 = ngữ cảnh không chứa thông tin cần thiết để trả lời
1.0 = ngữ cảnh chứa đầy đủ thông tin để đưa ra câu trả lời chuẩn"""


# ── relevance check (same as eval_retrieval.py) ───────────────────────────────

def parse_relevant_id(rel_id: str) -> tuple[str, int | None]:
    if ":d" in rel_id:
        doc_part, dieu_part = rel_id.rsplit(":d", 1)
        try:
            return doc_part, int(dieu_part)
        except ValueError:
            return rel_id, None
    return rel_id, None


def is_relevant(meta: dict[str, Any], relevant_ids: list[str]) -> bool:
    for rel_id in relevant_ids:
        doc_id, dieu = parse_relevant_id(rel_id)
        if meta.get("doc_id") == doc_id:
            if dieu is None or meta.get("dieu") == dieu:
                return True
    return False


# ── main eval ─────────────────────────────────────────────────────────────────

def run_eval(limit: int | None = None, top_k: int = 10):
    with open(EVAL_FILE, encoding="utf-8") as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    print("Loading vectorstore ...")
    db = ChromaVectorStoreManager(data_folder=config.DATA_FOLDER)
    docs = db.load_documents(config.PROCESSED_JSON_FILE)

    retrieval = Retrieval(vectorstore=db.vectorstore, documents=docs)

    # Override top_k
    if hasattr(retrieval.retriever, "search_kwargs"):
        retrieval.retriever.search_kwargs["k"] = top_k

    from langchain_aws import ChatBedrock
    judge_llm = ChatBedrock(
        model_id=config.LLM_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    scores: dict[str, list[float]] = {
        "answer_relevance": [],
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
    }
    latencies_ms: list[float] = []

    print(f"\nEvaluating {len(queries)} queries (top_k={top_k}) ...\n")

    for q in queries:
        print(f"  [{q['id']}] {q['query'][:60]}")

        # ── retrieve + generate ───────────────────────────────────────────────
        t0 = time.perf_counter()
        answer, retrieved_docs = retrieval.query_with_context(q["query"])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        context_text = _format_docs(retrieved_docs)

        # ── context precision: fraction of retrieved docs that are relevant ───
        rel_flags = [is_relevant(d.metadata, q["relevant_doc_ids"]) for d in retrieved_docs]
        cp = sum(rel_flags) / len(rel_flags) if rel_flags else 0.0
        scores["context_precision"].append(cp)

        # ── context recall: LLM checks if ground truth is inferrable ─────────
        cr = _llm_judge(judge_llm, CONTEXT_RECALL_PROMPT.format(
            context=context_text,
            ground_truth=q["ground_truth"],
        ))
        scores["context_recall"].append(cr)

        # ── faithfulness: LLM checks answer vs context ────────────────────────
        faith = _llm_judge(judge_llm, FAITHFULNESS_PROMPT.format(
            context=context_text,
            answer=answer,
        ))
        scores["faithfulness"].append(faith)

        # ── answer relevance: LLM checks answer vs question ───────────────────
        ar = _llm_judge(judge_llm, ANSWER_RELEVANCE_PROMPT.format(
            question=q["query"],
            answer=answer,
        ))
        scores["answer_relevance"].append(ar)

        print(
            f"    lat={latencies_ms[-1]:.0f}ms "
            f"CP={cp:.2f} CR={cr:.2f} Faith={faith:.2f} AR={ar:.2f}"
        )

    n = len(queries)
    lat_sorted = sorted(latencies_ms)

    print("\n" + "=" * 58)
    print(f"RAG QUALITY METRICS  (n={n})")
    print("=" * 58)
    print(f"{'Metric':<32} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 58)
    labels = {
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
        "faithfulness":      "Faithfulness",
        "answer_relevance":  "Answer Relevance",
    }
    for key, label in labels.items():
        vals = scores[key]
        print(
            f"  {label:<30} {sum(vals)/n:>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}"
        )
    print("-" * 58)
    print(f"  {'End-to-end latency avg (ms)':<30} {sum(latencies_ms)/n:>8.1f}")
    print(f"  {'End-to-end latency p50 (ms)':<30} {lat_sorted[int(0.5*n)]:>8.1f}")
    print(f"  {'End-to-end latency p95 (ms)':<30} {lat_sorted[min(int(0.95*n), n-1)]:>8.1f}")
    print("=" * 58)

    out: dict[str, Any] = {
        "n_queries": n,
        "top_k": top_k,
        "scores": {k: {
            "mean": round(sum(v) / n, 4),
            "min":  round(min(v), 4),
            "max":  round(max(v), 4),
            "all":  [round(x, 4) for x in v],
        } for k, v in scores.items()},
        "latency_ms": {
            "avg": round(sum(latencies_ms) / n, 2),
            "p50": round(lat_sorted[int(0.5 * n)], 2),
            "p95": round(lat_sorted[min(int(0.95 * n), n - 1)], 2),
        },
    }
    out_path = ROOT / "tests" / "eval_rag_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k",     type=int, default=10)
    args = ap.parse_args()
    run_eval(args.limit, args.k)
