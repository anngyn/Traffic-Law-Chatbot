"""
Retrieval-only evaluation: Precision@k, Recall@k, MRR@k, Hit Rate@k, Latency.

Run from AI002/src/domain/:
    python ../../tests/eval_retrieval.py [--k 1,3,5,10] [--limit N]
Or from AI002/:
    python tests/eval_retrieval.py
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

import config
from Retrieval.database import ChromaVectorStoreManager
from Retrieval.retrieval import Retrieval, _build_retriever
from Retrieval.chatbot import _detect_penalty_intent, _PENALTY_DOC_FILTER

EVAL_FILE = ROOT / "tests" / "eval_dataset.json"
DEFAULT_K = [1, 3, 5, 10]
WARMUP = 2


def parse_relevant_id(rel_id: str) -> tuple[str, int | None]:
    """'36/2024/QH15:d12' -> ('36/2024/QH15', 12)"""
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


def precision_at_k(rel: list[bool], k: int) -> float:
    top = rel[:k]
    return sum(top) / k if top else 0.0


def recall_at_k(rel: list[bool], k: int, n_gt: int) -> float:
    if n_gt == 0:
        return 1.0
    return min(sum(rel[:k]), n_gt) / n_gt


def reciprocal_rank(rel: list[bool]) -> float:
    for rank, hit in enumerate(rel, 1):
        if hit:
            return 1.0 / rank
    return 0.0


def run_eval(k_values: list[int], limit: int | None = None, no_graph: bool = False):
    with open(EVAL_FILE, encoding="utf-8") as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    print(f"Loading vectorstore ...")
    db = ChromaVectorStoreManager(data_folder=config.DATA_FOLDER)
    docs = db.load_documents(config.PROCESSED_JSON_FILE)

    # Use Retrieval.retrieve() so graph expansion is included (same pipeline as production)
    retrieval = Retrieval(vectorstore=db.vectorstore, documents=docs)
    expand_graph = not no_graph

    print(f"Warmup ({WARMUP} queries, graph={'on' if expand_graph else 'off'}) ...")
    for q in queries[:WARMUP]:
        retrieval.retrieve(q["query"], expand_graph=expand_graph)

    latencies_ms: list[float] = []
    rr_list: list[float] = []
    rel_cache: list[list[bool]] = []

    p_sums = {k: 0.0 for k in k_values}
    r_sums = {k: 0.0 for k in k_values}

    print(f"\nEvaluating {len(queries)} queries (graph_expand={expand_graph}) ...\n")

    for q in queries:
        # Simulate production routing (same as ChatBot.process_query)
        metadata_filter = _PENALTY_DOC_FILTER if _detect_penalty_intent(q["query"]) else None
        t0 = time.perf_counter()
        retrieved = retrieval.retrieve(q["query"], metadata_filter=metadata_filter, expand_graph=expand_graph)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        rel = [is_relevant(d.metadata, q["relevant_doc_ids"]) for d in retrieved]
        rel_cache.append(rel)
        n_gt = len(q["relevant_doc_ids"])

        rr = reciprocal_rank(rel)
        rr_list.append(rr)

        for k in k_values:
            p_sums[k] += precision_at_k(rel, k)
            r_sums[k] += recall_at_k(rel, k, n_gt)

        first_hit = next((i + 1 for i, r in enumerate(rel) if r), None)
        print(
            f"  [{q['id']}] {latencies_ms[-1]:.0f}ms "
            f"RR={rr:.2f} first_hit_rank={first_hit} retrieved={len(retrieved)}"
        )

    n = len(queries)
    mrr = sum(rr_list) / n
    lat_sorted = sorted(latencies_ms)

    print("\n" + "=" * 58)
    print(f"RETRIEVAL METRICS  (n={n})")
    print("=" * 58)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for k in k_values:
        p = p_sums[k] / n
        r = r_sums[k] / n
        hr = sum(1 for rel in rel_cache if any(rel[:k])) / n
        print(f"{'Precision@' + str(k):<30} {p:>10.4f}")
        print(f"{'Recall@' + str(k):<30} {r:>10.4f}")
        print(f"{'Hit Rate@' + str(k):<30} {hr:>10.4f}")
        print()
    print(f"{'MRR':<30} {mrr:>10.4f}")
    print("-" * 42)
    print(f"{'Latency avg (ms)':<30} {sum(latencies_ms)/n:>10.1f}")
    print(f"{'Latency p50 (ms)':<30} {lat_sorted[int(0.50 * n)]:>10.1f}")
    print(f"{'Latency p95 (ms)':<30} {lat_sorted[min(int(0.95 * n), n-1)]:>10.1f}")
    print(f"{'Latency p99 (ms)':<30} {lat_sorted[min(int(0.99 * n), n-1)]:>10.1f}")
    print("=" * 58)

    out: dict[str, Any] = {
        "n_queries": n,
        "mrr": round(mrr, 4),
        "metrics_by_k": {},
        "latency_ms": {
            "avg": round(sum(latencies_ms) / n, 2),
            "p50": round(lat_sorted[int(0.50 * n)], 2),
            "p95": round(lat_sorted[min(int(0.95 * n), n - 1)], 2),
            "p99": round(lat_sorted[min(int(0.99 * n), n - 1)], 2),
        },
    }
    for k in k_values:
        out["metrics_by_k"][f"k{k}"] = {
            "precision": round(p_sums[k] / n, 4),
            "recall":    round(r_sums[k] / n, 4),
            "hit_rate":  round(sum(1 for rel in rel_cache if any(rel[:k])) / n, 4),
        }

    out_path = ROOT / "tests" / "eval_retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,3,5,10", help="Comma-separated k values")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    ap.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    args = ap.parse_args()
    k_vals = [int(x) for x in args.k.split(",")]
    run_eval(k_vals, args.limit, no_graph=args.no_graph)
