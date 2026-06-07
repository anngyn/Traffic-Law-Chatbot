# eval/evaluate.py — RAGAS evaluation + recall@k + citation accuracy (run: python eval/evaluate.py).
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "domain"))
load_dotenv()

import config
from Retrieval.chatbot import ChatBot

from datasets import Dataset
from langchain_aws import BedrockEmbeddings, ChatBedrock
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness


def recall_at_k(retrieved_contents: list[str], ground_truth: str, k: int | None = None) -> float:
    """Fraction of ground-truth sentences found in retrieved contexts (proxy recall@k)."""
    gt_sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
    if not gt_sentences:
        return 0.0
    contexts = retrieved_contents[:k] if k else retrieved_contents
    joined = " ".join(contexts).lower()
    hits = sum(1 for s in gt_sentences if s.lower() in joined)
    return hits / len(gt_sentences)


def citation_accuracy(answer: str, retrieved_titles: list[str]) -> float:
    """Fraction of cited titles in the answer that actually exist in retrieved docs."""
    if not retrieved_titles:
        return 0.0
    cited = [t for t in retrieved_titles if t and t in answer]
    # If no citations found in answer, return 0
    return len(cited) / len(retrieved_titles) if retrieved_titles else 0.0


def main():
    bot = ChatBot(config.STOPWORDS_PATH, config.DATA_FOLDER, config.KEYWORD_FILE, config.PROCESSED_JSON_FILE)

    with open(os.path.join(ROOT, "eval", "testset.json"), encoding="utf-8") as f:
        testset = json.load(f)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    recall_scores: list[float] = []
    citation_scores: list[float] = []

    for item in testset:
        answer, docs = bot.retrieval.query_with_context(item["question"])
        contexts = [d.page_content for d in docs]
        titles = [d.metadata.get("title", "") for d in docs]

        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

        recall_scores.append(recall_at_k(contexts, item["ground_truth"]))
        citation_scores.append(citation_accuracy(answer, titles))

    # RAGAS metrics
    judge = LangchainLLMWrapper(ChatBedrock(model_id=config.LLM_MODEL_ID, region_name=config.AWS_REGION))
    emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(model_id=config.EMBEDDING_MODEL_ID, region_name=config.AWS_REGION))

    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge,
        embeddings=emb,
    )

    # Custom metrics
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0

    print("=== RAGAS Metrics ===")
    print(result)
    print(f"\n=== Custom Metrics ===")
    print(f"recall@k (avg):          {avg_recall:.4f}")
    print(f"citation_accuracy (avg): {avg_citation:.4f}")

    # Save results
    output = {
        "ragas": {k: float(v) for k, v in result.items()},
        "recall_at_k": avg_recall,
        "citation_accuracy": avg_citation,
        "per_question": [
            {"question": q, "recall": r, "citation": c}
            for q, r, c in zip(rows["question"], recall_scores, citation_scores)
        ],
    }
    out_path = os.path.join(ROOT, "eval", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
