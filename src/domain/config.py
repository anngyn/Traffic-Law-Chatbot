# config.py — single source of truth for paths and Bedrock settings.
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # .../AI002
DATA_DIR = ROOT_DIR / "data"

# Data files
STOPWORDS_PATH = str(DATA_DIR / "vietnamese-stopwords-dash.txt")
DATA_FOLDER = str(DATA_DIR)
KEYWORD_FILE = str(DATA_DIR / "top_keywords.txt")
PROCESSED_JSON_FILE = str(DATA_DIR / "output.json")

# AWS Bedrock
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# Retrieval
SIMILARITY_TOP_K = int(os.getenv("TOP_K_RESULTS", "10"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "AI002")

# Hybrid search + rerank
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
