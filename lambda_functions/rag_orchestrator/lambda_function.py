"""Lambda handler: RAG orchestrator — receives chat query, retrieves context, calls LLM."""
import json
import logging
import os

import boto3
import numpy as np

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET_NAME")
INDEX_PREFIX = os.getenv("S3_INDEX_PREFIX", "index/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL = os.getenv("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# In-memory cache (warm Lambda reuse)
_index_cache: dict = {}

SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo giúp trả lời các câu hỏi về luật giao thông đường bộ Việt Nam. "
    "Chỉ trả lời dựa trên ngữ cảnh được cung cấp, không thêm thông tin bên ngoài. "
    "Trích dẫn điều khoản cụ thể khi có thể."
)


def handler(event, context):
    """API Gateway POST /chat → {message, session_id?} → {answer}."""
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("message", "").strip()
        if not query:
            return _response(400, {"error": "Thiếu field 'message'."})
        if len(query) > 1000:
            return _response(400, {"error": "Câu hỏi quá dài (tối đa 1000 ký tự)."})

        # 1. Embed query
        query_vec = _embed(query)

        # 2. Load index + search
        chunks, scores = _search(query_vec, TOP_K)
        if not chunks:
            return _response(200, {"answer": "Không tìm thấy thông tin liên quan."})

        # 3. Build context
        context_str = "\n\n".join(
            f"Tiêu đề: {c['title']}\nNội dung: {c['content']}" for c in chunks
        )

        # 4. Call LLM
        answer = _invoke_llm(query, context_str)
        return _response(200, {"answer": answer})

    except Exception as e:
        logger.exception("RAG orchestrator error")
        return _response(500, {"error": "Lỗi xử lý. Vui lòng thử lại."})


def _embed(text: str) -> np.ndarray:
    """Get embedding vector from Bedrock Titan."""
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    return np.array(result["embedding"], dtype=np.float32)


def _load_index():
    """Load FAISS index + metadata from S3 (cached across warm invocations)."""
    if "vectors" in _index_cache:
        return _index_cache["vectors"], _index_cache["metadata"]

    # Download vectors and metadata from S3
    vectors_obj = s3.get_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}vectors.npy")
    vectors = np.frombuffer(vectors_obj["Body"].read(), dtype=np.float32)

    meta_obj = s3.get_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}metadata.json")
    metadata = json.loads(meta_obj["Body"].read())

    dim = len(metadata[0].get("embedding_dim", [])) or 1024
    vectors = vectors.reshape(-1, dim)

    _index_cache["vectors"] = vectors
    _index_cache["metadata"] = metadata
    return vectors, metadata


def _search(query_vec: np.ndarray, top_k: int):
    """Brute-force cosine similarity search (no FAISS dependency in Lambda)."""
    try:
        vectors, metadata = _load_index()
    except Exception as e:
        logger.warning(f"Index not found: {e}")
        return [], []

    # Cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    similarities = (vectors / norms) @ query_norm

    top_idx = np.argsort(similarities)[::-1][:top_k]
    chunks = [metadata[i] for i in top_idx]
    scores = [float(similarities[i]) for i in top_idx]
    return chunks, scores


def _invoke_llm(question: str, context: str) -> str:
    """Call Claude 3 Haiku via Bedrock."""
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Ngữ cảnh:\n---------------------\n{context}\n---------------------\n\n"
        f"Câu hỏi: {question}\n"
        f"Câu trả lời (kèm trích dẫn điều khoản):"
    )
    resp = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }
