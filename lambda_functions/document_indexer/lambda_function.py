"""Lambda handler: Document Indexer — triggered by S3 upload, extracts text, embeds, updates index."""
import io
import json
import logging
import os
import re

import boto3
import numpy as np

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET_NAME")
INDEX_PREFIX = os.getenv("S3_INDEX_PREFIX", "index/")
RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "raw/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# Regex patterns for Vietnamese legal structure
_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)", re.I)
_KHOAN = re.compile(r"^(\d+)\.\s")
_CHUONG = re.compile(r"^Chương\s+([IVXLCDM\d]+)\b", re.I)


def handler(event, context):
    """EventBridge S3 Object Created → parse PDF/JSON, embed, update index."""
    try:
        # Parse event (EventBridge format)
        detail = event.get("detail", {})
        bucket = detail.get("bucket", {}).get("name", BUCKET)
        key = detail.get("object", {}).get("key", "")

        if not key:
            logger.warning("No object key in event")
            return {"status": "skipped", "reason": "no key"}

        logger.info(f"Processing: s3://{bucket}/{key}")

        # Download file
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()

        # Parse into chunks based on file type
        if key.endswith(".json"):
            chunks = _parse_json(content)
        elif key.endswith(".txt"):
            chunks = _parse_text(content.decode("utf-8"), doc_id=key)
        else:
            logger.info(f"Unsupported file type: {key}")
            return {"status": "skipped", "reason": "unsupported format"}

        if not chunks:
            return {"status": "skipped", "reason": "no chunks extracted"}

        # Embed all chunks
        vectors = []
        for chunk in chunks:
            vec = _embed(chunk["content"][:8000])  # Titan max input
            vectors.append(vec)

        vectors_np = np.array(vectors, dtype=np.float32)

        # Load existing index (if any) and append
        existing_vectors, existing_meta = _load_existing_index()
        if existing_vectors is not None:
            vectors_np = np.vstack([existing_vectors, vectors_np])
            all_meta = existing_meta + chunks
        else:
            all_meta = chunks

        # Save updated index to S3
        _save_index(vectors_np, all_meta)

        logger.info(f"Indexed {len(chunks)} chunks from {key}. Total: {len(all_meta)}")
        return {"status": "success", "chunks_added": len(chunks), "total": len(all_meta)}

    except Exception as e:
        logger.exception("Document indexer error")
        return {"status": "error", "error": str(e)}


def _embed(text: str) -> np.ndarray:
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    return np.array(result["embedding"], dtype=np.float32)


def _parse_json(content: bytes) -> list[dict]:
    """Parse pre-processed JSON records (output.json format)."""
    data = json.loads(content)
    chunks = []
    for item in data:
        chunks.append({
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "doc_id": item.get("doc_id", ""),
            "doc_type": item.get("doc_type", ""),
            "dieu": item.get("dieu", ""),
            "khoan": item.get("khoan", ""),
            "chuong": item.get("chuong", ""),
        })
    return chunks


def _parse_text(text: str, doc_id: str = "") -> list[dict]:
    """Simple text chunking by Điều/Khoản structure."""
    chunks = []
    chuong = ""
    dieu_no = None
    dieu_title = ""
    buf = []

    def flush():
        nonlocal buf
        if dieu_no and buf:
            body = " ".join(buf).strip()
            chunks.append({
                "title": f"{doc_id} - Điều {dieu_no}. {dieu_title}",
                "content": f"Điều {dieu_no}. {dieu_title}\n{body}",
                "doc_id": doc_id,
                "doc_type": "text",
                "dieu": dieu_no,
                "khoan": 0,
                "chuong": chuong,
            })
        buf = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (m := _CHUONG.match(line)):
            flush()
            chuong = m.group(1)
            continue
        if (m := _DIEU.match(line)):
            flush()
            dieu_no = int(m.group(1))
            dieu_title = m.group(2).strip()
            continue
        buf.append(line)
    flush()
    return chunks


def _load_existing_index():
    """Load existing vectors + metadata from S3."""
    try:
        vec_obj = s3.get_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}vectors.npy")
        meta_obj = s3.get_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}metadata.json")
        vectors = np.frombuffer(vec_obj["Body"].read(), dtype=np.float32)
        metadata = json.loads(meta_obj["Body"].read())
        dim = 1024  # Titan v2 dimension
        vectors = vectors.reshape(-1, dim)
        return vectors, metadata
    except s3.exceptions.NoSuchKey:
        return None, []
    except Exception:
        return None, []


def _save_index(vectors: np.ndarray, metadata: list[dict]):
    """Save vectors + metadata to S3."""
    # Save vectors as raw numpy bytes
    buf = io.BytesIO()
    buf.write(vectors.tobytes())
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}vectors.npy", Body=buf.getvalue())

    # Save metadata as JSON
    meta_json = json.dumps(metadata, ensure_ascii=False)
    s3.put_object(Bucket=BUCKET, Key=f"{INDEX_PREFIX}metadata.json", Body=meta_json.encode("utf-8"))
