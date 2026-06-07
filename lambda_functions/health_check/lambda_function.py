"""Lambda handler: Health check — verifies S3 bucket and Bedrock connectivity."""
import json
import os
import time

import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET_NAME")

s3 = boto3.client("s3", region_name=REGION)


def handler(event, context):
    checks = {}
    start = time.time()

    # S3 check
    try:
        s3.head_bucket(Bucket=BUCKET)
        checks["s3"] = "ok"
    except Exception as e:
        checks["s3"] = f"error: {e}"

    # Bedrock check (lightweight list call)
    try:
        br = boto3.client("bedrock", region_name=REGION)
        br.list_foundation_models(byOutputModality="TEXT")
        checks["bedrock"] = "ok"
    except Exception as e:
        checks["bedrock"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "statusCode": 200 if all_ok else 503,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
            "latency_ms": int((time.time() - start) * 1000),
        }),
    }
