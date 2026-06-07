# 08 - CDK Deployment Guide

## Yêu cầu

- Node.js >= 18 (cho CDK CLI)
- Python >= 3.11
- AWS CLI configured (`aws configure`)
- AWS Account có quyền: CloudFormation, Lambda, S3, API Gateway, EventBridge, IAM

## Cài đặt

```bash
# 1. Cài CDK CLI
npm install -g aws-cdk

# 2. Cài Python dependencies
pip install -r requirements-cdk.txt

# 3. Bootstrap CDK (chỉ lần đầu, mỗi account/region)
cdk bootstrap aws://ACCOUNT_ID/us-east-1
```

## Deploy

```bash
# Xem CloudFormation template sẽ tạo
cdk synth

# Deploy lên AWS
cdk deploy
```

Output sẽ hiển thị API Gateway URL, VD:
```
Outputs:
AI002-ChatbotGiaothong.ChatbotApiEndpoint = https://xxxxx.execute-api.us-east-1.amazonaws.com/prod/
```

## Upload dữ liệu ban đầu

Sau khi deploy, upload file `output.json` (đã trích xuất từ PDF) lên S3:

```bash
# Upload pre-processed data → triggers document_indexer tự động
aws s3 cp data/output.json s3://ai002-rag-traffic-ACCOUNT_ID/raw/output.json
```

EventBridge sẽ tự động trigger `document_indexer` Lambda → embed + tạo index.

## Test API

```bash
# Health check
curl https://xxxxx.execute-api.us-east-1.amazonaws.com/prod/health

# Chat
curl -X POST https://xxxxx.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mức phạt vượt đèn đỏ là bao nhiêu?"}'
```

## Kiến trúc triển khai

```
S3 (raw/)  ──EventBridge──►  Lambda (document_indexer)  ──►  S3 (index/)
                                                                  │
API Gateway  ──POST /chat──►  Lambda (rag_orchestrator)  ◄────────┘
             ──GET /health─►  Lambda (health_check)              │
                                                                  ▼
                                                          Bedrock (Titan + Claude)
```

## Cấu trúc S3 Bucket

```
ai002-rag-traffic-{account_id}/
├── raw/              ← Upload documents vào đây
│   └── output.json
├── index/            ← Auto-generated bởi document_indexer
│   ├── vectors.npy   (numpy float32 array)
│   └── metadata.json (chunk metadata)
└── logs/             ← (reserved)
```

## Cập nhật code

```bash
# Sau khi sửa Lambda code hoặc CDK stack
cdk deploy
```

## Xóa toàn bộ

```bash
cdk destroy
# Lưu ý: S3 bucket có RemovalPolicy.RETAIN → cần xóa thủ công
aws s3 rb s3://ai002-rag-traffic-ACCOUNT_ID --force
```

## Chi phí ước tính (Free Tier)

| Service | Free Tier | Dự kiến sử dụng |
|---------|-----------|-----------------|
| Lambda | 1M requests/month | < 10K |
| API Gateway | 1M requests/month | < 10K |
| S3 | 5GB storage | < 100MB |
| Bedrock | Pay-per-token | ~$0.5-2/month |
| CloudWatch | 5GB logs | < 1GB |

**Tổng ước tính**: ~$1-3/month (chủ yếu Bedrock token cost).
