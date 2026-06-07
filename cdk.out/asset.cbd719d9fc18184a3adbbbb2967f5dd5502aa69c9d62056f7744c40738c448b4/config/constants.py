"""
Constants for AWS Bedrock RAG Chatbot
"""

# AWS Service Names
AWS_BEDROCK = "bedrock-runtime"
AWS_S3 = "s3"
AWS_LAMBDA = "lambda"
AWS_EVENTBRIDGE = "events"
AWS_CLOUDWATCH = "cloudwatch"

# Bedrock Model IDs
BEDROCK_MODELS = {
    "embedding": {
        "primary": "amazon.titan-embed-text-v2:0",
        "fallback": "amazon.titan-embed-text-v1"
    },
    "llm": {
        "primary": "meta.llama3-8b-instruct-v1:0",
        "fallback": "mistral.mistral-small-v1:0"
    }
}

# Local Model Names
LOCAL_MODELS = {
    "embedding": "intfloat/multilingual-e5-small"
}

# Vector Search Configuration
VECTOR_CONFIG = {
    "faiss_index_type": "IVF1024,Flat",
    "embedding_dimensions": {
        "titan-v2": 1024,
        "titan-v1": 1536,
        "e5-small": 384
    },
    "nlist": 1024,
    "default_top_k": 5,
    "max_top_k": 20,
    "min_confidence": 0.3,
    "default_confidence": 0.7
}

# Text Processing Configuration
TEXT_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "max_chunk_size": 1024,
    "min_chunk_size": 100,
    "max_tokens": 8000,
    "max_query_length": 1000
}

# S3 Structure
S3_STRUCTURE = {
    "raw": {
        "pdf": "raw/pdf/",
        "txt": "raw/txt/",
        "html": "raw/html/"
    },
    "index": {
        "faiss": "index/faiss.index",
        "metadata": "index/metadata.json",
        "manifest": "index/manifest.json"
    },
    "logs": {
        "processing": "logs/processing/",
        "errors": "logs/errors/"
    }
}

# Lambda Configuration
LAMBDA_CONFIG = {
    "rag_orchestrator": {
        "memory": 1024,
        "timeout": 30,
        "runtime": "python3.12"
    },
    "document_indexer": {
        "memory": 512,
        "timeout": 30,
        "runtime": "python3.12"
    },
    "health_check": {
        "memory": 128,
        "timeout": 10,
        "runtime": "python3.12"
    }
}

# API Endpoints
API_ENDPOINTS = {
    "chat": "/chat",
    "ingest": "/ingest",
    "health": "/health"
}

# HTTP Status Codes
HTTP_STATUS = {
    "OK": 200,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_query": "Truy vấn không hợp lệ. Vui lòng nhập câu hỏi về luật giao thông.",
    "empty_query": "Vui lòng nhập câu hỏi.",
    "query_too_long": f"Câu hỏi quá dài. Tối đa {TEXT_CONFIG['max_query_length']} ký tự.",
    "service_unavailable": "Dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau.",
    "no_results": "Không tìm thấy thông tin liên quan. Vui lòng thử câu hỏi khác.",
    "processing_error": "Lỗi xử lý. Vui lòng thử lại.",
    "unsupported_format": "Định dạng file không được hỗ trợ. Chỉ hỗ trợ PDF, TXT, HTML."
}

# Success Messages
SUCCESS_MESSAGES = {
    "document_processed": "Tài liệu đã được xử lý thành công.",
    "index_updated": "Chỉ mục đã được cập nhật.",
    "system_healthy": "Hệ thống hoạt động bình thường."
}

# Vietnamese Language Configuration
VIETNAMESE_CONFIG = {
    "stopwords_file": "data/vietnamese-stopwords-dash.txt",
    "enable_segmentation": True,
    "normalize_unicode": True,
    "remove_accents": False  # Keep accents for Vietnamese
}

# Free Tier Limits (for monitoring)
FREE_TIER_LIMITS = {
    "s3_storage_gb": 5,
    "lambda_requests_monthly": 1_000_000,
    "api_requests_monthly": 1_000_000,
    "cloudwatch_logs_gb": 5,
    "bedrock_tokens_monthly": 20_000  # Approximate free tier limit
}

# Retry Configuration
RETRY_CONFIG = {
    "max_attempts": 3,
    "backoff_multiplier": 1,
    "min_wait": 4,
    "max_wait": 10,
    "retryable_errors": [
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerError",
        "TimeoutError"
    ]
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_guardrails": True,
    "mask_pii_in_logs": True,
    "allowed_file_extensions": [".pdf", ".txt", ".html"],
    "max_file_size_mb": 10,
    "cors_origins": ["*"],  # Configure appropriately for production
    "rate_limit_per_minute": 60
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "enable_structured": True,
    "enable_metrics": True,
    "enable_tracing": False
}

# System Prompts
SYSTEM_PROMPTS = {
    "vietnamese_rag": """
Bạn là trợ lý AI chuyên về luật giao thông Việt Nam. 

QUY TẮC QUAN TRỌNG:
- Chỉ trả lời các câu hỏi liên quan đến luật giao thông Việt Nam
- KHÔNG đưa ra tư vấn pháp lý tuyệt đối
- Luôn khuyến khích tham khảo ý kiến chuyên gia pháp lý
- Trích dẫn chính xác điều khoản pháp luật từ ngữ cảnh được cung cấp
- Sử dụng ngôn ngữ rõ ràng, dễ hiểu
- Nếu không chắc chắn, hãy nói rõ sự không chắc chắn

Nếu câu hỏi không liên quan đến luật giao thông Việt Nam, hãy từ chối trả lời một cách lịch sự và hướng dẫn người dùng đặt câu hỏi phù hợp.

Ngữ cảnh: {context}
Câu hỏi: {query}

Trả lời:""",
    
    "query_refinement": """
Bạn là chuyên gia về luật giao thông Việt Nam. Hãy cải thiện câu truy vấn sau để tìm kiếm thông tin chính xác hơn trong cơ sở dữ liệu pháp luật.

Câu truy vấn gốc: {original_query}

Hãy tạo 2-3 phiên bản cải thiện của câu truy vấn, tập trung vào:
- Sử dụng thuật ngữ pháp lý chính xác
- Bao gồm từ khóa liên quan
- Làm rõ ý định tìm kiếm

Các phiên bản cải thiện:"""
}