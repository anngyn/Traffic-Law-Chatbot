"""
Utility functions for AWS Bedrock RAG Chatbot
"""

from .aws_clients import AWSClientManager, get_aws_clients
from .error_handling import (
    RAGChatbotError,
    BedrockError,
    VectorSearchError,
    DocumentProcessingError,
    ValidationError,
    create_retry_decorator,
    handle_bedrock_error,
    handle_s3_error,
    validate_query,
    validate_file_upload,
    create_error_response,
    log_error
)
from .logging_utils import StructuredLogger, get_logger
from .text_processing import VietnameseTextProcessor, get_text_processor
from .vector_operations import VectorStoreManager, EmbeddingManager, get_vector_store_manager, get_embedding_manager

__all__ = [
    "AWSClientManager",
    "get_aws_clients",
    "RAGChatbotError",
    "BedrockError", 
    "VectorSearchError",
    "DocumentProcessingError",
    "ValidationError",
    "create_retry_decorator",
    "handle_bedrock_error",
    "handle_s3_error",
    "validate_query",
    "validate_file_upload",
    "create_error_response",
    "log_error",
    "StructuredLogger",
    "get_logger",
    "VietnameseTextProcessor",
    "get_text_processor",
    "VectorStoreManager",
    "EmbeddingManager",
    "get_vector_store_manager",
    "get_embedding_manager"
]