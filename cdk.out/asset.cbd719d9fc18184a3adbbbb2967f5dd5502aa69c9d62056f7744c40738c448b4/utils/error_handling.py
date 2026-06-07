"""
Error handling utilities for AWS Bedrock RAG Chatbot
Provides consistent error handling and retry logic
"""

import time
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.constants import RETRY_CONFIG, ERROR_MESSAGES, HTTP_STATUS
from .logging_utils import get_logger

logger = get_logger(__name__)

class RAGChatbotError(Exception):
    """Base exception for RAG Chatbot errors"""
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class BedrockError(RAGChatbotError):
    """Bedrock service related errors"""
    def __init__(self, message: str, model_id: Optional[str] = None):
        self.model_id = model_id
        super().__init__(message, "BEDROCK_ERROR", 503)

class VectorSearchError(RAGChatbotError):
    """Vector search related errors"""
    def __init__(self, message: str):
        super().__init__(message, "VECTOR_SEARCH_ERROR", 500)

class DocumentProcessingError(RAGChatbotError):
    """Document processing related errors"""
    def __init__(self, message: str, filename: Optional[str] = None):
        self.filename = filename
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", 400)

class ValidationError(RAGChatbotError):
    """Input validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message, "VALIDATION_ERROR", 400)

def create_retry_decorator(
    max_attempts: int = RETRY_CONFIG['max_attempts'],
    backoff_multiplier: int = RETRY_CONFIG['backoff_multiplier'],
    min_wait: int = RETRY_CONFIG['min_wait'],
    max_wait: int = RETRY_CONFIG['max_wait'],
    retryable_exceptions: List[Type[Exception]] = None
):
    """Create a retry decorator with configurable parameters"""
    
    if retryable_exceptions is None:
        retryable_exceptions = [
            ClientError,
            BotoCoreError,
            BedrockError,
            VectorSearchError
        ]
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(tuple(retryable_exceptions)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.next_action} after {retry_state.outcome.exception()}"
        )
    )

def handle_bedrock_error(func: Callable) -> Callable:
    """Decorator to handle Bedrock-specific errors"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ThrottlingException':
                raise BedrockError(f"Bedrock API rate limit exceeded: {error_message}")
            elif error_code == 'ValidationException':
                raise ValidationError(f"Invalid Bedrock request: {error_message}")
            elif error_code == 'ResourceNotFoundException':
                raise BedrockError(f"Bedrock model not found: {error_message}")
            elif error_code == 'AccessDeniedException':
                raise BedrockError(f"Access denied to Bedrock: {error_message}")
            else:
                raise BedrockError(f"Bedrock error ({error_code}): {error_message}")
        except Exception as e:
            if not isinstance(e, RAGChatbotError):
                raise BedrockError(f"Unexpected Bedrock error: {str(e)}")
            raise
    
    return wrapper

def handle_s3_error(func: Callable) -> Callable:
    """Decorator to handle S3-specific errors"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'NoSuchBucket':
                raise RAGChatbotError(f"S3 bucket not found: {error_message}", "S3_BUCKET_NOT_FOUND", 404)
            elif error_code == 'NoSuchKey':
                raise RAGChatbotError(f"S3 object not found: {error_message}", "S3_OBJECT_NOT_FOUND", 404)
            elif error_code == 'AccessDenied':
                raise RAGChatbotError(f"Access denied to S3: {error_message}", "S3_ACCESS_DENIED", 403)
            else:
                raise RAGChatbotError(f"S3 error ({error_code}): {error_message}", "S3_ERROR", 500)
        except Exception as e:
            if not isinstance(e, RAGChatbotError):
                raise RAGChatbotError(f"Unexpected S3 error: {str(e)}", "S3_UNEXPECTED_ERROR", 500)
            raise
    
    return wrapper

def validate_query(query: str) -> None:
    """Validate user query input"""
    if not query or not query.strip():
        raise ValidationError(ERROR_MESSAGES['empty_query'], 'query')
    
    if len(query) > 1000:  # From TEXT_CONFIG['max_query_length']
        raise ValidationError(ERROR_MESSAGES['query_too_long'], 'query')
    
    # Check for potentially malicious content
    suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
    query_lower = query.lower()
    for pattern in suspicious_patterns:
        if pattern in query_lower:
            raise ValidationError("Truy vấn chứa nội dung không được phép.", 'query')

def validate_file_upload(filename: str, file_size: int, content_type: Optional[str] = None) -> None:
    """Validate file upload parameters"""
    if not filename:
        raise ValidationError("Tên file không được để trống.", 'filename')
    
    # Check file extension
    allowed_extensions = ['.pdf', '.txt', '.html', '.htm']
    file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    if file_extension not in allowed_extensions:
        raise ValidationError(ERROR_MESSAGES['unsupported_format'], 'filename')
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if file_size > max_size:
        raise ValidationError(f"File quá lớn. Tối đa {max_size // (1024*1024)}MB.", 'file_size')
    
    if file_size <= 0:
        raise ValidationError("File rỗng hoặc không hợp lệ.", 'file_size')

def create_error_response(error: Union[RAGChatbotError, Exception]) -> Dict[str, Any]:
    """Create standardized error response"""
    if isinstance(error, RAGChatbotError):
        return {
            'statusCode': error.status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': {
                'error': error.message,
                'error_code': error.error_code,
                'timestamp': time.time()
            }
        }
    else:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {str(error)}")
        return {
            'statusCode': HTTP_STATUS['INTERNAL_ERROR'],
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': {
                'error': ERROR_MESSAGES['processing_error'],
                'error_code': 'INTERNAL_ERROR',
                'timestamp': time.time()
            }
        }

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with context information"""
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {}
    }
    
    if isinstance(error, RAGChatbotError):
        error_details['error_code'] = error.error_code
        error_details['status_code'] = error.status_code
    
    logger.error("Error occurred", extra=error_details)