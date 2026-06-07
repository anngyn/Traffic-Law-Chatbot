"""
Logging utilities for AWS Bedrock RAG Chatbot
Provides structured logging with PII masking and metrics
"""

import logging
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

from ..config.settings import get_config

class StructuredLogger:
    """Structured logger with PII masking and metrics support"""
    
    def __init__(self, name: str):
        self.config = get_config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Configure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _mask_pii(self, data: Any) -> Any:
        """Mask personally identifiable information in log data"""
        if not self.config.mask_pii_in_logs:
            return data
        
        if isinstance(data, str):
            # Mask IP addresses
            data = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '***.***.***.**', data)
            # Mask email addresses
            data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.***', data)
            # Mask phone numbers (Vietnamese format)
            data = re.sub(r'\b0\d{9,10}\b', '0*********', data)
            return data
        
        elif isinstance(data, dict):
            return {key: self._mask_pii(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self._mask_pii(item) for item in data]
        
        return data
    
    def _create_log_entry(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create structured log entry"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'service': 'bedrock-rag-chatbot'
        }
        
        if extra:
            log_entry.update(self._mask_pii(extra))
        
        return log_entry
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        if self.config.enable_metrics:
            log_entry = self._create_log_entry('INFO', message, extra)
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.info(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        if self.config.enable_metrics:
            log_entry = self._create_log_entry('ERROR', message, extra)
            self.logger.error(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.error(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        if self.config.enable_metrics:
            log_entry = self._create_log_entry('WARNING', message, extra)
            self.logger.warning(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.warning(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        if self.config.debug:
            if self.config.enable_metrics:
                log_entry = self._create_log_entry('DEBUG', message, extra)
                self.logger.debug(json.dumps(log_entry, ensure_ascii=False))
            else:
                self.logger.debug(message)
    
    def log_request(self, request_id: str, method: str, path: str, query_params: Optional[Dict] = None):
        """Log incoming request"""
        self.info(
            f"Incoming request: {method} {path}",
            extra={
                'request_id': request_id,
                'method': method,
                'path': path,
                'query_params': query_params or {}
            }
        )
    
    def log_response(self, request_id: str, status_code: int, latency_ms: int):
        """Log outgoing response"""
        self.info(
            f"Response sent: {status_code}",
            extra={
                'request_id': request_id,
                'status_code': status_code,
                'latency_ms': latency_ms
            }
        )
    
    def log_bedrock_call(self, model_id: str, input_tokens: int, output_tokens: int, latency_ms: int):
        """Log Bedrock API call"""
        self.info(
            f"Bedrock API call: {model_id}",
            extra={
                'model_id': model_id,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency_ms': latency_ms,
                'service': 'bedrock'
            }
        )
    
    def log_vector_search(self, query_length: int, results_count: int, latency_ms: int, confidence_scores: list):
        """Log vector search operation"""
        self.info(
            f"Vector search completed: {results_count} results",
            extra={
                'query_length': query_length,
                'results_count': results_count,
                'latency_ms': latency_ms,
                'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0,
                'service': 'vector_search'
            }
        )
    
    def log_document_processing(self, filename: str, file_size: int, chunks_created: int, processing_time_ms: int):
        """Log document processing operation"""
        self.info(
            f"Document processed: {filename}",
            extra={
                'filename': filename,
                'file_size_bytes': file_size,
                'chunks_created': chunks_created,
                'processing_time_ms': processing_time_ms,
                'service': 'document_indexer'
            }
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        self.warning(
            f"Security event: {event_type}",
            extra={
                'event_type': event_type,
                'details': details,
                'service': 'security'
            }
        )

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)