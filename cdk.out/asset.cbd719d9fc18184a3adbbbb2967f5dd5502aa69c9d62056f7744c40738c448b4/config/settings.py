"""
Configuration management for AWS Bedrock RAG Chatbot
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from .constants import (
    BEDROCK_MODELS, LOCAL_MODELS, VECTOR_CONFIG, TEXT_CONFIG,
    LAMBDA_CONFIG, FREE_TIER_LIMITS, RETRY_CONFIG, SECURITY_CONFIG,
    LOGGING_CONFIG, VIETNAMESE_CONFIG
)

# Load environment variables
load_dotenv()

@dataclass
class AWSConfig:
    """AWS service configuration"""
    region: str
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        return cls(
            region=os.getenv('AWS_REGION', 'us-east-1'),
            access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

@dataclass
class S3Config:
    """S3 configuration"""
    bucket_name: str
    raw_prefix: str = "raw/"
    index_prefix: str = "index/"
    logs_prefix: str = "logs/"
    
    @classmethod
    def from_env(cls) -> 'S3Config':
        return cls(
            bucket_name=os.getenv('S3_BUCKET_NAME', 'rag-traffic-vi'),
            raw_prefix=os.getenv('S3_RAW_PREFIX', 'raw/'),
            index_prefix=os.getenv('S3_INDEX_PREFIX', 'index/'),
            logs_prefix=os.getenv('S3_LOGS_PREFIX', 'logs/')
        )

@dataclass
class BedrockConfig:
    """Bedrock configuration"""
    region: str
    embedding_model_id: str
    llm_model_id: str
    fallback_llm_model_id: str
    
    @classmethod
    def from_env(cls) -> 'BedrockConfig':
        return cls(
            region=os.getenv('BEDROCK_REGION', 'us-east-1'),
            embedding_model_id=os.getenv('EMBEDDING_MODEL_ID', BEDROCK_MODELS['embedding']['primary']),
            llm_model_id=os.getenv('LLM_MODEL_ID', BEDROCK_MODELS['llm']['primary']),
            fallback_llm_model_id=os.getenv('FALLBACK_LLM_MODEL_ID', BEDROCK_MODELS['llm']['fallback'])
        )

@dataclass
class VectorConfig:
    """Vector search configuration"""
    faiss_index_type: str
    embedding_dimension: int
    nlist: int
    top_k_results: int
    confidence_threshold: float
    
    @classmethod
    def from_env(cls) -> 'VectorConfig':
        return cls(
            faiss_index_type=os.getenv('FAISS_INDEX_TYPE', VECTOR_CONFIG['faiss_index_type']),
            embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', VECTOR_CONFIG['embedding_dimensions']['titan-v2'])),
            nlist=int(os.getenv('NLIST', VECTOR_CONFIG['nlist'])),
            top_k_results=int(os.getenv('TOP_K_RESULTS', VECTOR_CONFIG['default_top_k'])),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', VECTOR_CONFIG['default_confidence']))
        )

@dataclass
class TextConfig:
    """Text processing configuration"""
    chunk_size: int
    chunk_overlap: int
    max_tokens: int
    max_query_length: int
    
    @classmethod
    def from_env(cls) -> 'TextConfig':
        return cls(
            chunk_size=int(os.getenv('CHUNK_SIZE', TEXT_CONFIG['chunk_size'])),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', TEXT_CONFIG['chunk_overlap'])),
            max_tokens=int(os.getenv('MAX_TOKENS', TEXT_CONFIG['max_tokens'])),
            max_query_length=int(os.getenv('MAX_QUERY_LENGTH', TEXT_CONFIG['max_query_length']))
        )

@dataclass
class AppConfig:
    """Main application configuration"""
    aws: AWSConfig
    s3: S3Config
    bedrock: BedrockConfig
    vector: VectorConfig
    text: TextConfig
    
    # Environment settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Feature flags
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_guardrails: bool = True
    mask_pii_in_logs: bool = True
    
    # Vietnamese processing
    vietnamese_stopwords_file: str = "data/vietnamese-stopwords-dash.txt"
    enable_vietnamese_segmentation: bool = True
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        return cls(
            aws=AWSConfig.from_env(),
            s3=S3Config.from_env(),
            bedrock=BedrockConfig.from_env(),
            vector=VectorConfig.from_env(),
            text=TextConfig.from_env(),
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            enable_tracing=os.getenv('ENABLE_TRACING', 'false').lower() == 'true',
            enable_guardrails=os.getenv('ENABLE_GUARDRAILS', 'true').lower() == 'true',
            mask_pii_in_logs=os.getenv('MASK_PII_IN_LOGS', 'true').lower() == 'true',
            vietnamese_stopwords_file=os.getenv('VIETNAMESE_STOPWORDS_FILE', VIETNAMESE_CONFIG['stopwords_file']),
            enable_vietnamese_segmentation=os.getenv('ENABLE_VIETNAMESE_SEGMENTATION', 'true').lower() == 'true'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'aws': {
                'region': self.aws.region,
                'access_key_id': '***' if self.aws.access_key_id else None,
                'secret_access_key': '***' if self.aws.secret_access_key else None
            },
            's3': {
                'bucket_name': self.s3.bucket_name,
                'raw_prefix': self.s3.raw_prefix,
                'index_prefix': self.s3.index_prefix,
                'logs_prefix': self.s3.logs_prefix
            },
            'bedrock': {
                'region': self.bedrock.region,
                'embedding_model_id': self.bedrock.embedding_model_id,
                'llm_model_id': self.bedrock.llm_model_id,
                'fallback_llm_model_id': self.bedrock.fallback_llm_model_id
            },
            'vector': {
                'faiss_index_type': self.vector.faiss_index_type,
                'embedding_dimension': self.vector.embedding_dimension,
                'nlist': self.vector.nlist,
                'top_k_results': self.vector.top_k_results,
                'confidence_threshold': self.vector.confidence_threshold
            },
            'text': {
                'chunk_size': self.text.chunk_size,
                'chunk_overlap': self.text.chunk_overlap,
                'max_tokens': self.text.max_tokens,
                'max_query_length': self.text.max_query_length
            },
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'enable_metrics': self.enable_metrics,
            'enable_tracing': self.enable_tracing,
            'enable_guardrails': self.enable_guardrails,
            'mask_pii_in_logs': self.mask_pii_in_logs,
            'vietnamese_stopwords_file': self.vietnamese_stopwords_file,
            'enable_vietnamese_segmentation': self.enable_vietnamese_segmentation
        }

# Global configuration instance
config = AppConfig.from_env()

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def reload_config() -> AppConfig:
    """Reload configuration from environment"""
    global config
    config = AppConfig.from_env()
    return config