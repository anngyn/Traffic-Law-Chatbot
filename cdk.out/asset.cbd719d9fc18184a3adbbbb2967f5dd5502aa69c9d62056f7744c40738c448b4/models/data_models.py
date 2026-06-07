"""
Data models for AWS Bedrock RAG Chatbot
Defines data structures for documents, queries, and responses
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    TXT = "txt"
    HTML = "html"

class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    chunk_id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Document-specific metadata
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    article_number: Optional[str] = None
    law_reference: Optional[str] = None
    chunk_index: int = 0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Document:
    """Represents a source document"""
    document_id: str
    filename: str
    document_type: DocumentType
    file_size: int
    content: Optional[str] = None
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Document metadata
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    law_type: Optional[str] = None  # e.g., "Nghị định", "Thông tư"
    law_number: Optional[str] = None  # e.g., "100/2019/NĐ-CP"
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

@dataclass
class Citation:
    """Represents a citation from source material"""
    source: str
    article: Optional[str] = None
    content: str = ""
    relevance_score: float = 0.0
    page_number: Optional[int] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

@dataclass
class QueryRequest:
    """Represents an incoming query request"""
    query: str
    request_id: Optional[str] = None
    max_results: int = 5
    confidence_threshold: float = 0.7
    enable_refinement: bool = True
    
    # Request metadata
    timestamp: Optional[datetime] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None  # Will be masked in logs
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.request_id is None:
            self.request_id = f"req_{int(self.timestamp.timestamp() * 1000)}"

@dataclass
class QueryResponse:
    """Represents a query response"""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0
    latency_ms: int = 0
    
    # Response metadata
    request_id: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: int = 0
    refinement_applied: bool = False
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class VectorSearchResult:
    """Represents a vector search result"""
    chunk_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    document_id: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    article_number: Optional[str] = None

@dataclass
class IndexManifest:
    """Represents the vector index manifest"""
    version: str
    created_at: datetime
    total_chunks: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'version': self.version,
            'created_at': self.created_at.isoformat() + 'Z',
            'total_chunks': self.total_chunks,
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'documents': self.documents
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexManifest':
        """Create from dictionary"""
        return cls(
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            total_chunks=data['total_chunks'],
            embedding_model=data['embedding_model'],
            chunk_size=data['chunk_size'],
            chunk_overlap=data['chunk_overlap'],
            documents=data.get('documents', [])
        )

@dataclass
class HealthCheckResult:
    """Represents a health check result"""
    service_name: str
    status: str  # 'healthy', 'unhealthy', 'pending'
    message: str
    latency_ms: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SystemHealth:
    """Represents overall system health"""
    overall_status: str  # 'healthy', 'unhealthy', 'degraded'
    checks: List[HealthCheckResult] = field(default_factory=list)
    uptime_ms: int = 0
    version: str = "1.0.0"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            'status': self.overall_status,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'uptime_ms': self.uptime_ms,
            'version': self.version,
            'checks': {
                check.service_name: {
                    'status': check.status,
                    'message': check.message,
                    'latency_ms': check.latency_ms,
                    'details': check.details
                }
                for check in self.checks
            }
        }

@dataclass
class ProcessingMetrics:
    """Represents processing metrics for monitoring"""
    operation_type: str  # 'query', 'indexing', 'health_check'
    latency_ms: int
    success: bool
    error_message: Optional[str] = None
    
    # Resource usage
    tokens_used: int = 0
    memory_used_mb: Optional[float] = None
    
    # Operation-specific metrics
    results_count: Optional[int] = None
    confidence_score: Optional[float] = None
    
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow() 