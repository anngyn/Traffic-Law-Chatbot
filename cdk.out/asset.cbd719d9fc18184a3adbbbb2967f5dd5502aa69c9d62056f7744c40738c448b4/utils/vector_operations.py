"""
Vector operations utilities for FAISS and embedding management
Handles vector index creation, search, and management
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tempfile
from datetime import datetime

from ..config.settings import get_config
from ..models.data_models import VectorSearchResult, IndexManifest
from .logging_utils import get_logger
from .error_handling import VectorSearchError

logger = get_logger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        self.config = get_config()
        self.index = None
        self.metadata = {}
        self.manifest = None
        
    def create_index(self, dimension: int = None) -> 'faiss.Index':
        """
        Create a new FAISS index
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index
        """
        try:
            import faiss
        except ImportError:
            raise VectorSearchError("FAISS not installed. Install with: pip install faiss-cpu")
        
        dimension = dimension or self.config.vector.embedding_dimension
        index_type = self.config.vector.faiss_index_type
        
        logger.info(f"Creating FAISS index: {index_type}, dimension: {dimension}")
        
        if index_type == "Flat":
            index = faiss.IndexFlatL2(dimension)
        elif index_type.startswith("IVF"):
            # Parse IVF parameters (e.g., "IVF1024,Flat")
            parts = index_type.split(',')
            nlist = int(parts[0].replace('IVF', ''))
            quantizer_type = parts[1] if len(parts) > 1 else 'Flat'
            
            if quantizer_type == 'Flat':
                quantizer = faiss.IndexFlatL2(dimension)
            else:
                raise VectorSearchError(f"Unsupported quantizer type: {quantizer_type}")
            
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            raise VectorSearchError(f"Unsupported index type: {index_type}")
        
        self.index = index
        logger.info(f"Created FAISS index successfully")
        return index
    
    def train_index(self, embeddings: np.ndarray) -> None:
        """
        Train the FAISS index (required for IVF indices)
        
        Args:
            embeddings: Training embeddings
        """
        if self.index is None:
            raise VectorSearchError("Index not created. Call create_index() first.")
        
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info(f"Training index with {len(embeddings)} vectors")
            self.index.train(embeddings.astype(np.float32))
            logger.info("Index training completed")
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors to the index
        
        Args:
            embeddings: Embedding vectors to add
            metadata: Metadata for each vector
        """
        if self.index is None:
            raise VectorSearchError("Index not created. Call create_index() first.")
        
        if len(embeddings) != len(metadata):
            raise VectorSearchError("Number of embeddings must match number of metadata entries")
        
        # Convert to float32 for FAISS
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train_index(embeddings_f32)
        
        # Add vectors
        start_id = self.index.ntotal
        self.index.add(embeddings_f32)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            self.metadata[start_id + i] = meta
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = None, confidence_threshold: float = None) -> List[VectorSearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of search results
        """
        if self.index is None or self.index.ntotal == 0:
            raise VectorSearchError("Index is empty or not loaded")
        
        k = k or self.config.vector.top_k_results
        confidence_threshold = confidence_threshold or self.config.vector.confidence_threshold
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert distances to similarity scores (L2 distance -> similarity)
        # Lower distance = higher similarity
        max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        similarities = 1.0 - (distances[0] / (max_distance + 1e-8))
        
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities)):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            if similarity < confidence_threshold:
                continue
            
            metadata = self.metadata.get(idx, {})
            
            result = VectorSearchResult(
                chunk_id=metadata.get('chunk_id', f'chunk_{idx}'),
                content=metadata.get('content', ''),
                similarity_score=float(similarity),
                metadata=metadata,
                document_id=metadata.get('document_id'),
                source_file=metadata.get('source_file'),
                page_number=metadata.get('page_number'),
                article_number=metadata.get('article_number')
            )
            results.append(result)
        
        logger.info(f"Vector search returned {len(results)} results above threshold {confidence_threshold}")
        return results
    
    def save_index(self, index_path: str, metadata_path: str, manifest_path: str) -> None:
        """
        Save index, metadata, and manifest to files
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
            manifest_path: Path to save manifest
        """
        if self.index is None:
            raise VectorSearchError("No index to save")
        
        try:
            import faiss
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
            
            # Create and save manifest
            manifest = IndexManifest(
                version="1.0.0",
                created_at=datetime.utcnow(),
                total_chunks=self.index.ntotal,
                embedding_model=self.config.bedrock.embedding_model_id,
                chunk_size=self.config.text.chunk_size,
                chunk_overlap=self.config.text.chunk_overlap,
                documents=self._get_document_summary()
            )
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved manifest to {manifest_path}")
            
            self.manifest = manifest
            
        except Exception as e:
            raise VectorSearchError(f"Failed to save index: {str(e)}")
    
    def load_index(self, index_path: str, metadata_path: str, manifest_path: str = None) -> None:
        """
        Load index, metadata, and manifest from files
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            manifest_path: Path to manifest file (optional)
        """
        try:
            import faiss
            
            # Load FAISS index
            if not os.path.exists(index_path):
                raise VectorSearchError(f"Index file not found: {index_path}")
            
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}. Total vectors: {self.index.ntotal}")
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    # Convert string keys back to integers
                    metadata_raw = json.load(f)
                    self.metadata = {int(k): v for k, v in metadata_raw.items()}
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                self.metadata = {}
            
            # Load manifest if available
            if manifest_path and os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                    self.manifest = IndexManifest.from_dict(manifest_data)
                logger.info(f"Loaded manifest from {manifest_path}")
            
        except Exception as e:
            raise VectorSearchError(f"Failed to load index: {str(e)}")
    
    def _get_document_summary(self) -> List[Dict[str, Any]]:
        """Get summary of documents in the index"""
        doc_summary = {}
        
        for meta in self.metadata.values():
            doc_id = meta.get('document_id', 'unknown')
            if doc_id not in doc_summary:
                doc_summary[doc_id] = {
                    'id': doc_id,
                    'filename': meta.get('source_file', 'unknown'),
                    'chunks': 0,
                    'processed_at': datetime.utcnow().isoformat() + 'Z'
                }
            doc_summary[doc_id]['chunks'] += 1
        
        return list(doc_summary.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            return {'total_vectors': 0, 'is_trained': False}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'is_trained': getattr(self.index, 'is_trained', True),
            'index_type': type(self.index).__name__,
            'metadata_entries': len(self.metadata)
        }
        
        if self.manifest:
            stats['manifest_version'] = self.manifest.version
            stats['total_documents'] = len(self.manifest.documents)
        
        return stats

class EmbeddingManager:
    """Manages embedding generation and caching"""
    
    def __init__(self):
        self.config = get_config()
        self.cache = {}
    
    def generate_embeddings(self, texts: List[str], model_id: str = None) -> np.ndarray:
        """
        Generate embeddings for texts (placeholder - will be implemented in later tasks)
        
        Args:
            texts: List of texts to embed
            model_id: Model ID to use for embeddings
            
        Returns:
            Numpy array of embeddings
        """
        model_id = model_id or self.config.bedrock.embedding_model_id
        
        # This is a placeholder implementation
        # Real implementation will use Bedrock or local models
        logger.info(f"Generating embeddings for {len(texts)} texts using {model_id}")
        
        # Return dummy embeddings for now
        dimension = self.config.vector.embedding_dimension
        embeddings = np.random.random((len(texts), dimension)).astype(np.float32)
        
        return embeddings
    
    def get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding if available"""
        return self.cache.get(text_hash)
    
    def cache_embedding(self, text_hash: str, embedding: np.ndarray) -> None:
        """Cache embedding for future use"""
        self.cache[text_hash] = embedding

# Global instances
_vector_store_manager = None
_embedding_manager = None

def get_vector_store_manager() -> VectorStoreManager:
    """Get the global vector store manager instance"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager

def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager