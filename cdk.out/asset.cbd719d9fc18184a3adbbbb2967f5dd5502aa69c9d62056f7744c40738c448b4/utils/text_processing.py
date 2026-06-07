"""
Text processing utilities for Vietnamese language support
Handles text normalization, cleaning, and preprocessing for RAG
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..config.settings import get_config
from .logging_utils import get_logger

logger = get_logger(__name__)

class VietnameseTextProcessor:
    """Vietnamese text processing utilities"""
    
    def __init__(self):
        self.config = get_config()
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load Vietnamese stopwords from file"""
        try:
            with open(self.config.vietnamese_stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip().lower() for line in f if line.strip())
            logger.info(f"Loaded {len(stopwords)} Vietnamese stopwords")
            return stopwords
        except FileNotFoundError:
            logger.warning(f"Stopwords file not found: {self.config.vietnamese_stopwords_file}")
            return self._get_default_stopwords()
        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            return self._get_default_stopwords()
    
    def _get_default_stopwords(self) -> set:
        """Get default Vietnamese stopwords if file is not available"""
        return {
            'và', 'của', 'có', 'là', 'được', 'trong', 'với', 'để', 'cho', 'từ',
            'về', 'theo', 'như', 'khi', 'nếu', 'mà', 'hay', 'hoặc', 'nhưng',
            'vì', 'do', 'bởi', 'tại', 'trên', 'dưới', 'giữa', 'sau', 'trước',
            'này', 'đó', 'các', 'những', 'một', 'hai', 'ba', 'bốn', 'năm'
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization (NFC form for Vietnamese)
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str, remove_punctuation: bool = False) -> str:
        """
        Clean text for processing
        
        Args:
            text: Input text to clean
            remove_punctuation: Whether to remove punctuation
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize first
        text = self.normalize_text(text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove phone numbers (Vietnamese format)
        text = re.sub(r'\b0\d{9,10}\b', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        if remove_punctuation:
            # Keep Vietnamese characters, numbers, and basic punctuation
            text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,!?;:]', ' ', text)
        
        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """
        Segment Vietnamese text into words/tokens
        Note: This is a basic implementation. For production, consider using PyVi or underthesea
        
        Args:
            text: Input text to segment
            
        Returns:
            List of segmented tokens
        """
        if not text:
            return []
        
        # Basic word segmentation (split on whitespace and punctuation)
        # In a real implementation, you would use PyVi or underthesea here
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords if enabled
        if self.config.enable_vietnamese_segmentation:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from Vietnamese text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Clean and segment text
        cleaned_text = self.clean_text(text)
        tokens = self.segment_text(cleaned_text)
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for token in tokens:
            if len(token) > 2:  # Filter out very short words
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk (defaults to config value)
            overlap: Overlap between chunks (defaults to config value)
            
        Returns:
            List of text chunks with metadata
        """
        if not text:
            return []
        
        chunk_size = chunk_size or self.config.text.chunk_size
        overlap = overlap or self.config.text.chunk_overlap
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = self._split_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_size + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'chunk_index': chunk_index,
                    'content': current_chunk.strip(),
                    'size': current_size,
                    'start_pos': len(''.join(chunk['content'] for chunk in chunks[:chunk_index])),
                    'end_pos': len(''.join(chunk['content'] for chunk in chunks[:chunk_index])) + current_size
                })
                
                # Start new chunk with overlap
                if overlap > 0:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_length
                
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_length + (1 if current_chunk != sentence else 0)
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'chunk_index': chunk_index,
                'content': current_chunk.strip(),
                'size': current_size,
                'start_pos': len(''.join(chunk['content'] for chunk in chunks)),
                'end_pos': len(''.join(chunk['content'] for chunk in chunks)) + current_size
            })
        
        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (Vietnamese-aware)
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Vietnamese sentence endings
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query for better search results
        
        Args:
            query: User query
            
        Returns:
            Preprocessed query
        """
        if not query:
            return ""
        
        # Clean and normalize
        processed_query = self.clean_text(query)
        processed_query = self.normalize_text(processed_query)
        
        # Remove question words that might not help with search
        question_words = ['gì', 'sao', 'thế nào', 'như thế nào', 'tại sao', 'vì sao']
        for qword in question_words:
            processed_query = re.sub(rf'\b{qword}\b', '', processed_query, flags=re.IGNORECASE)
        
        # Clean up whitespace
        processed_query = re.sub(r'\s+', ' ', processed_query).strip()
        
        return processed_query
    
    def enhance_query(self, query: str) -> List[str]:
        """
        Generate enhanced versions of the query for better retrieval
        
        Args:
            query: Original query
            
        Returns:
            List of enhanced query variations
        """
        enhanced_queries = [query]
        
        # Add preprocessed version
        preprocessed = self.preprocess_query(query)
        if preprocessed and preprocessed != query:
            enhanced_queries.append(preprocessed)
        
        # Add keyword-based version
        keywords = self.extract_keywords(query, max_keywords=5)
        if keywords:
            keyword_query = ' '.join(keywords)
            if keyword_query not in enhanced_queries:
                enhanced_queries.append(keyword_query)
        
        # Add variations with common legal terms
        legal_terms = {
            'phạt': ['mức phạt', 'tiền phạt', 'xử phạt'],
            'lái xe': ['điều khiển phương tiện', 'người lái'],
            'giao thông': ['an toàn giao thông', 'trật tự giao thông'],
            'đường': ['đường bộ', 'tuyến đường'],
            'xe': ['phương tiện', 'xe cơ giới']
        }
        
        for term, variations in legal_terms.items():
            if term in query.lower():
                for variation in variations:
                    enhanced_query = query.lower().replace(term, variation)
                    if enhanced_query not in [q.lower() for q in enhanced_queries]:
                        enhanced_queries.append(enhanced_query)
        
        return enhanced_queries[:5]  # Limit to 5 variations

# Global text processor instance
_text_processor = None

def get_text_processor() -> VietnameseTextProcessor:
    """Get the global Vietnamese text processor instance"""
    global _text_processor
    if _text_processor is None:
        _text_processor = VietnameseTextProcessor()
    return _text_processor