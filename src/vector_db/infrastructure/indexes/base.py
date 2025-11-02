"""Base vector index implementation with concurrency control"""
from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, List, Tuple, Set, TYPE_CHECKING

from ...domain.interfaces import VectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk, DocumentID


class BaseVectorIndex(VectorIndex, ABC):
    """Base implementation for vector indexes with concurrency control"""
    
    def __init__(self):
        self._lock = RLock()
        self._chunks: Dict[str, 'Chunk'] = {}  # chunk_id -> chunk
        self._document_chunks: Dict['DocumentID', Set[str]] = {}  # document_id -> chunk_ids
    
    def add_chunks(self, document_id: 'DocumentID', chunks: List['Chunk']) -> None:
        """Add chunks from a document with thread safety"""
        with self._lock:
            # Remove existing chunks for this document first
            self._remove_document_impl(document_id)
            
            # Add new chunks
            chunk_ids = set()
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
                chunk_ids.add(chunk.id)
            
            self._document_chunks[document_id] = chunk_ids
            self._add_chunks_impl(chunks)
    
    def remove_document(self, document_id: 'DocumentID') -> None:
        """Remove all chunks belonging to a document with thread safety"""
        with self._lock:
            self._remove_document_impl(document_id)
    
    def search(self, query_embedding: List[float], k: int = 10, min_similarity: float = 0.0) -> List[Tuple['Chunk', float]]:
        """Search with thread safety"""
        with self._lock:
            return self._search_impl(query_embedding, k, min_similarity)
    
    def get_document_chunks(self, document_id: 'DocumentID') -> List['Chunk']:
        """Get all chunks for a document with thread safety"""
        with self._lock:
            if document_id not in self._document_chunks:
                return []
            
            chunk_ids = self._document_chunks[document_id]
            return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]
    
    def _get_all_chunks(self) -> List['Chunk']:
        """Get all indexed chunks - for len() implementation"""
        with self._lock:
            return list(self._chunks.values())
    
    def _remove_document_impl(self, document_id: 'DocumentID') -> None:
        """Internal implementation for removing a document"""
        if document_id not in self._document_chunks:
            return
        
        chunk_ids = self._document_chunks[document_id]
        chunk_ids_list = list(chunk_ids)
        
        # Remove from chunks storage
        for chunk_id in chunk_ids_list:
            self._chunks.pop(chunk_id, None)
        
        # Remove from document mapping
        del self._document_chunks[document_id]
        
        # Call implementation-specific removal
        self._remove_chunks_impl(chunk_ids_list)
    
    @abstractmethod
    def _add_chunks_impl(self, chunks: List['Chunk']) -> None:
        """Implementation-specific indexing logic"""
        pass
    
    @abstractmethod
    def _remove_chunks_impl(self, chunk_ids: List[str]) -> None:
        """Implementation-specific removal logic"""
        pass
    
    @abstractmethod
    def _search_impl(self, query_embedding: List[float], k: int, min_similarity: float) -> List[Tuple['Chunk', float]]:
        """Implementation-specific search logic"""
        pass