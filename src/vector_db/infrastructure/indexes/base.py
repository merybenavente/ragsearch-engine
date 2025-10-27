"""Base vector index implementation with concurrency control"""
from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...domain.interfaces import VectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk


class BaseVectorIndex(VectorIndex, ABC):
    """Base implementation for vector indexes with concurrency control"""
    
    def __init__(self):
        self._lock = RLock()
        self._chunks: Dict[str, 'Chunk'] = {}
    
    def index_chunks(self, chunks: List['Chunk']) -> None:
        """Index chunks with thread safety"""
        with self._lock:
            self._index_chunks_impl(chunks)
    
    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """Remove chunks with thread safety"""
        with self._lock:
            self._remove_chunks_impl(chunk_ids)
    
    def search(self, chunks: List['Chunk'], query_embedding: List[float], k: int = 10, min_similarity: float = 0.0) -> List[Tuple['Chunk', float]]:
        """Search with thread safety"""
        with self._lock:
            return self._search_impl(chunks, query_embedding, k, min_similarity)
    
    @abstractmethod
    def _index_chunks_impl(self, chunks: List['Chunk']) -> None:
        """Implementation-specific indexing logic"""
        pass
    
    @abstractmethod
    def _remove_chunks_impl(self, chunk_ids: List[str]) -> None:
        """Implementation-specific removal logic"""
        pass
    
    @abstractmethod
    def _search_impl(self, chunks: List['Chunk'], query_embedding: List[float], k: int, min_similarity: float) -> List[Tuple['Chunk', float]]:
        """Implementation-specific search logic"""
        pass