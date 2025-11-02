"""Naive vector index implementation using linear search with cosine similarity"""
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

from .base import BaseVectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk


class NaiveIndex(BaseVectorIndex):
    """Naive implementation using linear search with cosine similarity"""
    
    def _add_chunks_impl(self, chunks: List['Chunk']) -> None:
        """
        Naive index doesn't need special data structures for indexing.
        Chunks are stored in base class and accessed directly during search.
        """
        pass
    
    def _remove_chunks_impl(self, chunk_ids: List[str]) -> None:
        """
        Naive index doesn't need special cleanup for chunk removal.
        Chunks are removed from base class storage and no longer accessible.
        """
        pass
    
    def _search_impl(self, query_embedding: List[float], k: int, min_similarity: float) -> List[Tuple['Chunk', float]]:
        """Search using linear search with cosine similarity"""
        similarities = []
        
        for chunk in self._chunks.values():
            if not chunk.embedding:
                continue
                
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            if similarity >= min_similarity:
                similarities.append((chunk, similarity))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        return max(0.0, min(1.0, similarity))