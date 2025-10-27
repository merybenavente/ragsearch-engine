"""Naive vector index implementation using linear search with cosine similarity"""
import numpy as np
from typing import List, Tuple, TYPE_CHECKING

from .base import BaseVectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk


class NaiveIndex(BaseVectorIndex):
    """Naive implementation using linear search with cosine similarity"""
    
    def _index_chunks_impl(self, chunks: List['Chunk']) -> None:
        """Store chunks in memory for linear search"""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
    
    def _remove_chunks_impl(self, chunk_ids: List[str]) -> None:
        """Remove chunks from index"""
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
    
    def _search_impl(self, chunks: List['Chunk'], query_embedding: List[float], k: int, min_similarity: float) -> List[Tuple['Chunk', float]]:
        """Search using linear search with cosine similarity"""
        similarities = []
        
        for chunk in chunks:
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