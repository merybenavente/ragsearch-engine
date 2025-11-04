"""
Naive vector index implementation using linear search with cosine similarity

The naive approach performs exhaustive linear search through all vectors,
computing cosine similarity for each one. This serves as a baseline
implementation and is guaranteed to find the exact nearest neighbors.

Algorithm Overview:
1. Store all vectors in memory without any indexing structure
2. For each query, compute cosine similarity with every stored vector
3. Sort results by similarity and return top-k matches

Time Complexity:
- Indexing: O(1) - no preprocessing required
- Search: O(n * d) where n=number of vectors, d=dimensions
- Space: O(n * d)

Parameters:
- None - this is a parameter-free baseline implementation

Use this implementation for:
- Small datasets (< 10,000 vectors)
- Baseline comparisons and correctness verification
- When exact results are required and performance is not critical
"""
from typing import List, Tuple, TYPE_CHECKING

from .base import BaseVectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk


class NaiveIndex(BaseVectorIndex):
    """Naive implementation using linear search with cosine similarity"""

    def _add_chunks_impl(self, chunks: List["Chunk"]) -> None:
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

    def _search_impl(
        self, query_embedding: List[float], k: int, min_similarity: float
    ) -> List[Tuple["Chunk", float]]:
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
