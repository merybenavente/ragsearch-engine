"""
LSH (Locality Sensitive Hashing) vector index implementation

LSH is an approximation algorithm for solving the nearest neighbor search problem
in high-dimensional spaces. It works by hashing similar input vectors to the same
hash codes with high probability.

Algorithm Overview:
1. Generate random hyperplanes that divide the vector space
2. For each vector, compute a hash code based on which side of each hyperplane it falls
3. Store vectors in hash tables based on their hash codes
4. For search, compute the query's hash code and look up candidates in matching buckets

Time Complexity:
- Indexing: O(d * L * k) where d=dimensions, L=tables, k=hyperplanes
- Search: O(d * L * k + c) where c=candidates found
- Space: O(n * L) where n=number of vectors

Parameters:
- num_tables: More tables = better recall but slower search and more memory
- num_hyperplanes: More hyperplanes = more precise bucketing but potentially fewer matches
"""
import numpy as np
from typing import List, Tuple, Set, DefaultDict
from collections import defaultdict

from .base import BaseVectorIndex
from ...domain.models import Chunk, ChunkID


class LSHIndex(BaseVectorIndex):
    """
    LSH implementation for approximate nearest neighbor search

    Uses multiple hash tables with random hyperplanes to achieve sublinear search time.
    Trade-offs accuracy for significant performance improvements on large datasets.
    """

    # Use parameters optimized for semantic search with high-dimensional embeddings
    # For 1536-dim embeddings, use fewer hyperplanes for larger buckets = higher recall
    def __init__(self, num_tables: int = 6, num_hyperplanes: int = 4):
        super().__init__()
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes

        # Hash tables: each table contains buckets (hash_code -> set of chunk_ids)
        self.hash_tables: List[DefaultDict[str, Set[ChunkID]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        # Store hyperplanes for each table
        self.hyperplanes: List[np.ndarray] = []

        # Track vector dimensionality
        self.vector_dim: int = 0

    def _generate_hyperplanes(self, vector_dim: int) -> None:
        """Generate random hyperplanes for each hash table"""
        # Use thread-safe random number generator for reproducible results
        rng = np.random.RandomState(42)
        self.hyperplanes = []

        for _ in range(self.num_tables):
            # Generate random hyperplanes for this table
            table_hyperplanes = rng.randn(self.num_hyperplanes, vector_dim)
            # Normalize hyperplanes
            table_hyperplanes = table_hyperplanes / np.linalg.norm(
                table_hyperplanes, axis=1, keepdims=True
            )
            self.hyperplanes.append(table_hyperplanes)

    def _compute_hash_code(self, vector: List[float], table_idx: int) -> str:
        """Compute hash code for a vector using hyperplanes of a specific table"""
        if not self.hyperplanes:
            return ""

        vector_np = np.array(vector)
        hyperplanes = self.hyperplanes[table_idx]

        # Compute dot product with each hyperplane
        dot_products = np.dot(hyperplanes, vector_np)

        # Convert to binary hash code (1 if positive, 0 if negative)
        hash_bits = (dot_products > 0).astype(int)

        # Convert binary array to string for use as dictionary key
        return "".join(hash_bits.astype(str))

    def _add_chunks_impl(self, chunks: List[Chunk]) -> None:
        """Index chunks using LSH algorithm"""
        if not chunks:
            return

        # Initialize hyperplanes if this is the first time indexing
        if not self.hyperplanes:
            # Get vector dimension from first chunk with embedding
            for chunk in chunks:
                if chunk.embedding:
                    self.vector_dim = len(chunk.embedding)
                    break

            if self.vector_dim == 0:
                return  # No embeddings to index

            self._generate_hyperplanes(self.vector_dim)

        # Index each chunk
        for chunk in chunks:
            if not chunk.embedding:
                continue

            # Add chunk to each hash table
            for table_idx in range(self.num_tables):
                hash_code = self._compute_hash_code(chunk.embedding, table_idx)
                self.hash_tables[table_idx][hash_code].add(chunk.id)

    def _remove_chunks_impl(self, chunk_ids: List[ChunkID]) -> None:
        """Remove chunks from LSH index"""
        if not chunk_ids:
            return

        # Remove from all hash tables
        for table_idx in range(self.num_tables):
            hash_table = self.hash_tables[table_idx]

            # Remove chunk IDs from all buckets in this table
            for hash_code in list(hash_table.keys()):
                bucket = hash_table[hash_code]
                for chunk_id in chunk_ids:
                    bucket.discard(chunk_id)

                # Remove empty buckets to save memory
                if not bucket:
                    del hash_table[hash_code]

    def _search_impl(
        self, query_embedding: List[float], k: int, min_similarity: float
    ) -> List[Tuple[Chunk, float]]:
        """Search using LSH algorithm"""
        if not self.hyperplanes or not query_embedding:
            return []

        # Collect candidate chunk IDs from all tables
        candidate_chunk_ids = set()

        for table_idx in range(self.num_tables):
            hash_code = self._compute_hash_code(query_embedding, table_idx)
            if hash_code in self.hash_tables[table_idx]:
                candidate_chunk_ids.update(self.hash_tables[table_idx][hash_code])

        # Compute actual similarities for candidates
        similarities = []
        for chunk_id in candidate_chunk_ids:
            if chunk_id in self._chunks:
                chunk = self._chunks[chunk_id]
                if chunk.embedding:
                    similarity = self._cosine_similarity(
                        query_embedding, chunk.embedding
                    )
                    if similarity >= min_similarity:
                        similarities.append((chunk, similarity))

        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
