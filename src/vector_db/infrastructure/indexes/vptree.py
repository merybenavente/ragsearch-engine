"""
VP-Tree (Vantage Point Tree) vector index implementation

VP-Tree is an exact nearest neighbor search data structure for metric spaces.
It recursively partitions the space using vantage points and distance thresholds,
creating a binary tree that enables efficient pruning during search.

Algorithm Overview:
1. Select a random vantage point from the dataset
2. Calculate distances from vantage point to all other points
3. Choose median distance as threshold to split points into two groups
4. Recursively build left subtree (closer points) and right subtree (farther points)
5. For search, use triangle inequality to prune entire subtrees

Time Complexity:
- Building: O(n log n × d) where n=number of vectors, d=embedding dimension
- Search: O(log n × d) expected, O(n × d) worst case
- Space: O(n)

Parameters:
- leaf_size: Minimum number of points in leaf nodes (default: 20)
- random_seed: Seed for deterministic vantage point selection
"""
import heapq
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

from .base import BaseVectorIndex

if TYPE_CHECKING:
    from ...domain.models import Chunk


class VPTreeNode:
    """Node in a VP-Tree"""

    def __init__(
        self, chunks: List["Chunk"], threshold: float = 0.0, is_leaf: bool = False
    ):
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.left: Optional["VPTreeNode"] = None
        self.right: Optional["VPTreeNode"] = None
        self.chunks: Optional[List["Chunk"]]
        self.chunk: Optional["Chunk"]

        if is_leaf:
            self.chunks = chunks
            self.chunk = None
        else:
            self.chunk = chunks[0] if chunks else None
            self.chunks = None


class VPTreeIndex(BaseVectorIndex):
    """VP-Tree implementation for metric space nearest neighbor search"""

    def __init__(self, leaf_size: int = 20):
        super().__init__()
        self.leaf_size = leaf_size
        self.root: Optional[VPTreeNode] = None
        self._rng = np.random.RandomState(42)

    def _distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate distance between two embeddings (1 - cosine similarity)"""
        similarity = self._cosine_similarity(embedding1, embedding2)
        return 1.0 - similarity

    def _build_tree(self, chunks: List["Chunk"]) -> Optional[VPTreeNode]:
        """Recursively build VP-Tree from chunks"""
        if not chunks:
            return None

        if len(chunks) <= self.leaf_size:
            return VPTreeNode(chunks, is_leaf=True)

        # Choose vantage point (deterministic random selection)
        # Make a copy to avoid modifying the input list
        chunks_copy = chunks.copy()
        vantage_point_idx = self._rng.choice(len(chunks_copy))
        vantage_point = chunks_copy[vantage_point_idx]
        chunks_copy.remove(vantage_point)

        if not chunks_copy:
            return VPTreeNode([vantage_point], is_leaf=True)

        # Calculate distances from vantage point to all other chunks
        distances = []
        for chunk in chunks_copy:
            if chunk.embedding and vantage_point.embedding:
                dist = self._distance(vantage_point.embedding, chunk.embedding)
                distances.append((dist, chunk))

        if not distances:
            return VPTreeNode([vantage_point], is_leaf=True)

        # Sort by distance and find median point's distance as threshold
        distances.sort(key=lambda x: x[0])
        median_idx = len(distances) // 2

        # Use the median point's distance as threshold (not median of distances)
        if median_idx < len(distances):
            threshold = distances[median_idx][0]
        else:
            threshold = distances[-1][0]

        # Split chunks based on threshold
        left_chunks = [chunk for dist, chunk in distances if dist < threshold]
        right_chunks = [chunk for dist, chunk in distances if dist >= threshold]

        # Ensure balanced split - if all points have same distance, split evenly
        if not left_chunks or not right_chunks:
            mid = len(distances) // 2
            left_chunks = [chunk for _, chunk in distances[:mid]]
            right_chunks = [chunk for _, chunk in distances[mid:]]
            threshold = distances[mid - 1][0] if mid > 0 else 0.0

        # Create node and recursively build subtrees
        node = VPTreeNode([vantage_point], threshold, is_leaf=False)
        node.left = self._build_tree(left_chunks)
        node.right = self._build_tree(right_chunks)

        return node

    def _add_chunks_impl(self, chunks: List["Chunk"]) -> None:
        """Build VP-Tree from chunks"""
        if not chunks:
            return

        # Collect all chunks including existing ones
        all_chunks = list(self._chunks.values())

        # Filter chunks with embeddings
        chunks_with_embeddings = [chunk for chunk in all_chunks if chunk.embedding]

        if not chunks_with_embeddings:
            return

        # Rebuild tree with all chunks
        self.root = self._build_tree(chunks_with_embeddings.copy())

    def _remove_chunks_impl(self, chunk_ids: List[str]) -> None:
        """Remove chunks from VP-Tree by rebuilding"""
        if not chunk_ids:
            return

        # Collect remaining chunks
        remaining_chunks = [
            chunk
            for chunk in self._chunks.values()
            if chunk.id not in chunk_ids and chunk.embedding
        ]

        # Rebuild tree with remaining chunks
        self.root = self._build_tree(remaining_chunks) if remaining_chunks else None

    def _search_tree(
        self,
        node: Optional[VPTreeNode],
        query_embedding: List[float],
        k: int,
        min_similarity: float,
        results: List[Tuple[float, "Chunk"]],
    ) -> None:
        """Recursively search VP-Tree with proper pruning"""
        if not node:
            return

        # Handle leaf nodes
        if node.is_leaf and node.chunks:
            for chunk in node.chunks:
                if chunk.embedding:
                    similarity = self._cosine_similarity(
                        query_embedding, chunk.embedding
                    )
                    if similarity >= min_similarity:
                        # Use max-heap for efficient k-NN (negate similarity for max-heap)
                        if len(results) < k:
                            heapq.heappush(results, (-similarity, chunk))
                        elif -similarity > results[0][0]:  # Better than worst in heap
                            heapq.heapreplace(results, (-similarity, chunk))
            return

        # Handle internal nodes
        if not node.chunk or not node.chunk.embedding:
            return

        # Calculate similarity to vantage point
        similarity = self._cosine_similarity(query_embedding, node.chunk.embedding)
        if similarity >= min_similarity:
            # Use max-heap for efficient k-NN (negate similarity for max-heap)
            if len(results) < k:
                heapq.heappush(results, (-similarity, node.chunk))
            elif -similarity > results[0][0]:  # Better than worst in heap
                heapq.heapreplace(results, (-similarity, node.chunk))

        # Calculate distance to vantage point
        query_distance = self._distance(query_embedding, node.chunk.embedding)

        # Determine search radius for pruning
        search_radius = 1.0 - min_similarity  # Convert similarity to distance
        if len(results) == k:
            # Use current k-th best distance as radius
            worst_similarity = -results[0][0]
            search_radius = min(search_radius, 1.0 - worst_similarity)

        # Search left subtree (closer to vantage point)
        if query_distance <= node.threshold + search_radius:
            self._search_tree(node.left, query_embedding, k, min_similarity, results)

        # Search right subtree (farther from vantage point)
        if query_distance >= node.threshold - search_radius:
            self._search_tree(node.right, query_embedding, k, min_similarity, results)

    def _search_impl(
        self, query_embedding: List[float], k: int, min_similarity: float
    ) -> List[Tuple["Chunk", float]]:
        """Search using VP-Tree algorithm with proper k-NN"""
        if not self.root or not query_embedding:
            return []

        # Use heap for efficient k-NN search
        results_heap: List[Tuple[float, "Chunk"]] = []
        self._search_tree(self.root, query_embedding, k, min_similarity, results_heap)

        # Convert heap back to list and sort by similarity (descending)
        results = [(chunk, -neg_similarity) for neg_similarity, chunk in results_heap]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
