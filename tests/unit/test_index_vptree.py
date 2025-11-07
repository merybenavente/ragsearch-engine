"""Unit tests for VPTree-specific functionality"""
import numpy as np

from src.vector_db.infrastructure.indexes.vptree import VPTreeIndex, VPTreeNode
from src.vector_db.domain.models import Chunk


class TestVPTreeSpecific:
    """Test VPTree-specific functionality that differs from other indexes"""

    def test_vptree_node_structure(self):
        """Test VPTreeNode has correct structure for internal nodes"""
        chunk = Chunk(id="test", document_id="doc1", text="test", embedding=[1.0, 2.0])
        node = VPTreeNode([chunk], threshold=0.5, is_leaf=False)

        assert node.chunk == chunk
        assert node.chunks is None
        assert node.threshold == 0.5
        assert node.is_leaf is False
        assert node.left is None
        assert node.right is None

    def test_vptree_leaf_node_structure(self):
        """Test VPTreeNode has correct structure for leaf nodes"""
        chunks = [
            Chunk(id="test1", document_id="doc1", text="test1", embedding=[1.0, 2.0]),
            Chunk(id="test2", document_id="doc1", text="test2", embedding=[2.0, 1.0]),
        ]
        node = VPTreeNode(chunks, threshold=0.0, is_leaf=True)

        assert node.chunks == chunks
        assert node.threshold == 0.0
        assert node.is_leaf is True
        assert node.left is None
        assert node.right is None

    def test_leaf_size_parameter(self):
        """Test that leaf_size parameter is used correctly"""
        index = VPTreeIndex(leaf_size=3)
        assert index.leaf_size == 3

        # Create chunks that should form a leaf when <= leaf_size
        chunks = [
            Chunk(id="1", document_id="doc1", text="text1", embedding=[1.0, 0.0]),
            Chunk(id="2", document_id="doc1", text="text2", embedding=[0.0, 1.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)

        # With only 2 chunks and leaf_size=3, root should be a leaf
        assert index.root is not None
        assert index.root.is_leaf is True
        assert len(index.root.chunks) == 2

    def test_distance_is_one_minus_cosine(self):
        """Test that VPTree uses 1 - cosine_similarity as distance metric"""
        index = VPTreeIndex()

        # Identical vectors: cosine=1.0, distance=0.0
        vec = [1.0, 0.0, 0.0]
        assert abs(index._distance(vec, vec) - 0.0) < 1e-10

        # Orthogonal vectors: cosine=0.0, distance=1.0
        vec1, vec2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        assert abs(index._distance(vec1, vec2) - 1.0) < 1e-10

    def test_tree_rebuilds_on_add_remove(self):
        """Test that VPTree rebuilds entire tree on add/remove operations"""
        index = VPTreeIndex()

        # Add initial chunks
        chunks = [
            Chunk(id="1", document_id="doc1", text="text1", embedding=[1.0, 0.0]),
            Chunk(id="2", document_id="doc1", text="text2", embedding=[0.0, 1.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)
        first_root = index.root

        # Add more chunks - should rebuild
        new_chunk = Chunk(
            id="3", document_id="doc1", text="text3", embedding=[0.5, 0.5]
        )
        index._chunks[new_chunk.id] = new_chunk
        index._add_chunks_impl([new_chunk])

        assert index.root != first_root  # Tree was rebuilt

        # Remove chunks - should rebuild
        del index._chunks["1"]
        index._remove_chunks_impl(["1"])

        assert index.root != first_root  # Tree was rebuilt again

    def test_remove_all_chunks_clears_tree(self):
        """Test that removing all chunks sets root to None"""
        index = VPTreeIndex()

        chunks = [
            Chunk(id="1", document_id="doc1", text="text1", embedding=[1.0, 0.0]),
            Chunk(id="2", document_id="doc1", text="text2", embedding=[0.0, 1.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)
        assert index.root is not None

        # Remove all chunks
        for chunk in chunks:
            del index._chunks[chunk.id]

        index._remove_chunks_impl(["1", "2"])
        assert index.root is None

    def test_search_with_tree_structure(self):
        """Test that search traverses tree structure correctly"""
        index = VPTreeIndex()

        # Create chunks with known structure
        chunks = [
            Chunk(id="1", document_id="doc1", text="text1", embedding=[1.0, 0.0, 0.0]),
            Chunk(id="2", document_id="doc1", text="text2", embedding=[0.9, 0.1, 0.0]),
            Chunk(id="3", document_id="doc1", text="text3", embedding=[0.0, 1.0, 0.0]),
            Chunk(id="4", document_id="doc1", text="text4", embedding=[0.0, 0.0, 1.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        np.random.seed(42)  # For reproducible tree building
        index._add_chunks_impl(chunks)

        # Query should use tree structure for efficiency
        query = [0.95, 0.05, 0.0]
        results = index._search_impl(query, k=4, min_similarity=0.0)

        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(0.0 <= score <= 1.0 for _, score in results)
