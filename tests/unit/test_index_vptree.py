"""Unit tests for VPTree-specific functionality"""
from uuid import uuid4

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

    def test_distance_metric_conversion(self):
        """Test VP-Tree specific behavior: converts cosine similarity to distance"""
        index = VPTreeIndex()

        # VP-Tree uses distance = 1 - cosine_similarity
        # Identical vectors: cosine=1.0, distance=0.0
        vec = [1.0, 0.0, 0.0]
        distance = index._distance(vec, vec)
        assert abs(distance - 0.0) < 1e-10

        # Orthogonal vectors: cosine=0.0, distance=1.0
        vec1, vec2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
        distance = index._distance(vec1, vec2)
        assert abs(distance - 1.0) < 1e-10

        # Verify distance is complement of cosine similarity
        cosine_sim = index._cosine_similarity(vec1, vec2)
        distance = index._distance(vec1, vec2)
        assert abs(distance - (1.0 - cosine_sim)) < 1e-10

    def test_tree_rebuilds_on_add_remove(self):
        """Test VP-Tree specific behavior: rebuilds entire tree on add/remove"""
        index = VPTreeIndex()
        doc_id = str(uuid4())

        # Add initial chunks
        chunks = [
            Chunk(id="1", document_id=doc_id, text="text1", embedding=[1.0, 0.0]),
            Chunk(id="2", document_id=doc_id, text="text2", embedding=[0.0, 1.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)
        first_root = index.root

        # VP-Tree should rebuild completely (unlike incremental structures)
        # Add more chunks - should rebuild entire tree
        new_chunk = Chunk(
            id="3", document_id=doc_id, text="text3", embedding=[0.5, 0.5]
        )
        index._chunks[new_chunk.id] = new_chunk
        index._add_chunks_impl([new_chunk])

        # Root should be different object (tree rebuilt)
        assert index.root is not first_root

        # Remove chunks - should rebuild again
        del index._chunks["1"]
        index._remove_chunks_impl(["1"])

        # Root should be different again
        assert index.root is not first_root

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

    def test_search_with_tree_traversal(self):
        """Test VP-Tree specific behavior: search uses tree traversal with pruning"""
        index = VPTreeIndex(leaf_size=2)  # Small leaf size to force tree structure
        doc_id = str(uuid4())

        # Create chunks with known embeddings for predictable tree structure
        chunks = [
            Chunk(id="1", document_id=doc_id, text="text1", embedding=[1.0, 0.0, 0.0]),
            Chunk(id="2", document_id=doc_id, text="text2", embedding=[0.9, 0.1, 0.0]),
            Chunk(id="3", document_id=doc_id, text="text3", embedding=[0.0, 1.0, 0.0]),
            Chunk(id="4", document_id=doc_id, text="text4", embedding=[0.0, 0.0, 1.0]),
            Chunk(id="5", document_id=doc_id, text="text5", embedding=[-1.0, 0.0, 0.0]),
        ]

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)

        # Should have built a tree structure
        assert index.root is not None
        assert (
            not index.root.is_leaf
        )  # Should have internal nodes with 5 chunks and leaf_size=2

        # Query should use tree traversal
        query = [0.95, 0.05, 0.0]  # Very close to first chunk
        results = index._search_impl(query, k=3, min_similarity=0.0)

        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)

        # First result should be most similar (closest to query)
        if len(results) > 1:
            similarities = [score for _, score in results]
            assert similarities == sorted(similarities, reverse=True)

    def test_tree_pruning_efficiency(self):
        """Test VP-Tree specific behavior: triangle inequality pruning"""
        index = VPTreeIndex(leaf_size=1)  # Force deep tree
        doc_id = str(uuid4())

        # Create clusters of similar points to test pruning
        cluster1 = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
        cluster2 = [[-1.0, 0.0, 0.0], [-0.9, -0.1, 0.0], [-0.8, -0.2, 0.0]]

        chunks = []
        for i, embedding in enumerate(cluster1 + cluster2):
            chunks.append(
                Chunk(
                    id=f"chunk{i}",
                    document_id=doc_id,
                    text=f"text{i}",
                    embedding=embedding,
                )
            )

        for chunk in chunks:
            index._chunks[chunk.id] = chunk

        index._add_chunks_impl(chunks)

        # Query close to cluster1 should primarily return cluster1 results
        query = [1.0, 0.0, 0.0]
        results = index._search_impl(query, k=3, min_similarity=0.5)

        # Should find results efficiently using tree pruning
        assert len(results) >= 1
        # Best results should be from cluster1 (positive x values)
        best_chunk, best_score = results[0]
        assert best_chunk.embedding[0] > 0  # Should be from cluster1
