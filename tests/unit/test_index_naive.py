"""Unit tests for NaiveIndex-specific functionality"""
from uuid import uuid4

from src.vector_db.infrastructure.indexes.naive import NaiveIndex
from src.vector_db.domain.models import Chunk
from tests.utils import create_deterministic_embedding


class TestNaiveIndexSpecific:
    """Test NaiveIndex-specific behavior that differs from other indexes"""

    def setup_method(self):
        """Set up test fixtures"""
        self.index = NaiveIndex()
        self.doc_id = str(uuid4())

    def test_add_chunks_impl_does_nothing(self):
        """Test that _add_chunks_impl does nothing (no indexing overhead)"""
        chunks = [
            Chunk(
                id="chunk1",
                document_id=self.doc_id,
                text="text1",
                embedding=[1.0, 0.0, 0.0],
            )
        ]

        # Should complete without errors and without creating special structures
        self.index._add_chunks_impl(chunks)

        # Verify no special indexing attributes are created
        assert not hasattr(self.index, "hash_tables")  # No LSH structures
        assert not hasattr(self.index, "root")  # No tree structures
        assert not hasattr(self.index, "hyperplanes")  # No LSH hyperplanes

    def test_remove_chunks_impl_does_nothing(self):
        """Test that _remove_chunks_impl does nothing (no cleanup needed)"""
        # Should complete without errors
        self.index._remove_chunks_impl(["chunk1", "chunk2"])

        # No special cleanup should be required
        assert True  # Just verify no exceptions are raised

    def test_linear_search_examines_all_chunks(self):
        """Test that naive search examines all chunks (linear behavior)"""
        # Create chunks with predictable embeddings
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=self.doc_id,
                text=f"text{i}",
                embedding=create_deterministic_embedding(f"text{i}", 10),
            )
            for i in range(5)
        ]

        # Add chunks through base class interface
        self.index.add_chunks(self.doc_id, chunks)

        # Search with min_similarity=0 should return all chunks
        query = create_deterministic_embedding("query", 10)
        results = self.index._search_impl(query, k=10, min_similarity=0.0)

        # Should return all chunks since we examine each one
        assert len(results) == 5
        result_ids = {chunk.id for chunk, _ in results}
        expected_ids = {f"chunk{i}" for i in range(5)}
        assert result_ids == expected_ids

    def test_search_ignores_chunks_without_embeddings(self):
        """Test that search correctly ignores chunks without embeddings"""
        chunks = [
            Chunk(
                id="with_embedding",
                document_id=self.doc_id,
                text="has embedding",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                id="without_embedding",
                document_id=self.doc_id,
                text="no embedding",
                embedding=[],
            ),
        ]

        # Manually add to storage to test search behavior
        for chunk in chunks:
            self.index._chunks[chunk.id] = chunk

        query = [1.0, 0.0, 0.0]
        results = self.index._search_impl(query, k=10, min_similarity=0.0)

        # Should only return chunk with embedding
        assert len(results) == 1
        assert results[0][0].id == "with_embedding"

    def test_deterministic_linear_search_results(self):
        """Test that naive search produces deterministic results"""
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=self.doc_id,
                text=f"deterministic text {i}",
                embedding=create_deterministic_embedding(f"deterministic text {i}", 10),
            )
            for i in range(3)
        ]

        self.index.add_chunks(self.doc_id, chunks)

        query = create_deterministic_embedding("deterministic query", 10)

        # Run search multiple times
        results1 = self.index._search_impl(query, k=3, min_similarity=0.0)
        results2 = self.index._search_impl(query, k=3, min_similarity=0.0)

        # Results should be identical (same order, same scores)
        assert len(results1) == len(results2)
        for (chunk1, score1), (chunk2, score2) in zip(results1, results2):
            assert chunk1.id == chunk2.id
            assert abs(score1 - score2) < 1e-10

    def test_exact_cosine_similarity_baseline(self):
        """Test that naive index provides exact cosine similarity (baseline)"""
        # Create chunk with known embedding
        target_embedding = [1.0, 0.0, 0.0]
        chunk = Chunk(
            id="target",
            document_id=self.doc_id,
            text="target",
            embedding=target_embedding,
        )

        self.index.add_chunks(self.doc_id, [chunk])

        # Query with identical embedding should give perfect similarity
        results = self.index._search_impl(target_embedding, k=1, min_similarity=0.0)

        assert len(results) == 1
        _, similarity = results[0]
        assert abs(similarity - 1.0) < 1e-10  # Should be exactly 1.0

        # Query with orthogonal embedding should give zero similarity
        orthogonal_embedding = [0.0, 1.0, 0.0]
        results = self.index._search_impl(orthogonal_embedding, k=1, min_similarity=0.0)

        assert len(results) == 1
        _, similarity = results[0]
        assert abs(similarity - 0.0) < 1e-10  # Should be exactly 0.0
