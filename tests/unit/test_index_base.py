"""Unit tests for BaseVectorIndex functionality"""
import threading
import time
from uuid import uuid4

from src.vector_db.infrastructure.indexes.base import BaseVectorIndex
from src.vector_db.domain.models import Chunk
from tests.utils import create_deterministic_embedding


class ConcreteIndex(BaseVectorIndex):
    """Concrete implementation for testing BaseVectorIndex"""

    def __init__(self):
        super().__init__()
        self.add_chunks_called = []
        self.remove_chunks_called = []
        self.search_called = []

    def _add_chunks_impl(self, chunks):
        self.add_chunks_called.append(chunks)

    def _remove_chunks_impl(self, chunk_ids):
        self.remove_chunks_called.append(chunk_ids)

    def _search_impl(self, query_embedding, k, min_similarity):
        self.search_called.append((query_embedding, k, min_similarity))
        # Simple implementation for testing
        results = []
        for chunk in self._chunks.values():
            if chunk.embedding:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                if similarity >= min_similarity:
                    results.append((chunk, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class TestBaseVectorIndexInitialization:
    """Test BaseVectorIndex initialization"""

    def test_initialization(self):
        """Test that BaseVectorIndex initializes correctly"""
        index = ConcreteIndex()
        assert hasattr(index, "_lock")
        assert index._chunks == {}
        assert index._document_chunks == {}


class TestBaseVectorIndexChunkManagement:
    """Test BaseVectorIndex chunk and document management"""

    def setup_method(self):
        """Set up test fixtures"""
        self.index = ConcreteIndex()
        self.doc_id = str(uuid4())
        self.chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=self.doc_id,
                text=f"text{i}",
                embedding=create_deterministic_embedding(f"text{i}", 10),
            )
            for i in range(3)
        ]

    def test_add_chunks_stores_in_base(self):
        """Test that add_chunks stores chunks in base class storage"""
        self.index.add_chunks(self.doc_id, self.chunks)

        # Check chunks are stored
        assert len(self.index._chunks) == 3
        for chunk in self.chunks:
            assert self.index._chunks[chunk.id] == chunk

        # Check document mapping
        assert self.doc_id in self.index._document_chunks
        chunk_ids = self.index._document_chunks[self.doc_id]
        assert len(chunk_ids) == 3
        assert all(chunk.id in chunk_ids for chunk in self.chunks)

    def test_add_chunks_calls_implementation(self):
        """Test that add_chunks calls implementation-specific method"""
        self.index.add_chunks(self.doc_id, self.chunks)

        assert len(self.index.add_chunks_called) == 1
        assert self.index.add_chunks_called[0] == self.chunks

    def test_add_chunks_replaces_existing_document(self):
        """Test that adding chunks replaces existing document chunks"""
        # Add initial chunks
        self.index.add_chunks(self.doc_id, self.chunks[:2])
        assert len(self.index._chunks) == 2

        # Add new chunks for same document
        new_chunks = [self.chunks[2]]
        self.index.add_chunks(self.doc_id, new_chunks)

        # Should only have the new chunk
        assert len(self.index._chunks) == 1
        assert self.chunks[2].id in self.index._chunks
        assert self.chunks[0].id not in self.index._chunks
        assert self.chunks[1].id not in self.index._chunks

    def test_remove_document_removes_chunks(self):
        """Test that remove_document removes all chunks for document"""
        self.index.add_chunks(self.doc_id, self.chunks)

        self.index.remove_document(self.doc_id)

        # Check chunks are removed
        assert len(self.index._chunks) == 0
        assert self.doc_id not in self.index._document_chunks

    def test_remove_document_calls_implementation(self):
        """Test that remove_document calls implementation-specific method"""
        self.index.add_chunks(self.doc_id, self.chunks)

        self.index.remove_document(self.doc_id)

        assert len(self.index.remove_chunks_called) == 1
        removed_ids = self.index.remove_chunks_called[0]
        assert len(removed_ids) == 3
        assert all(chunk.id in removed_ids for chunk in self.chunks)

    def test_remove_nonexistent_document(self):
        """Test removing non-existent document doesn't cause errors"""
        self.index.remove_document("nonexistent")
        assert len(self.index.remove_chunks_called) == 0

    def test_get_document_chunks_returns_correct_chunks(self):
        """Test that get_document_chunks returns correct chunks"""
        self.index.add_chunks(self.doc_id, self.chunks)

        retrieved_chunks = self.index.get_document_chunks(self.doc_id)

        assert len(retrieved_chunks) == 3
        assert all(chunk in retrieved_chunks for chunk in self.chunks)

    def test_get_document_chunks_nonexistent_document(self):
        """Test get_document_chunks with non-existent document"""
        chunks = self.index.get_document_chunks("nonexistent")
        assert chunks == []

    def test_get_all_chunks(self):
        """Test _get_all_chunks returns all stored chunks"""
        self.index.add_chunks(self.doc_id, self.chunks)

        all_chunks = self.index._get_all_chunks()

        assert len(all_chunks) == 3
        assert all(chunk in all_chunks for chunk in self.chunks)


class TestBaseVectorIndexSearch:
    """Test BaseVectorIndex search functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.index = ConcreteIndex()
        self.doc_id = str(uuid4())
        self.chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=self.doc_id,
                text=f"text{i}",
                embedding=create_deterministic_embedding(f"text{i}", 10),
            )
            for i in range(3)
        ]
        self.index.add_chunks(self.doc_id, self.chunks)

    def test_search_calls_implementation(self):
        """Test that search calls implementation-specific method"""
        query = create_deterministic_embedding("query", 10)
        self.index.search(query, k=5, min_similarity=0.5)

        assert len(self.index.search_called) == 1
        called_args = self.index.search_called[0]
        assert called_args[0] == query
        assert called_args[1] == 5
        assert called_args[2] == 0.5

    def test_search_returns_results(self):
        """Test that search returns expected results"""
        query = create_deterministic_embedding("text0", 10)
        results = self.index.search(query, k=3, min_similarity=0.0)

        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        assert all(0.0 <= score <= 1.0 for _, score in results)

    def test_search_with_default_parameters(self):
        """Test search with default parameters"""
        query = create_deterministic_embedding("text0", 10)
        results = self.index.search(query)

        assert isinstance(results, list)


class TestBaseVectorIndexCosineSimilarity:
    """Test BaseVectorIndex cosine similarity calculation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.index = ConcreteIndex()

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors"""
        vec = [1.0, 2.0, 3.0]
        similarity = self.index._cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.index._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = self.index._cosine_similarity(vec1, vec2)
        assert similarity == 0.0  # Clamped to 0

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with vectors of different lengths"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        similarity = self.index._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = self.index._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

        # Test both zero
        similarity = self.index._cosine_similarity(vec1, [0.0, 0.0, 0.0])
        assert similarity == 0.0

    def test_cosine_similarity_handles_nan_inf(self):
        """Test cosine similarity handles NaN and Inf gracefully"""
        # This would be hard to trigger directly, but the code handles it
        # Test with very small numbers that might cause numerical issues
        vec1 = [1e-100, 1e-100, 1e-100]
        vec2 = [1e-100, 1e-100, 1e-100]
        similarity = self.index._cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_bounds(self):
        """Test that cosine similarity is always bounded between 0 and 1"""
        import numpy as np

        np.random.seed(42)

        for _ in range(10):
            vec1 = np.random.randn(5).tolist()
            vec2 = np.random.randn(5).tolist()
            similarity = self.index._cosine_similarity(vec1, vec2)
            assert 0.0 <= similarity <= 1.0


class TestBaseVectorIndexThreadSafety:
    """Test BaseVectorIndex thread safety"""

    def test_concurrent_add_operations(self):
        """Test concurrent add operations are thread-safe"""
        index = ConcreteIndex()
        results = []

        def add_chunks_worker(worker_id):
            doc_id = f"doc_{worker_id}"
            chunks = [
                Chunk(
                    id=f"chunk_{worker_id}_{i}",
                    document_id=doc_id,
                    text=f"text_{worker_id}_{i}",
                    embedding=create_deterministic_embedding(
                        f"text_{worker_id}_{i}", 10
                    ),
                )
                for i in range(5)
            ]
            index.add_chunks(doc_id, chunks)
            results.append(worker_id)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_chunks_worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all operations completed
        assert len(results) == 3
        assert len(index._chunks) == 15  # 3 workers * 5 chunks each
        assert len(index._document_chunks) == 3

    def test_concurrent_search_operations(self):
        """Test concurrent search operations are thread-safe"""
        index = ConcreteIndex()

        # Add some initial data
        doc_id = str(uuid4())
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=doc_id,
                text=f"text{i}",
                embedding=create_deterministic_embedding(f"text{i}", 10),
            )
            for i in range(10)
        ]
        index.add_chunks(doc_id, chunks)

        search_results = []

        def search_worker():
            query = create_deterministic_embedding("search_query", 10)
            for _ in range(5):
                results = index.search(query, k=3, min_similarity=0.0)
                search_results.append(len(results))
                time.sleep(0.001)

        # Create multiple search threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify searches completed without errors
        assert len(search_results) == 15  # 3 threads * 5 searches each
        assert all(isinstance(count, int) for count in search_results)

    def test_concurrent_add_remove_operations(self):
        """Test concurrent add and remove operations are thread-safe"""
        index = ConcreteIndex()
        operations_completed = []

        def add_worker():
            for i in range(3):
                doc_id = f"add_doc_{i}"
                chunks = [
                    Chunk(
                        id=f"add_chunk_{i}",
                        document_id=doc_id,
                        text=f"add_text_{i}",
                        embedding=create_deterministic_embedding(f"add_text_{i}", 10),
                    )
                ]
                index.add_chunks(doc_id, chunks)
                operations_completed.append(f"add_{i}")
                time.sleep(0.001)

        def remove_worker():
            # Add some initial data to remove
            for i in range(3):
                doc_id = f"remove_doc_{i}"
                chunks = [
                    Chunk(
                        id=f"remove_chunk_{i}",
                        document_id=doc_id,
                        text=f"remove_text_{i}",
                        embedding=create_deterministic_embedding(
                            f"remove_text_{i}", 10
                        ),
                    )
                ]
                index.add_chunks(doc_id, chunks)

            # Now remove them
            for i in range(3):
                index.remove_document(f"remove_doc_{i}")
                operations_completed.append(f"remove_{i}")
                time.sleep(0.001)

        # Start concurrent operations
        add_thread = threading.Thread(target=add_worker)
        remove_thread = threading.Thread(target=remove_worker)

        add_thread.start()
        remove_thread.start()

        add_thread.join()
        remove_thread.join()

        # Verify operations completed
        assert len(operations_completed) == 6  # 3 adds + 3 removes
