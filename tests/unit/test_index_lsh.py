"""Unit tests for LSH (Locality Sensitive Hashing) vector index"""
import numpy as np
from typing import List
import threading
import time
from uuid import uuid4

from src.vector_db.infrastructure.index_factory import IndexFactory
from src.vector_db.infrastructure.indexes.lsh import LSHIndex
from src.vector_db.domain.models import Chunk
from tests.utils import create_deterministic_embedding


class TestLSHIndex:
    """Test suite for LSH index implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.lsh_index = LSHIndex(num_tables=5, num_hyperplanes=8)

    def create_test_chunk(
        self, chunk_id: str, text: str, embedding: List[float]
    ) -> Chunk:
        """Create a test chunk with given parameters"""
        return Chunk(
            id=chunk_id, document_id=str(uuid4()), text=text, embedding=embedding
        )


class TestLSHInitialization:
    """Test LSH index initialization"""

    def test_default_initialization(self):
        """Test LSH index with default parameters"""
        index = LSHIndex()
        assert index.num_tables == 6
        assert index.num_hyperplanes == 4
        assert len(index.hash_tables) == 6
        assert len(index.hyperplanes) == 0
        assert index.vector_dim == 0

    def test_custom_initialization(self):
        """Test LSH index with custom parameters"""
        index = LSHIndex(num_tables=5, num_hyperplanes=8)
        assert index.num_tables == 5
        assert index.num_hyperplanes == 8
        assert len(index.hash_tables) == 5
        assert len(index.hyperplanes) == 0
        assert index.vector_dim == 0

    def test_custom_initialization_via_factory(self):
        """Test LSH index with custom parameters via factory"""
        factory = IndexFactory()
        index = factory.create_index("lsh", num_tables=5, num_hyperplanes=8)
        assert index.num_tables == 5
        assert index.num_hyperplanes == 8
        assert len(index.hash_tables) == 5
        assert len(index.hyperplanes) == 0
        assert index.vector_dim == 0

    def test_hash_tables_structure(self):
        """Test hash tables are properly initialized"""
        index = LSHIndex(num_tables=3, num_hyperplanes=4)

        # Each hash table should be a defaultdict
        for i in range(3):
            assert hasattr(index.hash_tables[i], "default_factory")
            # Test that it creates sets by default
            test_set = index.hash_tables[i]["test_key"]
            assert isinstance(test_set, set)


class TestLSHHashFunctions(TestLSHIndex):
    """Test LSH hash function generation and computation"""

    def test_hyperplane_generation(self):
        """Test hyperplane generation for given dimension"""
        vector_dim = 10
        self.lsh_index._generate_hyperplanes(vector_dim)

        # Should create hyperplanes for each table
        assert len(self.lsh_index.hyperplanes) == self.lsh_index.num_tables

        # Each table should have the right number of hyperplanes
        for table_hyperplanes in self.lsh_index.hyperplanes:
            assert table_hyperplanes.shape == (
                self.lsh_index.num_hyperplanes,
                vector_dim,
            )

            # Hyperplanes should be normalized
            norms = np.linalg.norm(table_hyperplanes, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_hyperplane_reproducibility(self):
        """Test that hyperplane generation is reproducible"""
        vector_dim = 10

        # Generate hyperplanes twice
        index1 = LSHIndex(num_tables=3, num_hyperplanes=4)
        index1._generate_hyperplanes(vector_dim)

        index2 = LSHIndex(num_tables=3, num_hyperplanes=4)
        index2._generate_hyperplanes(vector_dim)

        # Should be identical due to fixed seed
        for i in range(3):
            np.testing.assert_array_equal(index1.hyperplanes[i], index2.hyperplanes[i])

    def test_hash_code_computation(self):
        """Test hash code computation"""
        vector_dim = 6
        self.lsh_index._generate_hyperplanes(vector_dim)

        # Test with a simple vector
        test_vector = [1.0, 2.0, -1.0, 3.0, -2.0, 0.5]

        # Compute hash codes for each table
        for table_idx in range(self.lsh_index.num_tables):
            hash_code = self.lsh_index._compute_hash_code(test_vector, table_idx)

            # Should be a binary string
            assert isinstance(hash_code, str)
            assert len(hash_code) == self.lsh_index.num_hyperplanes
            assert all(bit in "01" for bit in hash_code)

    def test_hash_code_consistency(self):
        """Test that same vector produces same hash code"""
        vector_dim = 8
        self.lsh_index._generate_hyperplanes(vector_dim)

        test_vector = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]

        # Compute hash code multiple times
        hash_codes = []
        for _ in range(5):
            hash_code = self.lsh_index._compute_hash_code(test_vector, 0)
            hash_codes.append(hash_code)

        # All should be identical
        assert all(code == hash_codes[0] for code in hash_codes)

    def test_hash_code_without_hyperplanes(self):
        """Test hash code computation before hyperplanes are generated"""
        test_vector = [1.0, 2.0, 3.0]
        hash_code = self.lsh_index._compute_hash_code(test_vector, 0)
        assert hash_code == ""


class TestLSHIndexing(TestLSHIndex):
    """Test LSH indexing operations"""

    def test_add_chunks_empty_list(self):
        """Test adding empty chunk list"""
        self.lsh_index._add_chunks_impl([])
        assert len(self.lsh_index.hyperplanes) == 0
        assert self.lsh_index.vector_dim == 0

    def test_add_chunks_no_embeddings(self):
        """Test adding chunks without embeddings"""
        chunks = [
            self.create_test_chunk("chunk1", "text1", []),
            self.create_test_chunk("chunk2", "text2", []),
        ]

        self.lsh_index._add_chunks_impl(chunks)
        assert len(self.lsh_index.hyperplanes) == 0
        assert self.lsh_index.vector_dim == 0

    def test_add_chunks_with_embeddings(self):
        """Test adding chunks with embeddings"""
        embeddings = [
            create_deterministic_embedding("text1", 10),
            create_deterministic_embedding("text2", 10),
            create_deterministic_embedding("text3", 10),
        ]

        chunks = [
            self.create_test_chunk("chunk1", "text1", embeddings[0]),
            self.create_test_chunk("chunk2", "text2", embeddings[1]),
            self.create_test_chunk("chunk3", "text3", embeddings[2]),
        ]

        self.lsh_index._add_chunks_impl(chunks)

        # Hyperplanes should be generated
        assert len(self.lsh_index.hyperplanes) == self.lsh_index.num_tables
        assert self.lsh_index.vector_dim == 10

        # Chunks should be added to hash tables
        total_entries = 0
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                total_entries += len(bucket)

        # Each chunk should be in each table (num_chunks * num_tables)
        assert total_entries == 3 * self.lsh_index.num_tables

    def test_add_chunks_mixed_embeddings(self):
        """Test adding chunks with some having embeddings and some not"""
        embeddings = [
            create_deterministic_embedding("text1", 10),
            create_deterministic_embedding("text3", 10),
        ]

        chunks = [
            self.create_test_chunk("chunk1", "text1", embeddings[0]),
            self.create_test_chunk("chunk2", "text2", []),  # No embedding
            self.create_test_chunk("chunk3", "text3", embeddings[1]),
        ]

        self.lsh_index._add_chunks_impl(chunks)

        # Should process chunks with embeddings
        assert len(self.lsh_index.hyperplanes) == self.lsh_index.num_tables
        assert self.lsh_index.vector_dim == 10

        # Only chunks with embeddings should be indexed
        total_entries = 0
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                total_entries += len(bucket)

        assert total_entries == 2 * self.lsh_index.num_tables

    def test_hyperplane_initialization_once(self):
        """Test that hyperplanes are only initialized once"""
        embeddings = [
            create_deterministic_embedding("text1", 10),
            create_deterministic_embedding("text2", 10),
        ]

        chunks1 = [self.create_test_chunk("chunk1", "text1", embeddings[0])]
        chunks2 = [self.create_test_chunk("chunk2", "text2", embeddings[1])]

        # Add first batch
        self.lsh_index._add_chunks_impl(chunks1)
        first_hyperplanes = [hp.copy() for hp in self.lsh_index.hyperplanes]

        # Add second batch
        self.lsh_index._add_chunks_impl(chunks2)
        second_hyperplanes = self.lsh_index.hyperplanes

        # Hyperplanes should be identical
        for i in range(len(first_hyperplanes)):
            np.testing.assert_array_equal(first_hyperplanes[i], second_hyperplanes[i])


class TestLSHRemoval(TestLSHIndex):
    """Test LSH chunk removal operations"""

    def test_remove_empty_list(self):
        """Test removing empty chunk list"""
        self.lsh_index._remove_chunks_impl([])
        # Should not raise any errors

    def test_remove_from_empty_index(self):
        """Test removing chunks from empty index"""
        self.lsh_index._remove_chunks_impl(["chunk1", "chunk2"])
        # Should not raise any errors

    def test_remove_existing_chunks(self):
        """Test removing chunks that exist in the index"""
        # First add some chunks
        embeddings = [
            create_deterministic_embedding("text1", 10),
            create_deterministic_embedding("text2", 10),
            create_deterministic_embedding("text3", 10),
        ]

        chunks = [
            self.create_test_chunk("chunk1", "text1", embeddings[0]),
            self.create_test_chunk("chunk2", "text2", embeddings[1]),
            self.create_test_chunk("chunk3", "text3", embeddings[2]),
        ]

        self.lsh_index._add_chunks_impl(chunks)

        # Count initial entries
        initial_entries = 0
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                initial_entries += len(bucket)

        # Remove some chunks
        self.lsh_index._remove_chunks_impl(["chunk1", "chunk3"])

        # Count remaining entries
        remaining_entries = 0
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                remaining_entries += len(bucket)

        # Should have fewer entries
        assert remaining_entries < initial_entries

        # Verify specific chunks are gone
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                assert "chunk1" not in bucket
                assert "chunk3" not in bucket
                # chunk2 might still be there

    def test_remove_nonexistent_chunks(self):
        """Test removing chunks that don't exist"""
        # Add some chunks first
        embeddings = [create_deterministic_embedding("text1", 10)]
        chunks = [self.create_test_chunk("chunk1", "text1", embeddings[0])]

        self.lsh_index._add_chunks_impl(chunks)

        # Try to remove non-existent chunks
        self.lsh_index._remove_chunks_impl(["nonexistent1", "nonexistent2"])

        # Should not affect existing chunks
        found_chunk1 = False
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                if "chunk1" in bucket:
                    found_chunk1 = True
                    break

        assert found_chunk1

    def test_remove_empty_bucket_cleanup(self):
        """Test that empty buckets are cleaned up after removal"""
        # Add a single chunk
        embedding = create_deterministic_embedding("text1", 10)
        chunk = self.create_test_chunk("chunk1", "text1", embedding)

        self.lsh_index._add_chunks_impl([chunk])

        # Count non-empty buckets before removal
        buckets_before = 0
        for table in self.lsh_index.hash_tables:
            buckets_before += len(table)

        # Remove the chunk
        self.lsh_index._remove_chunks_impl(["chunk1"])

        # Count non-empty buckets after removal
        buckets_after = 0
        for table in self.lsh_index.hash_tables:
            buckets_after += len(table)

        # Should have fewer buckets (empty ones should be cleaned up)
        assert buckets_after < buckets_before


class TestLSHSearch(TestLSHIndex):
    """Test LSH search functionality"""

    def test_search_empty_index(self):
        """Test searching in empty index"""
        query_embedding = create_deterministic_embedding("query", 10)
        results = self.lsh_index._search_impl(query_embedding, k=5, min_similarity=0.0)
        assert results == []

    def test_search_no_hyperplanes(self):
        """Test searching before hyperplanes are generated"""
        query_embedding = create_deterministic_embedding("query", 10)
        results = self.lsh_index._search_impl(query_embedding, k=5, min_similarity=0.0)
        assert results == []

    def test_search_empty_query(self):
        """Test searching with empty query embedding"""
        results = self.lsh_index._search_impl([], k=5, min_similarity=0.0)
        assert results == []

    def test_search_basic_functionality(self):
        """Test basic search functionality"""
        # Use very similar text to create similar embeddings
        embeddings = [
            create_deterministic_embedding("machine learning", 10),
            create_deterministic_embedding("machine learninG", 10),  # One char diff
            create_deterministic_embedding("machine learnin", 10),  # One char diff
        ]

        chunks = [
            self.create_test_chunk("chunk1", "machine learning", embeddings[0]),
            self.create_test_chunk("chunk2", "machine learninG", embeddings[1]),
            self.create_test_chunk("chunk3", "machine learnin", embeddings[2]),
        ]

        # Add chunks to both LSH index and base class storage
        self.lsh_index._add_chunks_impl(chunks)
        for chunk in chunks:
            self.lsh_index._chunks[chunk.id] = chunk

        # Search with query very similar to indexed texts
        query_embedding = create_deterministic_embedding("machine learning", 10)
        results = self.lsh_index._search_impl(query_embedding, k=5, min_similarity=0.0)

        # Should return some results
        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        assert all(0.0 <= score <= 1.0 for _, score in results)

    def test_search_k_limit(self):
        """Test that search respects k limit"""
        # Add many chunks with very similar text
        embeddings = [
            create_deterministic_embedding(f"testing{i}", 10) for i in range(10)
        ]

        chunks = [
            self.create_test_chunk(f"chunk{i}", f"testing{i}", embeddings[i])
            for i in range(10)
        ]

        self.lsh_index._add_chunks_impl(chunks)
        for chunk in chunks:
            self.lsh_index._chunks[chunk.id] = chunk

        # Search with k=3 using similar text
        query_embedding = create_deterministic_embedding("testing1", 10)
        results = self.lsh_index._search_impl(query_embedding, k=3, min_similarity=0.0)

        # Should return at most k results
        assert len(results) <= 3

    def test_search_min_similarity_filter(self):
        """Test that search respects min_similarity threshold"""
        # Add chunks with very similar text
        embeddings = [
            create_deterministic_embedding("similar text", 10),
            create_deterministic_embedding("similar texT", 10),  # One char diff
        ]

        chunks = [
            self.create_test_chunk("chunk1", "similar text", embeddings[0]),
            self.create_test_chunk("chunk2", "similar texT", embeddings[1]),
        ]

        self.lsh_index._add_chunks_impl(chunks)
        for chunk in chunks:
            self.lsh_index._chunks[chunk.id] = chunk

        # Search with high similarity threshold
        query_embedding = create_deterministic_embedding("similar text", 10)
        results_high_threshold = self.lsh_index._search_impl(
            query_embedding, k=5, min_similarity=0.8
        )

        # Search with low similarity threshold
        results_low_threshold = self.lsh_index._search_impl(
            query_embedding, k=5, min_similarity=0.0
        )

        # High threshold should return fewer or equal results
        assert len(results_high_threshold) <= len(results_low_threshold)

    def test_search_results_sorted_by_similarity(self):
        """Test that search results are sorted by similarity in descending order"""
        # Add chunks with very similar text
        embeddings = [
            create_deterministic_embedding("sorttest", 10),
            create_deterministic_embedding("sorttesT", 10),  # One char diff
            create_deterministic_embedding("sortTesT", 10),  # Two char diff
        ]

        chunks = [
            self.create_test_chunk("chunk1", "sorttest", embeddings[0]),
            self.create_test_chunk("chunk2", "sorttesT", embeddings[1]),
            self.create_test_chunk("chunk3", "sortTesT", embeddings[2]),
        ]

        self.lsh_index._add_chunks_impl(chunks)
        for chunk in chunks:
            self.lsh_index._chunks[chunk.id] = chunk

        query_embedding = create_deterministic_embedding("sorttest", 10)
        results = self.lsh_index._search_impl(query_embedding, k=5, min_similarity=0.0)

        if len(results) > 1:
            # Check that results are sorted by similarity (descending)
            similarities = [score for _, score in results]
            assert similarities == sorted(similarities, reverse=True)


class TestLSHCosineSimilarity(TestLSHIndex):
    """Test cosine similarity calculation"""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors"""
        vec = [1.0, 2.0, 3.0]
        similarity = self.lsh_index._cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.lsh_index._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = self.lsh_index._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10  # Clamped to 0

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with vectors of different lengths"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        similarity = self.lsh_index._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = self.lsh_index._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_bounds(self):
        """Test that cosine similarity is bounded between 0 and 1"""
        for _ in range(10):
            vec1 = np.random.randn(5).tolist()
            vec2 = np.random.randn(5).tolist()
            similarity = self.lsh_index._cosine_similarity(vec1, vec2)
            assert 0.0 <= similarity <= 1.0


class TestLSHThreadSafety(TestLSHIndex):
    """Test LSH thread safety"""

    def test_concurrent_indexing(self):
        """Test concurrent chunk indexing"""
        # Pre-initialize hyperplanes to avoid race conditions in this test
        self.lsh_index.vector_dim = 10
        self.lsh_index._generate_hyperplanes(10)

        def add_chunks_worker(worker_id: int):
            embeddings = [
                create_deterministic_embedding(f"worker{worker_id}_text{i}", 10)
                for i in range(5)
            ]

            chunks = [
                self.create_test_chunk(
                    f"worker{worker_id}_chunk{i}", f"text{i}", embeddings[i]
                )
                for i in range(5)
            ]

            self.lsh_index._add_chunks_impl(chunks)
            for chunk in chunks:
                self.lsh_index._chunks[chunk.id] = chunk

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_chunks_worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all chunks were indexed
        total_chunks = 0
        for table in self.lsh_index.hash_tables:
            for bucket in table.values():
                total_chunks += len(bucket)

        # Should have entries for all chunks across all tables
        expected_entries = 15 * self.lsh_index.num_tables  # 3 workers * 5 chunks each
        assert total_chunks == expected_entries

    def test_concurrent_search_and_indexing(self):
        """Test concurrent search and indexing operations"""
        search_results = []

        def search_worker():
            query_embedding = create_deterministic_embedding("search query", 10)
            for _ in range(10):
                results = self.lsh_index._search_impl(
                    query_embedding, k=5, min_similarity=0.0
                )
                search_results.append(len(results))
                time.sleep(0.001)  # Small delay

        def index_worker():
            for i in range(5):
                embedding = create_deterministic_embedding(f"concurrent_text{i}", 10)
                chunk = self.create_test_chunk(
                    f"concurrent_chunk{i}", f"text{i}", embedding
                )
                self.lsh_index._add_chunks_impl([chunk])
                self.lsh_index._chunks[chunk.id] = chunk
                time.sleep(0.001)  # Small delay

        # Start concurrent operations
        search_thread = threading.Thread(target=search_worker)
        index_thread = threading.Thread(target=index_worker)

        search_thread.start()
        index_thread.start()

        search_thread.join()
        index_thread.join()

        # Should not have crashed and should have some results
        assert len(search_results) > 0


class TestLSHEdgeCases(TestLSHIndex):
    """Test LSH edge cases and error conditions"""

    def test_very_small_embeddings(self):
        """Test with very small embedding dimensions"""
        index = LSHIndex(num_tables=2, num_hyperplanes=2)

        embedding = [0.5, -0.5]
        chunk = self.create_test_chunk("chunk1", "text1", embedding)

        index._add_chunks_impl([chunk])
        index._chunks[chunk.id] = chunk

        # Should work without errors
        query_embedding = [0.3, -0.7]
        results = index._search_impl(query_embedding, k=1, min_similarity=0.0)
        assert len(results) <= 1

    def test_large_embeddings(self):
        """Test with large embedding dimensions"""
        index = LSHIndex(num_tables=3, num_hyperplanes=4)

        embedding = create_deterministic_embedding("large_text", 1000)
        chunk = self.create_test_chunk("chunk1", "text1", embedding)

        index._add_chunks_impl([chunk])
        index._chunks[chunk.id] = chunk

        # Should work without errors
        query_embedding = create_deterministic_embedding("large_query", 1000)
        results = index._search_impl(query_embedding, k=1, min_similarity=0.0)
        assert len(results) <= 1

    def test_extreme_parameters(self):
        """Test with extreme parameter values"""
        # Very small parameters
        index_small = LSHIndex(num_tables=1, num_hyperplanes=1)
        assert index_small.num_tables == 1
        assert index_small.num_hyperplanes == 1

        # Large parameters
        index_large = LSHIndex(num_tables=100, num_hyperplanes=50)
        assert index_large.num_tables == 100
        assert index_large.num_hyperplanes == 50

    def test_chunk_without_embedding_in_search(self):
        """Test search behavior when indexed chunk loses its embedding"""
        # Add chunk with embedding
        embedding = create_deterministic_embedding("text1", 10)
        chunk = self.create_test_chunk("chunk1", "text1", embedding)

        self.lsh_index._add_chunks_impl([chunk])

        # Modify chunk to have no embedding and add to base storage
        chunk_no_embedding = self.create_test_chunk("chunk1", "text1", [])
        self.lsh_index._chunks[chunk.id] = chunk_no_embedding

        # Search should handle this gracefully
        query_embedding = create_deterministic_embedding("query", 10)
        results = self.lsh_index._search_impl(query_embedding, k=5, min_similarity=0.0)

        # Should not crash and should not return the chunk without embedding
        assert isinstance(results, list)
