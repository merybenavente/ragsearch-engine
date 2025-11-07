"""Unit tests for VectorDBService search operations"""

import pytest
from uuid import uuid4

from src.vector_db.application.vector_db_service import VectorDBService
from src.vector_db.domain.models import Chunk, EMBEDDING_DIMENSION
from src.vector_db.infrastructure.repositories import RepositoryManager
from src.vector_db.infrastructure.search_index import RepositoryAwareSearchIndex
from src.vector_db.infrastructure.index_factory import IndexFactory
from src.vector_db.infrastructure.embedding_service import CohereEmbeddingService


class TestVectorDBServiceSearch:
    """Test cases for VectorDBService search operations"""

    def setup_method(self):
        """Setup test fixtures"""
        # Setup repositories
        self.repo_manager = RepositoryManager()
        self.search_index = RepositoryAwareSearchIndex(IndexFactory())
        self.embedding_service = CohereEmbeddingService()

        # Create VectorDBService instance
        self.vector_db_service = VectorDBService(
            self.repo_manager.get_document_repository(),
            self.repo_manager.get_library_repository(),
            self.search_index,
            self.embedding_service,
        )

    def test_search_library(self, mock_cohere_deterministic):
        """Test searching within a library"""
        # Create library and document
        library = self.vector_db_service.create_library("Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        )

        # Verify document was created and indexed
        assert len(document.chunks) > 0

        # Search for related content with very low threshold
        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="machine learning",  # Use simpler query that should match
            k=5,
            min_similarity=0.0,  # Use 0.0 to get all results
        )

        assert len(results) > 0
        # Results are List[Tuple[Chunk, float]], so extract chunks
        chunks = [chunk for chunk, score in results]
        assert all(chunk.document_id == document.id for chunk in chunks)

        # Verify chunks have similarity scores (via mock embeddings)
        for chunk, score in results:
            assert len(chunk.embedding) == EMBEDDING_DIMENSION
            assert isinstance(score, float)

    def test_search_library_not_found(self):
        """Test searching non-existent library raises error"""
        fake_library_id = str(uuid4())

        with pytest.raises(ValueError, match="Library .* not found"):
            self.vector_db_service.search_library(
                library_id=fake_library_id, query_text="test query"
            )

    def test_search_empty_library(self):
        """Test searching empty library returns empty results"""
        library = self.vector_db_service.create_library("Empty Library")

        results = self.vector_db_service.search_library(
            library_id=library.id, query_text="test query"
        )

        assert len(results) == 0

    def test_search_document(self, mock_cohere_deterministic):
        """Test searching within a specific document"""
        # Create library and document
        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="This is a test document about artificial intelligence and machine learning.",
        )

        results = self.vector_db_service.search_document(
            document_id=document.id,
            query_text="artificial intelligence",
            k=3,
            min_similarity=0.0,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        # Verify all results are from the target document
        for chunk, score in results:
            assert chunk.document_id == document.id

    def test_search_with_invalid_parameters(self):
        """Test search operations with invalid parameters"""
        library = self.vector_db_service.create_library(name="Test Library")

        # Test search with invalid k value
        with pytest.raises(ValueError):
            self.vector_db_service.search_library(
                library_id=library.id,
                query_text="test",
                k=-1,  # Invalid k value
            )

        # Test search with empty query
        with pytest.raises(ValueError):
            self.vector_db_service.search_library(
                library_id=library.id,
                query_text="",  # Empty query
                k=5,
            )
