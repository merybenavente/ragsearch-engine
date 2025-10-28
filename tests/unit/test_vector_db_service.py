"""Unit tests for VectorDBService"""

import pytest
import hashlib
import numpy as np
from uuid import uuid4

from src.vector_db.application.vector_db_service import VectorDBService
from src.vector_db.domain.models import Chunk
from src.vector_db.domain.interfaces import EmbeddingService
from src.vector_db.infrastructure.repositories import RepositoryManager
from src.vector_db.infrastructure.search_index import RepositoryAwareSearchIndex
from src.vector_db.infrastructure.index_factory import get_index_factory


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing"""

    def create_embedding(self, text: str, input_type: str = "search_document") -> list[float]:
        """Create deterministic mock embedding based on text hash"""
        # Create deterministic embedding based on text hash
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        embedding = np.random.randn(1536)
        return (embedding / np.linalg.norm(embedding)).tolist()


class TestVectorDBService:
    """Test cases for VectorDBService"""

    def setup_method(self):
        """Setup test fixtures"""
        # Setup repositories
        self.repo_manager = RepositoryManager()
        self.mock_embedding = MockEmbeddingService()
        self.search_index = RepositoryAwareSearchIndex(get_index_factory())

        # Create VectorDBService instance
        self.vector_db_service = VectorDBService(
            self.repo_manager.get_document_repository(),
            self.repo_manager.get_library_repository(),
            self.search_index,
            self.mock_embedding
        )

    def test_create_library(self):
        """Test creating a library"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1", "tag2"]
        index_type = "naive"

        library = self.vector_db_service.create_library(
            name=name,
            username=username,
            tags=tags,
            index_type=index_type
        )

        assert library.name == name
        assert library.metadata.username == username
        assert library.metadata.tags == tags
        assert library.index_type == index_type
        assert library.document_ids == set()

        # Verify library is persisted
        retrieved_library = self.vector_db_service.get_library(library.id)
        assert retrieved_library is not None
        assert retrieved_library.id == library.id

    def test_create_library_with_defaults(self):
        """Test creating a library with default values"""
        name = "Simple Library"

        library = self.vector_db_service.create_library(name=name)

        assert library.name == name
        assert library.metadata.username is None
        assert library.metadata.tags == []
        assert library.index_type == "naive"

    def test_create_document(self):
        """Test creating a document in a library"""
        # Create library first
        library = self.vector_db_service.create_library("Test Library")

        # Create document
        text = "This is a test document with multiple sentences. It should be chunked appropriately."
        username = "testuser"
        tags = ["doc_tag"]
        chunk_size = 50

        document = self.vector_db_service.create_document(
            library_id=library.id,
            text=text,
            username=username,
            tags=tags,
            chunk_size=chunk_size
        )

        assert document.library_id == library.id
        assert document.metadata.username == username
        assert document.metadata.tags == tags
        assert len(document.chunks) > 0

        # Verify document is added to library
        updated_library = self.vector_db_service.get_library(library.id)
        assert document.id in updated_library.document_ids

        # Verify chunks have embeddings
        for chunk in document.chunks:
            assert len(chunk.embedding) == 1536
            assert chunk.document_id == document.id

    def test_create_document_library_not_found(self):
        """Test creating document in non-existent library raises error"""
        fake_library_id = str(uuid4())

        with pytest.raises(ValueError, match="Library .* not found"):
            self.vector_db_service.create_document(
                library_id=fake_library_id,
                text="Test text"
            )

    def test_search_library(self):
        """Test searching within a library"""
        # Create library and document
        library = self.vector_db_service.create_library("Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms."
        )

        # Verify document was created and indexed
        assert len(document.chunks) > 0

        # Search for related content with very low threshold
        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="machine learning",  # Use simpler query that should match
            k=5,
            min_similarity=0.0  # Use 0.0 to get all results
        )

        assert len(results) > 0
        # Results are List[Tuple[Chunk, float]], so extract chunks
        chunks = [chunk for chunk, score in results]
        assert all(chunk.document_id == document.id for chunk in chunks)

        # Verify chunks have similarity scores (via mock embeddings)
        for chunk, score in results:
            assert len(chunk.embedding) == 1536
            assert isinstance(score, float)

    def test_search_library_not_found(self):
        """Test searching non-existent library raises error"""
        fake_library_id = str(uuid4())

        with pytest.raises(ValueError, match="Library .* not found"):
            self.vector_db_service.search_library(
                library_id=fake_library_id,
                query_text="test query"
            )

    def test_search_empty_library(self):
        """Test searching empty library returns empty results"""
        library = self.vector_db_service.create_library("Empty Library")

        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="test query"
        )

        assert len(results) == 0

    def test_update_library_metadata(self):
        """Test updating library metadata"""
        library = self.vector_db_service.create_library("Original Library")

        # Update name and tags
        new_name = "Updated Library"
        new_tags = ["updated", "tags"]

        updated_library = self.vector_db_service.update_library_metadata(
            library_id=library.id,
            name=new_name,
            tags=new_tags
        )

        assert updated_library.name == new_name
        assert updated_library.metadata.tags == new_tags

        # Verify persistence
        retrieved_library = self.vector_db_service.get_library(library.id)
        assert retrieved_library.name == new_name
        assert retrieved_library.metadata.tags == new_tags

    def test_delete_library(self):
        """Test deleting a library and its documents"""
        # Create library with document
        library = self.vector_db_service.create_library("Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Test document content"
        )

        # Delete library
        self.vector_db_service.delete_library(library.id)

        # Verify library is deleted
        assert self.vector_db_service.get_library(library.id) is None

        # Verify document is deleted
        assert self.vector_db_service.get_document(document.id) is None

    def test_get_library_documents(self):
        """Test getting all documents in a library"""
        library = self.vector_db_service.create_library("Test Library")

        # Create multiple documents
        doc1 = self.vector_db_service.create_document(
            library_id=library.id,
            text="First document content"
        )
        doc2 = self.vector_db_service.create_document(
            library_id=library.id,
            text="Second document content"
        )

        # Get all documents
        documents = self.vector_db_service.get_documents_in_library(library.id)

        assert len(documents) == 2
        doc_ids = {doc.id for doc in documents}
        assert doc1.id in doc_ids
        assert doc2.id in doc_ids

    def test_get_document(self):
        """Test getting a specific document"""
        library = self.vector_db_service.create_library("Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Test document content"
        )

        retrieved_document = self.vector_db_service.get_document(document.id)

        assert retrieved_document is not None
        assert retrieved_document.id == document.id
        assert retrieved_document.library_id == library.id

    def test_integration_workflow(self):
        """Test complete workflow integration"""
        # Create library
        library = self.vector_db_service.create_library(
            name="Integration Test Library",
            username="integrator",
            tags=["integration", "test"]
        )

        # Create multiple documents
        doc1 = self.vector_db_service.create_document(
            library_id=library.id,
            text="Python is a programming language that emphasizes readability and simplicity.",
            username="author1",
            tags=["python", "programming"]
        )

        doc2 = self.vector_db_service.create_document(
            library_id=library.id,
            text="Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
            username="author2",
            tags=["ml", "python", "scikit-learn"]
        )

        # Search for content
        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="Python programming",
            k=3,
            min_similarity=0.0  # Use low threshold to ensure results
        )

        # Verify results
        assert len(results) > 0
        # Results are List[Tuple[Chunk, float]], so check structure
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(isinstance(score, float) for score in scores)
        assert all(chunk.document_id in [doc1.id, doc2.id] for chunk in chunks)

        # Verify library contains both documents
        updated_library = self.vector_db_service.get_library(library.id)
        assert len(updated_library.document_ids) == 2
        assert doc1.id in updated_library.document_ids
        assert doc2.id in updated_library.document_ids

    def test_mock_embedding_consistency(self):
        """Test that mock embeddings are consistent for same text"""
        text = "Test text for consistency"

        embedding1 = self.mock_embedding.create_embedding(text)
        embedding2 = self.mock_embedding.create_embedding(text)

        assert embedding1 == embedding2
        assert len(embedding1) == 1536

        # Different text should produce different embeddings
        different_text = "Different text"
        different_embedding = self.mock_embedding.create_embedding(different_text)

        assert different_embedding != embedding1