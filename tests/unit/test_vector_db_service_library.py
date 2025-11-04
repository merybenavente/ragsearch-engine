"""Unit tests for VectorDBService library operations"""

import pytest
from uuid import uuid4

from src.vector_db.application.vector_db_service import VectorDBService
from src.vector_db.infrastructure.repositories import RepositoryManager
from src.vector_db.infrastructure.search_index import RepositoryAwareSearchIndex
from src.vector_db.infrastructure.index_factory import get_index_factory
from src.vector_db.infrastructure.embedding_service import CohereEmbeddingService


class TestVectorDBServiceLibrary:
    """Test cases for VectorDBService library operations"""

    def setup_method(self):
        """Setup test fixtures"""
        # Setup repositories
        self.repo_manager = RepositoryManager()
        self.search_index = RepositoryAwareSearchIndex(get_index_factory())
        self.embedding_service = CohereEmbeddingService()

        # Create VectorDBService instance
        self.vector_db_service = VectorDBService(
            self.repo_manager.get_document_repository(),
            self.repo_manager.get_library_repository(),
            self.search_index,
            self.embedding_service,
        )

    def test_create_library(self, mock_cohere_deterministic):
        """Test creating a library"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1", "tag2"]
        index_type = "naive"

        library = self.vector_db_service.create_library(
            name=name, username=username, tags=tags, index_type=index_type
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

    def test_create_library_with_defaults(self, mock_cohere_deterministic):
        """Test creating a library with default values"""
        name = "Simple Library"

        library = self.vector_db_service.create_library(name=name)

        assert library.name == name
        assert library.metadata.username is None
        assert library.metadata.tags == []
        assert library.index_type == "naive"

    def test_update_library_metadata(self, mock_cohere_deterministic):
        """Test updating library metadata"""
        library = self.vector_db_service.create_library("Original Library")

        # Update name and tags
        new_name = "Updated Library"
        new_tags = ["updated", "tags"]

        updated_library = self.vector_db_service.update_library_metadata(
            library_id=library.id, name=new_name, tags=new_tags
        )

        assert updated_library.name == new_name
        assert updated_library.metadata.tags == new_tags

        # Verify persistence
        retrieved_library = self.vector_db_service.get_library(library.id)
        assert retrieved_library.name == new_name
        assert retrieved_library.metadata.tags == new_tags

    def test_delete_library(self, mock_cohere_deterministic):
        """Test deleting a library and its documents"""
        # Create library with document
        library = self.vector_db_service.create_library("Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id, text="Test document content"
        )

        # Delete library
        self.vector_db_service.delete_library(library.id)

        # Verify library is deleted
        assert self.vector_db_service.get_library(library.id) is None

        # Verify document is deleted
        assert self.vector_db_service.get_document(document.id) is None

    def test_list_libraries(self, mock_cohere_deterministic):
        """Test listing all libraries"""
        # Create a few libraries
        self.vector_db_service.create_library(name="Library 1")
        self.vector_db_service.create_library(name="Library 2")

        libraries = self.vector_db_service.list_libraries()

        assert isinstance(libraries, list)
        assert len(libraries) == 2
        library_names = [lib.name for lib in libraries]
        assert "Library 1" in library_names
        assert "Library 2" in library_names

    def test_get_library_documents(self, mock_cohere_deterministic):
        """Test getting all documents in a library"""
        library = self.vector_db_service.create_library("Test Library")

        # Create multiple documents
        doc1 = self.vector_db_service.create_document(
            library_id=library.id, text="First document content"
        )
        doc2 = self.vector_db_service.create_document(
            library_id=library.id, text="Second document content"
        )

        # Get all documents
        documents = self.vector_db_service.get_documents_in_library(library.id)

        assert len(documents) == 2
        doc_ids = {doc.id for doc in documents}
        assert doc1.id in doc_ids
        assert doc2.id in doc_ids

    def test_library_operations_with_invalid_data(self, mock_cohere_deterministic):
        """Test library operations with invalid data"""
        # Test create library with empty name
        with pytest.raises(ValueError):
            self.vector_db_service.create_library(name="")

        # Test update library with invalid ID
        fake_id = str(uuid4())
        with pytest.raises(ValueError):
            self.vector_db_service.update_library_metadata(
                library_id=fake_id, name="Updated Name"
            )
