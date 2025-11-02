"""Unit tests for the repositories"""

from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.api.dependencies import get_vector_db_service, get_document_repository, get_library_repository
from src.vector_db.domain.models import EMBEDDING_DIMENSION


class TestDDDRepositories:
    """Test cases for DDD architecture repositories"""

    def setup_method(self):
        """Setup test fixtures"""
        self.vector_db_service = get_vector_db_service()
        self.document_repository = get_document_repository()
        self.library_repository = get_library_repository()

    def test_library_repository_operations(self):
        """Test basic library repository operations"""
        # Create a library
        library = self.vector_db_service.create_library(name="Test Library")

        # Test retrieval
        retrieved_library = self.library_repository.get(library.id)
        assert retrieved_library is not None
        assert retrieved_library.name == "Test Library"
        assert retrieved_library.id == library.id

        # Test listing
        libraries = self.library_repository.list_all()
        assert isinstance(libraries, list)
        assert len(libraries) >= 1
        library_ids = [lib.id for lib in libraries]
        assert library.id in library_ids

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_document_repository_operations(self, mock_co):
        """Test basic document repository operations"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]
        mock_co.embed.return_value = mock_response

        # Create a library and document
        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Test document content"
        )

        # Test retrieval
        retrieved_document = self.document_repository.get(document.id)
        assert retrieved_document is not None
        assert retrieved_document.id == document.id
        assert retrieved_document.library_id == library.id

        # Test listing by library
        documents = self.document_repository.get_by_library(library.id)
        assert isinstance(documents, list)
        assert len(documents) >= 1
        document_ids = [doc.id for doc in documents]
        assert document.id in document_ids

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_repository_consistency(self, mock_co):
        """Test that repositories maintain consistency"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]
        mock_co.embed.return_value = mock_response

        # Create a library
        library = self.vector_db_service.create_library(name="Test Library")

        # Create multiple documents
        doc1 = self.vector_db_service.create_document(library.id, "Document 1")
        doc2 = self.vector_db_service.create_document(library.id, "Document 2")

        # Check that library knows about documents
        updated_library = self.library_repository.get(library.id)
        assert len(updated_library.document_ids) == 2
        assert doc1.id in updated_library.document_ids
        assert doc2.id in updated_library.document_ids

        # Check that documents exist in document repository
        retrieved_doc1 = self.document_repository.get(doc1.id)
        retrieved_doc2 = self.document_repository.get(doc2.id)
        assert retrieved_doc1 is not None
        assert retrieved_doc2 is not None
        assert retrieved_doc1.library_id == library.id
        assert retrieved_doc2.library_id == library.id

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_delete_operations(self, mock_co):
        """Test delete operations maintain consistency"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]
        mock_co.embed.return_value = mock_response

        # Create a library with documents
        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Test document"
        )

        # Verify initial state
        assert self.library_repository.get(library.id) is not None
        assert self.document_repository.get(document.id) is not None

        # Delete the library
        self.vector_db_service.delete_library(library.id)

        # Verify library is deleted
        assert self.library_repository.get(library.id) is None

        # Verify associated documents are also cleaned up
        assert self.document_repository.get(document.id) is None

    def test_empty_results(self):
        """Test repository behavior with empty results"""
        # Test non-existent library
        non_existent_id = str(uuid4())
        assert self.library_repository.get(non_existent_id) is None
        assert self.document_repository.get(non_existent_id) is None

        # Test empty library
        library = self.vector_db_service.create_library(name="Empty Library")
        documents = self.document_repository.get_by_library(library.id)
        assert documents == []