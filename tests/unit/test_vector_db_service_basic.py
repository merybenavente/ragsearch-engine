"""Basic unit tests for VectorDBService"""

from unittest.mock import patch, MagicMock

from src.vector_db.api.dependencies import get_vector_db_service
from src.vector_db.domain.models import Document, Library, Chunk


class TestVectorDBServiceBasic:
    """Basic test cases for VectorDBService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.vector_db_service = get_vector_db_service()

    def test_create_library(self):
        """Test creating a library"""
        library = self.vector_db_service.create_library(name="Test Library")

        assert library.id is not None
        assert isinstance(library, Library)
        assert library.name == "Test Library"

    def test_get_library(self):
        """Test getting a library"""
        library = self.vector_db_service.create_library(name="Test Library")
        retrieved_library = self.vector_db_service.get_library(library.id)

        assert retrieved_library is not None
        assert retrieved_library.name == "Test Library"
        assert retrieved_library.id == library.id

    def test_list_libraries(self):
        """Test listing libraries"""
        # Create a few libraries
        lib1 = self.vector_db_service.create_library(name="Library 1")
        lib2 = self.vector_db_service.create_library(name="Library 2")

        libraries = self.vector_db_service.list_libraries()

        assert isinstance(libraries, list)
        assert len(libraries) == 2
        library_names = [lib.name for lib in libraries]
        assert "Library 1" in library_names
        assert "Library 2" in library_names

    def test_delete_library(self):
        """Test deleting a library"""
        library = self.vector_db_service.create_library(name="Test Library")

        # Verify library exists
        retrieved_library = self.vector_db_service.get_library(library.id)
        assert retrieved_library is not None

        # Delete library
        self.vector_db_service.delete_library(library.id)

        # Verify library is deleted
        deleted_library = self.vector_db_service.get_library(library.id)
        assert deleted_library is None

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_create_document(self, mock_co):
        """Test creating a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="This is a test document."
        )

        assert isinstance(document, Document)
        assert document.library_id == library.id
        assert len(document.chunks) > 0

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_document(self, mock_co):
        """Test getting a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="This is a test document."
        )

        retrieved_document = self.vector_db_service.get_document(document.id)

        assert retrieved_document is not None
        assert retrieved_document.id == document.id
        assert retrieved_document.library_id == library.id

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_library(self, mock_co):
        """Test searching in a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="This is a test document about machine learning."
        )

        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="machine learning",
            k=5,
            min_similarity=0.0
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_document(self, mock_co):
        """Test searching in a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="This is a test document about artificial intelligence and machine learning."
        )

        results = self.vector_db_service.search_document(
            document_id=document.id,
            query_text="artificial intelligence",
            k=3,
            min_similarity=0.0
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_create_empty_document(self):
        """Test creating an empty document"""
        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_empty_document(
            library_id=library.id,
            username="testuser",
            tags=["test", "empty"]
        )

        assert isinstance(document, Document)
        assert document.library_id == library.id
        assert document.metadata.username == "testuser"
        assert "test" in document.metadata.tags
        assert "empty" in document.metadata.tags
        assert len(document.chunks) == 0

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_update_document_content(self, mock_co):
        """Test updating document content"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        library = self.vector_db_service.create_library(name="Test Library")
        document = self.vector_db_service.create_document(
            library_id=library.id,
            text="Original content."
        )

        updated_document = self.vector_db_service.update_document_content(
            document_id=document.id,
            new_text="Updated content about machine learning.",
            chunk_size=100
        )

        assert isinstance(updated_document, Document)
        assert updated_document.id == document.id
        assert len(updated_document.chunks) > 0
        # Verify content was updated
        full_text = updated_document.get_full_text()
        assert "Updated content" in full_text

    def test_update_library_metadata(self):
        """Test updating library metadata"""
        library = self.vector_db_service.create_library(name="Original Name")

        updated_library = self.vector_db_service.update_library_metadata(
            library_id=library.id,
            name="Updated Name",
            tags=["updated", "test"]
        )

        assert updated_library.name == "Updated Name"
        assert "updated" in updated_library.metadata.tags
        assert "test" in updated_library.metadata.tags

    # TODO: review why the mocking is not consistent
    def test_get_documents_in_library(self):
        """Test getting documents in a library"""
        library = self.vector_db_service.create_library(name="Test Library")

        # Initially no documents
        documents = self.vector_db_service.get_documents_in_library(library.id)
        assert len(documents) == 0

        # Add some documents
        with patch('src.vector_db.infrastructure.embedding_service.co') as mock_co:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * 1536]
            mock_co.embed.return_value = mock_response

            doc1 = self.vector_db_service.create_document(library.id, "Document 1")
            doc2 = self.vector_db_service.create_document(library.id, "Document 2")

        documents = self.vector_db_service.get_documents_in_library(library.id)
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)