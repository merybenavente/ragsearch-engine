"""Unit tests for infrastructure repositories"""

from unittest.mock import patch, MagicMock
from uuid import uuid4
from threading import Thread
import time

from src.vector_db.infrastructure.repository import (
    InMemoryLibraryRepository,
    RepositoryBasedDocumentRepository,
    RepositoryBasedChunkRepository
)
from src.vector_db.domain.models import (
    Library, Document, Chunk
)
from src.vector_db.api.dependencies import get_document_service


class TestInMemoryLibraryRepository:
    """Test cases for InMemoryLibraryRepository"""

    def setup_method(self):
        """Setup test fixtures"""
        self.repository = InMemoryLibraryRepository()

    def test_save_new_library(self):
        """Test saving a new library"""
        library = Library(name="Test Library")

        saved_library = self.repository.save(library)

        assert saved_library.id == library.id
        assert saved_library.name == library.name

    def test_save_update_library(self):
        """Test updating an existing library"""
        library = Library(name="Original Name")

        # Save initial library
        self.repository.save(library)

        # Update library
        updated_library = library.model_copy(update={'name': 'Updated Name'})
        saved_library = self.repository.save(updated_library)

        assert saved_library.name == "Updated Name"

    def test_find_by_id_existing(self):
        """Test finding an existing library by ID"""
        library = Library(name="Test Library")
        self.repository.save(library)

        found_library = self.repository.find_by_id(library.id)

        assert found_library is not None
        assert found_library.id == library.id
        assert found_library.name == library.name

    def test_find_by_id_not_found(self):
        """Test finding a non-existent library by ID"""
        non_existent_id = str(uuid4())

        found_library = self.repository.find_by_id(non_existent_id)

        assert found_library is None

    def test_find_all_empty(self):
        """Test finding all libraries when repository is empty"""
        libraries = self.repository.find_all()

        assert libraries == []

    def test_find_all_with_libraries(self):
        """Test finding all libraries with multiple libraries"""
        library1 = Library(name="Library 1")
        library2 = Library(name="Library 2")

        self.repository.save(library1)
        self.repository.save(library2)

        libraries = self.repository.find_all()

        assert len(libraries) == 2
        library_names = [lib.name for lib in libraries]
        assert "Library 1" in library_names
        assert "Library 2" in library_names

    def test_delete_existing_library(self):
        """Test deleting an existing library"""
        library = Library(name="Test Library")
        self.repository.save(library)

        # Verify library exists
        assert self.repository.find_by_id(library.id) is not None

        # Delete library
        result = self.repository.delete(library.id)

        assert result is True
        assert self.repository.find_by_id(library.id) is None

    def test_delete_non_existent_library(self):
        """Test deleting a non-existent library"""
        non_existent_id = str(uuid4())

        result = self.repository.delete(non_existent_id)

        assert result is False

    def test_exists_existing_library(self):
        """Test checking if an existing library exists"""
        library = Library(name="Test Library")
        self.repository.save(library)

        result = self.repository.exists(library.id)

        assert result is True

    def test_exists_non_existent_library(self):
        """Test checking if a non-existent library exists"""
        non_existent_id = str(uuid4())

        result = self.repository.exists(non_existent_id)

        assert result is False

    def test_find_by_name_existing(self):
        """Test finding an existing library by name"""
        library = Library(name="Unique Library Name")
        self.repository.save(library)

        found_library = self.repository.find_by_name("Unique Library Name")

        assert found_library is not None
        assert found_library.id == library.id

    def test_find_by_name_not_found(self):
        """Test finding a non-existent library by name"""
        found_library = self.repository.find_by_name("Non-existent Library")

        assert found_library is None

    def test_find_by_name_multiple_libraries(self):
        """Test finding by name with multiple libraries"""
        library1 = Library(name="Library A")
        library2 = Library(name="Library B")

        self.repository.save(library1)
        self.repository.save(library2)

        found_library = self.repository.find_by_name("Library A")

        assert found_library is not None
        assert found_library.id == library1.id

    def test_thread_safety_concurrent_writes(self):
        """Test thread safety with concurrent writes"""
        libraries = []

        def create_and_save_library(index):
            library = Library(name=f"Library {index}")
            saved_library = self.repository.save(library)
            libraries.append(saved_library)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = Thread(target=create_and_save_library, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all libraries were saved
        assert len(libraries) == 10
        all_saved_libraries = self.repository.find_all()
        assert len(all_saved_libraries) == 10

    def test_thread_safety_concurrent_reads_writes(self):
        """Test thread safety with concurrent reads and writes"""
        library = Library(name="Test Library")
        self.repository.save(library)

        read_results = []
        write_results = []

        def read_library():
            for _ in range(5):
                result = self.repository.find_by_id(library.id)
                read_results.append(result is not None)

        def write_library():
            for i in range(5):
                updated_library = library.model_copy(update={'name': f'Updated {i}'})
                result = self.repository.save(updated_library)
                write_results.append(result is not None)

        # Create threads
        read_thread = Thread(target=read_library)
        write_thread = Thread(target=write_library)

        # Start threads
        read_thread.start()
        write_thread.start()

        # Wait for completion
        read_thread.join()
        write_thread.join()

        # Verify results
        assert len(read_results) == 5
        assert len(write_results) == 5
        assert all(read_results)  # All reads should succeed
        assert all(write_results)  # All writes should succeed

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_save_operation(self, mock_logger):
        """Test that save operations are logged"""
        library = Library(name="Test Library")

        self.repository.save(library)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Library saved" in call_args[0]

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_delete_operation(self, mock_logger):
        """Test that delete operations are logged"""
        library = Library(name="Test Library")
        self.repository.save(library)

        self.repository.delete(library.id)

        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        assert "Library deleted" in call_args[0]

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_delete_non_existent(self, mock_logger):
        """Test that delete operations on non-existent libraries are logged"""
        non_existent_id = str(uuid4())

        self.repository.delete(non_existent_id)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Library not found" in call_args[0]


class TestRepositoryBasedDocumentRepository:
    """Test cases for RepositoryBasedDocumentRepository"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_repository = InMemoryLibraryRepository()
        self.document_repository = RepositoryBasedDocumentRepository(self.library_repository)

    def test_find_by_id_existing_document(self):
        """Test finding an existing document by ID"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        document = Document(library_id=library.id)
        library = library.add_document(document)
        self.library_repository.save(library)

        found_document = self.document_repository.find_by_id(library.id, document.id)

        assert found_document is not None
        assert found_document.id == document.id
        assert found_document.library_id == library.id

    def test_find_by_id_library_not_found(self):
        """Test finding document when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        document_id = str(uuid4())

        found_document = self.document_repository.find_by_id(non_existent_library_id, document_id)

        assert found_document is None

    def test_find_by_id_document_not_found(self):
        """Test finding a non-existent document"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        non_existent_document_id = str(uuid4())
        found_document = self.document_repository.find_by_id(library.id, non_existent_document_id)

        assert found_document is None

    def test_find_all_in_library_existing(self):
        """Test finding all documents in an existing library"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        # Add documents
        doc1 = Document(library_id=library.id)
        doc2 = Document(library_id=library.id)
        library = library.add_document(doc1)
        library = library.add_document(doc2)
        self.library_repository.save(library)

        documents = self.document_repository.find_all_in_library(library.id)

        assert len(documents) == 2
        document_ids = [doc.id for doc in documents]
        assert doc1.id in document_ids
        assert doc2.id in document_ids

    def test_find_all_in_library_not_found(self):
        """Test finding documents in non-existent library"""
        non_existent_library_id = str(uuid4())

        documents = self.document_repository.find_all_in_library(non_existent_library_id)

        assert documents == []

    def test_find_all_in_library_empty(self):
        """Test finding documents in empty library"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        documents = self.document_repository.find_all_in_library(library.id)

        assert documents == []


class TestRepositoryBasedChunkRepository:
    """Test cases for RepositoryBasedChunkRepository"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_repository = InMemoryLibraryRepository()
        self.chunk_repository = RepositoryBasedChunkRepository(self.library_repository)

    @patch('src.vector_db.infrastructure.cohere_client.co')
    def test_find_by_id_existing_chunk(self, mock_co):
        """Test finding an existing chunk by ID"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] * 307]
        mock_co.embed.return_value = mock_response

        library = Library(name="Test Library")
        self.library_repository.save(library)

        document_service = get_document_service()
        document = document_service.create_document(library.id, "Test content")
        library = library.add_document(document)
        self.library_repository.save(library)

        chunk = document.chunks[0]
        found_chunk = self.chunk_repository.find_by_id(library.id, chunk.id)

        assert found_chunk is not None
        assert found_chunk.id == chunk.id
        assert found_chunk.document_id == document.id

    def test_find_by_id_library_not_found(self):
        """Test finding chunk when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        chunk_id = str(uuid4())

        found_chunk = self.chunk_repository.find_by_id(non_existent_library_id, chunk_id)

        assert found_chunk is None

    def test_find_by_id_chunk_not_found(self):
        """Test finding a non-existent chunk"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        non_existent_chunk_id = str(uuid4())
        found_chunk = self.chunk_repository.find_by_id(library.id, non_existent_chunk_id)

        assert found_chunk is None

    @patch('src.vector_db.infrastructure.cohere_client.co')
    def test_find_all_in_library_existing(self, mock_co):
        """Test finding all chunks in an existing library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] * 153]
        mock_co.embed.return_value = mock_response

        library = Library(name="Test Library")
        self.library_repository.save(library)

        # Add documents with content
        doc1 = Document(library_id=library.id)
        document_service = get_document_service()
        doc1 = document_service.update_document_content(doc1, "First document content")
        doc2 = Document(library_id=library.id)
        doc2 = document_service.update_document_content(doc2, "Second document content")

        library = library.add_document(doc1)
        library = library.add_document(doc2)
        self.library_repository.save(library)

        chunks = self.chunk_repository.find_all_in_library(library.id)

        expected_count = len(doc1.chunks) + len(doc2.chunks)
        assert len(chunks) == expected_count
        assert all(chunk.document_id in [doc1.id, doc2.id] for chunk in chunks)

    def test_find_all_in_library_not_found(self):
        """Test finding chunks in non-existent library"""
        non_existent_library_id = str(uuid4())

        chunks = self.chunk_repository.find_all_in_library(non_existent_library_id)

        assert chunks == []

    def test_find_all_in_library_empty(self):
        """Test finding chunks in empty library"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        chunks = self.chunk_repository.find_all_in_library(library.id)

        assert chunks == []

    @patch('src.vector_db.infrastructure.cohere_client.co')
    def test_find_all_in_document_existing(self, mock_co):
        """Test finding all chunks in an existing document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] * 153]
        mock_co.embed.return_value = mock_response

        library = Library(name="Test Library")
        self.library_repository.save(library)

        document_service = get_document_service()
        document = document_service.create_document(library.id, "Test content that will be chunked")
        library = library.add_document(document)
        self.library_repository.save(library)

        chunks = self.chunk_repository.find_all_in_document(library.id, document.id)

        assert len(chunks) == len(document.chunks)
        assert all(chunk.document_id == document.id for chunk in chunks)

    def test_find_all_in_document_library_not_found(self):
        """Test finding chunks in document when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        document_id = str(uuid4())

        chunks = self.chunk_repository.find_all_in_document(non_existent_library_id, document_id)

        assert chunks == []

    def test_find_all_in_document_document_not_found(self):
        """Test finding chunks in non-existent document"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        non_existent_document_id = str(uuid4())
        chunks = self.chunk_repository.find_all_in_document(library.id, non_existent_document_id)

        assert chunks == []

    def test_find_all_in_document_empty_document(self):
        """Test finding chunks in empty document"""
        library = Library(name="Test Library")
        self.library_repository.save(library)

        document = Document(library_id=library.id)  # Empty document
        library = library.add_document(document)
        self.library_repository.save(library)

        chunks = self.chunk_repository.find_all_in_document(library.id, document.id)

        assert chunks == []