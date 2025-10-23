"""Unit tests for infrastructure repositories"""

import pytest
from unittest.mock import patch
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
                time.sleep(0.001)  # Small delay
        
        def write_library():
            for i in range(5):
                updated_library = library.model_copy(update={'name': f'Updated {i}'})
                result = self.repository.save(updated_library)
                write_results.append(result is not None)
                time.sleep(0.001)  # Small delay
        
        # Create threads
        read_thread = Thread(target=read_library)
        write_thread = Thread(target=write_library)
        
        # Start threads
        read_thread.start()
        write_thread.start()
        
        # Wait for completion
        read_thread.join()
        write_thread.join()
        
        # Verify operations completed successfully
        assert all(read_results)
        assert all(write_results)

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_save_operation(self, mock_logger):
        """Test that save operations are logged"""
        library = Library(name="Test Library")
        
        self.repository.save(library)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[1]
        assert call_args['library_id'] == library.id
        assert call_args['library_name'] == library.name
        assert call_args['is_update'] is False

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_delete_operation(self, mock_logger):
        """Test that delete operations are logged"""
        library = Library(name="Test Library")
        self.repository.save(library)
        
        self.repository.delete(library.id)
        
        # Should have logged both save and delete
        assert mock_logger.info.call_count == 2
        
        # Check the delete log call
        delete_call = mock_logger.info.call_args_list[1]
        call_args = delete_call[1]
        assert call_args['library_id'] == library.id
        assert call_args['library_name'] == library.name

    @patch('src.vector_db.infrastructure.repository.LoggerMixin.logger')
    def test_logging_delete_non_existent(self, mock_logger):
        """Test that attempting to delete non-existent library is logged"""
        non_existent_id = str(uuid4())
        
        self.repository.delete(non_existent_id)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[1]
        assert call_args['library_id'] == non_existent_id


class TestRepositoryBasedDocumentRepository:
    """Test cases for RepositoryBasedDocumentRepository"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_repo = InMemoryLibraryRepository()
        self.document_repo = RepositoryBasedDocumentRepository(self.library_repo)

    def test_find_by_id_existing_document(self):
        """Test finding an existing document by ID"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        library = library.add_document(document)
        
        self.library_repo.save(library)
        
        found_document = self.document_repo.find_by_id(library.id, document.id)
        
        assert found_document is not None
        assert found_document.id == document.id

    def test_find_by_id_library_not_found(self):
        """Test finding document when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        non_existent_document_id = str(uuid4())
        
        found_document = self.document_repo.find_by_id(
            non_existent_library_id, 
            non_existent_document_id
        )
        
        assert found_document is None

    def test_find_by_id_document_not_found(self):
        """Test finding non-existent document in existing library"""
        library = Library(name="Test Library")
        self.library_repo.save(library)
        
        non_existent_document_id = str(uuid4())
        
        found_document = self.document_repo.find_by_id(
            library.id, 
            non_existent_document_id
        )
        
        assert found_document is None

    def test_find_all_in_library_existing(self):
        """Test finding all documents in an existing library"""
        library = Library(name="Test Library")
        document1 = Document(library_id=library.id)
        document2 = Document(library_id=library.id)
        
        library = library.add_document(document1)
        library = library.add_document(document2)
        self.library_repo.save(library)
        
        documents = self.document_repo.find_all_in_library(library.id)
        
        assert len(documents) == 2
        document_ids = [doc.id for doc in documents]
        assert document1.id in document_ids
        assert document2.id in document_ids

    def test_find_all_in_library_not_found(self):
        """Test finding all documents in a non-existent library"""
        non_existent_library_id = str(uuid4())
        
        documents = self.document_repo.find_all_in_library(non_existent_library_id)
        
        assert documents == []

    def test_find_all_in_library_empty(self):
        """Test finding all documents in an empty library"""
        library = Library(name="Empty Library")
        self.library_repo.save(library)
        
        documents = self.document_repo.find_all_in_library(library.id)
        
        assert documents == []


class TestRepositoryBasedChunkRepository:
    """Test cases for RepositoryBasedChunkRepository"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_repo = InMemoryLibraryRepository()
        self.chunk_repo = RepositoryBasedChunkRepository(self.library_repo)

    def test_find_by_id_existing_chunk(self):
        """Test finding an existing chunk by ID"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Test content for chunking")
        library = library.add_document(document)
        
        self.library_repo.save(library)
        
        # Get first chunk
        chunk = document.chunks[0]
        found_chunk = self.chunk_repo.find_by_id(library.id, chunk.id)
        
        assert found_chunk is not None
        assert found_chunk.id == chunk.id

    def test_find_by_id_library_not_found(self):
        """Test finding chunk when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        non_existent_chunk_id = str(uuid4())
        
        found_chunk = self.chunk_repo.find_by_id(
            non_existent_library_id, 
            non_existent_chunk_id
        )
        
        assert found_chunk is None

    def test_find_by_id_chunk_not_found(self):
        """Test finding non-existent chunk in existing library"""
        library = Library(name="Test Library")
        self.library_repo.save(library)
        
        non_existent_chunk_id = str(uuid4())
        
        found_chunk = self.chunk_repo.find_by_id(
            library.id, 
            non_existent_chunk_id
        )
        
        assert found_chunk is None

    def test_find_all_in_library_existing(self):
        """Test finding all chunks in an existing library"""
        library = Library(name="Test Library")
        document1 = Document(library_id=library.id)
        document1 = document1.replace_content("Content for document 1")
        document2 = Document(library_id=library.id)
        document2 = document2.replace_content("Content for document 2")
        
        library = library.add_document(document1)
        library = library.add_document(document2)
        self.library_repo.save(library)
        
        chunks = self.chunk_repo.find_all_in_library(library.id)
        
        expected_count = len(document1.chunks) + len(document2.chunks)
        assert len(chunks) == expected_count

    def test_find_all_in_library_not_found(self):
        """Test finding all chunks in a non-existent library"""
        non_existent_library_id = str(uuid4())
        
        chunks = self.chunk_repo.find_all_in_library(non_existent_library_id)
        
        assert chunks == []

    def test_find_all_in_library_empty(self):
        """Test finding all chunks in an empty library"""
        library = Library(name="Empty Library")
        self.library_repo.save(library)
        
        chunks = self.chunk_repo.find_all_in_library(library.id)
        
        assert chunks == []

    def test_find_all_in_document_existing(self):
        """Test finding all chunks in an existing document"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Test content for chunking into multiple pieces")
        library = library.add_document(document)
        
        self.library_repo.save(library)
        
        chunks = self.chunk_repo.find_all_in_document(library.id, document.id)
        
        assert len(chunks) == len(document.chunks)
        assert all(chunk.document_id == document.id for chunk in chunks)

    def test_find_all_in_document_library_not_found(self):
        """Test finding all chunks in document when library doesn't exist"""
        non_existent_library_id = str(uuid4())
        non_existent_document_id = str(uuid4())
        
        chunks = self.chunk_repo.find_all_in_document(
            non_existent_library_id, 
            non_existent_document_id
        )
        
        assert chunks == []

    def test_find_all_in_document_document_not_found(self):
        """Test finding all chunks in non-existent document"""
        library = Library(name="Test Library")
        self.library_repo.save(library)
        
        non_existent_document_id = str(uuid4())
        
        chunks = self.chunk_repo.find_all_in_document(
            library.id, 
            non_existent_document_id
        )
        
        assert chunks == []

    def test_find_all_in_document_empty_document(self):
        """Test finding all chunks in an empty document"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)  # Empty document
        library = library.add_document(document)
        
        self.library_repo.save(library)
        
        chunks = self.chunk_repo.find_all_in_document(library.id, document.id)
        
        assert chunks == []