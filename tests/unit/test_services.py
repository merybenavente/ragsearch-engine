"""Unit tests for application services"""

import pytest
from unittest.mock import patch
from uuid import uuid4

from src.vector_db.application.services import DocumentService, LibraryService, ChunkService
from src.vector_db.domain.models import (
    Document, Library, Chunk, Metadata
)


class TestDocumentService:
    """Test cases for DocumentService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())

    def test_create_document_with_content(self):
        """Test creating a document with content"""
        text = "This is a test document with some content."
        username = "testuser"
        tags = ["tag1", "tag2"]
        chunk_size = 20

        document = DocumentService.create_document(
            library_id=self.library_id,
            text=text,
            username=username,
            tags=tags,
            chunk_size=chunk_size
        )

        assert document.library_id == self.library_id
        assert document.has_content()
        assert len(document.chunks) > 1  # Should be chunked
        assert document.metadata.username == username
        assert document.metadata.tags == tags

    def test_create_document_with_defaults(self):
        """Test creating a document with default values"""
        text = "Test content"

        document = DocumentService.create_document(
            library_id=self.library_id,
            text=text
        )

        assert document.library_id == self.library_id
        assert document.has_content()
        assert document.metadata.username is None
        assert document.metadata.tags == []

    def test_create_empty_document(self):
        """Test creating an empty document"""
        username = "testuser"
        tags = ["tag1"]

        document = DocumentService.create_empty_document(
            library_id=self.library_id,
            username=username,
            tags=tags
        )

        assert document.library_id == self.library_id
        assert not document.has_content()
        assert document.chunks == []
        assert document.metadata.username == username
        assert document.metadata.tags == tags

    def test_create_empty_document_with_defaults(self):
        """Test creating an empty document with default values"""
        document = DocumentService.create_empty_document(
            library_id=self.library_id
        )

        assert document.library_id == self.library_id
        assert not document.has_content()
        assert document.metadata.username is None
        assert document.metadata.tags == []

    def test_update_document_content(self):
        """Test updating document content"""
        # Create initial document
        document = Document(library_id=self.library_id)
        document = document.replace_content("Original content")

        # Update content
        new_text = "Updated content that is much longer than the original content."
        updated_document = DocumentService.update_document_content(
            document=document,
            new_text=new_text,
            chunk_size=20
        )

        # Check that content was updated
        assert len(updated_document.chunks) > 1  # Should be chunked
        assert new_text == updated_document.get_full_text()

    @patch('src.vector_db.application.services.logger')
    def test_logging_document_creation(self, mock_logger):
        """Test that document creation is logged"""
        text = "Test content"
        username = "testuser"
        tags = ["tag1"]

        DocumentService.create_document(
            library_id=self.library_id,
            text=text,
            username=username,
            tags=tags
        )

        mock_logger.info.assert_called_once()
        # Check that logging was called with the expected message
        call_args = mock_logger.info.call_args
        assert "Document created" in call_args[0]


class TestLibraryService:
    """Test cases for library service functions"""

    def test_create_library(self):
        """Test creating a library"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1", "tag2"]

        library = LibraryService.create_library(
            name=name,
            username=username,
            tags=tags
        )

        assert library.name == name
        assert library.metadata.username == username
        assert library.metadata.tags == tags
        assert library.documents == []

    def test_create_library_with_defaults(self):
        """Test creating a library with default values"""
        name = "Test Library"

        library = LibraryService.create_library(name=name)

        assert library.name == name
        assert library.metadata.username is None
        assert library.metadata.tags == []

    def test_add_document_to_library(self):
        """Test adding a document to a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        updated_library = LibraryService.add_document_to_library(
            library=library,
            document=document
        )

        assert len(updated_library.documents) == 1
        assert updated_library.documents[0].id == document.id

    def test_add_document_to_library_duplicate(self):
        """Test adding a duplicate document to a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        # Add document first time
        library = LibraryService.add_document_to_library(
            library=library,
            document=document
        )

        # Try to add same document again
        with pytest.raises(ValueError, match="already exists"):
            LibraryService.add_document_to_library(
                library=library,
                document=document
            )

    def test_remove_document_from_library(self):
        """Test removing a document from a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        # Add document
        library = library.add_document(document)
        assert len(library.documents) == 1

        # Remove document
        updated_library = LibraryService.remove_document_from_library(
            library=library,
            document_id=document.id
        )

        assert len(updated_library.documents) == 0

    def test_remove_document_from_library_not_found(self):
        """Test removing a non-existent document from a library"""
        library = Library(name="Test Library")
        non_existent_id = str(uuid4())

        with pytest.raises(ValueError, match="not found"):
            LibraryService.remove_document_from_library(
                library=library,
                document_id=non_existent_id
            )

    def test_update_document_in_library(self):
        """Test updating a document in a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        # Add document
        library = library.add_document(document)

        # Update document
        updated_document = document.replace_content("New content")
        updated_library = LibraryService.update_document_in_library(
            library=library,
            updated_document=updated_document
        )

        # Verify update
        retrieved_doc = updated_library.get_document_by_id(document.id)
        assert retrieved_doc is not None
        assert retrieved_doc.has_content()

    def test_update_document_in_library_not_found(self):
        """Test updating a non-existent document in a library"""
        library = Library(name="Test Library")
        non_existent_doc = Document(library_id=library.id)

        with pytest.raises(ValueError, match="not found"):
            LibraryService.update_document_in_library(
                library=library,
                updated_document=non_existent_doc
            )

    def test_update_library_metadata(self):
        """Test updating library metadata"""
        library = Library(name="Original Name")

        new_name = "Updated Name"
        new_tags = ["new_tag1", "new_tag2"]

        updated_library = LibraryService.update_library_metadata(
            library=library,
            name=new_name,
            tags=new_tags
        )

        assert updated_library.name == new_name
        assert updated_library.metadata.tags == new_tags
        assert updated_library.metadata.last_update > library.metadata.last_update

    def test_update_library_metadata_partial(self):
        """Test updating library metadata partially"""
        library = Library(name="Original Name")
        original_name = library.name

        new_tags = ["new_tag"]

        updated_library = LibraryService.update_library_metadata(
            library=library,
            tags=new_tags
        )

        assert updated_library.name == original_name  # Should remain unchanged
        assert updated_library.metadata.tags == new_tags

    @patch('src.vector_db.application.services.logger')
    def test_logging_library_creation(self, mock_logger):
        """Test that library creation is logged"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1"]

        LibraryService.create_library(
            name=name,
            username=username,
            tags=tags
        )

        mock_logger.info.assert_called_once()
        # Check that logging was called with the expected message
        call_args = mock_logger.info.call_args
        assert "Library created" in call_args[0]


class TestChunkService:
    """Test cases for chunk service functions"""

    def setup_method(self):
        """Setup test fixtures"""
        pass

    def test_get_chunks_from_library(self):
        """Test getting chunks from a library"""
        library = Library(name="Test Library")

        # Add documents with content
        doc1 = Document(library_id=library.id).replace_content("Document 1 content")
        doc2 = Document(library_id=library.id).replace_content("Document 2 content")

        library = library.add_document(doc1)
        library = library.add_document(doc2)

        chunks = ChunkService.get_chunks_from_library(library)

        expected_count = len(doc1.chunks) + len(doc2.chunks)
        assert len(chunks) == expected_count

    def test_get_chunks_from_library_empty(self):
        """Test getting chunks from an empty library"""
        library = Library(name="Empty Library")

        chunks = ChunkService.get_chunks_from_library(library)

        assert chunks == []

    def test_get_chunks_from_document(self):
        """Test getting chunks from a document"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        document = document.replace_content("Test content for chunking")

        chunks = ChunkService.get_chunks_from_document(document)

        assert len(chunks) == len(document.chunks)
        assert all(chunk.document_id == document.id for chunk in chunks)

    def test_get_chunks_from_document_empty(self):
        """Test getting chunks from an empty document"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)

        chunks = ChunkService.get_chunks_from_document(document)

        assert chunks == []

    def test_get_chunk_from_library(self):
        """Test getting a specific chunk from a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Test content")
        library = library.add_document(document)

        # Get first chunk
        expected_chunk = document.chunks[0]
        found_chunk = ChunkService.get_chunk_from_library(
            library=library,
            chunk_id=expected_chunk.id
        )

        assert found_chunk is not None
        assert found_chunk.id == expected_chunk.id

    def test_get_chunk_from_library_not_found(self):
        """Test getting a non-existent chunk from a library"""
        library = Library(name="Test Library")
        non_existent_id = str(uuid4())

        found_chunk = ChunkService.get_chunk_from_library(
            library=library,
            chunk_id=non_existent_id
        )

        assert found_chunk is None

    def test_get_chunk_from_document(self):
        """Test getting a specific chunk from a document"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        document = document.replace_content("Test content")

        # Get first chunk
        expected_chunk = document.chunks[0]
        found_chunk = ChunkService.get_chunk_from_document(
            document=document,
            chunk_id=expected_chunk.id
        )

        assert found_chunk is not None
        assert found_chunk.id == expected_chunk.id

    def test_get_chunk_from_document_not_found(self):
        """Test getting a non-existent chunk from a document"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        document = document.replace_content("Test content")
        non_existent_id = str(uuid4())

        found_chunk = ChunkService.get_chunk_from_document(
            document=document,
            chunk_id=non_existent_id
        )

        assert found_chunk is None

    @patch('src.vector_db.application.services.logger')
    def test_logging_chunk_retrieval(self, mock_logger):
        """Test that chunk retrieval is logged"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Test content")
        library = library.add_document(document)

        ChunkService.get_chunks_from_library(library)

        mock_logger.debug.assert_called_once()
        # Check that logging was called with the expected message
        call_args = mock_logger.debug.call_args
        assert "Retrieved chunks from library" in call_args[0]