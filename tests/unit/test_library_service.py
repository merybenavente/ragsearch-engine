"""Unit tests for LibraryService"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.api.dependencies import get_library_service, get_document_service
from src.vector_db.domain.models import Document, Library


class TestLibraryService:
    """Test cases for library service functions"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_service = get_library_service()

    def test_create_library(self):
        """Test creating a library"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1", "tag2"]

        library = self.library_service.create_library(
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

        library = self.library_service.create_library(name=name)

        assert library.name == name
        assert library.metadata.username is None
        assert library.metadata.tags == []

    def test_add_document_to_library(self):
        """Test adding a document to a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        updated_library = self.library_service.add_document_to_library(
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
        library = self.library_service.add_document_to_library(
            library=library,
            document=document
        )

        # Try to add same document again
        with pytest.raises(ValueError, match="already exists"):
            self.library_service.add_document_to_library(
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
        updated_library = self.library_service.remove_document_from_library(
            library=library,
            document_id=document.id
        )

        assert len(updated_library.documents) == 0

    def test_remove_document_from_library_not_found(self):
        """Test removing a non-existent document from a library"""
        library = Library(name="Test Library")

        with pytest.raises(ValueError, match="not found"):
            self.library_service.remove_document_from_library(
                library=library,
                document_id="non-existent-id"
            )

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_update_document_in_library(self, mock_embed):
        """Test updating a document in a library"""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_embed.embed.return_value = mock_response

        library = Library(name="Test Library")
        document_service = get_document_service()
        document = document_service.create_document(library.id, "Original content")

        # Add document to library
        library = library.add_document(document)

        # Update document content
        updated_document = document_service.update_document_content(document, "Updated content")
        updated_library = self.library_service.update_document_in_library(
            library=library,
            updated_document=updated_document
        )

        # Check that document was updated
        assert len(updated_library.documents) == 1
        assert updated_library.documents[0].get_full_text() == "Updated content"

    def test_update_document_in_library_not_found(self):
        """Test updating a non-existent document in a library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)

        with pytest.raises(ValueError, match="not found"):
            self.library_service.update_document_in_library(
                library=library,
                updated_document=document
            )

    def test_update_library_metadata(self):
        """Test updating library metadata"""
        library = Library(name="Test Library")
        new_name = "Updated Library"
        new_tags = ["new_tag1", "new_tag2"]

        updated_library = self.library_service.update_library_metadata(
            library=library,
            name=new_name,
            tags=new_tags
        )

        assert updated_library.name == new_name
        assert updated_library.metadata.tags == new_tags

    def test_update_library_metadata_partial(self):
        """Test updating library metadata with partial updates"""
        library = Library(name="Test Library")
        new_tags = ["new_tag"]

        updated_library = self.library_service.update_library_metadata(
            library=library,
            tags=new_tags
        )

        assert updated_library.name == library.name  # Unchanged
        assert updated_library.metadata.tags == new_tags

    @patch('src.vector_db.application.services.logger')
    def test_logging_library_creation(self, mock_logger):
        """Test that library creation is logged"""
        name = "Test Library"
        username = "testuser"
        tags = ["tag1"]

        self.library_service.create_library(
            name=name,
            username=username,
            tags=tags
        )

        mock_logger.info.assert_called_once()