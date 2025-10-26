"""Unit tests for DocumentService"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.api.dependencies import get_document_service
from src.vector_db.domain.models import Document


class TestDocumentService:
    """Test cases for DocumentService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())
        self.document_service = get_document_service()

    def test_create_document_with_content(self, mock_cohere_embed):
        """Test creating a document with content"""

        text = "This is a test document with some content."
        username = "testuser"
        tags = ["tag1", "tag2"]
        chunk_size = 20

        document = self.document_service.create_document(
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

    def test_create_document_with_defaults(self, mock_cohere_embed):
        """Test creating a document with default values"""

        text = "Test content"

        document = self.document_service.create_document(
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

        document = self.document_service.create_empty_document(
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
        document = self.document_service.create_empty_document(
            library_id=self.library_id
        )

        assert document.library_id == self.library_id
        assert not document.has_content()
        assert document.metadata.username is None
        assert document.metadata.tags == []

    def test_update_document_content(self, mock_cohere_embed):
        """Test updating document content"""

        # Create initial document
        document = self.document_service.create_document(self.library_id, "Original content")

        # Update content
        new_text = "Updated content that is much longer than the original content."
        updated_document = self.document_service.update_document_content(
            document=document,
            new_text=new_text,
            chunk_size=20
        )

        # Check that content was updated
        assert len(updated_document.chunks) > 1  # Should be chunked
        assert new_text == updated_document.get_full_text()

    @patch('src.vector_db.infrastructure.embedding_service.co')
    @patch('src.vector_db.application.services.logger')
    def test_logging_document_creation(self, mock_logger, mock_co):
        """Test that document creation is logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        text = "Test content"
        username = "testuser"
        tags = ["tag1"]

        self.document_service.create_document(
            library_id=self.library_id,
            text=text,
            username=username,
            tags=tags
        )

        mock_logger.info.assert_called_once()