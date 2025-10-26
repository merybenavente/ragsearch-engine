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

    @pytest.mark.parametrize("text,username,tags,chunk_size,expected_username,expected_tags", [
        ("This is a test document with some content.", "testuser", ["tag1", "tag2"], 20, "testuser", ["tag1", "tag2"]),
        ("Test content", None, None, None, None, []),
        ("Another test document.", "user2", ["tag"], 30, "user2", ["tag"]),
        ("Short text", None, ["only_tag"], None, None, ["only_tag"]),
    ])
    def test_create_document_with_content(self, mock_cohere_embed, text, username, tags, chunk_size, expected_username, expected_tags):
        """Test creating a document with various content and metadata combinations"""
        kwargs = {"library_id": self.library_id, "text": text}
        if username is not None:
            kwargs["username"] = username
        if tags is not None:
            kwargs["tags"] = tags
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size

        document = self.document_service.create_document(**kwargs)

        assert document.library_id == self.library_id
        assert document.has_content()
        if len(text) > 20:  # Should be chunked if text is long enough
            assert len(document.chunks) >= 1
        assert document.metadata.username == expected_username
        assert document.metadata.tags == expected_tags

    @pytest.mark.parametrize("username,tags,expected_username,expected_tags", [
        ("testuser", ["tag1"], "testuser", ["tag1"]),
        (None, None, None, []),
        ("user2", ["tag1", "tag2"], "user2", ["tag1", "tag2"]),
        (None, ["only_tag"], None, ["only_tag"]),
    ])
    def test_create_empty_document(self, username, tags, expected_username, expected_tags):
        """Test creating an empty document with various metadata combinations"""
        kwargs = {"library_id": self.library_id}
        if username is not None:
            kwargs["username"] = username
        if tags is not None:
            kwargs["tags"] = tags

        document = self.document_service.create_empty_document(**kwargs)

        assert document.library_id == self.library_id
        assert not document.has_content()
        assert document.chunks == []
        assert document.metadata.username == expected_username
        assert document.metadata.tags == expected_tags

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
        
        # Verify log content contains key information
        call_args = mock_logger.info.call_args
        log_message = call_args[0][0]  # First positional argument
        assert "Document created" in log_message