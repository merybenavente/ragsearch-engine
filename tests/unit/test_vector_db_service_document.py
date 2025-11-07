"""Unit tests for VectorDBService document operations"""

import pytest
from uuid import uuid4

from src.vector_db.domain.models import Document, EMBEDDING_DIMENSION


class TestVectorDBServiceDocument:
    """Test cases for VectorDBService document operations"""

    def test_create_document(
        self, mock_cohere_deterministic, vector_db_service_instance
    ):
        """Test creating a document in a library"""
        # Create library first
        library = vector_db_service_instance.create_library("Test Library")

        # Create document
        text = "This is a test document with multiple sentences. It should be chunked appropriately."
        username = "testuser"
        tags = ["doc_tag"]
        chunk_size = 50

        document = vector_db_service_instance.create_document(
            library_id=library.id,
            text=text,
            username=username,
            tags=tags,
            chunk_size=chunk_size,
        )

        assert document.library_id == library.id
        assert document.metadata.username == username
        assert document.metadata.tags == tags
        assert len(document.chunks) > 0

        # Verify document is added to library
        updated_library = vector_db_service_instance.get_library(library.id)
        assert document.id in updated_library.document_ids

        # Verify chunks have embeddings
        for chunk in document.chunks:
            assert len(chunk.embedding) == EMBEDDING_DIMENSION
            assert chunk.document_id == document.id

    def test_create_document_library_not_found(self, vector_db_service_instance):
        """Test creating document in non-existent library raises error"""
        fake_library_id = str(uuid4())

        with pytest.raises(ValueError, match="Library .* not found"):
            vector_db_service_instance.create_document(
                library_id=fake_library_id, text="Test text"
            )

    def test_create_empty_document(self, vector_db_service_instance):
        """Test creating an empty document"""
        library = vector_db_service_instance.create_library(name="Test Library")
        document = vector_db_service_instance.create_empty_document(
            library_id=library.id, username="testuser", tags=["test", "empty"]
        )

        assert isinstance(document, Document)
        assert document.library_id == library.id
        assert document.metadata.username == "testuser"
        assert "test" in document.metadata.tags
        assert "empty" in document.metadata.tags
        assert len(document.chunks) == 0

    def test_get_document(self, mock_cohere_deterministic, vector_db_service_instance):
        """Test getting a specific document"""
        library = vector_db_service_instance.create_library("Test Library")
        document = vector_db_service_instance.create_document(
            library_id=library.id, text="Test document content"
        )

        retrieved_document = vector_db_service_instance.get_document(document.id)

        assert retrieved_document is not None
        assert retrieved_document.id == document.id
        assert retrieved_document.library_id == library.id

    def test_update_document_content(
        self, mock_cohere_deterministic, vector_db_service_instance
    ):
        """Test updating document content"""
        library = vector_db_service_instance.create_library(name="Test Library")
        document = vector_db_service_instance.create_document(
            library_id=library.id, text="Original content."
        )

        updated_document = vector_db_service_instance.update_document_content(
            document_id=document.id,
            new_text="Updated content about machine learning.",
            chunk_size=100,
        )

        assert isinstance(updated_document, Document)
        assert updated_document.id == document.id
        assert len(updated_document.chunks) > 0
        # Verify content was updated
        full_text = updated_document.get_full_text()
        assert "Updated content" in full_text

    def test_document_operations_with_invalid_ids(self, vector_db_service_instance):
        """Test document operations with invalid IDs"""
        fake_id = str(uuid4())

        # Test get document with invalid ID
        result = vector_db_service_instance.get_document(fake_id)
        assert result is None

        # Test update document with invalid ID
        with pytest.raises(ValueError):
            vector_db_service_instance.update_document_content(
                document_id=fake_id, new_text="Updated content"
            )

    def test_embedding_service_failure_handling(
        self, mock_embedding_service_failure, vector_db_service_instance
    ):
        """Test handling of embedding service failures"""
        library = vector_db_service_instance.create_library(name="Test Library")

        # Replace embedding service with failing one
        original_embedding_service = vector_db_service_instance.embedding_service
        vector_db_service_instance.embedding_service = mock_embedding_service_failure

        # Test that document creation fails gracefully
        with pytest.raises(RuntimeError):
            vector_db_service_instance.create_document(
                library_id=library.id, text="Test document"
            )

        # Restore original service
        vector_db_service_instance.embedding_service = original_embedding_service
