"""Unit tests for ChunkService"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.api.dependencies import get_chunk_service, get_document_service
from src.vector_db.domain.models import Document, Library, Chunk


class TestChunkService:
    """Test cases for ChunkService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())
        self.chunk_service = get_chunk_service()

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_chunks_from_library(self, mock_co):
        """Test getting chunks from a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a library with documents
        library = Library(name="Test Library")
        document_service = get_document_service()
        doc1 = document_service.create_document(library.id, "First document content.")
        doc2 = document_service.create_document(library.id, "Second document content.")
        library = library.add_document(doc1)
        library = library.add_document(doc2)

        chunks = self.chunk_service.get_chunks_from_library(library)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_get_chunks_from_library_empty(self):
        """Test getting chunks from an empty library"""
        library = Library(name="Test Library")

        chunks = self.chunk_service.get_chunks_from_library(library)

        assert chunks == []

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_chunks_from_document(self, mock_co):
        """Test getting chunks from a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "This is a test document with some content.")

        chunks = self.chunk_service.get_chunks_from_document(document)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_get_chunks_from_document_empty(self):
        """Test getting chunks from an empty document"""
        document = Document(library_id=self.library_id)

        chunks = self.chunk_service.get_chunks_from_document(document)

        assert chunks == []

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_chunk_from_library(self, mock_co):
        """Test getting a specific chunk from a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a library with a document
        library = Library(name="Test Library")
        document_service = get_document_service()
        document = document_service.create_document(library.id, "Test content.")
        library = library.add_document(document)

        # Get the first chunk
        chunk_id = document.chunks[0].id
        chunk = self.chunk_service.get_chunk_from_library(library, chunk_id)

        assert isinstance(chunk, Chunk)
        assert chunk.id == chunk_id

    def test_get_chunk_from_library_not_found(self):
        """Test getting a non-existent chunk from a library"""
        library = Library(name="Test Library")

        with pytest.raises(ValueError, match="not found"):
            self.chunk_service.get_chunk_from_library(library, "non-existent-id")

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_chunk_from_document(self, mock_embed):
        """Test getting a specific chunk from a document"""
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_embed.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "Test content.")

        # Get the first chunk
        chunk_id = document.chunks[0].id
        chunk = self.chunk_service.get_chunk_from_document(document, chunk_id)

        assert isinstance(chunk, Chunk)
        assert chunk.id == chunk_id

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_get_chunk_from_document_not_found(self, mock_co):
        """Test getting a non-existent chunk from a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "Test content.")

        with pytest.raises(ValueError, match="not found"):
            self.chunk_service.get_chunk_from_document(document, "non-existent-id")

    @patch('src.vector_db.infrastructure.embedding_service.co')
    @patch('src.vector_db.application.services.logger')
    def test_logging_chunk_retrieval(self, mock_logger, mock_co):
        """Test that chunk retrieval is logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "Test content.")

        chunk_id = document.chunks[0].id
        self.chunk_service.get_chunk_from_document(document, chunk_id)

        # Check for the chunk retrieval log specifically (multiple logs expected)
        chunk_retrieval_calls = [call for call in mock_logger.info.call_args_list 
                               if "Chunk retrieved" in str(call)]
        assert len(chunk_retrieval_calls) == 1