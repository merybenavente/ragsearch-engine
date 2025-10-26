"""Unit tests for SearchService integration"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.api.dependencies import get_search_service, get_document_service
from src.vector_db.domain.models import Document, Library, Chunk


class TestSearchService:
    """Test cases for SearchService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())
        self.search_service = get_search_service()

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_create_query_embedding(self, mock_co):
        """Test creating query embedding"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]  # 1536 dimensions
        mock_co.embed.return_value = mock_response

        query_text = "test query"
        embedding = self.search_service._create_query_embedding(query_text)

        # Verify Cohere API was called correctly
        mock_co.embed.assert_called_once_with(
            texts=[query_text],
            model="embed-v4.0",
            input_type="search_query",
            truncate="NONE"
        )

        # Verify embedding was returned
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_in_document(self, mock_co):
        """Test searching chunks in a document with text query"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]  # 1536 dimensions
        mock_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "This is a test document about machine learning and AI.")

        query_text = "machine learning"
        results = self.search_service.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=5,
            min_similarity=0.0
        )

        # Verify Cohere API was called (2 times: 1 for document + 1 for search query)
        assert mock_co.embed.call_count == 2

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Chunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)

        # Verify results are sorted by similarity (descending)
        similarities = [result[1] for result in results]
        assert similarities == sorted(similarities, reverse=True)

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_in_document_empty_document(self, mock_co):
        """Test searching chunks in an empty document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create an empty document
        document = Document(library_id=self.library_id)

        query_text = "test query"
        results = self.search_service.search_chunks_in_document(
            document=document,
            query_text=query_text
        )

        # Verify no results returned
        assert results == []

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_in_document_with_min_similarity(self, mock_co):
        """Test searching chunks with minimum similarity threshold"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "This is a test document about machine learning and AI.")

        query_text = "machine learning"
        results = self.search_service.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=10,
            min_similarity=0.8  # High threshold
        )

        # Verify results respect minimum similarity
        for chunk, similarity in results:
            assert similarity >= 0.8

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_in_document_k_limit(self, mock_co):
        """Test that search respects the k limit"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "This is a test document about machine learning and AI.")

        query_text = "machine learning"
        k = 2
        results = self.search_service.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=k
        )

        # Verify results don't exceed k
        assert len(results) <= k

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_in_library(self, mock_co):
        """Test searching chunks in a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a library with documents
        library = Library(name="Test Library")
        doc1 = Document(library_id=library.id)
        document_service = get_document_service()
        doc1 = document_service.update_document_content(doc1, "First document about machine learning.")
        doc2 = Document(library_id=library.id)
        doc2 = document_service.update_document_content(doc2, "Second document about artificial intelligence.")
        library = library.add_document(doc1)
        library = library.add_document(doc2)

        query_text = "machine learning"
        results = self.search_service.search_chunks(
            library=library,
            query_text=query_text,
            k=5
        )

        # Verify Cohere API was called (3 times: 2 for documents + 1 for search query)
        assert mock_co.embed.call_count == 3

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Chunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)

    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_chunks_empty_library(self, mock_co):
        """Test searching chunks in an empty library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create an empty library
        library = Library(name="Test Library")

        query_text = "test query"
        results = self.search_service.search_chunks(
            library=library,
            query_text=query_text
        )

        # Verify no results returned
        assert results == []

    @pytest.mark.parametrize("vec1,vec2,expected,tolerance,description", [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.001, "identical vectors"),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0, 0.001, "orthogonal vectors"),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 0.001, "opposite vectors"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0.0, "zero vectors"),
        ([1.0, 0.0, 0.0], [1.0, 0.0], 0.0, 0.0, "different dimensions"),
    ])
    def test_cosine_similarity(self, vec1, vec2, expected, tolerance, description):
        """Test cosine similarity calculation for various vector scenarios"""
        similarity = self.search_service._cosine_similarity(vec1, vec2)
        
        if tolerance > 0:
            assert abs(similarity - expected) < tolerance, f"Failed for {description}"
        else:
            assert similarity == expected, f"Failed for {description}"

    @patch('src.vector_db.application.services.logger')
    @patch('src.vector_db.infrastructure.embedding_service.co')
    def test_search_logging(self, mock_co, mock_logger):
        """Test that search operations are logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document_service = get_document_service()
        document = document_service.update_document_content(document, "This is a test document about machine learning.")

        query_text = "machine learning"
        self.search_service.search_chunks_in_document(
            document=document,
            query_text=query_text
        )

        # Verify logging was called for search operation
        # We expect multiple log calls (document creation + search), so check for the search log specifically
        search_calls = [call for call in mock_logger.info.call_args_list 
                       if "Document search completed" in str(call)]
        assert len(search_calls) == 1
        
        # Verify log content is about search completion
        call_args = search_calls[0]
        log_message = call_args[0][0]  # First positional argument
        assert "Document search completed" in log_message

    @patch('src.vector_db.application.services.logger')
    def test_logging_levels(self, mock_logger):
        """Test that appropriate log levels are used for different scenarios"""
        # Test info level for normal operations
        document = Document(library_id=self.library_id)
        
        try:
            self.search_service.search_chunks_in_document(document, "test query")
        except:
            pass  # We expect this to fail without proper setup
            
        # Should use info level for normal search operations
        assert mock_logger.info.called
        
        # Test that logger was used (verifies logging integration)
        assert mock_logger.call_count > 0