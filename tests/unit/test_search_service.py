"""Unit tests for SearchService"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4
import numpy as np

from src.vector_db.application.services import SearchService
from src.vector_db.domain.models import Document, Library, Chunk


class TestSearchService:
    """Test cases for SearchService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())

    @patch('src.vector_db.application.services.co')
    def test_create_query_embedding(self, mock_co):
        """Test creating query embedding"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_co.embed.return_value = mock_response

        query_text = "test query"
        embedding = SearchService._create_query_embedding(query_text)

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

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_in_document(self, mock_services_co, mock_domain_co):
        """Test searching chunks in a document with text query"""
        # Mock the Cohere API response for both domain and services
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document = document.replace_content("This is a test document about machine learning and AI.")

        query_text = "machine learning"
        results = SearchService.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=5,
            min_similarity=0.0
        )

        # Verify Cohere API was called
        mock_services_co.embed.assert_called_once()

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

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_in_document_empty_document(self, mock_services_co, mock_domain_co):
        """Test searching chunks in an empty document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create an empty document
        document = Document(library_id=self.library_id)

        query_text = "test query"
        results = SearchService.search_chunks_in_document(
            document=document,
            query_text=query_text
        )

        # Verify no results returned
        assert results == []

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_in_document_with_min_similarity(self, mock_services_co, mock_domain_co):
        """Test searching chunks with minimum similarity threshold"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document = document.replace_content("This is a test document about machine learning and AI.")

        query_text = "machine learning"
        results = SearchService.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=10,
            min_similarity=0.8  # High threshold
        )

        # Verify results respect minimum similarity
        for chunk, similarity in results:
            assert similarity >= 0.8

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_in_document_k_limit(self, mock_services_co, mock_domain_co):
        """Test that search respects the k limit"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document = document.replace_content("This is a test document about machine learning and AI.")

        query_text = "machine learning"
        k = 2
        results = SearchService.search_chunks_in_document(
            document=document,
            query_text=query_text,
            k=k
        )

        # Verify results don't exceed k
        assert len(results) <= k

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_in_library(self, mock_services_co, mock_domain_co):
        """Test searching chunks in a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create a library with documents
        library = Library(name="Test Library")
        doc1 = Document(library_id=library.id)
        doc1 = doc1.replace_content("First document about machine learning.")
        doc2 = Document(library_id=library.id)
        doc2 = doc2.replace_content("Second document about artificial intelligence.")
        library = library.add_document(doc1)
        library = library.add_document(doc2)

        query_text = "machine learning"
        results = SearchService.search_chunks(
            library=library,
            query_text=query_text,
            k=5
        )

        # Verify Cohere API was called
        mock_services_co.embed.assert_called_once()

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        assert all(isinstance(result[0], Chunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_chunks_empty_library(self, mock_services_co, mock_domain_co):
        """Test searching chunks in an empty library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create an empty library
        library = Library(name="Test Library")

        query_text = "test query"
        results = SearchService.search_chunks(
            library=library,
            query_text=query_text
        )

        # Verify no results returned
        assert results == []

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = SearchService._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = SearchService._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # Test opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = SearchService._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001  # Should be 0 due to max(0, similarity)

        # Test zero vectors
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = SearchService._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_different_dimensions(self):
        """Test cosine similarity with different vector dimensions"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0]  # Different dimension

        similarity = SearchService._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    @patch('src.vector_db.application.services.logger')
    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.co')
    def test_search_logging(self, mock_services_co, mock_domain_co, mock_logger):
        """Test that search operations are logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [np.random.uniform(-1, 1, 1536).tolist()]
        mock_services_co.embed.return_value = mock_response
        mock_domain_co.embed.return_value = mock_response

        # Create a document with content
        document = Document(library_id=self.library_id)
        document = document.replace_content("This is a test document about machine learning.")

        query_text = "machine learning"
        SearchService.search_chunks_in_document(
            document=document,
            query_text=query_text
        )

        # Verify logging was called
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Document search completed" in call_args[0] 