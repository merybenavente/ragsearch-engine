"""Unit tests for application services"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.vector_db.application.services import DocumentService, LibraryService, ChunkService, SearchService
from src.vector_db.domain.models import (
    Document, Library, Chunk
)


class TestDocumentService:
    """Test cases for DocumentService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())

    @pytest.fixture
    def mock_cohere_embed(self):
        with patch('src.vector_db.domain.models.co.embed') as mock:
            mock.return_value = MagicMock(embeddings=[[0.1] * 1536])
            yield mock

    def test_create_document_with_content(self, mock_cohere_embed):
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

    def test_create_document_with_defaults(self, mock_cohere_embed):
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

    def test_update_document_content(self, mock_cohere_embed):
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

    @patch('src.vector_db.domain.models.co')
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

        with pytest.raises(ValueError, match="not found"):
            LibraryService.remove_document_from_library(
                library=library,
                document_id="non-existent-id"
            )

    @patch('src.vector_db.domain.models.co.embed')
    def test_update_document_in_library(self, mock_embed):
        """Test updating a document in a library"""
        mock_embed.return_value = MagicMock(embeddings=[[0.1] * 1536])

        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Original content")

        # Add document to library
        library = library.add_document(document)

        # Update document content
        updated_document = document.replace_content("Updated content")
        updated_library = LibraryService.update_document_in_library(
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
            LibraryService.update_document_in_library(
                library=library,
                updated_document=document
            )

    def test_update_library_metadata(self):
        """Test updating library metadata"""
        library = Library(name="Test Library")
        new_name = "Updated Library"
        new_tags = ["new_tag1", "new_tag2"]

        updated_library = LibraryService.update_library_metadata(
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

        updated_library = LibraryService.update_library_metadata(
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

        LibraryService.create_library(
            name=name,
            username=username,
            tags=tags
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Library created" in call_args[0]


class TestChunkService:
    """Test cases for ChunkService"""

    def setup_method(self):
        """Setup test fixtures"""
        self.library_id = str(uuid4())

    @patch('src.vector_db.domain.models.co')
    def test_get_chunks_from_library(self, mock_co):
        """Test getting chunks from a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a library with documents
        library = Library(name="Test Library")
        doc1 = Document(library_id=library.id)
        doc1 = doc1.replace_content("First document content.")
        doc2 = Document(library_id=library.id)
        doc2 = doc2.replace_content("Second document content.")
        library = library.add_document(doc1)
        library = library.add_document(doc2)

        chunks = ChunkService.get_chunks_from_library(library)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_get_chunks_from_library_empty(self):
        """Test getting chunks from an empty library"""
        library = Library(name="Test Library")

        chunks = ChunkService.get_chunks_from_library(library)

        assert chunks == []

    @patch('src.vector_db.domain.models.co')
    def test_get_chunks_from_document(self, mock_co):
        """Test getting chunks from a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document = document.replace_content("This is a test document with some content.")

        chunks = ChunkService.get_chunks_from_document(document)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_get_chunks_from_document_empty(self):
        """Test getting chunks from an empty document"""
        document = Document(library_id=self.library_id)

        chunks = ChunkService.get_chunks_from_document(document)

        assert chunks == []

    @patch('src.vector_db.domain.models.co')
    def test_get_chunk_from_library(self, mock_co):
        """Test getting a specific chunk from a library"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        # Create a library with a document
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        document = document.replace_content("Test content.")
        library = library.add_document(document)

        # Get the first chunk
        chunk_id = document.chunks[0].id
        chunk = ChunkService.get_chunk_from_library(library, chunk_id)

        assert isinstance(chunk, Chunk)
        assert chunk.id == chunk_id

    def test_get_chunk_from_library_not_found(self):
        """Test getting a non-existent chunk from a library"""
        library = Library(name="Test Library")

        with pytest.raises(ValueError, match="not found"):
            ChunkService.get_chunk_from_library(library, "non-existent-id")

    @patch('src.vector_db.domain.models.co.embed')
    def test_get_chunk_from_document(self, mock_embed):
        """Test getting a specific chunk from a document"""
        mock_embed.return_value = MagicMock(embeddings=[[0.1] * 1536])

        document = Document(library_id=self.library_id)
        document = document.replace_content("Test content.")

        # Get the first chunk
        chunk_id = document.chunks[0].id
        chunk = ChunkService.get_chunk_from_document(document, chunk_id)

        assert isinstance(chunk, Chunk)
        assert chunk.id == chunk_id

    @patch('src.vector_db.domain.models.co')
    def test_get_chunk_from_document_not_found(self, mock_co):
        """Test getting a non-existent chunk from a document"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document = document.replace_content("Test content.")

        with pytest.raises(ValueError, match="not found"):
            ChunkService.get_chunk_from_document(document, "non-existent-id")

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.logger')
    def test_logging_chunk_retrieval(self, mock_logger, mock_co):
        """Test that chunk retrieval is logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
        mock_co.embed.return_value = mock_response

        document = Document(library_id=self.library_id)
        document = document.replace_content("Test content.")

        chunk_id = document.chunks[0].id
        ChunkService.get_chunk_from_document(document, chunk_id)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Chunk retrieved" in call_args[0]


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
        mock_response.embeddings = [[0.1] * 1536]  # 1536 dimensions
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
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]  # 1536 dimensions
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
        mock_response.embeddings = [[0.1] * 1536]
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
        mock_response.embeddings = [[0.1] * 1536]
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
        mock_response.embeddings = [[0.1] * 1536]
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
        mock_response.embeddings = [[0.1] * 1536]
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
        mock_response.embeddings = [[0.1] * 1536]
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
        assert abs(similarity - 0.0) < 0.001  # Clipped to 0 (max(0, -1.0))

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

    @patch('src.vector_db.domain.models.co')
    @patch('src.vector_db.application.services.logger')
    @patch('src.vector_db.application.services.co')
    def test_search_logging(self, mock_services_co, mock_logger, mock_domain_co):
        """Test that search operations are logged"""
        # Mock the Cohere API response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1536]
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