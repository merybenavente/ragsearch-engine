"""Test basic API functionality and imports"""

import pytest
from unittest.mock import patch, MagicMock


def test_basic_imports():
    """Test that the API functions can be imported"""
    try:
        from src.vector_db.application import services
        from src.vector_db.domain.models import Document, Library, Metadata
        from src.vector_db.api.main import app
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@patch('src.vector_db.domain.models.co')
def test_create_document_function(mock_co):
    """Test the create_document service function"""
    # Mock the Cohere API response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] * 307]  # 1536 dimensions (rounded)
    mock_co.embed.return_value = mock_response

    from src.vector_db.application.services import DocumentService

    library_id = "lib_123"
    text = "Test document content"
    username = "testuser"
    tags = ["test", "example"]

    document = DocumentService.create_document(
        library_id=library_id,
        text=text,
        username=username,
        tags=tags
    )

    assert document.library_id == library_id
    assert document.metadata.username == username
    assert document.metadata.tags == tags
    assert len(document.chunks) > 0
    assert document.has_content()


def test_create_library_function():
    """Test the create_library service function"""
    from src.vector_db.application.services import LibraryService

    name = "Test Library"
    username = "testuser"
    tags = ["test", "example"]

    library = LibraryService.create_library(
        name=name,
        username=username,
        tags=tags
    )

    assert library.name == name
    assert library.metadata.username == username
    assert library.metadata.tags == tags
    assert len(library.documents) == 0


@patch('src.vector_db.domain.models.co')
def test_integration_example(mock_co):
    """Test creating library and adding documents"""
    # Mock the Cohere API response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] * 307]  # 1536 dimensions (rounded)
    mock_co.embed.return_value = mock_response

    from src.vector_db.application.services import LibraryService, DocumentService

    # Create library
    library = LibraryService.create_library(
        name="Music Collection",
        username="Maria",
        tags=["personal", "favorites"]
    )

    # Create document
    document = DocumentService.create_document(
        library_id=library.id,
        text="This is a song about the sea",
        username="Maria",
        tags=["catalan", "indie"]
    )

    # Add document to library
    updated_library = LibraryService.add_document_to_library(library, document)

    assert len(updated_library.documents) == 1
    assert updated_library.documents[0].id == document.id
    assert updated_library.documents[0].library_id == library.id