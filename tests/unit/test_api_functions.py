"""Test basic API functionality and imports"""

import pytest
from unittest.mock import patch, MagicMock
from src.vector_db.domain.models import EMBEDDING_DIMENSION


def test_basic_imports():
    """Test that the API functions can be imported"""
    try:
        from src.vector_db.application import services
        from src.vector_db.domain.models import Document, Library, Metadata
        from src.vector_db.api.main import app
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@patch('src.vector_db.infrastructure.embedding_service.co')
def test_create_document_function(mock_co):
    """Test the create_document service function"""
    # Mock the Cohere API response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]
    mock_co.embed.return_value = mock_response

    from src.vector_db.api.dependencies import get_document_service

    library_id = "lib_123"
    text = "Test document content"
    username = "testuser"
    tags = ["test", "example"]

    document_service = get_document_service()
    document = document_service.create_document(
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
    from src.vector_db.api.dependencies import get_library_service

    name = "Test Library"
    username = "testuser"
    tags = ["test", "example"]

    library_service = get_library_service()
    library = library_service.create_library(
        name=name,
        username=username,
        tags=tags
    )

    assert library.name == name
    assert library.metadata.username == username
    assert library.metadata.tags == tags
    assert len(library.documents) == 0


@patch('src.vector_db.infrastructure.embedding_service.co')
def test_integration_example(mock_co):
    """Test creating library and adding documents"""
    # Mock the Cohere API response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]
    mock_co.embed.return_value = mock_response

    from src.vector_db.api.dependencies import get_library_service, get_document_service

    # Create library
    library_service = get_library_service()
    library = library_service.create_library(
        name="Music Collection",
        username="Maria",
        tags=["personal", "favorites"]
    )

    # Create document
    document_service = get_document_service()
    document = document_service.create_document(
        library_id=library.id,
        text="This is a song about the sea",
        username="Maria",
        tags=["catalan", "indie"]
    )

    # Add document to library
    updated_library = library_service.add_document_to_library(library, document)

    assert len(updated_library.documents) == 1
    assert updated_library.documents[0].id == document.id
    assert updated_library.documents[0].library_id == library.id