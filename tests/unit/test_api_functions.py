"""Test basic API functionality and imports"""

import pytest


def test_basic_imports():
    """Test that the API functions can be imported"""
    try:
        from src.vector_db.domain.models import Document, Library, Metadata
        from src.vector_db.api.main import app

        # Use imports to avoid unused import warnings
        assert Document is not None
        assert Library is not None
        assert Metadata is not None
        assert app is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_create_document_function(mock_cohere_deterministic, vector_db_service):
    """Test the create_document service function"""
    text = "Test document content"
    username = "testuser"
    tags = ["test", "example"]

    # First create a library
    library = vector_db_service.create_library(name="Test Library", username="testuser")

    # Then create a document
    document = vector_db_service.create_document(
        library_id=library.id, text=text, username=username, tags=tags
    )

    assert document.library_id == library.id
    assert document.metadata.username == username
    assert document.metadata.tags == tags
    assert len(document.chunks) > 0
    assert document.has_content()


def test_create_library_function(vector_db_service):
    """Test the create_library service function"""
    name = "Test Library"
    username = "testuser"
    tags = ["test", "example"]

    library = vector_db_service.create_library(name=name, username=username, tags=tags)

    assert library.name == name
    assert library.metadata.username == username
    assert library.metadata.tags == tags
    assert len(library.document_ids) == 0


def test_library_document_workflow(mock_cohere_deterministic, vector_db_service):
    """Test complete workflow of creating library and adding documents"""
    # Create library
    library = vector_db_service.create_library(
        name="Music Collection", username="Maria", tags=["personal", "favorites"]
    )

    # Create document (this automatically adds it to the library)
    document = vector_db_service.create_document(
        library_id=library.id,
        text="This is a song about the sea",
        username="Maria",
        tags=["catalan", "indie"],
    )

    # Get updated library to check document was added
    updated_library = vector_db_service.get_library(library.id)
    documents = vector_db_service.get_documents_in_library(library.id)

    assert len(updated_library.document_ids) == 1
    assert document.id in updated_library.document_ids
    assert len(documents) == 1
    assert documents[0].id == document.id
    assert documents[0].library_id == library.id
