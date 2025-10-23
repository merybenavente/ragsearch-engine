"""Test basic API functionality and imports"""

import pytest


def test_basic_imports():
    """Test that the API functions can be imported"""
    try:
        from src.vector_db.application import services
        from src.vector_db.domain.models import Document, Library, Metadata
        from src.vector_db.api.main import app
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_create_document_function():
    """Test the create_document service function"""
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


def test_integration_example():
    """Test creating library and adding documents"""
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