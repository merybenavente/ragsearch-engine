"""Unit tests for VectorDBService library operations"""

import pytest
from uuid import uuid4


class TestVectorDBServiceLibrary:
    """Test cases for VectorDBService library operations"""

    def test_create_library(self, sample_library_data, vector_db_service_instance):
        """Test creating a library"""
        library = vector_db_service_instance.create_library(**sample_library_data)

        assert library.name == sample_library_data["name"]
        assert library.metadata.username == sample_library_data["username"]
        assert library.metadata.tags == sample_library_data["tags"]
        assert library.index_type == sample_library_data["index_type"]
        assert library.document_ids == set()

        # Verify library is persisted
        retrieved_library = vector_db_service_instance.get_library(library.id)
        assert retrieved_library is not None
        assert retrieved_library.id == library.id

    def test_create_library_when_using_defaults(self, vector_db_service_instance):
        """Test creating a library with default values"""
        name = "Simple Library"

        library = vector_db_service_instance.create_library(name=name)

        assert library.name == name
        assert library.metadata.username is None
        assert library.metadata.tags == []
        assert library.index_type == "naive"

    def test_update_library_metadata(self, vector_db_service_instance):
        """Test updating library metadata"""
        library = vector_db_service_instance.create_library("Original Library")

        # Update name and tags
        new_name = "Updated Library"
        new_tags = ["updated", "tags"]

        updated_library = vector_db_service_instance.update_library_metadata(
            library_id=library.id, name=new_name, tags=new_tags
        )

        assert updated_library.name == new_name
        assert updated_library.metadata.tags == new_tags

        # Verify persistence
        retrieved_library = vector_db_service_instance.get_library(library.id)
        assert retrieved_library.name == new_name
        assert retrieved_library.metadata.tags == new_tags

    def test_delete_library(
        self, mock_cohere_deterministic, vector_db_service_instance
    ):
        """Test deleting a library and its documents"""
        # Create library with document
        library = vector_db_service_instance.create_library("Test Library")
        document = vector_db_service_instance.create_document(
            library_id=library.id, text="Test document content"
        )

        # Delete library
        vector_db_service_instance.delete_library(library.id)

        # Verify library is deleted
        assert vector_db_service_instance.get_library(library.id) is None

        # Verify document is deleted
        assert vector_db_service_instance.get_document(document.id) is None

    def test_list_libraries(self, vector_db_service_instance):
        """Test listing all libraries"""
        # Create a few libraries
        vector_db_service_instance.create_library(name="Library 1")
        vector_db_service_instance.create_library(name="Library 2")

        libraries = vector_db_service_instance.list_libraries()

        assert isinstance(libraries, list)
        assert len(libraries) == 2
        library_names = [lib.name for lib in libraries]
        assert "Library 1" in library_names
        assert "Library 2" in library_names

    def test_get_library_documents(
        self, mock_cohere_deterministic, vector_db_service_instance
    ):
        """Test getting all documents in a library"""
        library = vector_db_service_instance.create_library("Test Library")

        # Create multiple documents
        doc1 = vector_db_service_instance.create_document(
            library_id=library.id, text="First document content"
        )
        doc2 = vector_db_service_instance.create_document(
            library_id=library.id, text="Second document content"
        )

        # Get all documents
        documents = vector_db_service_instance.get_documents_in_library(library.id)

        assert len(documents) == 2
        doc_ids = {doc.id for doc in documents}
        assert doc1.id in doc_ids
        assert doc2.id in doc_ids

    def test_library_operations_when_invalid_data(self, vector_db_service_instance):
        """Test library operations with invalid data"""
        # Test create library with empty name
        with pytest.raises(ValueError):
            vector_db_service_instance.create_library(name="")

        # Test update library with invalid ID
        fake_id = str(uuid4())
        with pytest.raises(ValueError):
            vector_db_service_instance.update_library_metadata(
                library_id=fake_id, name="Updated Name"
            )
