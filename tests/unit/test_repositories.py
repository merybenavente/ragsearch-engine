"""Unit tests for the repositories"""

from uuid import uuid4


class TestDDDRepositories:
    """Test cases for DDD architecture repositories"""

    def test_library_repository_operations(self, vector_db_service, library_repository):
        """Test basic library repository operations"""
        # Create a library
        library = vector_db_service.create_library(name="Test Library")

        # Test retrieval
        retrieved_library = library_repository.get(library.id)
        assert retrieved_library is not None
        assert retrieved_library.name == "Test Library"
        assert retrieved_library.id == library.id

        # Test listing
        libraries = library_repository.list_all()
        assert isinstance(libraries, list)
        assert len(libraries) >= 1
        library_ids = [lib.id for lib in libraries]
        assert library.id in library_ids

    def test_document_repository_operations(
        self, mock_cohere_deterministic, vector_db_service, document_repository
    ):
        """Test basic document repository operations"""
        # Create a library and document
        library = vector_db_service.create_library(name="Test Library")
        document = vector_db_service.create_document(
            library_id=library.id, text="Test document content"
        )

        # Test retrieval
        retrieved_document = document_repository.get(document.id)
        assert retrieved_document is not None
        assert retrieved_document.id == document.id
        assert retrieved_document.library_id == library.id

        # Test listing by library
        documents = document_repository.get_by_library(library.id)
        assert isinstance(documents, list)
        assert len(documents) >= 1
        document_ids = [doc.id for doc in documents]
        assert document.id in document_ids

    def test_repository_consistency(
        self,
        mock_cohere_deterministic,
        vector_db_service,
        library_repository,
        document_repository,
    ):
        """Test that repositories maintain consistency"""
        # Create a library
        library = vector_db_service.create_library(name="Test Library")

        # Create multiple documents
        doc1 = vector_db_service.create_document(library.id, "Document 1")
        doc2 = vector_db_service.create_document(library.id, "Document 2")

        # Check that library knows about documents
        updated_library = library_repository.get(library.id)
        assert len(updated_library.document_ids) == 2
        assert doc1.id in updated_library.document_ids
        assert doc2.id in updated_library.document_ids

        # Check that documents exist in document repository
        retrieved_doc1 = document_repository.get(doc1.id)
        retrieved_doc2 = document_repository.get(doc2.id)
        assert retrieved_doc1 is not None
        assert retrieved_doc2 is not None
        assert retrieved_doc1.library_id == library.id
        assert retrieved_doc2.library_id == library.id

    def test_delete_operations(
        self,
        mock_cohere_deterministic,
        vector_db_service,
        library_repository,
        document_repository,
    ):
        """Test delete operations maintain consistency"""
        # Create a library with documents
        library = vector_db_service.create_library(name="Test Library")
        document = vector_db_service.create_document(
            library_id=library.id, text="Test document"
        )

        # Verify initial state
        assert library_repository.get(library.id) is not None
        assert document_repository.get(document.id) is not None

        # Delete the library
        vector_db_service.delete_library(library.id)

        # Verify library is deleted
        assert library_repository.get(library.id) is None

        # Verify associated documents are also cleaned up
        assert document_repository.get(document.id) is None

    def test_repository_operations_when_empty_results(
        self, vector_db_service, library_repository, document_repository
    ):
        """Test repository behavior with empty results"""
        # Test non-existent library
        non_existent_id = str(uuid4())
        assert library_repository.get(non_existent_id) is None
        assert document_repository.get(non_existent_id) is None

        # Test empty library
        library = vector_db_service.create_library(name="Empty Library")
        documents = document_repository.get_by_library(library.id)
        assert documents == []
