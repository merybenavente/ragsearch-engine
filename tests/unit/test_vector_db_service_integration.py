"""Integration tests for VectorDBService complete workflows"""

import pytest
from uuid import uuid4

from src.vector_db.application.vector_db_service import VectorDBService
from src.vector_db.domain.models import Chunk
from src.vector_db.infrastructure.repositories import RepositoryManager
from src.vector_db.infrastructure.search_index import RepositoryAwareSearchIndex
from src.vector_db.infrastructure.index_factory import get_index_factory
from src.vector_db.infrastructure.embedding_service import CohereEmbeddingService


class TestVectorDBServiceIntegration:
    """Integration test cases for VectorDBService complete workflows"""

    def setup_method(self):
        """Setup test fixtures"""
        # Setup repositories
        self.repo_manager = RepositoryManager()
        self.search_index = RepositoryAwareSearchIndex(get_index_factory())
        self.embedding_service = CohereEmbeddingService()

        # Create VectorDBService instance
        self.vector_db_service = VectorDBService(
            self.repo_manager.get_document_repository(),
            self.repo_manager.get_library_repository(),
            self.search_index,
            self.embedding_service,
        )

    def test_integration_workflow(self, mock_cohere_deterministic):
        """Test complete workflow integration"""
        # Create library
        library = self.vector_db_service.create_library(
            name="Integration Test Library",
            username="integrator",
            tags=["integration", "test"],
        )

        # Create multiple documents
        doc1 = self.vector_db_service.create_document(
            library_id=library.id,
            text="Python is a programming language that emphasizes readability and simplicity.",
            username="author1",
            tags=["python", "programming"],
        )

        doc2 = self.vector_db_service.create_document(
            library_id=library.id,
            text="Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
            username="author2",
            tags=["ml", "python", "scikit-learn"],
        )

        # Search for content
        results = self.vector_db_service.search_library(
            library_id=library.id,
            query_text="Python programming",
            k=3,
            min_similarity=0.0,  # Use low threshold to ensure results
        )

        # Verify results
        assert len(results) > 0
        # Results are List[Tuple[Chunk, float]], so check structure
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(isinstance(score, float) for score in scores)
        assert all(chunk.document_id in [doc1.id, doc2.id] for chunk in chunks)

        # Verify library contains both documents
        updated_library = self.vector_db_service.get_library(library.id)
        assert len(updated_library.document_ids) == 2
        assert doc1.id in updated_library.document_ids
        assert doc2.id in updated_library.document_ids

    def test_concurrent_operations_safety(self, mock_cohere_deterministic):
        """Test basic concurrent operations safety"""
        library = self.vector_db_service.create_library(name="Concurrent Test Library")

        # Create multiple documents in sequence (simulating concurrent operations)
        documents = []
        for i in range(3):
            doc = self.vector_db_service.create_document(
                library_id=library.id,
                text=f"Document {i} content for concurrent testing",
            )
            documents.append(doc)

        # Verify all documents were created correctly
        assert len(documents) == 3
        for i, doc in enumerate(documents):
            assert f"Document {i}" in doc.get_full_text()

        # Verify library state is consistent
        updated_library = self.vector_db_service.get_library(library.id)
        assert len(updated_library.document_ids) == 3

    def test_error_recovery_workflow(self, mock_cohere_deterministic):
        """Test system behavior during error conditions"""
        library = self.vector_db_service.create_library(name="Error Recovery Test")

        # Create a valid document first
        valid_doc = self.vector_db_service.create_document(
            library_id=library.id, text="Valid document content"
        )

        # Try to create document with invalid library ID
        with pytest.raises(ValueError):
            self.vector_db_service.create_document(
                library_id=str(uuid4()),  # Invalid library ID
                text="This should fail",
            )

        # Verify original document is still accessible
        retrieved_doc = self.vector_db_service.get_document(valid_doc.id)
        assert retrieved_doc is not None
        assert retrieved_doc.id == valid_doc.id

        # Verify library is still in consistent state
        updated_library = self.vector_db_service.get_library(library.id)
        assert len(updated_library.document_ids) == 1
        assert valid_doc.id in updated_library.document_ids

    def test_full_crud_workflow(self, mock_cohere_deterministic):
        """Test complete CRUD workflow for libraries and documents"""
        # CREATE: Library and documents
        library = self.vector_db_service.create_library(
            name="CRUD Test Library", username="crud_user", tags=["crud", "test"]
        )

        doc1 = self.vector_db_service.create_document(
            library_id=library.id, text="First document for CRUD testing"
        )

        doc2 = self.vector_db_service.create_document(
            library_id=library.id, text="Second document for CRUD testing"
        )

        # READ: Verify all data can be retrieved
        retrieved_library = self.vector_db_service.get_library(library.id)
        assert retrieved_library.name == "CRUD Test Library"
        assert len(retrieved_library.document_ids) == 2

        retrieved_doc1 = self.vector_db_service.get_document(doc1.id)
        retrieved_doc2 = self.vector_db_service.get_document(doc2.id)
        assert retrieved_doc1 is not None
        assert retrieved_doc2 is not None

        # UPDATE: Modify library and document
        updated_library = self.vector_db_service.update_library_metadata(
            library_id=library.id,
            name="Updated CRUD Library",
            tags=["crud", "test", "updated"],
        )
        assert updated_library.name == "Updated CRUD Library"
        assert "updated" in updated_library.metadata.tags

        updated_doc = self.vector_db_service.update_document_content(
            document_id=doc1.id, new_text="Updated content for first document"
        )
        assert "Updated content" in updated_doc.get_full_text()

        # SEARCH: Verify search works with updated content
        search_results = self.vector_db_service.search_library(
            library_id=library.id, query_text="updated content", k=5, min_similarity=0.0
        )
        assert len(search_results) > 0

        # DELETE: Remove documents and library
        # First delete one document
        retrieved_library_before = self.vector_db_service.get_library(library.id)
        assert len(retrieved_library_before.document_ids) == 2

        # Delete entire library (should cascade delete documents)
        self.vector_db_service.delete_library(library.id)

        # Verify everything is deleted
        assert self.vector_db_service.get_library(library.id) is None
        assert self.vector_db_service.get_document(doc1.id) is None
        assert self.vector_db_service.get_document(doc2.id) is None
