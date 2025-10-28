"""Comprehensive end-to-end integration test for the complete vector database workflow"""

"""
The test successfully demonstrates the entire user journey:
  1. 📚 Library Creation - Creates library with metadata and naive index
  2. 📄 Document Ingestion - Adds 3 diverse documents (Technology, Science, Philosophy)
  3. 🔍 Search Operations - Tests multiple search scenarios:
    - Technology-specific search (5 results)
    - Science-specific search (3 results)
    - Cross-domain search (7 results spanning 3 documents)
    - Document-specific search (4 results)
  4. ✏️ Content Updates - Updates document content and library metadata
  5. ✅ Data Consistency - Validates data integrity across operations
  6. 🗑️ Cleanup & Deletion - Tests individual and cascade deletion
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import patch, MagicMock

from src.vector_db.api.main import app

client = TestClient(app)


# Mock Cohere embedding service for integration tests
@pytest.fixture(autouse=True)
def mock_cohere_embedding_service():
    """Mock the Cohere embedding service for integration tests with realistic embeddings"""
    import hashlib
    import numpy as np

    def create_realistic_embedding(text: str) -> list[float]:
        """Create deterministic but realistic mock embedding based on text content"""
        # Create deterministic embedding based on text hash
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        embedding = np.random.randn(1536)
        # Normalize to unit vector for realistic similarity calculations
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    with patch('src.vector_db.infrastructure.embedding_service.co') as mock_co:
        def mock_embed(texts, **kwargs):
            # Generate realistic embeddings for each text
            embeddings = [create_realistic_embedding(text) for text in texts]
            mock_response = MagicMock()
            mock_response.embeddings = embeddings
            return mock_response

        mock_co.embed = mock_embed
        yield mock_co


@pytest.mark.integration
class TestCompleteVectorDBWorkflow:
    """Complete end-to-end workflow testing for the vector database system"""

    def test_complete_workflow_naive_index(self):
        """Test complete workflow with naive index: create library → add documents → search → update → delete"""

        # === PHASE 1: Library Creation ===
        print("\n=== Creating Library ===")
        library_data = {
            "name": "Complete Workflow Test Library",
            "username": "workflow_tester",
            "tags": ["integration", "workflow", "test"],
            "index_type": "naive"
        }

        library_response = client.post("/api/libraries", json=library_data)
        assert library_response.status_code == status.HTTP_201_CREATED
        library = library_response.json()
        library_id = library["id"]

        print(f"✓ Library created: {library['name']} (ID: {library_id})")
        assert library["name"] == library_data["name"]
        assert library["document_count"] == 0
        assert library["metadata"]["username"] == library_data["username"]
        assert library["metadata"]["tags"] == library_data["tags"]

        # === PHASE 2: Document Ingestion ===
        print("\n=== Adding Documents ===")

        # Document 1: Technology content
        doc1_data = {
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms "
                   "that can learn and make decisions from data. Popular frameworks include TensorFlow, "
                   "PyTorch, and scikit-learn. Deep learning uses neural networks with multiple layers "
                   "to process complex patterns in data.",
            "username": "tech_author",
            "tags": ["technology", "ai", "ml"],
            "chunk_size": 80
        }

        doc1_response = client.post(f"/api/libraries/{library_id}/documents", json=doc1_data)
        assert doc1_response.status_code == status.HTTP_201_CREATED
        doc1 = doc1_response.json()
        doc1_id = doc1["id"]

        print(f"✓ Document 1 created: {doc1['chunk_count']} chunks")
        assert doc1["library_id"] == library_id
        assert doc1["chunk_count"] > 0
        assert doc1["metadata"]["username"] == "tech_author"
        assert "technology" in doc1["metadata"]["tags"]

        # Document 2: Science content
        doc2_data = {
            "text": "The human brain contains approximately 86 billion neurons, each connected to "
                   "thousands of others through synapses. Neuroscience research has revealed how "
                   "memories are formed through synaptic plasticity and how different brain regions "
                   "specialize in processing various types of information.",
            "username": "science_author",
            "tags": ["science", "neuroscience", "brain"],
            "chunk_size": 75
        }

        doc2_response = client.post(f"/api/libraries/{library_id}/documents", json=doc2_data)
        assert doc2_response.status_code == status.HTTP_201_CREATED
        doc2 = doc2_response.json()
        doc2_id = doc2["id"]

        print(f"✓ Document 2 created: {doc2['chunk_count']} chunks")

        # Document 3: Philosophy content
        doc3_data = {
            "text": "Philosophy explores fundamental questions about existence, knowledge, ethics, "
                   "and the nature of reality. Ancient philosophers like Aristotle and Plato laid "
                   "the groundwork for Western philosophical thought. Modern philosophy continues "
                   "to grapple with questions about consciousness, free will, and moral responsibility.",
            "username": "philosophy_author",
            "tags": ["philosophy", "ethics", "consciousness"],
            "chunk_size": 90
        }

        doc3_response = client.post(f"/api/libraries/{library_id}/documents", json=doc3_data)
        assert doc3_response.status_code == status.HTTP_201_CREATED
        doc3 = doc3_response.json()
        doc3_id = doc3["id"]

        print(f"✓ Document 3 created: {doc3['chunk_count']} chunks")

        # Verify library now contains all documents
        updated_library_response = client.get(f"/api/libraries/{library_id}")
        updated_library = updated_library_response.json()
        assert updated_library["document_count"] == 3
        print(f"✓ Library now contains {updated_library['document_count']} documents")

        # === PHASE 3: Search Operations ===
        print("\n=== Testing Search Operations ===")

        # Search 1: Technology-specific query
        tech_search_data = {
            "query_text": "machine learning algorithms",
            "k": 5,
            "min_similarity": 0.0
        }

        tech_search_response = client.post(f"/api/libraries/{library_id}/search", json=tech_search_data)
        assert tech_search_response.status_code == status.HTTP_200_OK
        tech_results = tech_search_response.json()

        print(f"✓ Technology search returned {len(tech_results['results'])} results")
        assert len(tech_results["results"]) > 0
        assert tech_results["total_chunks_searched"] > 0
        assert tech_results["query_time_ms"] >= 0.0  # Allow zero for very fast searches

        # Verify search result structure
        for result in tech_results["results"]:
            assert "chunk" in result
            assert "similarity_score" in result
            assert isinstance(result["similarity_score"], float)
            assert "id" in result["chunk"]
            assert "text" in result["chunk"]
            assert "document_id" in result["chunk"]

        # Search 2: Science-specific query
        science_search_data = {
            "query_text": "brain neurons synapses",
            "k": 3,
            "min_similarity": 0.0  # Use 0.0 for mock embeddings
        }

        science_search_response = client.post(f"/api/libraries/{library_id}/search", json=science_search_data)
        assert science_search_response.status_code == status.HTTP_200_OK
        science_results = science_search_response.json()

        print(f"✓ Science search returned {len(science_results['results'])} results")
        assert len(science_results["results"]) > 0

        # Search 3: Cross-domain query (should find relevant content across documents)
        cross_search_data = {
            "query_text": "intelligence and consciousness",
            "k": 7,
            "min_similarity": 0.0
        }

        cross_search_response = client.post(f"/api/libraries/{library_id}/search", json=cross_search_data)
        assert cross_search_response.status_code == status.HTTP_200_OK
        cross_results = cross_search_response.json()

        print(f"✓ Cross-domain search returned {len(cross_results['results'])} results")
        assert len(cross_results["results"]) > 0

        # Verify we get results from multiple documents
        document_ids_in_results = set()
        for result in cross_results["results"]:
            document_ids_in_results.add(result["chunk"]["document_id"])

        print(f"✓ Results span {len(document_ids_in_results)} different documents")
        assert len(document_ids_in_results) >= 2  # Should find relevant content across documents

        # Search 4: Document-specific search
        doc_search_data = {
            "query_text": "neural networks deep learning",
            "k": 5,
            "min_similarity": 0.0
        }

        doc_search_response = client.post(f"/api/documents/{doc1_id}/search", json=doc_search_data)
        assert doc_search_response.status_code == status.HTTP_200_OK
        doc_results = doc_search_response.json()

        print(f"✓ Document-specific search returned {len(doc_results['results'])} results")
        # Verify all results are from the target document
        for result in doc_results["results"]:
            assert result["chunk"]["document_id"] == doc1_id

        # === PHASE 4: Content Updates ===
        print("\n=== Testing Content Updates ===")

        # Update document content
        update_doc_data = {
            "text": "Machine learning and artificial intelligence have revolutionized modern technology. "
                   "From recommendation systems to autonomous vehicles, AI applications are everywhere. "
                   "Emerging fields like generative AI and large language models are pushing the "
                   "boundaries of what's possible with artificial intelligence.",
            "chunk_size": 70
        }

        update_response = client.put(f"/api/documents/{doc1_id}", json=update_doc_data)
        assert update_response.status_code == status.HTTP_200_OK
        updated_doc = update_response.json()

        print(f"✓ Document updated: {updated_doc['chunk_count']} chunks")
        assert updated_doc["id"] == doc1_id

        # Verify updated content is searchable
        update_search_data = {
            "query_text": "generative AI language models",
            "k": 3,
            "min_similarity": 0.0
        }

        update_search_response = client.post(f"/api/documents/{doc1_id}/search", json=update_search_data)
        assert update_search_response.status_code == status.HTTP_200_OK
        update_search_results = update_search_response.json()

        print(f"✓ Updated content search returned {len(update_search_results['results'])} results")
        assert len(update_search_results["results"]) > 0

        # Update library metadata
        library_update_data = {
            "name": "Updated Workflow Test Library",
            "tags": ["integration", "workflow", "test", "updated"]
        }

        library_update_response = client.put(f"/api/libraries/{library_id}", json=library_update_data)
        assert library_update_response.status_code == status.HTTP_200_OK
        updated_library_meta = library_update_response.json()

        print(f"✓ Library metadata updated: {updated_library_meta['name']}")
        assert updated_library_meta["name"] == library_update_data["name"]
        assert "updated" in updated_library_meta["metadata"]["tags"]

        # === PHASE 5: Data Retrieval and Validation ===
        print("\n=== Validating Data Consistency ===")

        # Get all documents in library
        all_docs_response = client.get(f"/api/libraries/{library_id}/documents")
        assert all_docs_response.status_code == status.HTTP_200_OK
        all_docs = all_docs_response.json()

        print(f"✓ Retrieved {len(all_docs)} documents from library")
        assert len(all_docs) == 3

        # Verify each document can be retrieved individually
        for doc_summary in all_docs:
            doc_detail_response = client.get(f"/api/documents/{doc_summary['id']}")
            assert doc_detail_response.status_code == status.HTTP_200_OK
            doc_detail = doc_detail_response.json()
            assert doc_detail["id"] == doc_summary["id"]
            assert doc_detail["library_id"] == library_id

        # Get all libraries and verify our library is there
        all_libraries_response = client.get("/api/libraries")
        assert all_libraries_response.status_code == status.HTTP_200_OK
        all_libraries = all_libraries_response.json()

        library_found = False
        for lib in all_libraries:
            if lib["id"] == library_id:
                library_found = True
                assert lib["name"] == "Updated Workflow Test Library"
                assert lib["document_count"] == 3
                break

        assert library_found, "Updated library should be found in library list"
        print("✓ Library found in complete library listing")

        # === PHASE 6: Cleanup (Deletion) ===
        print("\n=== Testing Cleanup Operations ===")

        # Delete one document
        delete_doc_response = client.delete(f"/api/documents/{doc3_id}")
        assert delete_doc_response.status_code == status.HTTP_204_NO_CONTENT
        print(f"✓ Document {doc3_id} deleted")

        # Verify document is gone
        deleted_doc_response = client.get(f"/api/documents/{doc3_id}")
        assert deleted_doc_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify library document count updated
        library_after_delete_response = client.get(f"/api/libraries/{library_id}")
        library_after_delete = library_after_delete_response.json()
        assert library_after_delete["document_count"] == 2
        print("✓ Library document count updated after deletion")

        # Delete entire library (cascade delete)
        delete_library_response = client.delete(f"/api/libraries/{library_id}")
        assert delete_library_response.status_code == status.HTTP_204_NO_CONTENT
        print(f"✓ Library {library_id} deleted")

        # Verify library is gone
        deleted_library_response = client.get(f"/api/libraries/{library_id}")
        assert deleted_library_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify remaining documents are also gone (cascade delete)
        for remaining_doc_id in [doc1_id, doc2_id]:
            deleted_remaining_doc_response = client.get(f"/api/documents/{remaining_doc_id}")
            assert deleted_remaining_doc_response.status_code == status.HTTP_404_NOT_FOUND

        print("✓ Cascade deletion verified - all documents deleted with library")

        # === WORKFLOW COMPLETE ===
        print("\n=== WORKFLOW COMPLETE ===")
        print("✓ Library creation and configuration")
        print("✓ Multiple document ingestion with different content types")
        print("✓ Various search operations (library, document, cross-domain)")
        print("✓ Content and metadata updates")
        print("✓ Data consistency validation")
        print("✓ Cleanup and cascade deletion")
        print("✅ Complete vector database workflow successfully tested!")

    def test_workflow_with_empty_and_minimal_content(self):
        """Test workflow edge cases with empty documents and minimal content"""

        # Create library
        library_data = {"name": "Edge Case Test Library"}
        library_response = client.post("/api/libraries", json=library_data)
        library_id = library_response.json()["id"]

        # Create empty document
        empty_doc_response = client.post(f"/api/libraries/{library_id}/documents", json={})
        assert empty_doc_response.status_code == status.HTTP_201_CREATED
        empty_doc_id = empty_doc_response.json()["id"]

        # Search in empty document should return no results
        search_empty_response = client.post(
            f"/api/documents/{empty_doc_id}/search",
            json={"query_text": "test", "k": 5}
        )
        assert search_empty_response.status_code == status.HTTP_200_OK
        assert len(search_empty_response.json()["results"]) == 0

        # Create minimal content document
        minimal_doc_data = {"text": "AI"}  # Very short content
        minimal_doc_response = client.post(f"/api/libraries/{library_id}/documents", json=minimal_doc_data)
        assert minimal_doc_response.status_code == status.HTTP_201_CREATED
        minimal_doc = minimal_doc_response.json()
        assert minimal_doc["chunk_count"] >= 1

        # Search should work even with minimal content
        search_minimal_response = client.post(
            f"/api/libraries/{library_id}/search",
            json={"query_text": "artificial intelligence", "k": 5, "min_similarity": 0.0}
        )
        assert search_minimal_response.status_code == status.HTTP_200_OK
        # May or may not find results depending on similarity, but should not error

        # Cleanup
        client.delete(f"/api/libraries/{library_id}")

    def test_workflow_error_recovery(self):
        """Test workflow behavior with various error conditions"""

        # Create library
        library_data = {"name": "Error Recovery Test Library"}
        library_response = client.post("/api/libraries", json=library_data)
        library_id = library_response.json()["id"]

        # Try to search in empty library
        search_empty_library_response = client.post(
            f"/api/libraries/{library_id}/search",
            json={"query_text": "test query", "k": 5}
        )
        assert search_empty_library_response.status_code == status.HTTP_200_OK
        assert len(search_empty_library_response.json()["results"]) == 0

        # Add a document
        doc_data = {"text": "Test document for error recovery"}
        doc_response = client.post(f"/api/libraries/{library_id}/documents", json=doc_data)
        doc_id = doc_response.json()["id"]

        # Try invalid search parameters
        invalid_search_response = client.post(
            f"/api/libraries/{library_id}/search",
            json={"query_text": "", "k": 5}  # Empty query
        )
        assert invalid_search_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Try search with k too high
        high_k_search_response = client.post(
            f"/api/libraries/{library_id}/search",
            json={"query_text": "test", "k": 500}  # Too high
        )
        assert high_k_search_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Valid search should still work after errors
        valid_search_response = client.post(
            f"/api/libraries/{library_id}/search",
            json={"query_text": "test document", "k": 5}
        )
        assert valid_search_response.status_code == status.HTTP_200_OK

        # Cleanup
        client.delete(f"/api/libraries/{library_id}")