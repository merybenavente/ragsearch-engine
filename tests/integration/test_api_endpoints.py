"""Integration tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from uuid import uuid4
from unittest.mock import patch, MagicMock

from src.vector_db.api.main import app
from src.vector_db.domain.models import EMBEDDING_DIMENSION

client = TestClient(app)


# Mock Cohere embedding service for integration tests
@pytest.fixture(autouse=True)
def mock_cohere_embedding_service():
    """Mock the Cohere embedding service for integration tests"""
    with patch("src.vector_db.infrastructure.embedding_service.co") as mock_co:
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * EMBEDDING_DIMENSION]  # Mock embedding
        mock_co.embed.return_value = mock_response
        yield mock_co


class TestHealthEndpoints:
    """Test cases for health check endpoints"""

    def test_root_endpoint(self):
        """Test root health check endpoint"""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Vector Database API is running"

    def test_health_endpoint(self):
        """Test detailed health check endpoint"""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "vector-db"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data


class TestLibraryEndpoints:
    """Test cases for library endpoints"""

    def test_create_library_success(self, sample_library_data):
        """Test successful library creation"""
        response = client.post("/api/v1/libraries", json=sample_library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_library_data["name"]
        assert data["document_count"] == 0
        assert "id" in data
        assert "metadata" in data
        assert data["metadata"]["username"] == sample_library_data["username"]
        assert data["metadata"]["tags"] == sample_library_data["tags"]

    def test_create_library_minimal(self):
        """Test library creation with minimal data"""
        library_data = {"name": "Minimal Library"}

        response = client.post("/api/v1/libraries", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0
        assert data["metadata"]["username"] is None
        assert data["metadata"]["tags"] == []

    def test_create_library_with_index_params(self):
        """Test library creation with custom index parameters"""
        library_data = {
            "name": "LSH Library with Custom Params",
            "username": "testuser",
            "tags": ["lsh", "custom"],
            "index_type": "lsh",
            "index_params": {"num_tables": 10, "num_hyperplanes": 8},
        }

        response = client.post("/api/v1/libraries", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0
        assert data["metadata"]["username"] == library_data["username"]
        assert data["metadata"]["tags"] == library_data["tags"]

    def test_create_library_with_index_params_naive(self):
        """Test library creation with index params for naive index (should be ignored)"""
        library_data = {
            "name": "Naive Library with Params",
            "index_type": "naive",
            "index_params": {"some_param": "value"},
        }

        response = client.post("/api/v1/libraries", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0

    def test_create_library_with_vptree_params(self):
        """Test library creation with VPTree index parameters"""
        library_data = {
            "name": "VPTree Library",
            "index_type": "vptree",
            "index_params": {"leaf_size": 20},
        }

        response = client.post("/api/v1/libraries", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0

    def test_create_document_success(self):
        """Test successful document creation in a library"""
        # First create a library
        library_data = {"name": "Document Test Library"}
        library_response = client.post("/api/v1/libraries", json=library_data)
        assert library_response.status_code == status.HTTP_201_CREATED
        library_id = library_response.json()["id"]

        # Then create a document
        document_data = {
            "text": "This is a test document with some content that will be chunked for vector search.",
            "username": "testuser",
            "tags": ["doc_tag"],
            "chunk_size": 50,
        }

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=document_data
        )

        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error: {response.status_code} - {response.text}")
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["library_id"] == library_id
        assert data["chunk_count"] > 0
        assert data["text_preview"].startswith("This is a test document")
        assert data["metadata"]["username"] == document_data["username"]
        assert data["metadata"]["tags"] == document_data["tags"]

    def test_search_library(self):
        """Test searching within a library"""
        # Create library and document
        library_data = {"name": "Search Test Library"}
        library_response = client.post("/api/v1/libraries", json=library_data)
        library_id = library_response.json()["id"]

        document_data = {
            "text": "Machine learning algorithms are used in artificial intelligence applications."
        }
        client.post(f"/api/v1/libraries/{library_id}/documents", json=document_data)

        # Search for content
        search_data = {"query_text": "machine learning", "k": 5, "min_similarity": 0.0}

        response = client.post(
            f"/api/v1/libraries/{library_id}/search", json=search_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) > 0

        # Verify search result structure
        for result in data["results"]:
            assert "chunk" in result
            assert "similarity_score" in result
            assert isinstance(result["similarity_score"], float)
            assert "id" in result["chunk"]
            assert "text" in result["chunk"]

    def test_create_library_duplicate_name(self):
        """Test creating library with duplicate name"""
        library_data = {"name": "Duplicate Library"}

        # Check current libraries first
        client.get("/api/v1/libraries")

        # Create first library
        response1 = client.post("/api/v1/libraries", json=library_data)
        assert response1.status_code == status.HTTP_201_CREATED

        # Try to create second library with same name
        response2 = client.post("/api/v1/libraries", json=library_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

    def test_create_library_invalid_data(self):
        """Test library creation with invalid data"""
        invalid_data = {"name": ""}  # Empty name

        response = client.post("/api/v1/libraries", json=invalid_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_libraries_empty(self):
        """Test getting libraries when none exist"""
        response = client.get("/api/v1/libraries")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_libraries_with_data(self, sample_library_data):
        """Test getting libraries when some exist"""
        # Create a library first
        create_response = client.post("/api/v1/libraries", json=sample_library_data)
        assert create_response.status_code == status.HTTP_201_CREATED

        # Get all libraries
        response = client.get("/api/v1/libraries")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == sample_library_data["name"]

    def test_get_library_by_id_success(self, sample_library_data):
        """Test getting library by ID successfully"""
        # Create a library first
        create_response = client.post("/api/v1/libraries", json=sample_library_data)
        created_library = create_response.json()
        library_id = created_library["id"]

        # Get library by ID
        response = client.get(f"/api/v1/libraries/{library_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == sample_library_data["name"]

    def test_get_library_by_id_not_found(self):
        """Test getting library by non-existent ID"""
        non_existent_id = str(uuid4())

        response = client.get(f"/api/v1/libraries/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_library_success(self, sample_library_data):
        """Test successful library update"""
        # Create a library first
        create_response = client.post("/api/v1/libraries", json=sample_library_data)
        created_library = create_response.json()
        library_id = created_library["id"]

        # Update library
        update_data = {"name": "Updated Library", "tags": ["new_tag"]}
        response = client.put(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_data["name"]

    def test_update_library_not_found(self):
        """Test updating non-existent library"""
        non_existent_id = str(uuid4())
        update_data = {"name": "Updated Library"}

        response = client.put(f"/api/v1/libraries/{non_existent_id}", json=update_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_library_duplicate_name(self):
        """Test updating library with existing name"""
        # Create two libraries
        lib1_data = {"name": "Library 1"}
        lib2_data = {"name": "Library 2"}

        response1 = client.post("/api/v1/libraries", json=lib1_data)
        client.post("/api/v1/libraries", json=lib2_data)

        lib1_id = response1.json()["id"]

        # Try to update lib1 with lib2's name
        update_data = {"name": lib2_data["name"]}
        response = client.put(f"/api/v1/libraries/{lib1_id}", json=update_data)

        assert response.status_code == status.HTTP_409_CONFLICT

    def test_delete_library_success(self, sample_library_data):
        """Test successful library deletion"""
        create_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = create_response.json()["id"]

        # Delete library
        response = client.delete(f"/api/v1/libraries/{library_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify library is deleted
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_library_not_found(self):
        """Test deleting non-existent library"""
        non_existent_id = str(uuid4())

        response = client.delete(f"/api/v1/libraries/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentEndpoints:
    """Test cases for document endpoints"""

    def test_create_document_success(self, sample_library_data, sample_document_data):
        """Test successful document creation"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["library_id"] == library_id
        assert data["chunk_count"] > 0
        assert "id" in data
        assert "text_preview" in data

    def test_create_document_library_not_found(self, sample_document_data):
        """Test creating document in non-existent library"""
        non_existent_library_id = str(uuid4())

        response = client.post(
            f"/api/v1/libraries/{non_existent_library_id}/documents",
            json=sample_document_data,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_document_invalid_data(self, sample_library_data):
        """Test creating document with invalid data"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Try to create document with empty text
        invalid_data = {"text": ""}

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=invalid_data
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_documents_in_library(self, sample_library_data, sample_document_data):
        """Test getting documents in a library"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )
        assert doc_response.status_code == status.HTTP_201_CREATED

        # Get documents
        response = client.get(f"/api/v1/libraries/{library_id}/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["library_id"] == library_id

    def test_get_document_by_id_success(
        self, sample_library_data, sample_document_data
    ):
        """Test getting document by ID successfully"""
        # Create library and document
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Get document by ID
        response = client.get(f"/api/v1/libraries/{library_id}/documents/{document_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id
        assert data["library_id"] == library_id

    def test_get_document_by_id_not_found(self, sample_library_data):
        """Test getting document by non-existent ID"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        non_existent_doc_id = str(uuid4())

        response = client.get(
            f"/api/v1/libraries/{library_id}/documents/{non_existent_doc_id}"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_document_success(self, sample_library_data, sample_document_data):
        """Test successful document update"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Update document
        update_data = {
            "text": "Updated content that is much longer than the original content.",
            "chunk_size": 30,
        }
        response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", json=update_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id

    def test_delete_document_success(self, sample_library_data, sample_document_data):
        """Test successful document deletion"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Delete document
        response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify document is deleted
        get_response = client.get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestSearchEndpoints:
    """Test cases for search endpoints"""

    def test_search_document_success(self, sample_library_data, sample_document_data):
        """Test successful document-specific search"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Search within the specific document
        search_data = {"query_text": "machine learning", "k": 5, "min_similarity": 0.0}

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/{document_id}/search",
            json=search_data,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) > 0

        # Verify all results are from the same document
        for result in data["results"]:
            assert result["chunk"]["document_id"] == document_id

    def test_search_document_not_found(self, sample_library_data):
        """Test search in non-existent document"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        non_existent_document_id = str(uuid4())
        search_data = {"query_text": "test query", "k": 5}

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/{non_existent_document_id}/search",
            json=search_data,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_search_empty_document(self, sample_library_data):
        """Test search in document with no content"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create empty document (no text)
        document_data = {}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents", json=document_data
        )
        document_id = doc_response.json()["id"]

        # Search in empty document
        search_data = {"query_text": "test search", "k": 5}

        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/{document_id}/search",
            json=search_data,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["results"] == []

    def test_search_validation_errors(self, sample_library_data):
        """Test search with various validation errors"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Test empty query text
        search_data = {"query_text": "", "k": 5}

        response = client.post(
            f"/api/v1/libraries/{library_id}/search", json=search_data
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid k value (too high)
        search_data = {
            "query_text": "test query",
            "k": 500,  # Exceeds max of 100
        }

        response = client.post(
            f"/api/v1/libraries/{library_id}/search", json=search_data
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid k value (zero)
        search_data = {"query_text": "test query", "k": 0}

        response = client.post(
            f"/api/v1/libraries/{library_id}/search", json=search_data
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_response_structure_comprehensive(self, sample_library_data):
        """Test comprehensive search response structure validation"""
        lib_response = client.post("/api/v1/libraries", json=sample_library_data)
        library_id = lib_response.json()["id"]

        document_data = {
            "text": "Test content for comprehensive structure validation with multiple sentences.",
            "username": "test_user",
            "tags": ["test", "validation"],
            "chunk_size": 25,
        }
        client.post(f"/api/v1/libraries/{library_id}/documents", json=document_data)

        # Perform search
        search_data = {
            "query_text": "test content validation",
            "k": 3,
            "min_similarity": 0.0,
        }

        response = client.post(
            f"/api/v1/libraries/{library_id}/search", json=search_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Validate top-level response structure
        assert "results" in data
        assert "total_chunks_searched" in data
        assert "query_time_ms" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total_chunks_searched"], int)
        assert isinstance(data["query_time_ms"], float)

        # Validate individual result structure
        if data["results"]:
            result = data["results"][0]
            assert "chunk" in result
            assert "similarity_score" in result
            assert isinstance(result["similarity_score"], float)

            chunk = result["chunk"]
            assert "id" in chunk
            assert "document_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk

            # Validate chunk metadata structure
            metadata = chunk["metadata"]
            assert "creation_time" in metadata
            assert "last_update" in metadata
            assert "username" in metadata
            assert "tags" in metadata
            assert metadata["username"] == "test_user"
            assert metadata["tags"] == ["test", "validation"]


class TestErrorHandling:
    """Test cases for error handling"""

    def test_invalid_json(self):
        """Test sending invalid JSON"""
        response = client.post(
            "/api/v1/libraries",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_content_type(self):
        """Test sending data without proper content type"""
        response = client.post(
            "/api/v1/libraries",
            content='{"name": "Test"}',
            headers={"Content-Type": "text/plain"},
        )

        # Should still work with FastAPI's automatic parsing
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_201_CREATED,
        ]

    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/api/v1/nonexistent/")

        assert response.status_code == status.HTTP_404_NOT_FOUND
