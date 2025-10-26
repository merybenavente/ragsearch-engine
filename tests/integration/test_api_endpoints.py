"""Integration tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from uuid import uuid4

from src.vector_db.api.main import app

client = TestClient(app)


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

    def test_create_library_success(self):
        """Test successful library creation"""
        library_data = {
            "name": "Test Library",
            "username": "testuser",
            "tags": ["tag1", "tag2"]
        }
        
        response = client.post("/api/v1/libraries/", json=library_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0
        assert "id" in data
        assert "metadata" in data

    def test_create_library_minimal(self):
        """Test library creation with minimal data"""
        library_data = {"name": "Minimal Library"}
        
        response = client.post("/api/v1/libraries/", json=library_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]

    def test_create_library_duplicate_name(self):
        """Test creating library with duplicate name"""
        library_data = {"name": "Duplicate Library"}
        
        # Check current libraries first
        response_check = client.get("/api/v1/libraries/")
        
        # Create first library
        response1 = client.post("/api/v1/libraries/", json=library_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Try to create second library with same name
        response2 = client.post("/api/v1/libraries/", json=library_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

    def test_create_library_invalid_data(self):
        """Test library creation with invalid data"""
        invalid_data = {"name": ""}  # Empty name
        
        response = client.post("/api/v1/libraries/", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_libraries_empty(self):
        """Test getting libraries when none exist"""
        response = client.get("/api/v1/libraries/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_libraries_with_data(self):
        """Test getting libraries when some exist"""
        # Create a library first
        library_data = {"name": "Test Library"}
        create_response = client.post("/api/v1/libraries/", json=library_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        
        # Get all libraries
        response = client.get("/api/v1/libraries/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == library_data["name"]

    def test_get_library_by_id_success(self):
        """Test getting library by ID successfully"""
        # Create a library first
        library_data = {"name": "Test Library"}
        create_response = client.post("/api/v1/libraries/", json=library_data)
        created_library = create_response.json()
        library_id = created_library["id"]
        
        # Get library by ID
        response = client.get(f"/api/v1/libraries/{library_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == library_data["name"]

    def test_get_library_by_id_not_found(self):
        """Test getting library by non-existent ID"""
        non_existent_id = str(uuid4())
        
        response = client.get(f"/api/v1/libraries/{non_existent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_library_success(self):
        """Test successful library update"""
        # Create a library first
        library_data = {"name": "Original Library"}
        create_response = client.post("/api/v1/libraries/", json=library_data)
        created_library = create_response.json()
        library_id = created_library["id"]
        
        # Update library
        update_data = {
            "name": "Updated Library",
            "tags": ["new_tag"]
        }
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
        
        response1 = client.post("/api/v1/libraries/", json=lib1_data)
        response2 = client.post("/api/v1/libraries/", json=lib2_data)
        
        lib1_id = response1.json()["id"]
        
        # Try to update lib1 with lib2's name
        update_data = {"name": "Library 2"}
        response = client.put(f"/api/v1/libraries/{lib1_id}", json=update_data)
        
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_delete_library_success(self):
        """Test successful library deletion"""
        # Create a library first
        library_data = {"name": "To Delete Library"}
        create_response = client.post("/api/v1/libraries/", json=library_data)
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

    def test_create_document_success(self):
        """Test successful document creation"""
        # Create a library first
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Create document
        document_data = {
            "text": "This is a test document with some content.",
            "username": "testuser",
            "tags": ["doc_tag"],
            "chunk_size": 50
        }
        
        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["library_id"] == library_id
        assert data["chunk_count"] > 0
        assert "id" in data
        assert "text_preview" in data

    def test_create_document_library_not_found(self):
        """Test creating document in non-existent library"""
        non_existent_library_id = str(uuid4())
        document_data = {"text": "Test content"}
        
        response = client.post(
            f"/api/v1/libraries/{non_existent_library_id}/documents/", 
            json=document_data
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_document_invalid_data(self):
        """Test creating document with invalid data"""
        # Create a library first
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Try to create document with empty text
        invalid_data = {"text": ""}
        
        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=invalid_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_documents_in_library(self):
        """Test getting documents in a library"""
        # Create a library
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Create a document
        document_data = {"text": "Test document content"}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        assert doc_response.status_code == status.HTTP_201_CREATED
        
        # Get documents
        response = client.get(f"/api/v1/libraries/{library_id}/documents/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["library_id"] == library_id

    def test_get_document_by_id_success(self):
        """Test getting document by ID successfully"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Test document content"}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Get document by ID
        response = client.get(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id
        assert data["library_id"] == library_id

    def test_get_document_by_id_not_found(self):
        """Test getting document by non-existent ID"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_doc_id = str(uuid4())
        
        response = client.get(f"/api/v1/libraries/{library_id}/documents/{non_existent_doc_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_document_success(self):
        """Test successful document update"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Original content"}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Update document
        update_data = {
            "text": "Updated content that is much longer than the original content.",
            "chunk_size": 30
        }
        response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", 
            json=update_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id

    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "To delete content"}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Delete document
        response = client.delete(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify document is deleted
        get_response = client.get(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestChunkEndpoints:
    """Test cases for chunk endpoints"""

    def test_get_chunks_in_library(self):
        """Test getting chunks in a library"""
        # Create library and document with content
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "This is test content that should be chunked into multiple pieces."}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        assert doc_response.status_code == status.HTTP_201_CREATED
        
        # Get chunks
        response = client.get(f"/api/v1/libraries/{library_id}/chunks/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        assert all("text" in chunk for chunk in data)
        assert all("embedding" in chunk for chunk in data)

    def test_get_chunks_library_not_found(self):
        """Test getting chunks from non-existent library"""
        non_existent_library_id = str(uuid4())
        
        response = client.get(f"/api/v1/libraries/{non_existent_library_id}/chunks/")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_chunk_by_id_success(self):
        """Test getting chunk by ID successfully"""
        # Create library, document, and get chunk ID
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Test content for chunking"}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        
        # Get chunks to find a chunk ID
        chunks_response = client.get(f"/api/v1/libraries/{library_id}/chunks/")
        chunks = chunks_response.json()
        chunk_id = chunks[0]["id"]
        
        # Get specific chunk
        response = client.get(f"/api/v1/libraries/{library_id}/chunks/{chunk_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == chunk_id

    def test_get_chunk_by_id_not_found(self):
        """Test getting chunk by non-existent ID"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_chunk_id = str(uuid4())
        
        response = client.get(f"/api/v1/libraries/{library_id}/chunks/{non_existent_chunk_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_document_chunks(self):
        """Test getting chunks from a specific document"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "This is specific document content for chunk testing."}
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Get document chunks
        response = client.get(
            f"/api/v1/libraries/{library_id}/chunks/documents/{document_id}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        assert all(chunk["document_id"] == document_id for chunk in data)

    def test_get_document_chunks_document_not_found(self):
        """Test getting chunks from non-existent document"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_document_id = str(uuid4())
        
        response = client.get(
            f"/api/v1/libraries/{library_id}/chunks/documents/{non_existent_document_id}"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestErrorHandling:
    """Test cases for error handling"""

    def test_invalid_json(self):
        """Test sending invalid JSON"""
        response = client.post(
            "/api/v1/libraries/",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_content_type(self):
        """Test sending data without proper content type"""
        response = client.post(
            "/api/v1/libraries/",
            content='{"name": "Test"}',
            headers={"Content-Type": "text/plain"}
        )
        
        # Should still work with FastAPI's automatic parsing
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_201_CREATED]

    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/api/v1/nonexistent/")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND