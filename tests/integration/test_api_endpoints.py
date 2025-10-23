"""Integration tests for API endpoints"""

import pytest
from httpx import AsyncClient
from fastapi import status
from uuid import uuid4

from src.vector_db.api.main import app


@pytest.fixture
async def client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    """Test cases for health check endpoints"""

    async def test_root_endpoint(self, client: AsyncClient):
        """Test root health check endpoint"""
        response = await client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Vector Database API is running"

    async def test_health_endpoint(self, client: AsyncClient):
        """Test detailed health check endpoint"""
        response = await client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "vector-db"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data


class TestLibraryEndpoints:
    """Test cases for library endpoints"""

    async def test_create_library_success(self, client: AsyncClient):
        """Test successful library creation"""
        library_data = {
            "name": "Test Library",
            "username": "testuser",
            "tags": ["tag1", "tag2"]
        }
        
        response = await client.post("/api/v1/libraries/", json=library_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]
        assert data["document_count"] == 0
        assert "id" in data
        assert "metadata" in data

    async def test_create_library_minimal(self, client: AsyncClient):
        """Test library creation with minimal data"""
        library_data = {"name": "Minimal Library"}
        
        response = await client.post("/api/v1/libraries/", json=library_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == library_data["name"]

    async def test_create_library_duplicate_name(self, client: AsyncClient):
        """Test creating library with duplicate name"""
        library_data = {"name": "Duplicate Library"}
        
        # Create first library
        response1 = await client.post("/api/v1/libraries/", json=library_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Try to create second library with same name
        response2 = await client.post("/api/v1/libraries/", json=library_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

    async def test_create_library_invalid_data(self, client: AsyncClient):
        """Test library creation with invalid data"""
        invalid_data = {"name": ""}  # Empty name
        
        response = await client.post("/api/v1/libraries/", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_get_libraries_empty(self, client: AsyncClient):
        """Test getting libraries when none exist"""
        response = await client.get("/api/v1/libraries/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    async def test_get_libraries_with_data(self, client: AsyncClient):
        """Test getting libraries when some exist"""
        # Create a library first
        library_data = {"name": "Test Library"}
        create_response = await client.post("/api/v1/libraries/", json=library_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        
        # Get all libraries
        response = await client.get("/api/v1/libraries/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == library_data["name"]

    async def test_get_library_by_id_success(self, client: AsyncClient):
        """Test getting library by ID successfully"""
        # Create a library first
        library_data = {"name": "Test Library"}
        create_response = await client.post("/api/v1/libraries/", json=library_data)
        created_library = create_response.json()
        library_id = created_library["id"]
        
        # Get library by ID
        response = await client.get(f"/api/v1/libraries/{library_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == library_data["name"]

    async def test_get_library_by_id_not_found(self, client: AsyncClient):
        """Test getting library by non-existent ID"""
        non_existent_id = str(uuid4())
        
        response = await client.get(f"/api/v1/libraries/{non_existent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_library_success(self, client: AsyncClient):
        """Test successful library update"""
        # Create a library first
        library_data = {"name": "Original Library"}
        create_response = await client.post("/api/v1/libraries/", json=library_data)
        created_library = create_response.json()
        library_id = created_library["id"]
        
        # Update library
        update_data = {
            "name": "Updated Library",
            "tags": ["new_tag"]
        }
        response = await client.put(f"/api/v1/libraries/{library_id}", json=update_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_data["name"]

    async def test_update_library_not_found(self, client: AsyncClient):
        """Test updating non-existent library"""
        non_existent_id = str(uuid4())
        update_data = {"name": "Updated Library"}
        
        response = await client.put(f"/api/v1/libraries/{non_existent_id}", json=update_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_library_duplicate_name(self, client: AsyncClient):
        """Test updating library with existing name"""
        # Create two libraries
        lib1_data = {"name": "Library 1"}
        lib2_data = {"name": "Library 2"}
        
        response1 = await client.post("/api/v1/libraries/", json=lib1_data)
        response2 = await client.post("/api/v1/libraries/", json=lib2_data)
        
        lib1_id = response1.json()["id"]
        
        # Try to update lib1 with lib2's name
        update_data = {"name": "Library 2"}
        response = await client.put(f"/api/v1/libraries/{lib1_id}", json=update_data)
        
        assert response.status_code == status.HTTP_409_CONFLICT

    async def test_delete_library_success(self, client: AsyncClient):
        """Test successful library deletion"""
        # Create a library first
        library_data = {"name": "To Delete Library"}
        create_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = create_response.json()["id"]
        
        # Delete library
        response = await client.delete(f"/api/v1/libraries/{library_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify library is deleted
        get_response = await client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_library_not_found(self, client: AsyncClient):
        """Test deleting non-existent library"""
        non_existent_id = str(uuid4())
        
        response = await client.delete(f"/api/v1/libraries/{non_existent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentEndpoints:
    """Test cases for document endpoints"""

    async def test_create_document_success(self, client: AsyncClient):
        """Test successful document creation"""
        # Create a library first
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Create document
        document_data = {
            "text": "This is a test document with some content.",
            "username": "testuser",
            "tags": ["doc_tag"],
            "chunk_size": 50
        }
        
        response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["library_id"] == library_id
        assert data["chunk_count"] > 0
        assert "id" in data
        assert "text_preview" in data

    async def test_create_document_library_not_found(self, client: AsyncClient):
        """Test creating document in non-existent library"""
        non_existent_library_id = str(uuid4())
        document_data = {"text": "Test content"}
        
        response = await client.post(
            f"/api/v1/libraries/{non_existent_library_id}/documents/", 
            json=document_data
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_create_document_invalid_data(self, client: AsyncClient):
        """Test creating document with invalid data"""
        # Create a library first
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Try to create document with empty text
        invalid_data = {"text": ""}
        
        response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=invalid_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_get_documents_in_library(self, client: AsyncClient):
        """Test getting documents in a library"""
        # Create a library
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        # Create a document
        document_data = {"text": "Test document content"}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        assert doc_response.status_code == status.HTTP_201_CREATED
        
        # Get documents
        response = await client.get(f"/api/v1/libraries/{library_id}/documents/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["library_id"] == library_id

    async def test_get_document_by_id_success(self, client: AsyncClient):
        """Test getting document by ID successfully"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Test document content"}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Get document by ID
        response = await client.get(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id
        assert data["library_id"] == library_id

    async def test_get_document_by_id_not_found(self, client: AsyncClient):
        """Test getting document by non-existent ID"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_doc_id = str(uuid4())
        
        response = await client.get(f"/api/v1/libraries/{library_id}/documents/{non_existent_doc_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_document_success(self, client: AsyncClient):
        """Test successful document update"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Original content"}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Update document
        update_data = {
            "text": "Updated content that is much longer than the original content.",
            "chunk_size": 30
        }
        response = await client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", 
            json=update_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == document_id

    async def test_delete_document_success(self, client: AsyncClient):
        """Test successful document deletion"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "To delete content"}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Delete document
        response = await client.delete(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify document is deleted
        get_response = await client.get(f"/api/v1/libraries/{library_id}/documents/{document_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestChunkEndpoints:
    """Test cases for chunk endpoints"""

    async def test_get_chunks_in_library(self, client: AsyncClient):
        """Test getting chunks in a library"""
        # Create library and document with content
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "This is test content that should be chunked into multiple pieces."}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        assert doc_response.status_code == status.HTTP_201_CREATED
        
        # Get chunks
        response = await client.get(f"/api/v1/libraries/{library_id}/chunks/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        assert all("text" in chunk for chunk in data)
        assert all("embedding" in chunk for chunk in data)

    async def test_get_chunks_library_not_found(self, client: AsyncClient):
        """Test getting chunks from non-existent library"""
        non_existent_library_id = str(uuid4())
        
        response = await client.get(f"/api/v1/libraries/{non_existent_library_id}/chunks/")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_chunk_by_id_success(self, client: AsyncClient):
        """Test getting chunk by ID successfully"""
        # Create library, document, and get chunk ID
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "Test content for chunking"}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        
        # Get chunks to find a chunk ID
        chunks_response = await client.get(f"/api/v1/libraries/{library_id}/chunks/")
        chunks = chunks_response.json()
        chunk_id = chunks[0]["id"]
        
        # Get specific chunk
        response = await client.get(f"/api/v1/libraries/{library_id}/chunks/{chunk_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == chunk_id

    async def test_get_chunk_by_id_not_found(self, client: AsyncClient):
        """Test getting chunk by non-existent ID"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_chunk_id = str(uuid4())
        
        response = await client.get(f"/api/v1/libraries/{library_id}/chunks/{non_existent_chunk_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_document_chunks(self, client: AsyncClient):
        """Test getting chunks from a specific document"""
        # Create library and document
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        document_data = {"text": "This is specific document content for chunk testing."}
        doc_response = await client.post(
            f"/api/v1/libraries/{library_id}/documents/", 
            json=document_data
        )
        document_id = doc_response.json()["id"]
        
        # Get document chunks
        response = await client.get(
            f"/api/v1/libraries/{library_id}/chunks/documents/{document_id}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        assert all(chunk["document_id"] == document_id for chunk in data)

    async def test_get_document_chunks_document_not_found(self, client: AsyncClient):
        """Test getting chunks from non-existent document"""
        # Create library
        library_data = {"name": "Test Library"}
        lib_response = await client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]
        
        non_existent_document_id = str(uuid4())
        
        response = await client.get(
            f"/api/v1/libraries/{library_id}/chunks/documents/{non_existent_document_id}"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestErrorHandling:
    """Test cases for error handling"""

    async def test_invalid_json(self, client: AsyncClient):
        """Test sending invalid JSON"""
        response = await client.post(
            "/api/v1/libraries/",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_missing_content_type(self, client: AsyncClient):
        """Test sending data without proper content type"""
        response = await client.post(
            "/api/v1/libraries/",
            content='{"name": "Test"}',
            headers={"Content-Type": "text/plain"}
        )
        
        # Should still work with FastAPI's automatic parsing
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_201_CREATED]

    async def test_nonexistent_endpoint(self, client: AsyncClient):
        """Test accessing non-existent endpoint"""
        response = await client.get("/api/v1/nonexistent/")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND