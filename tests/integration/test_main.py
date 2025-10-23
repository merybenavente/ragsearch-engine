import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Vector Database API is running"}


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "vector-db"
    assert data["version"] == "0.1.0"


def test_create_vector():
    """Test creating a vector"""
    vector_data = {
        "data": [1.0, 2.0, 3.0],
        "metadata": {"test": "data"}
    }
    response = client.post("/vectors", json=vector_data)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["data"] == [1.0, 2.0, 3.0]
    assert data["metadata"] == {"test": "data"}


def test_get_vector():
    """Test retrieving a vector by ID"""
    response = client.get("/vectors/test_id")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test_id"
    assert data["data"] == [1.0, 2.0, 3.0]


def test_search_vectors():
    """Test vector search functionality"""
    query_data = {
        "data": [1.0, 2.0, 3.0]
    }
    response = client.post("/vectors/search", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10  # Default limit
    
    # Test with custom limit
    response = client.post("/vectors/search?limit=2", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 2


def test_create_vector_with_custom_id():
    """Test creating a vector with a custom ID"""
    vector_data = {
        "id": "custom_id_123",
        "data": [4.0, 5.0, 6.0],
        "metadata": {"custom": True}
    }
    response = client.post("/vectors", json=vector_data)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "custom_id_123"
    assert data["data"] == [4.0, 5.0, 6.0]

