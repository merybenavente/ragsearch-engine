"""Pytest configuration and fixtures"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.vector_db.api.main import app
from src.vector_db.infrastructure.logging import configure_logging, LogLevel


def pytest_configure(config):
    """Configure pytest settings"""
    # Configure logging for tests
    configure_logging(level=LogLevel.ERROR, json_format=False)


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    with patch('src.vector_db.infrastructure.logging.get_logger') as mock:
        yield mock


@pytest.fixture(autouse=True)
def reset_repositories():
    """Reset repository state between tests"""
    # Clear the singleton repository instances
    from src.vector_db.api.dependencies import _library_repo
    if hasattr(_library_repo, '_libraries'):
        _library_repo._libraries.clear()
    yield


@pytest.fixture
def sample_library_data():
    """Sample library data for testing"""
    return {
        "name": "Test Library",
        "username": "testuser",
        "tags": ["tag1", "tag2"]
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "text": "This is a test document with some content that will be chunked.",
        "username": "testuser",
        "tags": ["doc_tag"],
        "chunk_size": 50
    }