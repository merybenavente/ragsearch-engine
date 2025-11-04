"""Pytest configuration and fixtures"""

import hashlib
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock

from src.vector_db.infrastructure.logging import configure_logging, LogLevel


def pytest_configure(config):
    """Configure pytest settings"""
    # Configure logging for tests
    configure_logging(level=LogLevel.ERROR, json_format=False)


# Configure asyncio mode for pytest-asyncio
pytest_asyncio.asyncio_mode = "auto"


@pytest.fixture(autouse=True)
def reset_repositories():
    """Reset repository state between tests"""
    # Clear the singleton repository instances using the new architecture
    from src.vector_db.api.dependencies import clear_all_data

    clear_all_data()
    yield


@pytest.fixture
def sample_library_data():
    """Sample library data for testing"""
    return {
        "name": "Test Library",
        "username": "testuser",
        "tags": ["tag1", "tag2"],
        "index_type": "naive",
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "text": "This is a test document with some content that will be chunked.",
        "username": "testuser",
        "tags": ["doc_tag"],
        "chunk_size": 50,
    }


@pytest.fixture
def mock_cohere_deterministic():
    """Mock Cohere embedding API with deterministic embeddings based on text content"""
    from src.vector_db.domain.models import EMBEDDING_DIMENSION

    def create_deterministic_embedding(text: str) -> list[float]:
        """Create deterministic mock embedding based on text hash"""
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        embedding = np.random.randn(EMBEDDING_DIMENSION)
        return (embedding / np.linalg.norm(embedding)).tolist()

    with patch("src.vector_db.infrastructure.embedding_service.co") as mock_co:

        def mock_embed(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = [create_deterministic_embedding(text) for text in texts]
            mock_response = MagicMock()
            mock_response.embeddings = embeddings
            return mock_response

        mock_co.embed = mock_embed
        yield mock_co
