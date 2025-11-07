"""Pytest configuration and fixtures"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.vector_db.infrastructure.logging import configure_logging, LogLevel
from tests.utils import create_deterministic_embedding


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


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


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================


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


# ============================================================================
# PRIMARY MOCKS
# ============================================================================


@pytest.fixture
def mock_cohere_deterministic():
    """Mock Cohere embedding API with deterministic embeddings based on text content"""
    from src.vector_db.domain.models import EMBEDDING_DIMENSION

    with patch("src.vector_db.infrastructure.embedding_service.co") as mock_co:

        def mock_embed(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = [
                create_deterministic_embedding(text, EMBEDDING_DIMENSION)
                for text in texts
            ]
            mock_response = MagicMock()
            mock_response.embeddings = embeddings
            return mock_response

        mock_co.embed = mock_embed
        yield mock_co


# ============================================================================
# DOMAIN LAYER MOCKS
# ============================================================================


@pytest.fixture
def mock_datetime():
    """Mock datetime.now() for deterministic timestamp testing"""
    fixed_time = datetime(2025, 12, 31, 23, 59, 59)  # Far future time

    with patch("src.vector_db.domain.models.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        # Keep the real datetime class for other uses
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock_dt


# ============================================================================
# SPECIALIZED MOCKS
# ============================================================================


@pytest.fixture
def mock_embedding_service_failure():
    """Mock embedding service that always fails - for error testing"""
    from src.vector_db.domain.interfaces import EmbeddingService

    class FailingEmbeddingService(EmbeddingService):
        def create_embedding(
            self, text: str, input_type: str = "search_document"
        ) -> list[float]:
            raise RuntimeError("Embedding service failed")

    return FailingEmbeddingService()


@pytest.fixture
def deterministic_embeddings():
    """Generate deterministic embeddings for testing index implementations"""
    return create_deterministic_embedding


# ============================================================================
# SERVICE LAYER FIXTURES
# ============================================================================


@pytest.fixture
def vector_db_service():
    """Provide a clean VectorDBService instance for each test"""
    from src.vector_db.api.dependencies import get_vector_db_service

    return get_vector_db_service()


@pytest.fixture
def document_repository():
    """Provide a clean document repository for each test"""
    from src.vector_db.api.dependencies import get_document_repository

    return get_document_repository()


@pytest.fixture
def library_repository():
    """Provide a clean library repository for each test"""
    from src.vector_db.api.dependencies import get_library_repository

    return get_library_repository()


@pytest.fixture
def vector_db_service_instance():
    """Provide a fully configured VectorDBService instance for tests that need manual setup"""
    from src.vector_db.application.vector_db_service import VectorDBService
    from src.vector_db.infrastructure.repositories import RepositoryManager
    from src.vector_db.infrastructure.search_index import RepositoryAwareSearchIndex
    from src.vector_db.infrastructure.index_factory import IndexFactory
    from src.vector_db.infrastructure.embedding_service import CohereEmbeddingService

    repo_manager = RepositoryManager()
    search_index = RepositoryAwareSearchIndex(IndexFactory())
    embedding_service = CohereEmbeddingService()

    return VectorDBService(
        repo_manager.get_document_repository(),
        repo_manager.get_library_repository(),
        search_index,
        embedding_service,
    )


@pytest.fixture
def index_factory_instance():
    """Provide an IndexFactory instance for testing"""
    from src.vector_db.infrastructure.index_factory import IndexFactory

    return IndexFactory()
