# New DDD Architecture Dependencies
from ..infrastructure.repositories import RepositoryManager
from ..infrastructure.embedding_service import CohereEmbeddingService
from ..infrastructure.search_index import RepositoryAwareSearchIndex
from ..infrastructure.index_factory import get_index_factory
from ..application.vector_db_service import VectorDBService
from ..domain.interfaces import (
    DocumentRepository,
    LibraryRepository,
    SearchIndex,
    EmbeddingService,
)

from ..infrastructure.logging import get_logger

logger = get_logger(__name__)

# New DDD Architecture (Recommended)
_repo_manager = RepositoryManager()
_document_repository: DocumentRepository = _repo_manager.get_document_repository()
_library_repository: LibraryRepository = _repo_manager.get_library_repository()
_embedding_service: EmbeddingService = CohereEmbeddingService()
_index_factory = get_index_factory()
_search_index: SearchIndex = RepositoryAwareSearchIndex(_index_factory)
_vector_db_service = VectorDBService(
    _document_repository, _library_repository, _search_index, _embedding_service
)


# New DDD Architecture Dependencies
def get_vector_db_service() -> VectorDBService:
    """Dependency injection for the main VectorDB service (recommended)"""
    return _vector_db_service


def get_document_repository() -> DocumentRepository:
    """Dependency injection for document repository"""
    return _document_repository


def get_library_repository() -> LibraryRepository:
    """Dependency injection for library repository"""
    return _library_repository


def get_search_index() -> SearchIndex:
    """Dependency injection for search index"""
    return _search_index


def get_embedding_service() -> EmbeddingService:
    """Dependency injection for embedding service"""
    return _embedding_service


# Utility functions
def clear_all_data() -> None:
    """Clear all data from repositories (for testing)"""
    _repo_manager.clear_all()


def get_repository_manager() -> RepositoryManager:
    """Get the repository manager for advanced use cases"""
    return _repo_manager
