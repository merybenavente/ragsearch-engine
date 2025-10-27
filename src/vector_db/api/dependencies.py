# New DDD Architecture Dependencies
from ..infrastructure.repositories import (
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
    RepositoryManager
)
from ..infrastructure.embedding_service import CohereEmbeddingService
from ..infrastructure.search_index import RepositoryAwareSearchIndex
from ..infrastructure.index_factory import get_index_factory
from ..application.vector_db_service import VectorDBService
from ..domain.interfaces import (
    DocumentRepository,
    LibraryRepository,
    SearchIndex,
    EmbeddingService
)

# Legacy imports for backward compatibility
from ..infrastructure.repository import (
    LibraryRepository as LegacyLibraryRepository,
    InMemoryLibraryRepository as LegacyInMemoryLibraryRepository,
    DocumentRepository as LegacyDocumentRepository,
    ChunkRepository,
    RepositoryBasedDocumentRepository,
    RepositoryBasedChunkRepository,
)
from ..application.services import DocumentService, SearchService, LibraryService, ChunkService
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
    _document_repository,
    _library_repository,
    _search_index,
    _embedding_service
)

# Legacy services for backward compatibility
try:
    from ..application.index_service import IndexService
    from ..infrastructure.indexes.naive import NaiveIndex
    
    _legacy_library_repo: LegacyLibraryRepository = LegacyInMemoryLibraryRepository()
    _legacy_document_repo: LegacyDocumentRepository = RepositoryBasedDocumentRepository(_legacy_library_repo)
    _chunk_repo: ChunkRepository = RepositoryBasedChunkRepository(_legacy_library_repo)
    
    # Create IndexService with proper parameters
    _naive_index = NaiveIndex()
    _legacy_index_service = IndexService(_naive_index, _embedding_service)
    
    _document_service: DocumentService = DocumentService(_embedding_service)
    _search_service: SearchService = SearchService(_embedding_service, _legacy_index_service)
    _library_service: LibraryService = LibraryService(_embedding_service)
    _chunk_service: ChunkService = ChunkService()
    
    logger.info("Legacy services initialized successfully")
except ImportError as e:
    # Fallback if legacy services are not available
    logger.warning(f"Legacy services not available: {e}")
    _legacy_library_repo = None
    _legacy_document_repo = None
    _chunk_repo = None
    _document_service = None
    _search_service = None
    _library_service = None
    _chunk_service = None


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


# Legacy Dependencies (for backward compatibility)
def get_legacy_library_repository() -> LegacyLibraryRepository:
    """Dependency injection for legacy library repository"""
    return _legacy_library_repo


def get_legacy_document_repository() -> LegacyDocumentRepository:
    """Dependency injection for legacy document repository"""
    return _legacy_document_repo


def get_chunk_repository() -> ChunkRepository:
    """Dependency injection for chunk repository"""
    return _chunk_repo


def get_document_service() -> DocumentService:
    """Dependency injection for document service (legacy)"""
    if _document_service is None:
        raise RuntimeError("Legacy document service not available")
    return _document_service


def get_search_service() -> SearchService:
    """Dependency injection for search service (legacy)"""
    if _search_service is None:
        raise RuntimeError("Legacy search service not available")
    return _search_service


def get_library_service() -> LibraryService:
    """Dependency injection for library service (legacy)"""
    if _library_service is None:
        raise RuntimeError("Legacy library service not available")
    return _library_service


def get_chunk_service() -> ChunkService:
    """Dependency injection for chunk service (legacy)"""
    if _chunk_service is None:
        raise RuntimeError("Legacy chunk service not available")
    return _chunk_service


# Utility functions
def clear_all_data() -> None:
    """Clear all data from repositories (for testing)"""
    _repo_manager.clear_all()


def get_repository_manager() -> RepositoryManager:
    """Get the repository manager for advanced use cases"""
    return _repo_manager