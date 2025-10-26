from ..infrastructure.repository import (
    LibraryRepository,
    InMemoryLibraryRepository,
    DocumentRepository,
    ChunkRepository,
    RepositoryBasedDocumentRepository,
    RepositoryBasedChunkRepository,
)
from ..infrastructure.embedding_service import CohereEmbeddingService
from ..application.services import DocumentService, SearchService, LibraryService, ChunkService
from ..domain.interfaces import EmbeddingService

# Repository instances (singletons)
_library_repo: LibraryRepository = InMemoryLibraryRepository()
_document_repo: DocumentRepository = RepositoryBasedDocumentRepository(_library_repo)
_chunk_repo: ChunkRepository = RepositoryBasedChunkRepository(_library_repo)

# Service instances (singletons)
_embedding_service: EmbeddingService = CohereEmbeddingService()
_document_service: DocumentService = DocumentService(_embedding_service)
_search_service: SearchService = SearchService(_embedding_service)
_library_service: LibraryService = LibraryService()
_chunk_service: ChunkService = ChunkService()


def get_library_repository() -> LibraryRepository:
    """Dependency injection for library repository"""
    return _library_repo


def get_document_repository() -> DocumentRepository:
    """Dependency injection for document repository"""
    return _document_repo


def get_chunk_repository() -> ChunkRepository:
    """Dependency injection for chunk repository"""
    return _chunk_repo


def get_embedding_service() -> EmbeddingService:
    """Dependency injection for embedding service"""
    return _embedding_service


def get_document_service() -> DocumentService:
    """Dependency injection for document service"""
    return _document_service


def get_search_service() -> SearchService:
    """Dependency injection for search service"""
    return _search_service


def get_library_service() -> LibraryService:
    """Dependency injection for library service"""
    return _library_service


def get_chunk_service() -> ChunkService:
    """Dependency injection for chunk service"""
    return _chunk_service