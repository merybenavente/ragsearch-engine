from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from threading import RLock
from ..domain.models import Library, LibraryID, Document, DocumentID, Chunk, ChunkID
from .logging import LoggerMixin, log_performance
import time


class LibraryRepository(ABC):
    """Abstract repository for library operations"""

    @abstractmethod
    def save(self, library: Library) -> Library:
        """Save a library"""
        pass

    @abstractmethod
    def find_by_id(self, library_id: LibraryID) -> Optional[Library]:
        """Find library by ID"""
        pass

    @abstractmethod
    def find_all(self) -> List[Library]:
        """Get all libraries"""
        pass

    @abstractmethod
    def delete(self, library_id: LibraryID) -> bool:
        """Delete a library"""
        pass

    @abstractmethod
    def exists(self, library_id: LibraryID) -> bool:
        """Check if library exists"""
        pass


class InMemoryLibraryRepository(LibraryRepository, LoggerMixin):
    """In-memory implementation of library repository with thread safety"""

    def __init__(self):
        self._libraries: Dict[LibraryID, Library] = {}
        self._lock = RLock()  # Reentrant lock for thread safety
        self.logger.info("InMemoryLibraryRepository initialized")

    def save(self, library: Library) -> Library:
        """Save a library (insert or update)"""
        start_time = time.time()
        with self._lock:
            is_update = library.id in self._libraries
            self._libraries[library.id] = library
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(
                operation="library_save",
                duration_ms=duration_ms,
                library_id=library.id,
                is_update=is_update
            )
            
            self.logger.info(
                "Library saved",
                library_id=library.id,
                library_name=library.name,
                is_update=is_update,
                document_count=len(library.documents)
            )
            return library

    def find_by_id(self, library_id: LibraryID) -> Optional[Library]:
        """Find library by ID"""
        with self._lock:
            library = self._libraries.get(library_id)
            self.logger.debug(
                "Library lookup",
                library_id=library_id,
                found=library is not None
            )
            return library

    def find_all(self) -> List[Library]:
        """Get all libraries"""
        with self._lock:
            libraries = list(self._libraries.values())
            self.logger.debug(
                "Retrieved all libraries",
                count=len(libraries)
            )
            return libraries

    def delete(self, library_id: LibraryID) -> bool:
        """Delete a library"""
        with self._lock:
            if library_id in self._libraries:
                library = self._libraries[library_id]
                del self._libraries[library_id]
                self.logger.info(
                    "Library deleted",
                    library_id=library_id,
                    library_name=library.name,
                    document_count=len(library.documents)
                )
                return True
            else:
                self.logger.warning(
                    "Library not found",
                    library_id=library_id
                )
                return False

    def exists(self, library_id: LibraryID) -> bool:
        """Check if library exists"""
        with self._lock:
            return library_id in self._libraries

    def find_by_name(self, name: str) -> Optional[Library]:
        """Find library by name"""
        with self._lock:
            for library in self._libraries.values():
                if library.name == name:
                    return library
            return None


class DocumentRepository(ABC):
    """Abstract repository for document operations"""

    @abstractmethod
    def find_by_id(self, library_id: LibraryID, document_id: DocumentID) -> Optional[Document]:
        """Find document by ID within a library"""
        pass

    @abstractmethod
    def find_all_in_library(self, library_id: LibraryID) -> List[Document]:
        """Get all documents in a library"""
        pass


class ChunkRepository(ABC):
    """Abstract repository for chunk operations"""

    @abstractmethod
    def find_by_id(self, library_id: LibraryID, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find chunk by ID within a library"""
        pass

    @abstractmethod
    def find_all_in_library(self, library_id: LibraryID) -> List[Chunk]:
        """Get all chunks in a library"""
        pass

    @abstractmethod
    def find_all_in_document(self, library_id: LibraryID, document_id: DocumentID) -> List[Chunk]:
        """Get all chunks in a document"""
        pass


class RepositoryBasedDocumentRepository(DocumentRepository):
    """Document repository that works with library repository"""

    def __init__(self, library_repo: LibraryRepository):
        self.library_repo = library_repo

    def find_by_id(self, library_id: LibraryID, document_id: DocumentID) -> Optional[Document]:
        """Find document by ID within a library"""
        library = self.library_repo.find_by_id(library_id)
        if not library:
            return None
        return library.get_document_by_id(document_id)

    def find_all_in_library(self, library_id: LibraryID) -> List[Document]:
        """Get all documents in a library"""
        library = self.library_repo.find_by_id(library_id)
        if not library:
            return []
        return library.documents


class RepositoryBasedChunkRepository(ChunkRepository):
    """Chunk repository that works with library repository"""

    def __init__(self, library_repo: LibraryRepository):
        self.library_repo = library_repo

    def find_by_id(self, library_id: LibraryID, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find chunk by ID within a library"""
        library = self.library_repo.find_by_id(library_id)
        if not library:
            return None

        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                return chunk
        return None

    def find_all_in_library(self, library_id: LibraryID) -> List[Chunk]:
        """Get all chunks in a library"""
        library = self.library_repo.find_by_id(library_id)
        if not library:
            return []
        return library.get_all_chunks()

    def find_all_in_document(self, library_id: LibraryID, document_id: DocumentID) -> List[Chunk]:
        """Get all chunks in a document"""
        library = self.library_repo.find_by_id(library_id)
        if not library:
            return []

        document = library.get_document_by_id(document_id)
        if not document:
            return []

        return document.chunks