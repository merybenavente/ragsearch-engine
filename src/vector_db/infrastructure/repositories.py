"""
In-memory repository implementations for testing and development.

These repositories store data in memory and are suitable for development and testing.
For production, these would be replaced with database-backed implementations.
"""
from typing import Dict, List, Optional
from threading import RLock
from ..domain.models import Document, DocumentID, Library, LibraryID
from ..domain.interfaces import DocumentRepository, LibraryRepository
from ..infrastructure.logging import get_logger

logger = get_logger(__name__)


class InMemoryDocumentRepository(DocumentRepository):
    """In-memory implementation of DocumentRepository"""
    
    def __init__(self):
        self._documents: Dict[DocumentID, Document] = {}
        self._library_documents: Dict[LibraryID, List[DocumentID]] = {}
        self._lock = RLock()  # Thread safety for concurrent access
    
    def save(self, document: Document) -> None:
        """Save a document"""
        with self._lock:
            self._documents[document.id] = document
            
            # Update library mapping
            if document.library_id not in self._library_documents:
                self._library_documents[document.library_id] = []
            
            # Add to library mapping if not already present
            if document.id not in self._library_documents[document.library_id]:
                self._library_documents[document.library_id].append(document.id)
            
            logger.debug("Document saved", doc_id=document.id, lib_id=document.library_id)
    
    def get(self, document_id: DocumentID) -> Optional[Document]:
        """Get a document by ID"""
        with self._lock:
            document = self._documents.get(document_id)
            logger.debug("Document retrieved", doc_id=document_id, found=document is not None)
            return document
    
    def get_by_library(self, library_id: LibraryID) -> List[Document]:
        """Get all documents in a library"""
        with self._lock:
            if library_id not in self._library_documents:
                logger.debug("No documents found in library", lib_id=library_id)
                return []
            
            documents = []
            for doc_id in self._library_documents[library_id]:
                document = self._documents.get(doc_id)
                if document:
                    documents.append(document)
            
            logger.debug("Documents retrieved from library", lib_id=library_id, count=len(documents))
            return documents
    
    def delete(self, document_id: DocumentID) -> None:
        """Delete a document"""
        with self._lock:
            document = self._documents.get(document_id)
            if not document:
                logger.warning("Document not found for deletion", doc_id=document_id)
                return
            
            # Remove from documents
            del self._documents[document_id]
            
            # Remove from library mapping
            if document.library_id in self._library_documents:
                try:
                    self._library_documents[document.library_id].remove(document_id)
                    # Clean up empty library mappings
                    if not self._library_documents[document.library_id]:
                        del self._library_documents[document.library_id]
                except ValueError:
                    pass  # Document wasn't in the library mapping
            
            logger.debug("Document deleted", doc_id=document_id, lib_id=document.library_id)
    
    def exists(self, document_id: DocumentID) -> bool:
        """Check if a document exists"""
        with self._lock:
            exists = document_id in self._documents
            logger.debug("Document existence check", doc_id=document_id, exists=exists)
            return exists
    
    def clear(self) -> None:
        """Clear all documents (for testing)"""
        with self._lock:
            self._documents.clear()
            self._library_documents.clear()
            logger.debug("All documents cleared")


class InMemoryLibraryRepository(LibraryRepository):
    """In-memory implementation of LibraryRepository"""
    
    def __init__(self):
        self._libraries: Dict[LibraryID, Library] = {}
        self._lock = RLock()  # Thread safety for concurrent access
    
    def save(self, library: Library) -> None:
        """Save a library"""
        with self._lock:
            self._libraries[library.id] = library
            logger.debug("Library saved", lib_id=library.id, name=library.name)
    
    def get(self, library_id: LibraryID) -> Optional[Library]:
        """Get a library by ID"""
        with self._lock:
            library = self._libraries.get(library_id)
            logger.debug("Library retrieved", lib_id=library_id, found=library is not None)
            return library
    
    def delete(self, library_id: LibraryID) -> None:
        """Delete a library"""
        with self._lock:
            if library_id in self._libraries:
                library_name = self._libraries[library_id].name
                del self._libraries[library_id]
                logger.debug("Library deleted", lib_id=library_id, name=library_name)
            else:
                logger.warning("Library not found for deletion", lib_id=library_id)
    
    def exists(self, library_id: LibraryID) -> bool:
        """Check if a library exists"""
        with self._lock:
            exists = library_id in self._libraries
            logger.debug("Library existence check", lib_id=library_id, exists=exists)
            return exists
    
    def list_all(self) -> List[Library]:
        """List all libraries"""
        with self._lock:
            libraries = list(self._libraries.values())
            logger.debug("All libraries retrieved", count=len(libraries))
            return libraries
    
    def clear(self) -> None:
        """Clear all libraries (for testing)"""
        with self._lock:
            self._libraries.clear()
            logger.debug("All libraries cleared")


class RepositoryManager:
    """
    Utility class to manage repository lifecycle and provide factory methods.
    
    This is useful for testing and dependency injection setup.
    """
    
    def __init__(self):
        self.document_repo = InMemoryDocumentRepository()
        self.library_repo = InMemoryLibraryRepository()
        self._lock = RLock()  # Thread safety for coordination operations
    
    def clear_all(self) -> None:
        """Clear all repositories (for testing)"""
        with self._lock:
            self.document_repo.clear()
            self.library_repo.clear()
            logger.info("All repositories cleared")
    
    def get_document_repository(self) -> DocumentRepository:
        """Get the document repository instance"""
        return self.document_repo
    
    def get_library_repository(self) -> LibraryRepository:
        """Get the library repository instance"""
        return self.library_repo