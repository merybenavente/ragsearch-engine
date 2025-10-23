"""Application services - business logic"""
from typing import List, Optional
from datetime import datetime

from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata
from ..infrastructure.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Service layer for document operations"""

    @staticmethod
    def create_document(
        library_id: LibraryID,
        text: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None,
        chunk_size: int = 500
    ) -> Document:
        """Create a new document with content"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        document = Document(library_id=library_id, metadata=metadata)
        result = document.replace_content(text, chunk_size)

        logger.info("Document created", doc_id=result.id, chunks=len(result.chunks))
        return result

    @staticmethod
    def create_empty_document(
        library_id: LibraryID,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Document:
        """Create an empty document (no content, no chunks)"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        document = Document(library_id=library_id, metadata=metadata)

        logger.info("Empty document created", doc_id=document.id)
        return document

    @staticmethod
    def update_document_content(
        document: Document,
        new_text: str,
        chunk_size: int = 500
    ) -> Document:
        """Update document content (replaces all existing content and chunks)"""
        old_chunk_count = len(document.chunks)
        result = document.replace_content(new_text, chunk_size)

        logger.info("Document updated", doc_id=document.id, chunks=f"{old_chunk_count}→{len(result.chunks)}")
        return result


class LibraryService:
    """Service layer for library operations"""

    @staticmethod
    def create_library(
        name: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Library:
        """Create a new library"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        library = Library(name=name, metadata=metadata)

        logger.info("Library created", lib_id=library.id, name=name)
        return library

    @staticmethod
    def add_document_to_library(library: Library, document: Document) -> Library:
        """Add a document to a library"""
        if library.document_exists(document.id):
            logger.warning("Duplicate document", doc_id=document.id)
            raise ValueError(f"Document {document.id} already exists in library")

        result = library.add_document(document)
        logger.info("Document added", doc_id=document.id, total_docs=len(result.documents))
        return result

    @staticmethod
    def remove_document_from_library(library: Library, document_id: DocumentID) -> Library:
        """Remove a document from a library"""
        if not library.document_exists(document_id):
            raise ValueError(f"Document {document_id} not found in library")

        result = library.remove_document(document_id)
        logger.info("Document removed", doc_id=document_id, total_docs=len(result.documents))
        return result

    @staticmethod
    def update_document_in_library(library: Library, updated_document: Document) -> Library:
        """Update a document within a library"""
        if not library.document_exists(updated_document.id):
            raise ValueError(f"Document {updated_document.id} not found in library")

        result = library.update_document(updated_document)
        logger.info("Document updated in library", doc_id=updated_document.id)
        return result

    @staticmethod
    def update_library_metadata(
        library: Library,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Library:
        """Update library metadata"""
        updates = {'metadata': library.metadata.update_timestamp()}

        if name is not None:
            updates['name'] = name
        if tags is not None:
            updates['metadata'] = library.metadata.model_copy(update={
                'tags': tags,
                'last_update': datetime.now()
            })

        result = library.model_copy(update=updates)
        logger.info("Library metadata updated", lib_id=library.id)
        return result


class ChunkService:
    """
    Service layer for chunk operations (read-only).

    Chunks are derived data from documents. All chunk modifications
    happen through document content updates which regenerate chunks automatically.

    This service provides read-only access to chunks for search and retrieval.
    """

    @staticmethod
    def get_chunks_from_library(library: Library) -> List[Chunk]:
        """Get all chunks from a library (read-only)"""
        chunks = library.get_all_chunks()
        logger.debug("Retrieved chunks from library", lib_id=library.id, count=len(chunks))
        return chunks

    @staticmethod
    def get_chunks_from_document(document: Document) -> List[Chunk]:
        """Get all chunks from a document (read-only)"""
        chunks = document.chunks
        logger.debug("Retrieved chunks from document", doc_id=document.id, count=len(chunks))
        return chunks

    @staticmethod
    def get_chunk_from_library(library: Library, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a library (read-only)"""
        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                logger.debug("Found chunk", chunk_id=chunk_id, doc_id=document.id)
                return chunk

        logger.debug("Chunk not found", chunk_id=chunk_id, lib_id=library.id)
        return None

    @staticmethod
    def get_chunk_from_document(document: Document, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a document (read-only)"""
        chunk = document.get_chunk_by_id(chunk_id)
        logger.debug("Chunk lookup", doc_id=document.id, chunk_id=chunk_id, found=chunk is not None)
        return chunk