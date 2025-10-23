from typing import List, Optional
from datetime import datetime
from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata
from ..infrastructure.logging import LoggerMixin

class DocumentService(LoggerMixin):
    """Service layer for document operations"""

    def create_document(
        self,
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

        # Set content using the single content update method
        result = document.replace_content(text, chunk_size)

        self.logger.info(
            "Document created",
            library_id=library_id,
            document_id=result.id,
            text_length=len(text),
            chunk_size=chunk_size,
            chunks_created=len(result.chunks),
            username=username,
            tags=tags
        )

        return result

    def create_empty_document(
        self,
        library_id: LibraryID,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Document:
        """Create an empty document (no content, no chunks)"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        document = Document(library_id=library_id, metadata=metadata)

        self.logger.info(
            "Empty document created",
            library_id=library_id,
            document_id=document.id,
            username=username,
            tags=tags
        )

        return document

    def update_document_content(
        self,
        document: Document,
        new_text: str,
        chunk_size: int = 500
    ) -> Document:
        """Update document content (replaces all existing content and chunks)"""
        old_chunk_count = len(document.chunks)
        result = document.replace_content(new_text, chunk_size)

        self.logger.info(
            "Document content updated",
            document_id=document.id,
            text_length=len(new_text),
            chunk_size=chunk_size,
            old_chunk_count=old_chunk_count,
            new_chunk_count=len(result.chunks)
        )

        return result


class LibraryService(LoggerMixin):
    """Service layer for library operations"""

    def create_library(
        self,
        name: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Library:
        """Create a new library"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        library = Library(name=name, metadata=metadata)

        self.logger.info(
            "Library created",
            library_id=library.id,
            name=name,
            username=username,
            tags=tags
        )

        return library

    def add_document_to_library(
        self,
        library: Library,
        document: Document
    ) -> Library:
        """Add a document to a library"""
        if library.document_exists(document.id):
            self.logger.warning(
                "Attempted to add duplicate document",
                library_id=library.id,
                document_id=document.id
            )
            raise ValueError(f"Document {document.id} already exists in library")

        result = library.add_document(document)

        self.logger.info(
            "Document added to library",
            library_id=library.id,
            document_id=document.id,
            library_document_count=len(result.documents)
        )

        return result

    @staticmethod
    def remove_document_from_library(library: Library, document_id: DocumentID) -> Library:
        """Remove a document from a library"""
        if not library.document_exists(document_id):
            raise ValueError(f"Document {document_id} not found in library")

        return library.remove_document(document_id)

    @staticmethod
    def update_document_in_library(library: Library, updated_document: Document) -> Library:
        """Update a document within a library"""
        if not library.document_exists(updated_document.id):
            raise ValueError(f"Document {updated_document.id} not found in library")

        return library.update_document(updated_document)

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

        return library.model_copy(update=updates)


class ChunkService(LoggerMixin):
    """
    Service layer for chunk operations (read-only).

    Chunks are derived data from documents. All chunk modifications
    happen through document content updates which regenerate chunks automatically.

    This service provides read-only access to chunks for search and retrieval.
    """

    def get_chunks_from_library(self, library: Library) -> List[Chunk]:
        """Get all chunks from a library (read-only)"""
        chunks = library.get_all_chunks()
        self.logger.debug(
            "Retrieved chunks from library",
            library_id=library.id,
            chunk_count=len(chunks)
        )
        return chunks

    def get_chunks_from_document(self, document: Document) -> List[Chunk]:
        """Get all chunks from a document (read-only)"""
        chunks = document.chunks
        self.logger.debug(
            "Retrieved chunks from document",
            document_id=document.id,
            chunk_count=len(chunks)
        )
        return chunks

    def get_chunk_from_library(self, library: Library, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a library (read-only)"""
        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                self.logger.debug(
                    "Found chunk in library",
                    library_id=library.id,
                    chunk_id=chunk_id,
                    document_id=document.id
                )
                return chunk

        self.logger.debug(
            "Chunk not found in library",
            library_id=library.id,
            chunk_id=chunk_id
        )
        return None

    def get_chunk_from_document(self, document: Document, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a document (read-only)"""
        chunk = document.get_chunk_by_id(chunk_id)
        self.logger.debug(
            "Chunk lookup in document",
            document_id=document.id,
            chunk_id=chunk_id,
            found=chunk is not None
        )
        return chunk