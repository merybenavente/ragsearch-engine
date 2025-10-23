from typing import List, Optional
from datetime import datetime
from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata


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

        # Set content using the single content update method
        return document.replace_content(text, chunk_size)

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
        return Document(library_id=library_id, metadata=metadata)

    @staticmethod
    def update_document_content(
        document: Document,
        new_text: str,
        chunk_size: int = 500
    ) -> Document:
        """Update document content (replaces all existing content and chunks)"""
        return document.replace_content(new_text, chunk_size)


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
        return Library(name=name, metadata=metadata)

    @staticmethod
    def add_document_to_library(library: Library, document: Document) -> Library:
        """Add a document to a library"""
        if library.document_exists(document.id):
            raise ValueError(f"Document {document.id} already exists in library")

        return library.add_document(document)

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
        return library.get_all_chunks()

    @staticmethod
    def get_chunks_from_document(document: Document) -> List[Chunk]:
        """Get all chunks from a document (read-only)"""
        return document.chunks

    @staticmethod
    def get_chunk_from_library(library: Library, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a library (read-only)"""
        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                return chunk
        return None

    @staticmethod
    def get_chunk_from_document(document: Document, chunk_id: ChunkID) -> Optional[Chunk]:
        """Find a specific chunk in a document (read-only)"""
        return document.get_chunk_by_id(chunk_id)