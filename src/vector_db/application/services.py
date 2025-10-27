"""Application services - business logic"""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
import numpy as np

from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata, EMBEDDING_MODEL
from ..domain.interfaces import EmbeddingService, VectorIndex
from ..infrastructure.logging import get_logger

if TYPE_CHECKING:
    from .index_service import IndexService

logger = get_logger(__name__)

class DocumentService:
    """Service layer for document operations"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self._embedding_service = embedding_service

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
        result = self._replace_document_content(document, text, chunk_size)

        logger.info("Document created", doc_id=result.id, chunks=len(result.chunks))
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

        logger.info("Empty document created", doc_id=document.id)
        return document

    def update_document_content(
        self,
        document: Document,
        new_text: str,
        chunk_size: int = 500
    ) -> Document:
        """Update document content (replaces all existing content and chunks)"""
        old_chunk_count = len(document.chunks)
        result = self._replace_document_content(document, new_text, chunk_size)

        logger.info("Document updated", doc_id=document.id, chunks=f"{old_chunk_count}→{len(result.chunks)}")
        return result

    def _replace_document_content(self, document: Document, text: str, chunk_size: int = 500) -> Document:
        """
        Replace document content with proper chunking and embedding generation.
        
        This method handles:
        1. Text chunking 
        2. Embedding generation for each chunk
        3. Document update with new chunks
        """
        if not text:
            # Empty content = no chunks
            return document.model_copy(update={
                'chunks': [],
                'metadata': document.metadata.update_timestamp()
            })

        # Generate chunks with embeddings
        chunks = self._create_chunks_from_text(document.id, text, chunk_size)
        
        return document.model_copy(update={
            'chunks': chunks,
            'metadata': document.metadata.update_timestamp()
        })

    def _create_chunks_from_text(self, document_id: DocumentID, text: str, chunk_size: int = 500) -> List[Chunk]:
        """
        Split text into chunks and create Chunk objects with embeddings.
        
        This method handles both chunking and embedding generation to ensure
        chunks are never created without embeddings.
        """
        chunks = []

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text:  # Only create non-empty chunks
                # Generate embedding for this chunk
                embedding = self._embedding_service.create_embedding(chunk_text)
                
                chunk = Chunk(
                    document_id=document_id,
                    text=chunk_text,
                    embedding=embedding
                )
                chunks.append(chunk)

        return chunks


class LibraryService:
    """Service layer for library operations - pure data management"""

    def __init__(self, embedding_service: EmbeddingService):
        self._embedding_service = embedding_service

    def create_library(
        self,
        name: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None,
        index_type: str = "naive"
    ) -> Library:
        """Create a new library with specified index type"""
        if tags is None:
            tags = []

        metadata = Metadata(username=username, tags=tags)
        library = Library(name=name, metadata=metadata, index_type=index_type)

        logger.info("Library created", lib_id=library.id, name=name, index_type=index_type)
        return library


    def add_document_to_library(self, library: Library, document: Document) -> Library:
        """Add a document to a library - pure data management"""
        if library.document_exists(document.id):
            logger.warning("Duplicate document", doc_id=document.id)
            raise ValueError(f"Document {document.id} already exists in library")

        result = library.add_document(document)
        logger.info("Document added", doc_id=document.id, total_docs=len(result.documents))
        return result

    def remove_document_from_library(self, library: Library, document_id: DocumentID) -> Library:
        """Remove a document from a library - pure data management"""
        if not library.document_exists(document_id):
            raise ValueError(f"Document {document_id} not found in library")

        result = library.remove_document(document_id)
        logger.info("Document removed", doc_id=document_id, total_docs=len(result.documents))
        return result

    # TODO: this should probably be moved to the document service?
    def update_document_in_library(self, library: Library, updated_document: Document) -> Library:
        """Update a document within a library - pure data management"""
        if not library.document_exists(updated_document.id):
            raise ValueError(f"Document {updated_document.id} not found in library")

        result = library.update_document(updated_document)
        logger.info("Document updated in library", doc_id=updated_document.id)
        return result

    def update_library_metadata(
        self,
        library: Library,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Library:
        """Update library metadata"""
        # Early return if no changes
        if name is None and tags is None:
            return library

        updates = {}

        if name is not None:
            updates['name'] = name

        # Actualizar los tags y el timestamp correctamente
        new_metadata = library.metadata
        if tags is not None:
            new_metadata = new_metadata.model_copy(update={'tags': tags})
        new_metadata = new_metadata.update_timestamp()
        updates['metadata'] = new_metadata

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

    def get_chunks_from_library(self, library: Library) -> List[Chunk]:
        """Get all chunks from a library (read-only)"""
        chunks = library.get_all_chunks()
        logger.debug("Retrieved chunks from library", lib_id=library.id, count=len(chunks))
        return chunks

    def get_chunks_from_document(self, document: Document) -> List[Chunk]:
        """Get all chunks from a document (read-only)"""
        chunks = document.chunks
        logger.debug("Retrieved chunks from document", doc_id=document.id, count=len(chunks))
        return chunks

    def get_chunk_from_library(self, library: Library, chunk_id: ChunkID) -> Chunk:
        """Find a specific chunk in a library (read-only)"""
        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                logger.debug("Found chunk", chunk_id=chunk_id, doc_id=document.id)
                return chunk

        logger.debug("Chunk not found", chunk_id=chunk_id, lib_id=library.id)
        raise ValueError(f"Chunk with id {chunk_id} not found in library")

    def get_chunk_from_document(self, document: Document, chunk_id: ChunkID) -> Chunk:
        """Find a specific chunk in a document (read-only)"""
        chunk = document.get_chunk_by_id(chunk_id)
        logger.info("Chunk retrieved", doc_id=document.id, chunk_id=chunk_id, found=chunk is not None)
        if chunk is None:
            raise ValueError(f"Chunk with id {chunk_id} not found in document")
        return chunk


class SearchService:
    """
    Service layer for vector similarity search operations.

    This service orchestrates all search operations and manages library-scoped indexing.
    It maintains indexes per library and provides search functionality.
    """
    
    def __init__(self, embedding_service: EmbeddingService, index_service: 'IndexService'):
        self._embedding_service = embedding_service
        self._index_service = index_service
        self._library_indexes: Dict[LibraryID, VectorIndex] = {}
        self._library_index_types: Dict[LibraryID, str] = {}

    def create_library_index(self, library_id: LibraryID, index_type: str = "naive") -> None:
        """Create an index for a library"""
        self._library_indexes[library_id] = self._index_service.create_index(index_type)
        self._library_index_types[library_id] = index_type
        logger.info("Library index created", lib_id=library_id, index_type=index_type)

    def _get_library_index(self, library_id: LibraryID) -> VectorIndex:
        """Get the index for a library, creating it with default type if missing"""
        if library_id not in self._library_indexes:
            index_type = self._library_index_types.get(library_id, "naive")
            self._library_indexes[library_id] = self._index_service.create_index(index_type)
            if library_id not in self._library_index_types:
                self._library_index_types[library_id] = index_type
        return self._library_indexes[library_id]

    def index_document_chunks(self, library_id: LibraryID, chunks: List[Chunk]) -> None:
        """Add document chunks to the library's index"""
        if chunks:
            index = self._get_library_index(library_id)
            index.index_chunks(chunks)
            logger.debug("Indexed document chunks", lib_id=library_id, chunk_count=len(chunks))

    def remove_document_chunks(self, library_id: LibraryID, chunk_ids: List[str]) -> None:
        """Remove document chunks from the library's index"""
        if chunk_ids:
            index = self._get_library_index(library_id)
            index.remove_chunks(chunk_ids)
            logger.debug("Removed document chunks from index", lib_id=library_id, chunk_count=len(chunk_ids))


    def search_chunks(
        self,
        library: Library,
        query_text: str,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the k most similar chunks in a library.

        Args:
            library: Library to search in
            query_text: Query text to search for
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity descending
        """
        chunks = library.get_all_chunks()
        if not chunks:
            logger.debug("No chunks found in library", lib_id=library.id)
            return []

        index = self._get_library_index(library.id)
        query_embedding = self._embedding_service.create_embedding(query_text, input_type="search_query")
        
        results = index.search(chunks, query_embedding, k, min_similarity)
        logger.info("Library search completed", lib_id=library.id, results_count=len(results))
        return results

    def search_chunks_in_document(
        self,
        library: Library,
        document_id: DocumentID,
        query_text: str,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the k most similar chunks in a specific document.

        Args:
            library: Library containing the document
            document_id: ID of the document to search in
            query_text: Query text to search for
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity descending
        """
        if not library.document_exists(document_id):
            raise ValueError(f"Document {document_id} not found in library")

        document = library.get_document_by_id(document_id)
        if not document or not document.chunks:
            logger.debug("Document has no chunks", doc_id=document_id)
            return []

        index = self._get_library_index(library.id)
        query_embedding = self._embedding_service.create_embedding(query_text, input_type="search_query")
        
        results = index.search(document.chunks, query_embedding, k, min_similarity)
        logger.info("Document search completed", doc_id=document_id, results_count=len(results))
        return results

