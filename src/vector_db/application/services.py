"""Application services - business logic"""
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata, EMBEDDING_MODEL
from ..infrastructure.logging import get_logger
from ..infrastructure.cohere_client import co

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
    def get_chunk_from_library(library: Library, chunk_id: ChunkID) -> Chunk:
        """Find a specific chunk in a library (read-only)"""
        for document in library.documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                logger.debug("Found chunk", chunk_id=chunk_id, doc_id=document.id)
                return chunk

        logger.debug("Chunk not found", chunk_id=chunk_id, lib_id=library.id)
        raise ValueError(f"Chunk with id {chunk_id} not found in library")

    @staticmethod
    def get_chunk_from_document(document: Document, chunk_id: ChunkID) -> Chunk:
        """Find a specific chunk in a document (read-only)"""
        chunk = document.get_chunk_by_id(chunk_id)
        logger.info("Chunk retrieved", doc_id=document.id, chunk_id=chunk_id, found=chunk is not None)
        if chunk is None:
            raise ValueError(f"Chunk with id {chunk_id} not found in document")
        return chunk


class SearchService:
    """
    Service layer for vector similarity search operations.
    
    This service implements k-nearest neighbor search using cosine similarity
    to find the most relevant chunks for a given query embedding.
    """
    
    @staticmethod
    def _create_query_embedding(text: str) -> List[float]:
        """Convert query text to embedding using same model as chunks"""
        if co is None:
            raise RuntimeError("Cohere client not available. Please set COHERE_API_KEY environment variable.")
        
        resp = co.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_query",
            truncate="NONE"
        )
        embeddings = np.array(resp.embeddings)
        # Normalize the embedding
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings[0].tolist()

    @staticmethod
    def _perform_search(
        chunks: List[Chunk],
        query_text: str,
        k: int,
        min_similarity: float,
        context: dict
    ) -> List[Tuple[Chunk, float]]:
        """
        Common search logic for both library and document searches.
        
        Args:
            chunks: List of chunks to search in
            query_text: Query text to search for
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            context: Dictionary with context info for logging (e.g., lib_id, doc_id)
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity descending
        """
        query_embedding = SearchService._create_query_embedding(query_text)

        # Calculate similarities for all chunks
        similarities = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning("Chunk has no embedding", chunk_id=chunk.id)
                continue
                
            similarity = SearchService._cosine_similarity(query_embedding, chunk.embedding)
            if similarity >= min_similarity:
                similarities.append((chunk, similarity))

        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Handle case where we have fewer results than requested k
        if len(similarities) < k:
            logger.warning(
                "Fewer results than requested",
                requested_k=k,
                available_results=len(similarities),
                **context
            )
        
        results = similarities[:k]

        logger.info(
            "Document search completed",
            query_length=len(query_embedding),
            total_chunks=len(chunks),
            results_returned=len(results),
            min_similarity=min_similarity,
            **context
        )

        return results

    @staticmethod
    def search_chunks(
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

        return SearchService._perform_search(
            chunks=chunks,
            query_text=query_text,
            k=k,
            min_similarity=min_similarity,
            context={"lib_id": library.id}
        )

    @staticmethod
    def search_chunks_in_document(
        document: Document,
        query_text: str,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the k most similar chunks in a specific document.
        
        Args:
            document: Document to search in
            query_text: Query text to search for
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity descending
        """
        if not document.chunks:
            logger.debug("Document has no chunks", doc_id=document.id)
            return []

        return SearchService._perform_search(
            chunks=document.chunks,
            query_text=query_text,
            k=k,
            min_similarity=min_similarity,
            context={"doc_id": document.id}
        )

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(vec1) != len(vec2):
            logger.error("Vector dimensions don't match", dim1=len(vec1), dim2=len(vec2))
            return 0.0

        # Convert to numpy arrays for efficient computation
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))