"""
Single Application Service - VectorDBService

This service orchestrates all vector database operations following Domain-Driven Design principles.
It coordinates between domain entities, repositories, and infrastructure services.
"""
from typing import List, Optional, Tuple
from ..domain.models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID, Metadata
from ..domain.interfaces import (
    DocumentRepository, 
    LibraryRepository, 
    SearchIndex, 
    EmbeddingService
)
from .document_indexing_service import DocumentIndexingService
from ..infrastructure.logging import get_logger
from ..infrastructure.index_factory import AVAILABLE_INDEX_TYPES

logger = get_logger(__name__)


class VectorDBService:
    """
    Single application service that orchestrates all vector database operations.
    
    This service follows DDD principles:
    - Coordinates between domain entities, repositories, and infrastructure
    - Ensures operations are atomic and consistent
    - Handles cross-cutting concerns like logging and error handling
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
        search_index: SearchIndex,
        embedding_service: EmbeddingService
    ):
        self.document_repo = document_repository
        self.library_repo = library_repository
        self.search_index = search_index
        self.embedding_service = embedding_service
    
    def _get_document_indexing_service(self, library_id: LibraryID) -> DocumentIndexingService:
        """Get a DocumentIndexingService for a specific library"""
        library_index = self.search_index.get_library_index(library_id)
        return DocumentIndexingService(library_index, self.embedding_service)
        
    
    # Library Operations
    
    def create_library(
        self,
        name: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None,
        index_type: str = "naive"
    ) -> Library:
        """Create a new library with search index"""
        # Validate parameters
        if not name or name.strip() == "":
            raise ValueError("Library name cannot be empty")
        
        # Validate index type
        if index_type not in AVAILABLE_INDEX_TYPES:
            raise ValueError(f"Invalid index type '{index_type}'. Valid types are: {', '.join(AVAILABLE_INDEX_TYPES)}")
        
        # Check for duplicate names
        existing_libraries = self.library_repo.list_all()
        if any(lib.name == name for lib in existing_libraries):
            raise ValueError(f"Library with name '{name}' already exists")
            
        library = Library.create(name, username, tags, index_type)
        
        # Explicitly create the search index for this library
        # This validates the index type and ensures the index is ready for use
        self.search_index.create_library_index(library.id, index_type)
        
        # Persist library
        self.library_repo.save(library)
        
        logger.info("Library created", lib_id=library.id, name=name, index_type=index_type)
        return library
    
    def get_library(self, library_id: LibraryID) -> Optional[Library]:
        """Get a library by ID"""
        return self.library_repo.get(library_id)
    
    def _ensure_library_exists(self, library_id: LibraryID) -> Library:
        """Get library or raise ValueError if not found"""
        library = self.library_repo.get(library_id)
        if not library:
            raise ValueError(f"Library {library_id} not found")
        return library
    
    def update_library_metadata(
        self,
        library_id: LibraryID,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Library:
        """Update library metadata"""
        library = self._ensure_library_exists(library_id)
        
        # Check for duplicate names (if name is being updated)
        if name and name != library.name:
            existing_libraries = self.library_repo.list_all()
            if any(lib.name == name and lib.id != library_id for lib in existing_libraries):
                raise ValueError(f"Library with name '{name}' already exists")
        
        updated_library = library.update_metadata(name, tags)
        self.library_repo.save(updated_library)
        
        logger.info("Library metadata updated", lib_id=library_id)
        return updated_library
    
    def delete_library(self, library_id: LibraryID) -> None:
        """Delete a library and all its documents"""
        library = self._ensure_library_exists(library_id)
        
        # Delete all documents in the library
        documents = self.document_repo.get_by_library(library_id)
        indexing_service = self._get_document_indexing_service(library_id)
        for document in documents:
            # Remove from search index first (while we still have access to chunks)
            indexing_service.remove_document(document.id)
            # Then delete from repository
            self.document_repo.delete(document.id)
        
        # Delete the library
        self.library_repo.delete(library_id)
        
        logger.info("Library deleted", lib_id=library_id, documents_deleted=len(documents))
    
    def list_libraries(self) -> List[Library]:
        """List all libraries"""
        return self.library_repo.list_all()
    
    # Document Operations
    
    def create_document(
        self,
        library_id: LibraryID,
        text: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None,
        chunk_size: int = 500
    ) -> Document:
        """Create a new document with content in a library"""
        # Verify library exists
        library = self._ensure_library_exists(library_id)
        
        # Create document first (domain logic)
        document = Document.create_empty(library_id, username, tags)
        
        # Create chunks with embeddings if text provided (application layer responsibility)
        if text:
            chunks = self._create_chunks_from_text(document.id, text, chunk_size, document.metadata)
            document = document.update_chunks(chunks)
        
        try:
            # Persist document
            self.document_repo.save(document)
            
            # Update library membership
            updated_library = library.add_document_reference(document.id)
            self.library_repo.save(updated_library)
            
            # Index document for search using DocumentIndexingService
            indexing_service = self._get_document_indexing_service(library_id)
            indexing_service.index_document(document)
            
            logger.info("Document created", doc_id=document.id, lib_id=library_id, chunks=len(document.chunks))
            return document
            
        except Exception as e:
            # Rollback: Try to clean up any partial state
            try:
                self.document_repo.delete(document.id)
                logger.warning("Rolled back document creation", doc_id=document.id, error=str(e))
            except Exception:
                logger.error("Failed to rollback document creation", doc_id=document.id)
            raise
    
    def create_empty_document(
        self,
        library_id: LibraryID,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Document:
        """Create an empty document in a library"""
        # Verify library exists
        library = self._ensure_library_exists(library_id)
        
        # Create empty document (domain logic)
        document = Document.create_empty(library_id, username, tags)
        
        try:
            # Persist document
            self.document_repo.save(document)
            
            # Update library membership
            updated_library = library.add_document_reference(document.id)
            self.library_repo.save(updated_library)
            
            logger.info("Empty document created", doc_id=document.id, lib_id=library_id)
            return document
            
        except Exception as e:
            # Rollback: Try to clean up any partial state
            try:
                self.document_repo.delete(document.id)
                logger.warning("Rolled back empty document creation", doc_id=document.id, error=str(e))
            except Exception:
                logger.error("Failed to rollback empty document creation", doc_id=document.id)
            raise
    
    def get_document(self, document_id: DocumentID) -> Optional[Document]:
        """Get a document by ID"""
        return self.document_repo.get(document_id)
    
    def _ensure_document_exists(self, document_id: DocumentID) -> Document:
        """Get document or raise ValueError if not found"""
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        return document
    
    def update_document_content(
        self,
        document_id: DocumentID,
        new_text: str,
        chunk_size: int = 500
    ) -> Document:
        """Update document content and re-index"""
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Create new chunks with embeddings (application layer responsibility)
        old_chunk_count = len(document.chunks)
        new_chunks = self._create_chunks_from_text(document.id, new_text, chunk_size, document.metadata) if new_text else []
        
        # Update document with new chunks (domain logic)
        updated_document = document.update_chunks(new_chunks)
        
        # Persist changes
        self.document_repo.save(updated_document)
        
        # Re-index document using DocumentIndexingService
        indexing_service = self._get_document_indexing_service(document.library_id)
        indexing_service.index_document(updated_document)
        
        logger.info("Document updated", doc_id=document_id, chunks=f"{old_chunk_count}→{len(updated_document.chunks)}")
        return updated_document
    
    def delete_document(self, document_id: DocumentID) -> None:
        """Delete a document and remove from library"""
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Remove from search index using DocumentIndexingService
        indexing_service = self._get_document_indexing_service(document.library_id)
        indexing_service.remove_document(document_id)
        
        # Remove from library membership
        library = self.library_repo.get(document.library_id)
        if library:
            updated_library = library.remove_document_reference(document_id)
            self.library_repo.save(updated_library)
        
        # Delete document last
        self.document_repo.delete(document_id)
        
        logger.info("Document deleted", doc_id=document_id, lib_id=document.library_id)
    
    def get_documents_in_library(self, library_id: LibraryID) -> List[Document]:
        """Get all documents in a library"""
        return self.document_repo.get_by_library(library_id)
    
    # Search Operations
    
    def search_library(
        self,
        library_id: LibraryID,
        query_text: str,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for chunks in a library"""
        # Validate parameters
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not query_text or query_text.strip() == "":
            raise ValueError("query_text cannot be empty")
        
        # Verify library exists
        library = self.library_repo.get(library_id)
        if not library:
            raise ValueError(f"Library {library_id} not found")
        
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query_text, input_type="search_query")
        
        # Search using index
        results = self.search_index.search_chunks(library_id, query_embedding, k, min_similarity)
        
        logger.info("Library search completed", lib_id=library_id, results_count=len(results))
        return results
    
    def search_document(
        self,
        document_id: DocumentID,
        query_text: str,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for chunks in a specific document"""
        # Validate parameters
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not query_text or query_text.strip() == "":
            raise ValueError("query_text cannot be empty")
        
        # Verify document exists
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        if not document.chunks:
            logger.debug("Document has no chunks", doc_id=document_id)
            return []
        
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query_text, input_type="search_query")
        
        # Get document chunks from the indexing service for this library
        indexing_service = self._get_document_indexing_service(document.library_id)
        document_chunks = indexing_service.get_document_chunks(document_id)
        
        if not document_chunks:
            logger.debug("No indexed chunks found for document", doc_id=document_id)
            return []
        
        # Search across all chunks in the library but then filter by document
        all_results = self.search_index.search_chunks(document.library_id, query_embedding, k * 5, min_similarity)
        
        # Filter results to only include chunks from this document
        results = [(chunk, score) for chunk, score in all_results if chunk.document_id == document_id][:k]
        
        logger.info("Document search completed", doc_id=document_id, results_count=len(results))
        return results
    
    # Chunk Operations (Read-only)
    
    def get_chunk(self, library_id: LibraryID, chunk_id: ChunkID) -> Optional[Chunk]:
        """Get a specific chunk from a library"""
        documents = self.document_repo.get_by_library(library_id)
        for document in documents:
            chunk = document.get_chunk_by_id(chunk_id)
            if chunk:
                logger.debug("Found chunk", chunk_id=chunk_id, doc_id=document.id)
                return chunk
        
        logger.debug("Chunk not found", chunk_id=chunk_id, lib_id=library_id)
        return None
    
    def get_chunks_from_document(self, document_id: DocumentID) -> List[Chunk]:
        """Get all chunks from a document"""
        document = self.document_repo.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        logger.debug("Retrieved chunks from document", doc_id=document_id, count=len(document.chunks))
        return document.chunks
    
    def get_chunks_from_library(self, library_id: LibraryID) -> List[Chunk]:
        """Get all chunks from a library"""
        documents = self.document_repo.get_by_library(library_id)
        chunks = []
        for document in documents:
            chunks.extend(document.chunks)
        
        logger.debug("Retrieved chunks from library", lib_id=library_id, count=len(chunks))
        return chunks
    
    def _create_chunks_from_text(self, document_id: DocumentID, text: str, chunk_size: int = 500, metadata=None) -> List[Chunk]:
        """Create chunks from text with embeddings (application layer logic)"""
        
        # Use provided metadata or create default
        if metadata is None:
            metadata = Metadata()
        
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text:  # Only create non-empty chunks
                # Create embedding for this chunk
                embedding = self.embedding_service.create_embedding(
                    chunk_text, 
                    input_type="search_document"
                )
                
                chunk = Chunk(
                    document_id=document_id,
                    text=chunk_text,
                    embedding=embedding,  # Embedding created here
                    metadata=metadata  # Inherit document metadata
                )
                chunks.append(chunk)
        
        return chunks