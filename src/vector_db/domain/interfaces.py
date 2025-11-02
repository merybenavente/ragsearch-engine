"""Domain interfaces for dependency injection"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .models import Document, DocumentID, Library, LibraryID, Chunk, ChunkID


class EmbeddingService(ABC):
    """Abstract interface for embedding services"""
    
    @abstractmethod
    def create_embedding(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Create embedding for the given text.
        
        Args:
            text: The text to embed
            input_type: Type of input ("search_document" or "search_query")
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            RuntimeError: If the embedding service is not available
        """
        pass


class VectorIndex(ABC):
    """Abstract interface for vector indexes"""
    
    @abstractmethod
    def add_chunks(self, document_id: DocumentID, chunks: List['Chunk']) -> None:
        """
        Add chunks from a document to the index.
        
        Args:
            document_id: ID of the document containing these chunks
            chunks: List of chunks to index
        """
        pass
    
    @abstractmethod
    def remove_document(self, document_id: DocumentID) -> None:
        """
        Remove all chunks belonging to a document from the index.
        
        Args:
            document_id: ID of the document to remove
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 10, min_similarity: float = 0.0) -> List[Tuple['Chunk', float]]:
        """
        Search for the k most similar chunks to the query across all indexed chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of (chunk, similarity_score) tuples ordered by similarity
        """
        pass
    
    def get_library_index(self, library_id: LibraryID) -> 'SearchIndex':
        """
        Get the search index for a specific library.
        
        Args:
            library_id: ID of the library
            
        Returns:
            SearchIndex instance for the library
        """
        raise NotImplementedError("This method should be implemented by SearchIndex implementations that support library-specific indexes")
    
    @abstractmethod
    def get_document_chunks(self, document_id: DocumentID) -> List['Chunk']:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of chunks belonging to the document
        """
        pass
    
    def __contains__(self, document_id: DocumentID) -> bool:
        """Check if a document is indexed"""
        return len(self.get_document_chunks(document_id)) > 0
    
    def __len__(self) -> int:
        """Get total number of indexed chunks"""
        return len(self._get_all_chunks())
    
    @abstractmethod
    def _get_all_chunks(self) -> List['Chunk']:
        """Get all indexed chunks - for len() implementation"""
        pass


class DocumentRepository(ABC):
    """Abstract interface for document persistence"""
    
    @abstractmethod
    def save(self, document: Document) -> None:
        """Save a document"""
        pass
    
    @abstractmethod
    def get(self, document_id: DocumentID) -> Optional[Document]:
        """Get a document by ID"""
        pass
    
    @abstractmethod
    def get_by_library(self, library_id: LibraryID) -> List[Document]:
        """Get all documents in a library"""
        pass
    
    @abstractmethod
    def delete(self, document_id: DocumentID) -> None:
        """Delete a document"""
        pass
    
    @abstractmethod
    def exists(self, document_id: DocumentID) -> bool:
        """Check if a document exists"""
        pass


class LibraryRepository(ABC):
    """Abstract interface for library persistence"""
    
    @abstractmethod
    def save(self, library: Library) -> None:
        """Save a library"""
        pass
    
    @abstractmethod
    def get(self, library_id: LibraryID) -> Optional[Library]:
        """Get a library by ID"""
        pass
    
    @abstractmethod
    def delete(self, library_id: LibraryID) -> None:
        """Delete a library"""
        pass
    
    @abstractmethod
    def exists(self, library_id: LibraryID) -> bool:
        """Check if a library exists"""
        pass
    
    @abstractmethod
    def list_all(self) -> List[Library]:
        """List all libraries"""
        pass


class SearchIndex(ABC):
    """Abstract interface for search indexing - infrastructure concern"""
    
    @abstractmethod
    def index_document(self, document: Document) -> None:
        """Index a document and all its chunks"""
        pass
    
    @abstractmethod
    def remove_document(self, document_id: DocumentID) -> None:
        """Remove a document and all its chunks from the index"""
        pass
    
    @abstractmethod
    def search_chunks(self, library_id: LibraryID, query_embedding: List[float], k: int = 10, min_similarity: float = 0.0) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks within the specified library"""
        pass
    
    @abstractmethod
    def create_library_index(self, library_id: LibraryID, index_type: str) -> None:
        """Create an index for a library"""
        pass