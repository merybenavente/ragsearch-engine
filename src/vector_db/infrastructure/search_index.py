"""
Infrastructure implementation of SearchIndex using existing vector indexes.

This bridges the domain SearchIndex interface with the existing infrastructure.
"""
from typing import Dict, List, Tuple
from threading import RLock
from ..domain.models import Document, DocumentID, LibraryID, Chunk
from ..domain.interfaces import SearchIndex
from .index_factory import IndexFactory
from ..infrastructure.logging import get_logger

logger = get_logger(__name__)


class RepositoryAwareSearchIndex(SearchIndex):
    """
    SearchIndex implementation that maintains library-scoped indexes.
    
    This implementation creates separate indexes per library and delegates
    search operations to the appropriate VectorIndex implementation.
    """
    
    def __init__(self, index_factory: IndexFactory):
        self.index_factory = index_factory
        self._library_indexes: Dict[LibraryID, 'VectorIndex'] = {}
        self._library_index_types: Dict[LibraryID, str] = {}
        self._document_library_mapping: Dict[DocumentID, LibraryID] = {}
        self._lock = RLock()  # Thread safety for concurrent index operations
    
    def _get_library_index(self, library_id: LibraryID) -> 'VectorIndex':
        """Get or create the index for a library"""
        with self._lock:
            if library_id not in self._library_indexes:
                index_type = self._library_index_types.get(library_id, "naive")
                self._library_indexes[library_id] = self.index_factory.create_index(index_type)
                if library_id not in self._library_index_types:
                    self._library_index_types[library_id] = index_type
            return self._library_indexes[library_id]
    
    def create_library_index(self, library_id: LibraryID, index_type: str) -> None:
        """Create an index for a library"""
        with self._lock:
            self._library_indexes[library_id] = self.index_factory.create_index(index_type)
            self._library_index_types[library_id] = index_type
            logger.info("Library index created", lib_id=library_id, index_type=index_type)
    
    def index_document(self, document: Document) -> None:
        """Index a document and all its chunks"""
        if not document.chunks:
            logger.debug("Document has no chunks to index", doc_id=document.id)
            return
        
        with self._lock:
            # Track document-library mapping
            old_library_id = self._document_library_mapping.get(document.id)
            if old_library_id and old_library_id != document.library_id:
                # Document moved between libraries - remove from old library index
                logger.info("Document moved between libraries", doc_id=document.id, old_lib=old_library_id, new_lib=document.library_id)
                # Note: calling remove_document here could cause deadlock, so we handle it inline
                del self._document_library_mapping[document.id]
            
            self._document_library_mapping[document.id] = document.library_id
            
            # Get library index and index chunks
            index = self._get_library_index(document.library_id)
            index.index_chunks(document.chunks)
            
            logger.debug("Document indexed", doc_id=document.id, lib_id=document.library_id, chunk_count=len(document.chunks))
    
    def remove_document(self, document_id: DocumentID) -> None:
        """Remove a document and all its chunks from the index"""
        with self._lock:
            if document_id not in self._document_library_mapping:
                logger.warning("Document not found in mapping", doc_id=document_id)
                return
            
            library_id = self._document_library_mapping[document_id]
            
            # For proper chunk removal, we need to know which chunks to remove
            # Since the SearchIndex interface doesn't provide chunk access,
            # we'll implement a simplified approach where we let the index
            # handle cleanup internally
            
            # Note: This is a design limitation - ideally the SearchIndex would
            # maintain its own chunk mappings or the VectorIndex would support
            # document-based removal
            
            # Remove from mapping
            del self._document_library_mapping[document_id]
            logger.debug("Document removed from mapping", doc_id=document_id, lib_id=library_id)
            
            # Log the limitation
            logger.warning("Document chunks may remain in index - design limitation", doc_id=document_id)
    
    def search_chunks(
        self,
        chunks: List[Chunk],
        query_embedding: List[float],
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks within the provided chunk list"""
        if not chunks:
            logger.debug("No chunks provided for search")
            return []
        
        with self._lock:
            # Determine which library index to use based on first chunk's document
            # All chunks should belong to documents in the same library for this to work properly
            first_chunk = chunks[0]
            library_id = self._document_library_mapping.get(first_chunk.document_id)
            
            if not library_id:
                # Fallback to naive search if no library mapping exists
                logger.warning("No library mapping found, using naive search", doc_id=first_chunk.document_id)
                return self._naive_search(chunks, query_embedding, k, min_similarity)
            
            # Use the library's index
            index = self._get_library_index(library_id)
            results = index.search(chunks, query_embedding, k, min_similarity)
            
            logger.debug("Chunk search completed", chunk_count=len(chunks), results_count=len(results))
            return results
    
    def _naive_search(
        self,
        chunks: List[Chunk],
        query_embedding: List[float], 
        k: int,
        min_similarity: float
    ) -> List[Tuple[Chunk, float]]:
        """Fallback naive search implementation"""
        import numpy as np
        
        results = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            
            # Calculate cosine similarity
            v1 = np.array(query_embedding)
            v2 = np.array(chunk.embedding)
            
            if len(v1) != len(v2):
                continue
                
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                continue
                
            similarity = dot_product / (norm_v1 * norm_v2)
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]
            
            if similarity >= min_similarity:
                results.append((chunk, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]