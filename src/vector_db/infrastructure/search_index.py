"""
Infrastructure implementation of SearchIndex using existing vector indexes.

This bridges the domain SearchIndex interface with the existing infrastructure.
"""
from typing import Dict, List, Tuple, Optional
from threading import RLock
from ..domain.models import Document, DocumentID, LibraryID, Chunk
from ..domain.interfaces import SearchIndex
from .index_factory import IndexFactory
from .indexes.base import VectorIndex
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
        self._library_indexes: Dict[LibraryID, VectorIndex] = {}
        self._document_library_mapping: Dict[DocumentID, LibraryID] = {}
        self._lock = RLock()  # Thread safety for concurrent index operations

    def get_library_index(self, library_id: LibraryID) -> VectorIndex:
        """Get the index for a library"""
        with self._lock:
            if library_id not in self._library_indexes:
                raise ValueError(
                    f"No index found for library {library_id}. Create one first using create_library_index()."
                )
            return self._library_indexes[library_id]

    def create_library_index(
        self,
        library_id: LibraryID,
        index_type: str,
        index_params: Optional[dict] = None,
    ) -> None:
        """Create an index for a library"""
        with self._lock:
            params = index_params or {}
            self._library_indexes[library_id] = self.index_factory.create_index(
                index_type, **params
            )
            logger.info(
                "Library index created",
                lib_id=library_id,
                index_type=index_type,
                params=params,
            )

    def index_document(self, document: Document) -> None:
        """Index a document and all its chunks"""
        if not document.chunks:
            logger.debug("Document has no chunks to index", doc_id=document.id)
            return

        with self._lock:
            self._document_library_mapping[document.id] = document.library_id

            # Get library index and add document chunks
            index = self.get_library_index(document.library_id)
            index.add_chunks(document.id, document.chunks)

            logger.debug(
                "Document indexed",
                doc_id=document.id,
                lib_id=document.library_id,
                chunk_count=len(document.chunks),
            )

    def remove_document(self, document_id: DocumentID) -> None:
        """Remove a document and all its chunks from the index"""
        with self._lock:
            if document_id not in self._document_library_mapping:
                logger.warning("Document not found in mapping", doc_id=document_id)
                return

            library_id = self._document_library_mapping[document_id]

            # Remove document from the vector index
            index = self.get_library_index(library_id)
            index.remove_document(document_id)

            # Remove from mapping
            del self._document_library_mapping[document_id]
            logger.debug("Document removed", doc_id=document_id, lib_id=library_id)

    def search_chunks(
        self,
        library_id: LibraryID,
        query_embedding: List[float],
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks within the specified library"""
        with self._lock:
            # TODO: review
            index = self.get_library_index(library_id)
            results = index.search(query_embedding, k, min_similarity)

            logger.debug(
                "Chunk search completed", lib_id=library_id, results_count=len(results)
            )
            return results
