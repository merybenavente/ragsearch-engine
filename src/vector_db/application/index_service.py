"""Index service for vector search operations"""
from typing import List, Tuple, TYPE_CHECKING

from ..domain.models import Document, Library
if TYPE_CHECKING:
    from ..domain.models import Chunk
from ..domain.interfaces import EmbeddingService, VectorIndex
from ..infrastructure.logging import get_logger

logger = get_logger(__name__)


class IndexService:
    """Service layer for vector index operations"""
    
    def __init__(self, vector_index: VectorIndex, embedding_service: EmbeddingService):
        self._vector_index = vector_index
        self._embedding_service = embedding_service
    
    def index_library(self, library: Library) -> None:
        """Index all chunks from a library"""
        chunks = library.get_all_chunks()
        if chunks:
            self._vector_index.index_chunks(chunks)
            logger.info("Library indexed", lib_id=library.id, chunks_count=len(chunks))
    
    def index_document(self, document: Document) -> None:
        """Index all chunks from a document"""
        if document.chunks:
            self._vector_index.index_chunks(document.chunks)
            logger.info("Document indexed", doc_id=document.id, chunks_count=len(document.chunks))
    
    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """Remove chunks from the index"""
        self._vector_index.remove_chunks(chunk_ids)
        logger.info("Chunks removed from index", chunks_count=len(chunk_ids))
    
    def create_index(self, index_type: str) -> VectorIndex:
        """Create a vector index of the specified type"""
        match index_type:
            case "naive":
                from ..infrastructure.indexes.naive import NaiveIndex
                return NaiveIndex()
            case "lsh":
                from ..infrastructure.indexes.lsh import LSHIndex
                return LSHIndex()
            case "vptree":
                from ..infrastructure.indexes.vptree import VPTreeIndex
                return VPTreeIndex()
            case _:
                raise ValueError(f"Unknown index type: {index_type}")

    def search(self, chunks: List['Chunk'], query_text: str, k: int = 10, min_similarity: float = 0.0) -> List[Tuple['Chunk', float]]:
        """Search for similar chunks within the given chunks"""
        query_embedding = self._embedding_service.create_embedding(query_text, input_type="search_query")
        results = self._vector_index.search(chunks, query_embedding, k, min_similarity)
        logger.info("Index search completed", query_length=len(query_text), results_count=len(results))
        return results
    
