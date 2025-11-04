"""Domain service for document indexing operations"""
from typing import List

from ..domain.models import Document, DocumentID, Chunk
from ..domain.interfaces import VectorIndex, EmbeddingService


class DocumentIndexingService:
    """
    Domain service that handles document indexing operations.

    This service sits in the application layer and orchestrates the
    interaction between domain entities and infrastructure services
    while maintaining DDD principles.
    """

    def __init__(self, vector_index: VectorIndex, embedding_service: EmbeddingService):
        self._vector_index = vector_index
        self._embedding_service = embedding_service

    def index_document(self, document: Document) -> None:
        """
        Index a document by ensuring its chunks have embeddings and adding them to the vector index.

        Args:
            document: The document to index
        """
        if not document.chunks:
            return

        # Ensure all chunks have embeddings
        chunks_with_embeddings = []
        for chunk in document.chunks:
            if not chunk.embedding:
                # Create embedding for chunk
                embedding = self._embedding_service.create_embedding(
                    chunk.text, input_type="search_document"
                )
                chunk_with_embedding = chunk.model_copy(update={"embedding": embedding})
                chunks_with_embeddings.append(chunk_with_embedding)
            else:
                chunks_with_embeddings.append(chunk)

        # Add chunks to vector index
        self._vector_index.add_chunks(document.id, chunks_with_embeddings)

    def remove_document(self, document_id: DocumentID) -> None:
        """
        Remove a document from the vector index.

        Args:
            document_id: ID of the document to remove
        """
        self._vector_index.remove_document(document_id)

    def is_document_indexed(self, document_id: DocumentID) -> bool:
        """
        Check if a document is indexed.

        Args:
            document_id: ID of the document to check

        Returns:
            True if the document is indexed, False otherwise
        """
        return document_id in self._vector_index

    def get_document_chunks(self, document_id: DocumentID) -> List[Chunk]:
        """
        Get all indexed chunks for a document.

        Args:
            document_id: ID of the document

        Returns:
            List of chunks for the document
        """
        return self._vector_index.get_document_chunks(document_id)

    def get_total_chunks_count(self) -> int:
        """
        Get the total number of indexed chunks.

        Returns:
            Total number of chunks in the index
        """
        return len(self._vector_index)
