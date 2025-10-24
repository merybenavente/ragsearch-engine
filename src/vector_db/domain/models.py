import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

from ..infrastructure.cohere_client import co


ChunkID = str
DocumentID = str
LibraryID = str

# Embedding model constants
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSION = 1536


class Metadata(BaseModel):
    creation_time: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now)
    username: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    def update_timestamp(self) -> 'Metadata':
        """Return a copy with updated timestamp"""
        return self.model_copy(update={'last_update': datetime.now()})


class Chunk(BaseModel):
    """
    Immutable chunk representing a piece of text with embedding.

    Chunks are derived data - they should only be created through document processing,
    never updated directly. This ensures data integrity between document text and chunks.
    """
    id: ChunkID = Field(default_factory=lambda: str(uuid4()))
    document_id: DocumentID
    text: str = Field(..., min_length=1, description="Chunk text content")
    embedding: List[float] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    def __init__(self, **data):
        super().__init__(**data)
        # Generate embedding after initialization if not provided
        if not self.embedding and self.text:
            self.embedding = self._create_embedding(self.text)

    def _create_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        WARNING: This method is not thread-safe due to shared Cohere client access.
        The global 'co' client may experience race conditions when multiple threads
        call this method simultaneously. Consider:
        - Using a connection pool for the Cohere client
        - Implementing per-thread clients
        - Adding synchronization locks around API calls
        - Moving to async implementation with semaphores
        """
        if co is None:
            raise RuntimeError("Cohere client not available. Please set COHERE_API_KEY environment variable.")

        resp = co.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
            truncate="NONE"
        )

        embeddings = np.array(resp.embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings[0].tolist()


class Document(BaseModel):
    """
    Document containing text content that gets automatically chunked.

    Documents are the unit of authoring - users create and update documents,
    and the system automatically generates chunks for search/retrieval.
    """
    id: DocumentID = Field(default_factory=lambda: str(uuid4()))
    library_id: LibraryID
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    def get_chunk_ids(self) -> List[ChunkID]:
        """Get list of chunk IDs in this document"""
        return [chunk.id for chunk in self.chunks]

    def has_content(self) -> bool:
        """Check if document has any chunks"""
        return len(self.chunks) > 0

    def get_full_text(self) -> str:
        """Reconstruct full document text from chunks (in order)"""
        return "".join(chunk.text for chunk in self.chunks)

    def get_chunk_by_id(self, chunk_id: ChunkID) -> Optional[Chunk]:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def replace_content(self, text: str, chunk_size: int = 500) -> 'Document':
        """
        Replace entire document content and regenerate all chunks.

        This is the ONLY way to modify document content, ensuring:
        - Data consistency between text and chunks
        - Atomic updates (all or nothing)
        - Predictable chunking behavior
        """
        if not text:
            # Empty content = no chunks
            return self.model_copy(update={
                'chunks': [],
                'metadata': self.metadata.update_timestamp()
            })

        # Generate new chunks from text
        new_chunks = self._create_chunks_from_text(text, chunk_size)

        return self.model_copy(update={
            'chunks': new_chunks,
            'metadata': self.metadata.update_timestamp()
        })

    def _create_chunks_from_text(self, text: str, chunk_size: int = 500) -> List[Chunk]:
        """
        Split text into chunks and create Chunk objects.

        Current implementation uses simple character-based chunking.

        TODO: Implement smart chunking strategies:
        - Paragraph-based chunking (split on \n\n)
        - Sentence-based chunking (using NLP sentence boundaries)
        - Semantic chunking (split on topic boundaries)
        - Sliding window chunking (with overlap)
        - Custom delimiter chunking (user-defined separators)
        """
        chunks = []

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text:  # Only create non-empty chunks
                chunk = Chunk(
                    document_id=self.id,
                    text=chunk_text
                )
                chunks.append(chunk)

        return chunks


class Library(BaseModel):
    """
    Library containing multiple documents with metadata.

    Libraries are the top-level organizational unit for documents.
    """
    id: LibraryID = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, description="Library name")
    documents: List[Document] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    def get_document_ids(self) -> List[DocumentID]:
        """Get list of document IDs in this library"""
        return [document.id for document in self.documents]

    def document_exists(self, document_id: DocumentID) -> bool:
        """Check if document exists in library"""
        return document_id in self.get_document_ids()

    def get_document_by_id(self, document_id: DocumentID) -> Optional[Document]:
        """Get a specific document by ID"""
        for document in self.documents:
            if document.id == document_id:
                return document
        return None

    def add_document(self, document: Document) -> 'Library':
        """Add a document to the library"""
        # Ensure document has correct library_id
        if document.library_id != self.id:
            document = document.model_copy(update={'library_id': self.id})

        updated_documents = self.documents + [document]
        return self.model_copy(update={
            'documents': updated_documents,
            'metadata': self.metadata.update_timestamp()
        })

    def remove_document(self, document_id: DocumentID) -> 'Library':
        """Remove a document from the library"""
        updated_documents = [doc for doc in self.documents if doc.id != document_id]
        return self.model_copy(update={
            'documents': updated_documents,
            'metadata': self.metadata.update_timestamp()
        })

    def update_document(self, updated_document: Document) -> 'Library':
        """Replace a document in the library with an updated version"""
        updated_documents = []
        for doc in self.documents:
            if doc.id == updated_document.id:
                # Ensure correct library_id
                if updated_document.library_id != self.id:
                    updated_document = updated_document.model_copy(update={'library_id': self.id})
                updated_documents.append(updated_document)
            else:
                updated_documents.append(doc)

        return self.model_copy(update={
            'documents': updated_documents,
            'metadata': self.metadata.update_timestamp()
        })

    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks from all documents in the library"""
        chunks = []
        for document in self.documents:
            chunks.extend(document.chunks)
        return chunks

    def get_all_chunk_ids(self) -> List[ChunkID]:
        """Get all chunk IDs from all documents in the library"""
        return [chunk.id for chunk in self.get_all_chunks()]