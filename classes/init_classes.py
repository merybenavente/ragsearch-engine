from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from datetime import datetime
from uuid import uuid4
import random
from typing import Dict

# Custom ID types for better type safety
ChunkID = Annotated[str, Field(description="Unique chunk identifier")]
DocumentID = Annotated[str, Field(description="Unique document identifier")]

class Metadata(BaseModel):
    creation_time: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)

class ChunkMetadata(Metadata):
    document_id: DocumentID

class DocumentMetadata(Metadata):
    # TODO: implement Library
    library_id: str


class Chunk(BaseModel):
    id: ChunkID = Field(default_factory=lambda: str(uuid4()))
    metadata: ChunkMetadata
    text: str = Field(..., min_length=1, description="Chunk text content")
    # TODO: add embedding logic to generate actual embeddings
    embedding: List[float] = Field(default_factory=lambda: [random.random() for _ in range(768)])

    def update(self, text: str) -> 'Chunk':
        # TODO: add embedding logic to generate actual embeddings
        new_embedding = [random.random() for _ in range(768)]  # placeholder
        updated_metadata = self.metadata.model_copy(update={
            'last_update': datetime.now()
        })
        return self.model_copy(update={
            'text': text,
            'embedding': new_embedding,
            'metadata': updated_metadata
        })


class Document(BaseModel):
    id: DocumentID = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    chunks: List[Chunk] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, metadata: DocumentMetadata, chunk_size: int = 500) -> 'Document':
        """Create a document by splitting text into chunks."""
        document = cls(metadata=metadata)

        # TODO: implement something smarter than character-based chunking, like paragraph
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():  # Skip empty chunks
                chunk_metadata = ChunkMetadata(document_id=document.id)
                chunk = Chunk(text=chunk_text, metadata=chunk_metadata)
                chunks.append(chunk)

        return document.model_copy(update={'chunks': chunks})

    def add_chunk(self, text: str) -> 'Document':
        chunk_metadata = ChunkMetadata(document_id=self.id)
        chunk = Chunk(text=text, metadata=chunk_metadata)

        return self.model_copy(update={
            'chunks': self.chunks + [chunk]
        })

    def update_chunk(self, chunk_id: ChunkID, text: str) -> 'Document':
        updated_chunks = []
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                updated_chunks.append(chunk.update(text))
            else:
                updated_chunks.append(chunk)
        return self.model_copy(update={'chunks': updated_chunks})


class VectorDB(BaseModel):
    chunk_store: Dict[ChunkID, Chunk] = Field(default_factory=dict)
    vector_index: Dict[ChunkID, List[float]] = Field(default_factory=dict)  # Simple dict for now

    def add_document(self, document: Document) -> 'VectorDB':
        """Add a document and index all its chunks."""
        new_chunk_store = self.chunk_store.copy()
        new_vector_index = self.vector_index.copy()

        # Add all chunks from the document
        for chunk in document.chunks:
            new_chunk_store[chunk.id] = chunk
            new_vector_index[chunk.id] = chunk.embedding

        return self.model_copy(update={
            'chunk_store': new_chunk_store,
            'vector_index': new_vector_index
        })

    def update_document(self, document: Document) -> 'VectorDB':
        """Update a document and re-index its chunks."""
        new_chunk_store = self.chunk_store.copy()
        new_vector_index = self.vector_index.copy()

        # Remove old chunks from this document
        old_chunk_ids = [chunk_id for chunk_id, chunk in self.chunk_store.items()
                        if chunk.metadata.document_id == document.id]
        for chunk_id in old_chunk_ids:
            new_chunk_store.pop(chunk_id, None)
            new_vector_index.pop(chunk_id, None)

        # Add updated document chunks
        for chunk in document.chunks:
            new_chunk_store[chunk.id] = chunk
            new_vector_index[chunk.id] = chunk.embedding

        return self.model_copy(update={
            'chunk_store': new_chunk_store,
            'vector_index': new_vector_index
        })

    def get_chunk(self, chunk_id: ChunkID) -> Optional[Chunk]:
        return self.chunk_store.get(chunk_id)

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Chunk]:
        # TODO: implement proper vector index
        raise NotImplementedError("TODO")


class Library(BaseModel):
    # TODO
    pass