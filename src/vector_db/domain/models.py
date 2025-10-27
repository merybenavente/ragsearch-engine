import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Set
from datetime import datetime
from uuid import uuid4



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
        # Embedding should be provided explicitly during chunk creation
        # If not provided, it will remain empty (to be filled by application layer)



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

    @classmethod
    def create_with_chunks(
        cls,
        library_id: LibraryID,
        chunks: List[Chunk],
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'Document':
        """Create a new document with pre-created chunks"""
        if tags is None:
            tags = []
        
        metadata = Metadata(username=username, tags=tags)
        return cls(library_id=library_id, chunks=chunks, metadata=metadata)

    @classmethod
    def create_empty(
        cls,
        library_id: LibraryID,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'Document':
        """Create an empty document (no content, no chunks)"""
        if tags is None:
            tags = []
        
        metadata = Metadata(username=username, tags=tags)
        return cls(library_id=library_id, metadata=metadata)

    def update_chunks(self, new_chunks: List[Chunk]) -> 'Document':
        """Update document with new chunks"""
        return self.model_copy(update={
            'chunks': new_chunks,
            'metadata': self.metadata.update_timestamp()
        })

    def clear_content(self) -> 'Document':
        """Remove all chunks from document"""
        return self.model_copy(update={
            'chunks': [],
            'metadata': self.metadata.update_timestamp()
        })



class Library(BaseModel):
    """
    Library containing document references with metadata.

    Libraries are the top-level organizational unit for documents.
    In DDD approach, Library manages document membership, not document lifecycle.
    """
    id: LibraryID = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, description="Library name")
    document_ids: Set[DocumentID] = Field(default_factory=set, description="References to documents in this library")
    metadata: Metadata = Field(default_factory=Metadata)
    index_type: str = Field(default="naive", description="Preferred index type for this library")

    @classmethod
    def create(
        cls,
        name: str,
        username: Optional[str] = None,
        tags: Optional[List[str]] = None,
        index_type: str = "naive"
    ) -> 'Library':
        """Create a new library"""
        if tags is None:
            tags = []
        
        metadata = Metadata(username=username, tags=tags)
        return cls(name=name, metadata=metadata, index_type=index_type)

    def add_document_reference(self, document_id: DocumentID) -> 'Library':
        """Add a document reference to this library"""
        if document_id in self.document_ids:
            raise ValueError(f"Document {document_id} already exists in library")
        
        updated_document_ids = self.document_ids.copy()
        updated_document_ids.add(document_id)
        return self.model_copy(update={
            'document_ids': updated_document_ids,
            'metadata': self.metadata.update_timestamp()
        })

    def remove_document_reference(self, document_id: DocumentID) -> 'Library':
        """Remove a document reference from this library"""
        updated_document_ids = self.document_ids.copy()
        updated_document_ids.discard(document_id)  # discard doesn't raise if not present
        return self.model_copy(update={
            'document_ids': updated_document_ids,
            'metadata': self.metadata.update_timestamp()
        })

    def has_document(self, document_id: DocumentID) -> bool:
        """Check if library has a document reference"""
        return document_id in self.document_ids

    def update_metadata(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'Library':
        """Update library metadata"""
        if name is None and tags is None:
            return self
        
        updates = {}
        if name is not None:
            updates['name'] = name
        
        new_metadata = self.metadata
        if tags is not None:
            new_metadata = new_metadata.model_copy(update={'tags': tags})
        new_metadata = new_metadata.update_timestamp()
        updates['metadata'] = new_metadata
        
        return self.model_copy(update=updates)

