from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from datetime import datetime
from uuid import uuid4
import random


ChunkID = Annotated[str, Field(description="Unique chunk identifier")]
DocumentID = Annotated[str, Field(description="Unique document identifier")]
LibraryID = Annotated[str, Field(description="Unique library identifier")]

class Metadata(BaseModel):
    creation_time: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now)
    username: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)

class Chunk(BaseModel):
    id: ChunkID = Field(default_factory=lambda: str(uuid4()))
    document_id: DocumentID
    text: str = Field(..., min_length=1, description="Chunk text content")
    embedding: List[float]
    metadata: Metadata = Field(default_factory=Metadata)

    def __init__(self):
        self.embedding: List[float] = self.__create_embedding(self.text)

    def __create_embedding(self, text: str) -> float:
        # TODO: add embedding logic to generate actual embeddings
        return [random.random() for _ in range(768)]

    def update(self, text: str) -> 'Chunk':
        updated_metadata = self.metadata.model_copy(update={
            'last_update': datetime.now()
        })
        return self.model_copy(update={
            'text': text,
            'embedding': self.__create_embedding(text),
            'metadata': updated_metadata
        })


class Document(BaseModel):
    id: DocumentID = Field(default_factory=lambda: str(uuid4()))
    library_id: LibraryID
    # name: str = Field(..., min_length=1, description="Document name")
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    def __init__(self, text: Optional[str]):
        if text:
            self.populate_content(text)

    def get_chunks(self) -> List[ChunkID]:
        return [chunk.id for chunk in self.chunks]

    def __create_chunks(self, text: str, chunk_size: int) -> List[Chunk]:
        # TODO: implement something smarter than character-based chunking, like paragraph
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunk = Chunk(text=chunk_text, document_id=self.id)
                chunks.append(chunk)
        return chunks

    @classmethod
    def set_content(cls, text: str, chunk_size: int = 500) -> 'Document':
        chunks = cls.__create_chunks(text, chunk_size)
        cls.metadata.last_update = datetime.now
        return cls.model_copy(update={'chunks': chunks})

    def has_content(self, text: str) -> bool:
        return bool(self.text)

    def update_chunk(self, chunk_id: ChunkID, text: str) -> 'Document':
        raise NotImplementedError("TODO")

class Library(BaseModel):
    id: LibraryID = Field(default_factory=lambda: str(uuid4()))
    # name: str = Field(..., min_length=1, description="Library name")
    documents: List[Document] = Field(default_factory=list)

    def get_documents(self) -> List[DocumentID]:
        return [document.id for document in self.documents]

    def document_exists(self, document_id: str) -> bool:
        return bool(document_id in self.get_documents())

    def get_chunks(self) -> List[ChunkID]:
        return [
            chunk.id
            for document in self.documents
            for chunk in document.get_chunks()
        ]


class SemanticSearch(BaseModel):
    # TODO: implement the different algorithm

    # TODO:
    def similarity_search(self, text: str, chunk_ids: List[ChunkID], k: int = 5) -> List[Chunk]:
        # TODO: implement proper vector index
        # For now, access embeddings from chunks via self.chunk_store[chunk_id].embedding
        raise NotImplementedError("TODO")
