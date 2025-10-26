"""API request/response schemas"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..domain.models import LibraryID, DocumentID, ChunkID


class MetadataResponse(BaseModel):
    """Metadata response schema"""
    creation_time: datetime
    last_update: datetime
    username: Optional[str] = None
    tags: List[str] = []


class CreateLibraryRequest(BaseModel):
    """Request schema for creating a library"""
    name: str = Field(..., min_length=1, description="Library name")
    username: Optional[str] = Field(None, description="Username")
    tags: List[str] = Field(default_factory=list, description="Tags")


class UpdateLibraryRequest(BaseModel):
    """Request schema for updating a library"""
    name: Optional[str] = Field(None, min_length=1, description="Library name")
    tags: Optional[List[str]] = Field(None, description="Tags")


class LibraryResponse(BaseModel):
    """Response schema for library operations"""
    id: LibraryID
    name: str
    metadata: MetadataResponse
    document_count: int = Field(description="Number of documents in library")


class CreateDocumentRequest(BaseModel):
    """Request schema for creating a document"""
    text: str = Field(..., min_length=1, description="Document content")
    username: Optional[str] = Field(None, description="Username")
    tags: List[str] = Field(default_factory=list, description="Tags")
    chunk_size: int = Field(default=500, ge=1, le=2000, description="Chunk size for splitting")


class UpdateDocumentRequest(BaseModel):
    """Request schema for updating a document"""
    text: str = Field(..., min_length=1, description="New document content")
    chunk_size: int = Field(default=500, ge=1, le=2000, description="Chunk size for splitting")


class DocumentResponse(BaseModel):
    """Response schema for document operations"""
    id: DocumentID
    library_id: LibraryID
    metadata: MetadataResponse
    chunk_count: int = Field(description="Number of chunks in document")
    text_preview: str = Field(description="First 200 characters of document text")


class ChunkResponse(BaseModel):
    """Response schema for chunk operations"""
    id: ChunkID
    document_id: DocumentID
    text: str
    embedding: List[float]
    metadata: MetadataResponse


class SearchRequest(BaseModel):
    """Request schema for vector search"""
    query_text: str = Field(..., min_length=1, description="Query text for similarity search")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")


class SearchResult(BaseModel):
    """Individual search result"""
    chunk: ChunkResponse
    similarity_score: float = Field(description="Similarity score (0-1)")


class SearchResponse(BaseModel):
    """Response schema for search operations"""
    results: List[SearchResult]
    total_chunks_searched: int = Field(description="Total number of chunks in library")
    query_time_ms: float = Field(description="Query execution time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[dict] = None