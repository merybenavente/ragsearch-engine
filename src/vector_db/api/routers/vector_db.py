"""
New DDD-style API router using VectorDBService.

This router demonstrates the clean architecture approach with a single
orchestrating service that coordinates all operations.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..schemas import (
    CreateLibraryRequest,
    UpdateLibraryRequest,
    LibraryResponse,
    CreateDocumentRequest,
    UpdateDocumentRequest,
    DocumentResponse,
    SearchRequest,
    SearchResponse,
    MetadataResponse
)
from ..dependencies import get_vector_db_service
from ...application.vector_db_service import VectorDBService
from ...domain.models import LibraryID, DocumentID, ChunkID
from ...infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v2", tags=["vector-db-v2"])


def _to_library_response(library) -> LibraryResponse:
    """Convert domain Library to LibraryResponse"""
    return LibraryResponse(
        id=library.id,
        name=library.name,
        metadata=MetadataResponse(
            creation_time=library.metadata.creation_time,
            last_update=library.metadata.last_update,
            username=library.metadata.username,
            tags=library.metadata.tags,
        ),
        document_count=len(library.document_ids),  # Fixed to use document_ids
    )


def _to_document_response(document) -> DocumentResponse:
    """Convert domain Document to DocumentResponse"""
    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        chunks=[
            {
                "id": chunk.id,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "metadata": {
                    "creation_time": chunk.metadata.creation_time,
                    "last_update": chunk.metadata.last_update,
                    "username": chunk.metadata.username,
                    "tags": chunk.metadata.tags,
                }
            }
            for chunk in document.chunks
        ],
        metadata=MetadataResponse(
            creation_time=document.metadata.creation_time,
            last_update=document.metadata.last_update,
            username=document.metadata.username,
            tags=document.metadata.tags,
        ),
    )


# Library Operations

@router.post("/libraries", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(
    request: CreateLibraryRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Create a new library"""
    try:
        library = vector_db.create_library(
            name=request.name,
            username=request.username,
            tags=request.tags,
            index_type=request.index_type  # Now properly defined in schema
        )
        return _to_library_response(library)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create library",
        )


@router.get("/libraries", response_model=List[LibraryResponse])
async def get_libraries(
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Get all libraries"""
    try:
        libraries = vector_db.list_libraries()
        return [_to_library_response(library) for library in libraries]
    except Exception as e:
        logger.error(f"Unexpected error retrieving libraries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve libraries",
        )


@router.get("/libraries/{library_id}", response_model=LibraryResponse)
async def get_library(
    library_id: LibraryID,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Get a specific library by ID"""
    try:
        library = vector_db.get_library(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )
        return _to_library_response(library)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve library",
        )


@router.put("/libraries/{library_id}", response_model=LibraryResponse)
async def update_library(
    library_id: LibraryID,
    request: UpdateLibraryRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Update a library"""
    try:
        updated_library = vector_db.update_library_metadata(
            library_id=library_id,
            name=request.name,
            tags=request.tags,
        )
        return _to_library_response(updated_library)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error updating library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update library",
        )


@router.delete("/libraries/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(
    library_id: LibraryID,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Delete a library and all its documents"""
    try:
        vector_db.delete_library(library_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete library",
        )


# Document Operations

@router.post("/libraries/{library_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    library_id: LibraryID,
    request: CreateDocumentRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Create a new document in a library"""
    try:
        if request.text:
            document = vector_db.create_document(
                library_id=library_id,
                text=request.text,
                username=request.username,
                tags=request.tags,
                chunk_size=request.chunk_size  # Now properly defined in schema
            )
        else:
            document = vector_db.create_empty_document(
                library_id=library_id,
                username=request.username,
                tags=request.tags,
            )
        return _to_document_response(document)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document",
        )


@router.get("/libraries/{library_id}/documents", response_model=List[DocumentResponse])
async def get_documents_in_library(
    library_id: LibraryID,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Get all documents in a library"""
    try:
        documents = vector_db.get_documents_in_library(library_id)
        return [_to_document_response(doc) for doc in documents]
    except Exception as e:
        logger.error(f"Unexpected error retrieving documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: DocumentID,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Get a specific document by ID"""
    try:
        document = vector_db.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found",
            )
        return _to_document_response(document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: DocumentID,
    request: UpdateDocumentRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Update document content"""
    try:
        updated_document = vector_db.update_document_content(
            document_id=document_id,
            new_text=request.text,
            chunk_size=request.chunk_size  # Now properly defined in schema
        )
        return _to_document_response(updated_document)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error updating document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document",
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: DocumentID,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Delete a document"""
    try:
        vector_db.delete_document(document_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )


# Search Operations

@router.post("/libraries/{library_id}/search", response_model=SearchResponse)
async def search_library(
    library_id: LibraryID,
    request: SearchRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Search for chunks in a library"""
    try:
        results = vector_db.search_library(
            library_id=library_id,
            query_text=request.query,
            k=request.k,
            min_similarity=request.min_similarity,
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                {
                    "chunk": {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "embedding": chunk.embedding,
                        "metadata": {
                            "creation_time": chunk.metadata.creation_time,
                            "last_update": chunk.metadata.last_update,
                            "username": chunk.metadata.username,
                            "tags": chunk.metadata.tags,
                        }
                    },
                    "similarity": similarity
                }
                for chunk, similarity in results
            ],
            total_results=len(results)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error searching library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search library",
        )


@router.post("/documents/{document_id}/search", response_model=SearchResponse)
async def search_document(
    document_id: DocumentID,
    request: SearchRequest,
    vector_db: VectorDBService = Depends(get_vector_db_service),
):
    """Search for chunks in a specific document"""
    try:
        results = vector_db.search_document(
            document_id=document_id,
            query_text=request.query,
            k=request.k,
            min_similarity=request.min_similarity,
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                {
                    "chunk": {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "embedding": chunk.embedding,
                        "metadata": {
                            "creation_time": chunk.metadata.creation_time,
                            "last_update": chunk.metadata.last_update,
                            "username": chunk.metadata.username,
                            "tags": chunk.metadata.tags,
                        }
                    },
                    "similarity": similarity
                }
                for chunk, similarity in results
            ],
            total_results=len(results)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error searching document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search document",
        )