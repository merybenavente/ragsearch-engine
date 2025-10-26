from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..schemas import ChunkResponse, MetadataResponse, SearchRequest, SearchResponse, SearchResult
from ..dependencies import get_library_repository, get_chunk_service, get_search_service
from ...infrastructure.repository import LibraryRepository
from ...application.services import ChunkService, SearchService
from ...domain.models import LibraryID, DocumentID, ChunkID
from ...infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/libraries/{library_id}/chunks", tags=["chunks"])


def _to_chunk_response(chunk) -> ChunkResponse:
    """Convert domain Chunk to ChunkResponse"""
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=MetadataResponse(
            creation_time=chunk.metadata.creation_time,
            last_update=chunk.metadata.last_update,
            username=chunk.metadata.username,
            tags=chunk.metadata.tags,
        ),
    )


@router.get("/", response_model=List[ChunkResponse])
async def get_chunks(
    library_id: LibraryID,
    library_repo: LibraryRepository = Depends(get_library_repository),
    chunk_service: ChunkService = Depends(get_chunk_service),
):
    """Get all chunks in a library"""
    try:
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        chunks = chunk_service.get_chunks_from_library(library)
        return [_to_chunk_response(chunk) for chunk in chunks]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chunks",
        )


@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    library_id: LibraryID,
    chunk_id: ChunkID,
    library_repo: LibraryRepository = Depends(get_library_repository),
    chunk_service: ChunkService = Depends(get_chunk_service),
):
    """Get a specific chunk by ID"""
    try:
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        try:
            chunk = chunk_service.get_chunk_from_library(library, chunk_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk with ID '{chunk_id}' not found",
            )

        return _to_chunk_response(chunk)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving chunk: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chunk",
        )


@router.get("/documents/{document_id}", response_model=List[ChunkResponse])
async def get_document_chunks(
    library_id: LibraryID,
    document_id: DocumentID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Get all chunks from a specific document"""
    try:
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        document = library.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found",
            )

        return [_to_chunk_response(chunk) for chunk in document.chunks]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving document chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document chunks",
        )


@router.post("/search", response_model=SearchResponse)
async def search_chunks(
    library_id: LibraryID,
    request: SearchRequest,
    library_repo: LibraryRepository = Depends(get_library_repository),
    search_service: SearchService = Depends(get_search_service),
):
    """Perform vector similarity search on chunks in a library"""
    try:
        import time
        start_time = time.time()
        
        # Check if library exists
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        # Perform search
        search_results = search_service.search_chunks(
            library=library,
            query_text=request.query_text,
            k=request.k,
        )

        # Calculate total chunks in library
        total_chunks = sum(len(doc.chunks) for doc in library.documents)
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000

        # Convert to response format
        results = [
            SearchResult(
                chunk=_to_chunk_response(chunk),
                similarity_score=similarity_score,
            )
            for chunk, similarity_score in search_results
        ]

        return SearchResponse(
            results=results,
            total_chunks_searched=total_chunks,
            query_time_ms=query_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error performing search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search",
        )