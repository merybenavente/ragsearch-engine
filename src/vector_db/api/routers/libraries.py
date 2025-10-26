from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..schemas import (
    CreateLibraryRequest,
    UpdateLibraryRequest,
    LibraryResponse,
    MetadataResponse
)
from ..dependencies import get_library_repository, get_library_service
from ...infrastructure.repository import LibraryRepository
from ...application.services import LibraryService
from ...domain.models import LibraryID
from ...domain.exceptions import DuplicateLibraryException
from ...infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/libraries", tags=["libraries"])


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
        document_count=len(library.documents),
    )


@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(
    request: CreateLibraryRequest,
    library_repo: LibraryRepository = Depends(get_library_repository),
    library_service: LibraryService = Depends(get_library_service),
):
    """Create a new library"""
    try:
        # Check if library with same name already exists
        existing = library_repo.find_by_name(request.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Library with name '{request.name}' already exists",
            )

        # Create library
        library = library_service.create_library(
            name=request.name,
            username=request.username,
            tags=request.tags,
        )

        # Save to repository
        saved_library = library_repo.save(library)
        return _to_library_response(saved_library)

    except HTTPException:
        raise
    except DuplicateLibraryException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error creating library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create library",
        )


@router.get("/", response_model=List[LibraryResponse])
async def get_libraries(
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Get all libraries"""
    try:
        libraries = library_repo.find_all()
        return [_to_library_response(library) for library in libraries]
    except Exception as e:
        logger.error(f"Unexpected error retrieving libraries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve libraries",
        )


@router.get("/{library_id}", response_model=LibraryResponse)
async def get_library(
    library_id: LibraryID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Get a specific library by ID"""
    try:
        library = library_repo.find_by_id(library_id)
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


@router.put("/{library_id}", response_model=LibraryResponse)
async def update_library(
    library_id: LibraryID,
    request: UpdateLibraryRequest,
    library_repo: LibraryRepository = Depends(get_library_repository),
    library_service: LibraryService = Depends(get_library_service),
):
    """Update a library"""
    try:
        # Find existing library
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        # Check for name conflicts if name is being updated
        if request.name and request.name != library.name:
            existing = library_repo.find_by_name(request.name)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Library with name '{request.name}' already exists",
                )

        # Update library
        updated_library = library_service.update_library_metadata(
            library=library,
            name=request.name,
            tags=request.tags,
        )

        # Save to repository
        saved_library = library_repo.save(updated_library)
        return _to_library_response(saved_library)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update library",
        )


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(
    library_id: LibraryID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Delete a library"""
    try:
        # Check if library exists
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        # Delete library
        success = library_repo.delete(library_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete library",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting library: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete library",
        )