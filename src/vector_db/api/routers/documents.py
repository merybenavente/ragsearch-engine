from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..schemas import (
    CreateDocumentRequest,
    UpdateDocumentRequest,
    DocumentResponse,
    MetadataResponse,
)
from ..dependencies import get_library_repository, get_document_service
from ...infrastructure.repository import LibraryRepository
from ...application.services import DocumentService
from ...domain.models import LibraryID, DocumentID
from ...infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])


def _to_document_response(document) -> DocumentResponse:
    """Convert domain Document to DocumentResponse"""
    full_text = document.get_full_text()
    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        metadata=MetadataResponse(
            creation_time=document.metadata.creation_time,
            last_update=document.metadata.last_update,
            username=document.metadata.username,
            tags=document.metadata.tags,
        ),
        chunk_count=len(document.chunks),
        text_preview=full_text[:200] + "..." if len(full_text) > 200 else full_text,
    )


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    library_id: LibraryID,
    request: CreateDocumentRequest,
    library_repo: LibraryRepository = Depends(get_library_repository),
    document_service: DocumentService = Depends(get_document_service),
):
    """Create a new document in a library"""
    try:
        # Check if library exists
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        # Create document
        document = document_service.create_document(
            library_id=library_id,
            text=request.text,
            username=request.username,
            tags=request.tags,
            chunk_size=request.chunk_size,
        )

        # Add document to library
        updated_library = library.add_document(document)
        library_repo.save(updated_library)

        return _to_document_response(document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document",
        )


@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    library_id: LibraryID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Get all documents in a library"""
    try:
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        return [_to_document_response(document) for document in library.documents]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    library_id: LibraryID,
    document_id: DocumentID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Get a specific document by ID"""
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

        return _to_document_response(document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    library_id: LibraryID,
    document_id: DocumentID,
    request: UpdateDocumentRequest,
    library_repo: LibraryRepository = Depends(get_library_repository),
    document_service: DocumentService = Depends(get_document_service),
):
    """Update a document's content"""
    try:
        # Find library and document
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

        # Update document content
        updated_document = document_service.update_document_content(
            document=document,
            new_text=request.text,
            chunk_size=request.chunk_size,
        )

        # Update document in library
        updated_library = library.update_document(updated_document)
        library_repo.save(updated_library)

        return _to_document_response(updated_document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document",
        )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    library_id: LibraryID,
    document_id: DocumentID,
    library_repo: LibraryRepository = Depends(get_library_repository),
):
    """Delete a document from a library"""
    try:
        # Find library
        library = library_repo.find_by_id(library_id)
        if not library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Library with ID '{library_id}' not found",
            )

        # Check if document exists
        document = library.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found",
            )

        # Remove document from library
        updated_library = library.remove_document(document_id)
        library_repo.save(updated_library)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )