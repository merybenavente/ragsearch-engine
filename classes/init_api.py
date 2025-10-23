from classes.init_classes import (
    Document, DocumentID,
    Library, LibraryID, Metadata
)
from typing import List, Optional


def create_document(
    # name: str,
    library_id: LibraryID,
    text: Optional[str],
    username: Optional[str] = None,
    tags: Optional[List[str]] = [],
):
    document = Document(
        library_id=library_id,
        # name=name,
        metadata=Metadata(username, tags)
    )
    if text:
        document.populate_content(text)
    return document

def update_document(
    library: Library,
    document: Document,
    text: str
) -> bool:
    if not library.document_exists(document.id):
        print("warning or error: document non found")
        return
    if document.has_content():
        print("warning or error: over writing document content")
    return document.set_content(text)

def delete_document(
    library: Library,
    document: Document,
) -> bool:
    library.document_exists(document.id)
    library.pop(document)

def create_library(
        name: str,
        documents: Optional[List[DocumentID]] = [],
        username: Optional[str] = None,
        tags: Optional[List[str]] = [],
    ) -> bool:
    library = Library(
        name=name,
        documents=documents,
        metadata=Metadata(username, tags)
    )
    # LIBRARIES[library.id] = library

def search_in_library(library: id):
    pass