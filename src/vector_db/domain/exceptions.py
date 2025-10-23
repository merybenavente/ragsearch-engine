"""Custom exceptions for the vector database API"""


class VectorDBException(Exception):
    """Base exception for vector database operations"""
    pass


class DuplicateLibraryException(VectorDBException):
    """Raised when trying to create a library that already exists"""
    def __init__(self, library_name: str):
        self.library_name = library_name
        super().__init__(f"Library with name '{library_name}' already exists")