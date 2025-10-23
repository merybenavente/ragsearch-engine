"""Unit tests for domain models"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.vector_db.domain.models import (
    Chunk, Document, Library, Metadata,
    ChunkID, DocumentID, LibraryID
)


class TestMetadata:
    """Test cases for Metadata model"""

    def test_metadata_creation_with_defaults(self):
        """Test metadata creation with default values"""
        metadata = Metadata()
        
        assert metadata.username is None
        assert metadata.tags == []
        assert isinstance(metadata.creation_time, datetime)
        assert isinstance(metadata.last_update, datetime)

    def test_metadata_creation_with_values(self):
        """Test metadata creation with provided values"""
        tags = ["tag1", "tag2"]
        username = "testuser"
        
        metadata = Metadata(username=username, tags=tags)
        
        assert metadata.username == username
        assert metadata.tags == tags

    def test_metadata_update_timestamp(self):
        """Test updating metadata timestamp"""
        metadata = Metadata()
        original_time = metadata.last_update
        
        # Wait a tiny bit to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        updated_metadata = metadata.update_timestamp()
        
        assert updated_metadata.last_update > original_time
        assert updated_metadata.creation_time == metadata.creation_time


class TestChunk:
    """Test cases for Chunk model"""

    def test_chunk_creation_with_text(self):
        """Test chunk creation with text content"""
        document_id = str(uuid4())
        text = "This is a test chunk"
        
        chunk = Chunk(document_id=document_id, text=text)
        
        assert chunk.document_id == document_id
        assert chunk.text == text
        assert isinstance(chunk.id, str)
        assert len(chunk.embedding) == 768  # Default embedding size
        assert isinstance(chunk.metadata, Metadata)

    def test_chunk_creation_with_embedding(self):
        """Test chunk creation with provided embedding"""
        document_id = str(uuid4())
        text = "Test text"
        embedding = [0.1, 0.2, 0.3]
        
        chunk = Chunk(document_id=document_id, text=text, embedding=embedding)
        
        assert chunk.embedding == embedding

    def test_chunk_creation_requires_text(self):
        """Test that chunk creation requires text"""
        document_id = str(uuid4())
        
        with pytest.raises(ValueError):
            Chunk(document_id=document_id, text="")

    def test_chunk_immutability(self):
        """Test that chunks are immutable"""
        document_id = str(uuid4())
        text = "Test text"
        
        chunk = Chunk(document_id=document_id, text=text)
        
        # Chunks should be immutable - attempting to modify should create new instance
        with pytest.raises(AttributeError):
            chunk.text = "New text"


class TestDocument:
    """Test cases for Document model"""

    def test_document_creation_empty(self):
        """Test creation of empty document"""
        library_id = str(uuid4())
        
        document = Document(library_id=library_id)
        
        assert document.library_id == library_id
        assert isinstance(document.id, str)
        assert document.chunks == []
        assert isinstance(document.metadata, Metadata)

    def test_document_has_content(self):
        """Test document content detection"""
        library_id = str(uuid4())
        
        # Empty document
        document = Document(library_id=library_id)
        assert not document.has_content()
        
        # Document with content
        document_with_content = document.replace_content("Some content")
        assert document_with_content.has_content()

    def test_document_replace_content(self):
        """Test document content replacement"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        
        text = "This is a test document with some content that should be chunked."
        updated_doc = document.replace_content(text, chunk_size=20)
        
        assert updated_doc.has_content()
        assert len(updated_doc.chunks) > 1  # Should be chunked
        assert updated_doc.get_full_text() == text

    def test_document_replace_content_empty(self):
        """Test replacing content with empty string"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        
        # Add content first
        document = document.replace_content("Some content")
        assert document.has_content()
        
        # Replace with empty content
        empty_doc = document.replace_content("")
        assert not empty_doc.has_content()
        assert empty_doc.chunks == []

    def test_document_get_chunk_by_id(self):
        """Test getting chunk by ID"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        
        document = document.replace_content("Test content")
        
        # Get first chunk
        chunk = document.chunks[0]
        found_chunk = document.get_chunk_by_id(chunk.id)
        
        assert found_chunk is not None
        assert found_chunk.id == chunk.id
        
        # Test non-existent chunk
        non_existent = document.get_chunk_by_id("non-existent-id")
        assert non_existent is None

    def test_document_get_chunk_ids(self):
        """Test getting all chunk IDs"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        
        document = document.replace_content("Test content")
        
        chunk_ids = document.get_chunk_ids()
        assert len(chunk_ids) == len(document.chunks)
        assert all(chunk_id in [c.id for c in document.chunks] for chunk_id in chunk_ids)

    def test_document_chunking_strategy(self):
        """Test document chunking with different chunk sizes"""
        library_id = str(uuid4())
        document = Document(library_id=library_id)
        
        text = "a" * 1000  # 1000 character text
        
        # Test different chunk sizes
        doc_100 = document.replace_content(text, chunk_size=100)
        doc_500 = document.replace_content(text, chunk_size=500)
        
        assert len(doc_100.chunks) > len(doc_500.chunks)
        assert all(len(chunk.text) <= 100 for chunk in doc_100.chunks)
        assert all(len(chunk.text) <= 500 for chunk in doc_500.chunks)


class TestLibrary:
    """Test cases for Library model"""

    def test_library_creation(self):
        """Test library creation"""
        name = "Test Library"
        
        library = Library(name=name)
        
        assert library.name == name
        assert isinstance(library.id, str)
        assert library.documents == []
        assert isinstance(library.metadata, Metadata)

    def test_library_add_document(self):
        """Test adding document to library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        
        updated_library = library.add_document(document)
        
        assert len(updated_library.documents) == 1
        assert updated_library.documents[0].id == document.id
        assert updated_library.documents[0].library_id == library.id

    def test_library_add_document_fixes_library_id(self):
        """Test that adding document fixes library_id if incorrect"""
        library = Library(name="Test Library")
        wrong_library_id = str(uuid4())
        document = Document(library_id=wrong_library_id)
        
        updated_library = library.add_document(document)
        
        # Should fix the library_id
        assert updated_library.documents[0].library_id == library.id

    def test_library_remove_document(self):
        """Test removing document from library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        
        # Add document
        library = library.add_document(document)
        assert len(library.documents) == 1
        
        # Remove document
        library = library.remove_document(document.id)
        assert len(library.documents) == 0

    def test_library_update_document(self):
        """Test updating document in library"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        
        # Add document
        library = library.add_document(document)
        
        # Update document content
        updated_document = document.replace_content("New content")
        library = library.update_document(updated_document)
        
        # Should have updated document
        retrieved_doc = library.get_document_by_id(document.id)
        assert retrieved_doc is not None
        assert retrieved_doc.has_content()

    def test_library_get_document_by_id(self):
        """Test getting document by ID"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        
        library = library.add_document(document)
        
        # Test existing document
        found_doc = library.get_document_by_id(document.id)
        assert found_doc is not None
        assert found_doc.id == document.id
        
        # Test non-existent document
        non_existent = library.get_document_by_id("non-existent-id")
        assert non_existent is None

    def test_library_document_exists(self):
        """Test checking if document exists"""
        library = Library(name="Test Library")
        document = Document(library_id=library.id)
        
        # Before adding
        assert not library.document_exists(document.id)
        
        # After adding
        library = library.add_document(document)
        assert library.document_exists(document.id)

    def test_library_get_all_chunks(self):
        """Test getting all chunks from library"""
        library = Library(name="Test Library")
        
        # Add documents with content
        doc1 = Document(library_id=library.id).replace_content("Document 1 content")
        doc2 = Document(library_id=library.id).replace_content("Document 2 content")
        
        library = library.add_document(doc1)
        library = library.add_document(doc2)
        
        all_chunks = library.get_all_chunks()
        
        expected_count = len(doc1.chunks) + len(doc2.chunks)
        assert len(all_chunks) == expected_count

    def test_library_get_all_chunk_ids(self):
        """Test getting all chunk IDs from library"""
        library = Library(name="Test Library")
        
        # Add document with content
        doc = Document(library_id=library.id).replace_content("Test content")
        library = library.add_document(doc)
        
        chunk_ids = library.get_all_chunk_ids()
        expected_ids = [chunk.id for chunk in doc.chunks]
        
        assert len(chunk_ids) == len(expected_ids)
        assert set(chunk_ids) == set(expected_ids)

    def test_library_get_document_ids(self):
        """Test getting all document IDs from library"""
        library = Library(name="Test Library")
        
        # Add multiple documents
        doc1 = Document(library_id=library.id)
        doc2 = Document(library_id=library.id)
        
        library = library.add_document(doc1)
        library = library.add_document(doc2)
        
        document_ids = library.get_document_ids()
        
        assert len(document_ids) == 2
        assert doc1.id in document_ids
        assert doc2.id in document_ids