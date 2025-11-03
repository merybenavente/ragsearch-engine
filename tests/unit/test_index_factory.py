"""Tests for IndexFactory"""
import pytest
from src.vector_db.infrastructure.index_factory import IndexFactory, get_index_factory
from src.vector_db.infrastructure.indexes.naive import NaiveIndex
from src.vector_db.infrastructure.indexes.lsh import LSHIndex
from src.vector_db.infrastructure.indexes.vptree import VPTreeIndex


class TestIndexFactory:
    """Test IndexFactory functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.factory = IndexFactory()

    def test_create_naive_index(self):
        """Test creating naive index"""
        index = self.factory.create_index("naive")
        assert isinstance(index, NaiveIndex)

    def test_create_lsh_index_default(self):
        """Test creating LSH index with default parameters"""
        index = self.factory.create_index("lsh")
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 8
        assert index.num_hyperplanes == 6

    def test_create_lsh_index_custom_params(self):
        """Test creating LSH index with custom parameters via factory"""
        index = self.factory.create_index("lsh", num_tables=5, num_hyperplanes=8)
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 5
        assert index.num_hyperplanes == 8
        assert len(index.hash_tables) == 5
        assert len(index.hyperplanes) == 0
        assert index.vector_dim == 0

    def test_create_lsh_index_partial_params(self):
        """Test creating LSH index with partial custom parameters"""
        index = self.factory.create_index("lsh", num_tables=10)
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 10
        assert index.num_hyperplanes == 6  # Default value

    def test_create_vptree_index(self):
        """Test creating VPTree index"""
        index = self.factory.create_index("vptree")
        assert isinstance(index, VPTreeIndex)

    @pytest.mark.skip(reason="VPTree is WIP")
    def test_create_vptree_index_with_params(self):
        """Test creating VPTree index with parameters"""
        index = self.factory.create_index("vptree", some_param="value")
        assert isinstance(index, VPTreeIndex)

    def test_create_naive_index_with_params(self):
        """Test creating naive index with parameters (should be ignored)"""
        index = self.factory.create_index("naive", some_param="value")
        assert isinstance(index, NaiveIndex)

    def test_unknown_index_type(self):
        """Test creating unknown index type defaults to naive"""
        index = self.factory.create_index("unknown")
        assert isinstance(index, NaiveIndex)

    def test_unknown_index_type_with_params(self):
        """Test creating unknown index type with params defaults to naive"""
        index = self.factory.create_index("unknown", some_param="value")
        assert isinstance(index, NaiveIndex)

    def test_get_index_factory(self):
        """Test getting global factory instance"""
        factory = get_index_factory()
        assert isinstance(factory, IndexFactory)

        # Should return the same instance
        factory2 = get_index_factory()
        assert factory is factory2