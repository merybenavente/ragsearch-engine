"""Tests for IndexFactory"""
import pytest
from src.vector_db.infrastructure.indexes.naive import NaiveIndex
from src.vector_db.infrastructure.indexes.lsh import LSHIndex
from src.vector_db.infrastructure.indexes.vptree import VPTreeIndex


class TestIndexFactory:
    """Test IndexFactory functionality"""

    def test_create_naive_index(self, index_factory_instance):
        """Test creating naive index"""
        index = index_factory_instance.create_index("naive")
        assert isinstance(index, NaiveIndex)

    def test_create_lsh_index_default(self, index_factory_instance):
        """Test creating LSH index with default parameters"""
        index = index_factory_instance.create_index("lsh")
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 6
        assert index.num_hyperplanes == 4

    def test_create_lsh_index_custom_params(self, index_factory_instance):
        """Test creating LSH index with custom parameters via factory"""
        index = index_factory_instance.create_index(
            "lsh", num_tables=5, num_hyperplanes=8
        )
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 5
        assert index.num_hyperplanes == 8
        assert len(index.hash_tables) == 5
        assert len(index.hyperplanes) == 0
        assert index.vector_dim == 0

    def test_create_lsh_index_partial_params(self, index_factory_instance):
        """Test creating LSH index with partial custom parameters"""
        index = index_factory_instance.create_index("lsh", num_tables=10)
        assert isinstance(index, LSHIndex)
        assert index.num_tables == 10
        assert index.num_hyperplanes == 4  # Default value

    def test_create_vptree_index(self, index_factory_instance):
        """Test creating VPTree index"""
        index = index_factory_instance.create_index("vptree")
        assert isinstance(index, VPTreeIndex)

    def test_unknown_index_type_raises_error(self, index_factory_instance):
        """Test creating unknown index type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown index type 'unknown'"):
            index_factory_instance.create_index("unknown")

    def test_unknown_index_type_with_params_raises_error(self, index_factory_instance):
        """Test creating unknown index type with params raises ValueError"""
        with pytest.raises(ValueError, match="Unknown index type 'typo'"):
            index_factory_instance.create_index("typo", some_param="value")
