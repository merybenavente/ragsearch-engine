"""
Factory for creating vector indexes in the new DDD architecture.
"""
from ..domain.interfaces import VectorIndex
from .indexes.naive import NaiveIndex
from .indexes.lsh import LSHIndex
from .indexes.vptree import VPTreeIndex
from ..infrastructure.logging import get_logger

logger = get_logger(__name__)

AVAILABLE_INDEX_TYPES = ["naive", "lsh", "vptree"]


class IndexFactory:
    """Factory for creating vector indexes"""

    def create_index(self, index_type: str, **kwargs) -> VectorIndex:
        """Create a vector index of the specified type"""
        if index_type == "naive":
            return NaiveIndex()
        elif index_type == "lsh":
            return LSHIndex(**kwargs)
        elif index_type == "vptree":
            return VPTreeIndex(**kwargs)
        else:
            logger.warning(f"Unknown index type '{index_type}', defaulting to naive")
            return NaiveIndex()


# Global factory instance
index_factory = IndexFactory()


def get_index_factory() -> IndexFactory:
    """Get the global index factory instance"""
    return index_factory
