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
        match index_type:
            case "naive":
                return NaiveIndex()
            case "lsh":
                return LSHIndex(**kwargs)
            case "vptree":
                return VPTreeIndex(**kwargs)
            case _:
                # TODO: it think I prefer to return an error than silently fallback to Naive
                logger.warning(
                    f"Unknown index type '{index_type}', defaulting to naive"
                )
                return NaiveIndex()
