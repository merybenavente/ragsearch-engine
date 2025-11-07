"""Shared test utilities for vector database tests"""

import numpy as np
from typing import List


def create_deterministic_embedding(text: str, dimension: int = 10) -> List[float]:
    """Create deterministic embedding based on text content

    Args:
        text: Input text to create embedding for
        dimension: Embedding dimension (default: 10)

    Returns:
        Normalized embedding vector as list of floats
    """
    seed = hash(text) % (2**32)
    np.random.seed(seed)
    embedding = np.random.randn(dimension)
    return (embedding / np.linalg.norm(embedding)).tolist()
