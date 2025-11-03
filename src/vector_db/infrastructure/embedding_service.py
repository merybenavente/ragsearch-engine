"""Cohere embedding service implementation"""
import numpy as np
from typing import List

from ..domain.interfaces import EmbeddingService
from ..domain.models import EMBEDDING_MODEL
from .cohere_client import co
from .logging import get_logger

logger = get_logger(__name__)


class CohereEmbeddingService(EmbeddingService):
    """Cohere implementation of the EmbeddingService interface"""

    def create_embedding(
        self, text: str, input_type: str = "search_document"
    ) -> List[float]:
        """
        Create embedding using Cohere's embedding model.

        Args:
            text: The text to embed
            input_type: Type of input ("search_document" or "search_query")

        Returns:
            Normalized embedding vector as list of floats

        Raises:
            RuntimeError: If Cohere client is not available
        """
        if co is None:
            raise RuntimeError(
                "Cohere client not available. Please set COHERE_API_KEY environment variable."
            )

        try:
            resp = co.embed(
                texts=[text],
                model=EMBEDDING_MODEL,
                input_type=input_type,
                truncate="NONE",
            )

            embeddings = np.array(resp.embeddings)
            # Normalize the embedding using L2 norm
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            result = embeddings[0].tolist()
            logger.debug(
                "Created embedding", text_length=len(text), embedding_dim=len(result)
            )
            return result

        except Exception as e:
            logger.error(
                "Failed to create embedding", error=str(e), text_length=len(text)
            )
            raise RuntimeError(f"Failed to create embedding: {str(e)}") from e
