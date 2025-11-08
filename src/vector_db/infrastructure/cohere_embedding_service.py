"""Cohere embedding service implementation"""
import os
import numpy as np
import cohere
from typing import List, Optional

from ..domain.interfaces import EmbeddingService
from ..domain.models import EMBEDDING_MODEL
from .logging import get_logger

logger = get_logger(__name__)


class CohereEmbeddingService(EmbeddingService):
    """Cohere implementation of the EmbeddingService interface"""

    def __init__(self):
        """Initialize the Cohere embedding service"""
        self._client: Optional[cohere.Client] = self._get_cohere_client()

    def _get_cohere_client(self) -> Optional[cohere.Client]:
        """Get Cohere client, returns None if API key is not available"""
        try:
            api_key = os.environ.get("COHERE_API_KEY")
            if api_key:
                timeout = float(
                    os.environ.get("COHERE_TIMEOUT", "30")
                )  # Default 30 seconds
                return cohere.Client(api_key, timeout=timeout)
            logger.warning("Cohere API key not found in environment variables")
            return None
        except Exception as e:
            logger.error("Failed to initialize Cohere client", error=str(e))
            return None

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
        if self._client is None:
            raise RuntimeError(
                "Cohere client not available. Please set COHERE_API_KEY environment variable."
            )

        try:
            resp = self._client.embed(
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
