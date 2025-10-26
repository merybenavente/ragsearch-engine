"""Domain interfaces for dependency injection"""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    """Abstract interface for embedding services"""
    
    @abstractmethod
    def create_embedding(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Create embedding for the given text.
        
        Args:
            text: The text to embed
            input_type: Type of input ("search_document" or "search_query")
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            RuntimeError: If the embedding service is not available
        """
        pass