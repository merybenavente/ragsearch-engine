"""Cohere client initialization"""
import os
import cohere
from .logging import get_logger

logger = get_logger(__name__)

def _get_cohere_client():
    """Get Cohere client, returns None if API key is not available"""
    try:
        api_key = os.environ.get("COHERE_API_KEY")
        if api_key:
            timeout = float(os.environ.get("COHERE_TIMEOUT", "30"))  # Default 30 seconds
            return cohere.Client(api_key, timeout=timeout)
        logger.warning("Cohere API key not found in environment variables")
        return None
    except Exception as e:
        logger.error("Failed to initialize Cohere client", error=str(e))
        return None

co = _get_cohere_client()