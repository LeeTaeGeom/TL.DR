from abc import ABC, abstractmethod
from typing import List

class Embedding(ABC):
    """An interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of documents.
        
        Args:
            texts: A list of strings to embed.
            
        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Create an embedding for a single query.
        
        Args:
            text: The query string to embed.
            
        Returns:
            A single embedding (list of floats).
        """
        pass
