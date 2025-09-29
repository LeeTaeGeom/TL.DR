"""Factory for creating embedding backends."""

from typing import Dict, List, Type

from src.interfaces.embedding_interface import Embedding

from .bge_embedding import BgeM3Embedding
from .e5_embedding import MultilingualE5LargeInstruct
from .jina_embedding import JinaEmbeddingsV3
from .qwen_embedding import Qwen3Embedding06B


class EmbeddingFactory:
    """Factory for creating embeddings."""

    _embeddings: Dict[str, Type[Embedding]] = {
        "e5": MultilingualE5LargeInstruct,
        "jina": JinaEmbeddingsV3,
        "bge-m3": BgeM3Embedding,
        "qwen3": Qwen3Embedding06B,
    }

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Type[Embedding]):
        """Register an embedding backend under a name."""
        cls._embeddings[name] = embedding_class

    @classmethod
    def create_embedding(cls, embedding_type: str = "e5") -> Embedding:
        """Create an embedding instance.

        Args:
            embedding_type: The embedding backend name.

        Returns:
            An instance of the embedding backend.

        Raises:
            ValueError: If the embedding type is unknown.
        """
        if embedding_type not in cls._embeddings:
            available = list(cls._embeddings.keys())
            raise ValueError(
                f"Unknown embedding: {embedding_type}. Available: {available}"
            )

        return cls._embeddings[embedding_type]()

    @classmethod
    def get_available_embeddings(cls) -> List[str]:
        """Get available embedding backend names."""
        return list(cls._embeddings.keys())


# Convenience functions
def get_embedding(embedding_type: str = "e5") -> Embedding:
    """Get an embedding backend instance."""
    return EmbeddingFactory.create_embedding(embedding_type)


def get_best_embedding() -> Embedding:
    """Get the preferred/default embedding backend (defaults to e5)."""
    return EmbeddingFactory.create_embedding("e5")
