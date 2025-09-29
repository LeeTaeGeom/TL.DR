"""Embeddings backends and factory exports."""

from .e5_embedding import MultilingualE5LargeInstruct
from .embedding_factory import (
    EmbeddingFactory,
    get_best_embedding,
    get_embedding,
)
from .jina_embedding import JinaEmbeddingsV3


__all__ = [
    "EmbeddingFactory",
    "get_embedding",
    "get_best_embedding",
    "MultilingualE5LargeInstruct",
    "JinaEmbeddingsV3",
]
