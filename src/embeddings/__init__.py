"""Embeddings backends and factory exports."""

from .bge_embedding import BgeM3Embedding
from .e5_embedding import MultilingualE5LargeInstruct
from .embedding_factory import (
    EmbeddingFactory,
    get_best_embedding,
    get_embedding,
)
from .jina_embedding import JinaEmbeddingsV3
from .qwen_embedding import Qwen3Embedding06B


__all__ = [
    "EmbeddingFactory",
    "get_embedding",
    "get_best_embedding",
    "MultilingualE5LargeInstruct",
    "JinaEmbeddingsV3",
    "BgeM3Embedding",
    "Qwen3Embedding06B",
]
