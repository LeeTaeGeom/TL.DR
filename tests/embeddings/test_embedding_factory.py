"""Tests for EmbeddingFactory."""

from typing import List

import pytest

from embeddings import (
    EmbeddingFactory,
    get_best_embedding,
    get_embedding,
)
from interfaces.embedding_interface import Embedding


class DummyEmbedding(Embedding):
    def __init__(self):
        self.created = True

    def embed_documents(self, texts: List[str]):
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str):
        return [float(len(text))]


class TestEmbeddingFactory:
    """Test EmbeddingFactory methods."""

    def test_register_embedding(self):
        """register_embedding should add a new backend name."""
        EmbeddingFactory.register_embedding("dummy", DummyEmbedding)
        assert "dummy" in EmbeddingFactory.get_available_embeddings()

    def test_create_embedding_default(self):
        """Default backend should create without error (no model download tested)."""
        # We avoid constructing real backends that may download models in unit tests.
        EmbeddingFactory.register_embedding("dummy_default", DummyEmbedding)
        emb = EmbeddingFactory.create_embedding("dummy_default")
        assert isinstance(emb, DummyEmbedding)

    def test_create_embedding_unknown(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding"):
            EmbeddingFactory.create_embedding("unknown")

    def test_get_available_embeddings(self):
        names = EmbeddingFactory.get_available_embeddings()
        assert isinstance(names, list)
        assert "e5" in names or "jina" in names


class TestConvenienceFunctions:
    def test_get_embedding_with_dummy(self):
        EmbeddingFactory.register_embedding("dummy2", DummyEmbedding)
        emb = get_embedding("dummy2")
        assert isinstance(emb, DummyEmbedding)

    def test_get_best_embedding_is_callable(self):
        # Not instantiating real heavy models; just ensuring the function resolves a known key.
        EmbeddingFactory.register_embedding("dummy_best", DummyEmbedding)
        # Temporarily override best by creating under the default key name
        EmbeddingFactory.register_embedding("e5", DummyEmbedding)
        emb = get_best_embedding()
        assert isinstance(emb, DummyEmbedding)
