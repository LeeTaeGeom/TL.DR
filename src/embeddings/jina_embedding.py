from typing import List

from sentence_transformers import SentenceTransformer

from src.interfaces.embedding_interface import Embedding


class JinaEmbeddingsV3(Embedding):
    def __init__(self):
        self.model = SentenceTransformer(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
