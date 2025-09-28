from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Document:
    """A simple data class for representing a document."""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class VectorDB(ABC):
    """An interface for vector databases."""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: A list of Document objects.
            embeddings: A list of corresponding embeddings for the documents.
        """
        pass

    @abstractmethod
    def search_by_embedding(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Document]:
        """
        Search for similar documents using a query embedding.

        Args:
            query_embedding: The embedding of the query.
            k: The number of similar documents to return.

        Returns:
            A list of the most similar Document objects found.
        """
        pass
