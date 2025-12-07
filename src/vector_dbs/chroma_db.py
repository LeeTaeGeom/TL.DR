"""ChromaDB implementation for vector storage."""

import uuid
from typing import List, Optional

import chromadb

from src.interfaces.vector_db_interface import Document, VectorDB


class ChromaVectorDB(VectorDB):
    """Vector database implementation using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "tldr_documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB.

        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory to persist the database. If None, uses in-memory.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: A list of Document objects.
            embeddings: A list of corresponding embeddings for the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

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
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                if results.get("distances"):
                    metadata["distance"] = results["distances"][0][i]
                documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
