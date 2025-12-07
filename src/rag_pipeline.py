"""RAG Pipeline - Combines document processing, embedding, vector DB, and LLM."""

from pathlib import Path
from typing import List

from src.interfaces.document_processor_interface import DocumentProcessor
from src.interfaces.embedding_interface import Embedding
from src.interfaces.llm_interface import LLM
from src.interfaces.vector_db_interface import Document, VectorDB


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding: Embedding,
        vector_db: VectorDB,
        llm: LLM,
    ):
        """
        Initialize RAG Pipeline.

        Args:
            document_processor: Document processor for loading documents.
            embedding: Embedding model for vectorizing text.
            vector_db: Vector database for storing and searching documents.
            llm: LLM for generating responses.
        """
        self.document_processor = document_processor
        self.embedding = embedding
        self.vector_db = vector_db
        self.llm = llm

    def ingest_document(self, file_path: str) -> int:
        """
        Ingest a document into the vector database.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks ingested.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process document into Document objects
        documents = self.document_processor.process_document(file_path)

        if not documents:
            return 0

        # Extract text content for embedding
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Store in vector database
        self.vector_db.add_documents(documents, embeddings)

        return len(documents)

    def ingest_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> int:
        """
        Ingest all documents from a directory.

        Args:
            directory_path: Path to the directory.
            extensions: List of file extensions to process (e.g., ['.pdf', '.txt']).
                       If None, processes all supported files.

        Returns:
            Total number of chunks ingested.
        """
        extensions = extensions or [".pdf"]
        path = Path(directory_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        total_chunks = 0
        for ext in extensions:
            for file_path in path.glob(f"**/*{ext}"):
                try:
                    chunks = self.ingest_document(str(file_path))
                    total_chunks += chunks
                    print(f"✓ Ingested {file_path.name}: {chunks} chunks")
                except Exception as e:
                    print(f"✗ Failed to ingest {file_path.name}: {e}")

        return total_chunks

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.
            k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """
        query_embedding = self.embedding.embed_query(query)
        return self.vector_db.search_by_embedding(query_embedding, k)

    def query(self, question: str, k: int = 5) -> str:
        """
        Answer a question using RAG.

        Args:
            question: The question to answer.
            k: Number of documents to retrieve for context.

        Returns:
            The generated answer.
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve(question, k)

        if not relevant_docs:
            return "관련 문서를 찾을 수 없습니다. 먼저 문서를 추가해주세요."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[문서 {i}] (출처: {Path(source).name})\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Create prompt
        prompt = self._create_rag_prompt(question, context)

        # Generate response
        return self.llm.generate(prompt)

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a RAG prompt with question and context."""
        return f"""다음 문서 내용을 참고하여 질문에 답변해주세요.

## 참고 문서
{context}

## 질문
{question}

## 답변
위의 문서 내용을 바탕으로 질문에 대해 상세하고 정확하게 답변해주세요.
문서에 없는 내용은 추측하지 말고, 문서에 관련 정보가 없다면 그렇다고 말씀해주세요."""

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "document_count": self.vector_db.get_document_count(),
        }
