from abc import ABC, abstractmethod
from typing import List

from .vector_db_interface import Document


class DocumentProcessor(ABC):
    """Simple interface for extracting text from documents for RAG."""

    @abstractmethod
    def process_document(self, file_path: str) -> List[Document]:
        """Extract text from document and return chunks ready for embedding."""
        pass

    @abstractmethod
    def get_processor_name(self) -> str:
        """Get processor name."""
        pass
