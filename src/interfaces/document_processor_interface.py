from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .vector_db_interface import Document


class ProcessedContent:
    """A class to represent processed content from a document."""

    def __init__(
        self,
        text: str = "",
        images: List[bytes] = None,
        tables: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.text = text
        self.images = images or []
        self.tables = tables or []
        self.metadata = metadata or {}

    def __repr__(self):
        return f"ProcessedContent(text_length={len(self.text)}, images_count={len(self.images)}, tables_count={len(self.tables)})"


class DocumentProcessor(ABC):
    """An interface for document processors."""

    @abstractmethod
    def process_document(
        self, file_path: str, chunk_strategy: str = "page"
    ) -> List[Document]:
        """
        Process a document and return a list of Document objects.

        Args:
            file_path: Path to the document file to process.
            chunk_strategy: Strategy for chunking content ("page", "section", "paragraph").

        Returns:
            A list of Document objects containing processed content.
        """
        pass

    @abstractmethod
    def extract_content(self, file_path: str) -> ProcessedContent:
        """
        Extract all content from a document (text, images, tables).

        Args:
            file_path: Path to the document file.

        Returns:
            ProcessedContent object containing all extracted data.
        """
        pass

    @abstractmethod
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic document information and metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary containing document metadata.
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx']).
        """
        pass

    @abstractmethod
    def get_processor_name(self) -> str:
        """
        Get the name of the processor implementation.

        Returns:
            Name of the processor (e.g., "UnstructuredProcessor", "PyMuPDFProcessor").
        """
        pass
