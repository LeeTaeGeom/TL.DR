"""Simple factory for PDF processors."""

from typing import Dict, Type

from src.interfaces.document_processor_interface import DocumentProcessor

from .pymupdf_processor import PyMuPDFProcessor
from .unstructured_processor import UnstructuredPDFProcessor


class ProcessorFactory:
    """Factory for creating PDF processors."""

    _processors: Dict[str, Type[DocumentProcessor]] = {
        "pymupdf": PyMuPDFProcessor,
        "unstructured": UnstructuredPDFProcessor,
    }

    @classmethod
    def register_processor(cls, name: str, processor_class: Type[DocumentProcessor]):
        """Register a processor."""
        cls._processors[name] = processor_class

    @classmethod
    def create_processor(cls, processor_type: str = "pymupdf") -> DocumentProcessor:
        """Create a processor instance."""
        if processor_type not in cls._processors:
            available = list(cls._processors.keys())
            raise ValueError(
                f"Unknown processor: {processor_type}. Available: {available}"
            )

        return cls._processors[processor_type]()

    @classmethod
    def get_available_processors(cls) -> list:
        """Get available processor names."""
        return list(cls._processors.keys())


# Convenience functions
def get_pdf_processor(processor_type: str = "pymupdf") -> DocumentProcessor:
    """Get a PDF processor."""
    return ProcessorFactory.create_processor(processor_type)


def get_best_pdf_processor() -> DocumentProcessor:
    """Get the best available processor (defaults to pymupdf)."""
    return ProcessorFactory.create_processor("pymupdf")
