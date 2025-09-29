"""Simple document processors for RAG."""

from .processor_factory import (
    ProcessorFactory,
    get_best_pdf_processor,
    get_pdf_processor,
)
from .pymupdf_processor import PyMuPDFProcessor


# Try to import unstructured processor
try:
    from .unstructured_processor import UnstructuredPDFProcessor

    __all__ = [
        "ProcessorFactory",
        "PyMuPDFProcessor",
        "UnstructuredPDFProcessor",
        "get_pdf_processor",
        "get_best_pdf_processor",
    ]
except ImportError:
    __all__ = [
        "ProcessorFactory",
        "PyMuPDFProcessor",
        "get_pdf_processor",
        "get_best_pdf_processor",
    ]
