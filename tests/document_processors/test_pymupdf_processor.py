"""Tests for PyMuPDFProcessor."""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from document_processors.pymupdf_processor import PyMuPDFProcessor


class TestPyMuPDFProcessor:
    """Test PyMuPDFProcessor methods."""

    def test_get_processor_name(self):
        """Test get_processor_name method."""
        processor = PyMuPDFProcessor()
        assert processor.get_processor_name() == "PyMuPDFProcessor"

    def test_process_document_file_not_found(self):
        """Test process_document with non-existent file."""
        processor = PyMuPDFProcessor()

        with pytest.raises(FileNotFoundError):
            processor.process_document("nonexistent.pdf")

    @pytest.mark.skipif(not Path("pdf").exists(), reason="No PDF directory")
    def test_process_document_with_real_pdf(self):
        """Test process_document with real PDF."""
        processor = PyMuPDFProcessor()
        pdf_files = list(Path("pdf").glob("*.pdf"))

        if pdf_files:
            documents = processor.process_document(str(pdf_files[0]))

            # Check results
            assert len(documents) > 0
            assert all(doc.page_content.strip() for doc in documents)
            assert all("source" in doc.metadata for doc in documents)
            assert all("page_number" in doc.metadata for doc in documents)
            assert all("processor" in doc.metadata for doc in documents)

            # Check first document
            first_doc = documents[0]
            assert first_doc.metadata["processor"] == "PyMuPDFProcessor"
            assert first_doc.metadata["page_number"] == 1
            assert isinstance(first_doc.page_content, str)
            assert len(first_doc.page_content) > 0
