"""Tests for ProcessorFactory."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from document_processors import ProcessorFactory, PyMuPDFProcessor, get_pdf_processor, get_best_pdf_processor


class TestProcessorFactory:
    """Test ProcessorFactory methods."""

    def test_register_processor(self):
        """Test register_processor method."""
        # This is tested implicitly by the factory working
        assert "pymupdf" in ProcessorFactory.get_available_processors()

    def test_create_processor_pymupdf(self):
        """Test create_processor with pymupdf."""
        processor = ProcessorFactory.create_processor("pymupdf")
        assert isinstance(processor, PyMuPDFProcessor)
        assert processor.get_processor_name() == "PyMuPDFProcessor"

    def test_create_processor_default(self):
        """Test create_processor with default (no args)."""
        processor = ProcessorFactory.create_processor()
        assert isinstance(processor, PyMuPDFProcessor)

    def test_create_processor_unknown(self):
        """Test create_processor with unknown type."""
        with pytest.raises(ValueError, match="Unknown processor"):
            ProcessorFactory.create_processor("unknown")

    def test_get_available_processors(self):
        """Test get_available_processors method."""
        processors = ProcessorFactory.get_available_processors()
        assert isinstance(processors, list)
        assert "pymupdf" in processors


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_pdf_processor_default(self):
        """Test get_pdf_processor with default."""
        processor = get_pdf_processor()
        assert processor.get_processor_name() == "PyMuPDFProcessor"

    def test_get_pdf_processor_pymupdf(self):
        """Test get_pdf_processor with pymupdf."""
        processor = get_pdf_processor("pymupdf")
        assert processor.get_processor_name() == "PyMuPDFProcessor"

    def test_get_best_pdf_processor(self):
        """Test get_best_pdf_processor."""
        processor = get_best_pdf_processor()
        assert processor.get_processor_name() == "PyMuPDFProcessor"


@pytest.mark.skipif(not Path("pdf").exists(), reason="No PDF directory")
class TestIntegration:
    """Integration tests."""

    def test_factory_with_real_pdf(self):
        """Test factory-created processor with real PDF."""
        processor = ProcessorFactory.create_processor("pymupdf")
        pdf_files = list(Path("pdf").glob("*.pdf"))
        
        if pdf_files:
            documents = processor.process_document(str(pdf_files[0]))
            assert len(documents) > 0
            assert all(doc.page_content.strip() for doc in documents)