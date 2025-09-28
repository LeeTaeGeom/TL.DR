"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def pdf_files():
    """Provide paths to test PDF files."""
    pdf_dir = Path(__file__).parent.parent / "pdf"
    return {
        "iphone": pdf_dir / "iPhone_17_Pro_and_iPhone_17_Pro_Max_PER_Sept2025.pdf",
        "mobile": pdf_dir / "mobile_20250910.pdf",
    }


@pytest.fixture
def sample_pdf_path(pdf_files):
    """Provide a single PDF file for basic tests."""
    # Use the smaller file for faster tests
    return pdf_files["mobile"]


@pytest.fixture
def all_pdf_paths(pdf_files):
    """Provide all PDF files for comprehensive tests."""
    return list(pdf_files.values())
