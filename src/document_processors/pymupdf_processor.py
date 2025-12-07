from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.interfaces.document_processor_interface import DocumentProcessor
from src.interfaces.vector_db_interface import Document


class PyMuPDFProcessor(DocumentProcessor):
    """Simple PDF text extractor using PyMuPDF."""

    def process_document(self, file_path: str) -> List[Document]:
        """Extract text from PDF and return page-by-page chunks."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        documents = []
        pdf_doc = fitz.open(file_path)

        try:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text().strip()

                if text:  # Only add pages with text
                    metadata = {
                        "source": file_path,
                        "page_number": page_num + 1,
                        "processor": self.get_processor_name(),
                    }
                    documents.append(Document(text, metadata))

        finally:
            pdf_doc.close()

        return documents

    def get_processor_name(self) -> str:
        """Get processor name."""
        return "PyMuPDFProcessor"
