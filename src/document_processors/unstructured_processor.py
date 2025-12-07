from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf

from src.interfaces.document_processor_interface import DocumentProcessor
from src.interfaces.vector_db_interface import Document


class UnstructuredPDFProcessor(DocumentProcessor):
    """Simple PDF text extractor using unstructured library."""

    def process_document(self, file_path: str) -> List[Document]:
        """Extract text from PDF and return page-by-page chunks."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract elements
        elements = partition_pdf(file_path)

        # Group by page
        page_content = {}
        for element in elements:
            page_num = (
                getattr(element.metadata, "page_number", 1)
                if hasattr(element, "metadata")
                else 1
            )
            if page_num not in page_content:
                page_content[page_num] = []
            page_content[page_num].append(str(element))

        # Create documents
        documents = []
        for page_num, texts in page_content.items():
            content = "\n\n".join(texts).strip()
            if content:  # Only add pages with content
                metadata = {
                    "source": file_path,
                    "page_number": page_num,
                    "processor": self.get_processor_name(),
                }
                documents.append(Document(content, metadata))

        return documents

    def get_processor_name(self) -> str:
        """Get processor name."""
        return "UnstructuredPDFProcessor"
