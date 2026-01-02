import pytest
from pathlib import Path
from rag.ingestion.loader import DocumentLoader
from rag.models import Document

loader = DocumentLoader()  # Create loader instance once for all tests

class TestDocumentLoader:  # Group all loader tests in a class

    def test_load_returns_document(self, tmp_path):  # tmp_path is a pytest fixture that creates a temp directory
        test_file = tmp_path / "test.txt"  # Create a test file path
        test_file.write_text("Hello World")  # Write some content to it
        
        doc = loader.load(test_file)  # Load the file
        
        assert isinstance(doc, Document)  # Check it returns a Document object
        assert doc.content == "Hello World"  # Check content matches
        assert doc.metadata["type"] == ".txt"  # Check file type is correct

    def test_load_generates_unique_id(self, tmp_path):  # Test that IDs are generated
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")
        
        doc = loader.load(test_file)
        
        assert doc.id is not None  # ID should exist
        assert len(doc.id) == 12  # ID should be 12 characters (from MD5 hash)

    def test_load_same_file_same_id(self, tmp_path):  # Test ID consistency
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")
        
        doc1 = loader.load(test_file)  # Load same file twice
        doc2 = loader.load(test_file)
        
        assert doc1.id == doc2.id  # Same file should produce same ID

    def test_load_stores_source_path(self, tmp_path):  # Test source is stored
        test_file = tmp_path / "myfile.txt"
        test_file.write_text("Content")
        
        doc = loader.load(test_file)
        
        assert doc.source == str(test_file)  # Source should be the file path
        assert doc.metadata["filename"] == "myfile.txt"  # Filename in metadata

    def test_load_unsupported_file_type_raises_error(self, tmp_path):  # Test error handling
        test_file = tmp_path / "data.csv"  # CSV is not supported
        test_file.write_text("a,b,c")
        
        with pytest.raises(ValueError) as exc_info:  # Expect a ValueError
            loader.load(test_file)
        
        assert "Unsupported file type" in str(exc_info.value)  # Check error message

    def test_load_markdown_uses_text_loader(self, tmp_path):  # Test .md files work
        test_file = tmp_path / "readme.md"
        test_file.write_text("# Hello Markdown")
        
        doc = loader.load(test_file)
        
        assert doc.content == "# Hello Markdown"  # Content should be raw markdown
        assert doc.metadata["type"] == ".md"

    def test_load_pdf_extracts_text(self):  # Test PDF loading with real file
        pdf_path = Path("/home/sujith/github-sujith/rag/rag_v1/data/documents/macbeth.pdf")
        
        doc = loader.load(pdf_path)  # Load the PDF
        
        assert isinstance(doc, Document)  # Check it returns a Document
        assert len(doc.content) > 0  # Check content was extracted
        assert doc.metadata["type"] == ".pdf"  # Check file type
        assert doc.metadata["filename"] == "macbeth.pdf"  # Check filename
        assert "macbeth" in doc.content.lower() or len(doc.content) > 100  # Check some content exists
        
        print(f"\n--- PDF Content Preview ---")  # Print header
        print(f"Document ID: {doc.id}")  # Show generated ID
        print(f"Content length: {len(doc.content)} characters")  # Show total length
        print(f"First 500 chars:\n{doc.content[:500]}")  # Show first 500 characters
        print(f"--- End Preview ---\n")
