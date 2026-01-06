# tests/test_chunker.py - Unit tests for SentenceChunker
import pytest
from rag.ingestion.chunker import SentenceChunker
from rag.models import Document, Chunk

chunker = SentenceChunker(chunk_size=50, overlap=10)  # Create chunker with small size for testing

class TestSentenceChunker:

    def test_chunk_returns_chunks(self):  # Basic test - does it return Chunk objects?
        doc = Document(id="test_doc", content="Hello world. This is a test.")
        
        chunks = list(chunker.chunk(doc))  # Convert generator to list
        
        assert len(chunks) > 0  # Should have at least one chunk
        assert all(isinstance(c, Chunk) for c in chunks)  # All should be Chunk objects

    def test_chunk_preserves_document_id(self):  # Chunks should link back to document
        doc = Document(id="my_doc_123", content="First sentence. Second sentence.")
        
        chunks = list(chunker.chunk(doc))
        
        for chunk in chunks:
            assert chunk.document_id == "my_doc_123"  # All chunks should have same parent ID

    def test_chunk_generates_unique_ids(self):  # Each chunk should have unique ID
        doc = Document(id="doc", content="One. Two. Three. Four. Five. Six. Seven. Eight.")
        
        chunks = list(chunker.chunk(doc))
        chunk_ids = [c.id for c in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicate IDs

    def test_chunk_respects_size_limit(self):  # Chunks should not exceed chunk_size (approximately)
        doc = Document(id="doc", content="A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P.")
        small_chunker = SentenceChunker(chunk_size=20, overlap=5)
        
        chunks = list(small_chunker.chunk(doc))
        
        for chunk in chunks:
            assert len(chunk.content) <= 30  # Allow some flexibility for sentence boundaries

    def test_chunk_has_overlap(self):  # Chunks should overlap with each other
        doc = Document(
            id="doc", 
            content="First sentence here. Second sentence there. Third sentence everywhere."
        )
        chunker_with_overlap = SentenceChunker(chunk_size=30, overlap=10)
        
        chunks = list(chunker_with_overlap.chunk(doc))
        
        if len(chunks) > 1:  # Only check if we have multiple chunks
            chunk1_end = chunks[0].content[-10:]  # Last 10 chars of chunk 1
            chunk2_start = chunks[1].content[:10]  # First 10 chars of chunk 2
            
            print(f"\nChunk 1 ends with: '{chunk1_end}'")
            print(f"Chunk 2 starts with: '{chunk2_start}'")
            
            assert chunk1_end in chunks[1].content  # Overlap should appear in next chunk

    def test_chunk_copies_metadata(self):  # Metadata should be copied to each chunk
        doc = Document(
            id="doc", 
            content="Hello world. Testing metadata.",
            metadata={"filename": "test.txt", "author": "John"}
        )
        
        chunks = list(chunker.chunk(doc))
        
        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.txt"
            assert chunk.metadata["author"] == "John"

    def test_chunk_tracks_positions(self):  # start_char and end_char should be set
        doc = Document(id="doc", content="Short text. More text here.")
        
        chunks = list(chunker.chunk(doc))
        
        for chunk in chunks:
            assert chunk.start_char >= 0  # Valid start position
            assert chunk.end_char > chunk.start_char  # End after start

    def test_empty_document(self):  # Should handle empty content
        doc = Document(id="empty", content="")
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) <= 1  # Either 0 or 1 empty chunk

    def test_single_sentence(self):  # Document with just one sentence
        doc = Document(id="single", content="This is the only sentence.")
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 1
        assert chunks[0].content == "This is the only sentence."

    def test_chunk_with_real_text(self):  # Test with more realistic content
        doc = Document(
            id="hamlet",
            content="To be or not to be. That is the question. Whether tis nobler in the mind. To suffer the slings and arrows."
        )
        
        chunks = list(chunker.chunk(doc))
        
        print(f"\n--- Chunks from Hamlet text ---")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: '{chunk.content}'")
        print(f"--- End ---\n")
        
        assert len(chunks) >= 1
        assert all(len(c.content) > 0 for c in chunks)  # No empty chunks
