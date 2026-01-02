"""
Data Models for RAG Application
================================

This file defines the data structures (models) used throughout the RAG system.
Think of these as "blueprints" that describe what data looks like.

We use Python's @dataclass decorator which:
  - Automatically creates __init__, __repr__, __eq__ methods
  - Makes the code cleaner and more readable
  - Provides type hints for IDE support

Usage:
    from rag.models import Document, Chunk, SearchResult, RAGResponse
"""

from dataclasses import dataclass, field  # dataclass = auto-generate class boilerplate
from typing import Optional               # Optional = value can be None
from datetime import datetime             # For timestamps


# ==============================================================================
# DOCUMENT MODEL
# ==============================================================================
# A Document represents a source file (PDF, TXT, DOCX) that was loaded into the system

@dataclass
class Document:
    """
    Represents a source document that was ingested into the RAG system.
    
    Example:
        doc = Document(
            id="doc_001",
            content="This is the full text content...",
            source="report.pdf"
        )
    """
    
    # Unique identifier for this document (e.g., UUID or filename hash)
    id: str
    
    # The full text content extracted from the document
    content: str
    
    # Extra information about the document (e.g., author, page count, file size)
    # field(default_factory=dict) creates a new empty dict for each Document
    # This avoids the Python mutable default argument bug
    metadata: dict = field(default_factory=dict)
    
    # Original file path or URL where the document came from
    source: str = ""
    
    # When this document was added to the system
    created_at: datetime = field(default_factory=datetime.now)


# ==============================================================================
# CHUNK MODEL
# ==============================================================================
# A Chunk is a small piece of a Document, used because:
#   1. LLMs have token limits - can't process entire documents
#   2. Smaller chunks give more precise search results
#   3. Embeddings work better on focused text segments

@dataclass
class Chunk:
    """
    A chunk of text extracted from a document.
    
    Documents are split into chunks for embedding and retrieval.
    Each chunk maintains a link back to its parent document.
    
    Example:
        chunk = Chunk(
            id="chunk_001",
            document_id="doc_001",
            content="This is paragraph 3 of the document...",
            start_char=1500,
            end_char=2000
        )
    """
    
    # Unique identifier for this chunk
    id: str
    
    # ID of the parent Document this chunk came from
    # Allows us to trace back to the original source
    document_id: str
    
    # The actual text content of this chunk
    content: str
    
    # The embedding vector (list of floats) for this chunk
    # Optional because embedding is generated after chunking
    # Example: [0.123, -0.456, 0.789, ...] (768 numbers for bge-base)
    embedding: Optional[list[float]] = None
    
    # Extra information (e.g., page number, section title)
    metadata: dict = field(default_factory=dict)
    
    # Character positions in the original document
    # Useful for highlighting where the chunk came from
    start_char: int = 0
    end_char: int = 0


# ==============================================================================
# SEARCH RESULT MODEL
# ==============================================================================
# When you search the vector database, you get back SearchResults

@dataclass
class SearchResult:
    """
    A single result from vector similarity search.
    
    Contains the matched chunk and its similarity score.
    Higher score = more similar to the query.
    
    Example:
        result = SearchResult(
            chunk=some_chunk,
            score=0.89  # 89% similar to the query
        )
    """
    
    # The chunk that matched the search query
    chunk: Chunk
    
    # Similarity score (0.0 to 1.0, higher = more relevant)
    # This is cosine similarity between query embedding and chunk embedding
    score: float


# ==============================================================================
# RAG RESPONSE MODEL
# ==============================================================================
# The final output of the RAG system - answer + sources

@dataclass
class RAGResponse:
    """
    Complete response from the RAG system.
    
    Contains the LLM-generated answer along with the source chunks
    that were used to generate it (for transparency/citations).
    
    Example:
        response = RAGResponse(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI that...",
            sources=[search_result_1, search_result_2]
        )
    """
    
    # The answer generated by the LLM
    answer: str
    
    # List of search results that were used as context for the answer
    # These are the "sources" or "citations" for the answer
    sources: list[SearchResult]
    
    # The original question that was asked
    query: str