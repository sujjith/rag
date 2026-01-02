from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Document:
    """Represents a source document."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Chunk:
    """A chunk of text from a document."""
    id: str
    document_id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

@dataclass
class SearchResult:
    """Result from vector search."""
    chunk: Chunk
    score: float

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: list[SearchResult]
    query: str