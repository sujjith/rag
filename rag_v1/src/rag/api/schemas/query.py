# src/rag/api/schemas/query.py
"""Query-related schemas."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SourceCitation(BaseModel):
    """Source citation in response."""
    document_id: str
    filename: str
    content_preview: str = Field(..., max_length=500)
    relevance_score: float
    start_char: int
    end_char: int


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = False
    filter_document_id: Optional[str] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    query: str
    sources: list[SourceCitation]
    processing_time_ms: float