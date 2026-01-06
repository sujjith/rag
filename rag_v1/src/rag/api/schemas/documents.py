# src/rag/api/schemas/documents.py
"""Document-related schemas."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Document metadata."""
    filename: str
    file_type: str
    file_size: int
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentResponse(BaseModel):
    """Response for a single document."""
    id: str
    source: str
    metadata: DocumentMetadata


class DocumentListResponse(BaseModel):
    """Response for document list."""
    documents: list[DocumentResponse]
    total: int


class IngestRequest(BaseModel):
    """Request for document ingestion (JSON metadata)."""
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class IngestResponse(BaseModel):
    """Response for document ingestion."""
    document_id: str
    filename: str
    chunks_created: int
    message: str