# src/rag/api/schemas/common.py
"""Common schema definitions."""
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str
    data: dict[str, Any] = Field(default_factory=dict)