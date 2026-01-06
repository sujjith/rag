# src/rag/exceptions.py
"""Custom exceptions for the RAG system."""

class RAGException(Exception):
    """Base exception for RAG system."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(RAGException):
    """Raised when a document is not found."""
    pass


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class LLMError(RAGException):
    """Raised when LLM generation fails."""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class RetryExhaustedError(RAGException):
    """Raised when all retry attempts are exhausted."""
    pass