"""
Configuration Management for RAG Application
=============================================

This file uses Pydantic Settings to manage all configuration values.
Settings can come from:
  1. Environment variables (e.g., export GROQ_API_KEY=xxx)
  2. A .env file in the project root
  3. Default values defined here

Usage in other files:
    from rag.config import settings
    print(settings.groq_api_key)
"""

from pydantic_settings import BaseSettings  # Handles loading from .env and environment
from pydantic import Field  # Used to add validation and metadata to fields


class Settings(BaseSettings):
    """
    All configuration settings for the RAG application.
    
    Each field follows the pattern:
        field_name: type = default_value
    
    If no default is provided (using Field(...)), the value MUST exist
    in .env or environment variables, otherwise the app will fail to start.
    """
    
    # ==================== GROQ API (LLM Provider) ====================
    # Groq provides fast LLM inference for models like Llama, Mixtral
    
    # REQUIRED: Your Groq API key from https://console.groq.com
    # The "..." means no default - this MUST be set in .env
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    
    # Which Groq model to use for generating answers
    # Options: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
    groq_model: str = "llama-3.1-8b-instant"

    # ==================== EMBEDDING MODEL ====================
    # Embeddings convert text into numerical vectors for similarity search
    
    # HuggingFace model name for generating embeddings
    # BGE (BAAI General Embedding) models are good for RAG
    # Options: BAAI/bge-small-en-v1.5 (faster), BAAI/bge-base-en-v1.5 (better quality)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    # Number of dimensions in the embedding vector
    # Must match the model: bge-small=384, bge-base=768, bge-large=1024
    embedding_dimension: int = 768

    # ==================== QDRANT (Vector Database) ====================
    # Qdrant stores embeddings and enables fast similarity search
    
    # Host where Qdrant is running (use "localhost" for local, or cloud URL)
    qdrant_host: str = "localhost"
    
    # Port for Qdrant (default is 6333)
    qdrant_port: int = 6333
    
    # Name of the collection (like a database table) to store documents
    collection_name: str = "documents"

    # ==================== CHUNKING SETTINGS ====================
    # Documents are split into smaller chunks before embedding
    
    # Maximum number of characters per chunk
    # Smaller = more precise but loses context, Larger = more context but less precise
    chunk_size: int = 512
    
    # How many characters to overlap between chunks
    # Overlap helps maintain context across chunk boundaries
    chunk_overlap: int = 50

    # ==================== RETRIEVAL SETTINGS ====================
    # Controls how documents are retrieved for answering questions
    
    # Number of top matching chunks to retrieve
    top_k: int = 5
    
    # Whether to use a reranker to improve search results
    # Reranking is slower but more accurate
    use_reranker: bool = False
    
    # Cross-encoder model for reranking (only used if use_reranker=True)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ==================== PYDANTIC CONFIG ====================
    class Config:
        """Pydantic configuration for loading settings."""
        
        # Path to the .env file (relative to where you run the app)
        env_file = ".env"
        
        # Encoding of the .env file
        env_file_encoding = "utf-8"


# Create a single instance of Settings that will be imported everywhere
# This loads all values from .env and environment variables when the app starts
settings = Settings()