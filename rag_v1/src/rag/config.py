# src/rag/config.py - Configuration management for RAG application
from pydantic_settings import BaseSettings, SettingsConfigDict  # Pydantic v2 settings
from pydantic import Field  # For field validation

class Settings(BaseSettings):  # All configuration settings loaded from .env or environment
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")  # Pydantic v2 config style
    
    groq_api_key: str = ""  # Groq API key - MUST be set in .env file (never commit API keys!)
    groq_model: str = "llama-3.1-8b-instant"  # Which Groq model to use
    
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # HuggingFace model for embeddings
    embedding_dimension: int = 768  # Vector dimensions (bge-small=384, bge-base=768)
    
    qdrant_host: str = "localhost"  # Where Qdrant is running
    qdrant_port: int = 6335  # Qdrant port (using 6335 to avoid conflict with existing instance on 6333)
    collection_name: str = "documents"  # Vector DB collection name
    
    chunk_size: int = 512  # Max characters per chunk
    chunk_overlap: int = 50  # Characters to overlap between chunks
    
    top_k: int = 5  # Number of results to retrieve
    use_reranker: bool = False  # Whether to use reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Reranker model

settings = Settings()  # Global settings instance - import as: from rag.config import settings