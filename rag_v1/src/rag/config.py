from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = "llama-3.1-8b-instant"

    # Embedding
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()