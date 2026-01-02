# src/rag/embeddings/embedder.py
from sentence_transformers import SentenceTransformer  # Library for generating text embeddings
from rag.config import settings  # Our app settings (embedding_model name, etc.)

class Embedder:  # Converts text into numerical vectors (embeddings) for similarity search

    def __init__(self, model_name: str = None):  # Initialize with optional model name
        self.model_name = model_name or settings.embedding_model  # Use provided name or default from config
        self.model = SentenceTransformer(self.model_name)  # Load the embedding model (downloads on first use)

    def embed(self, texts: list[str]) -> list[list[float]]:  # Embed multiple texts at once (batch processing)
        embeddings = self.model.encode(  # Convert texts to vectors
            texts,
            normalize_embeddings=True,  # Normalize for cosine similarity (vectors have length 1)
            show_progress_bar=True  # Show progress for large batches
        )
        return embeddings.tolist()  # Convert numpy array to Python list

    def embed_query(self, query: str) -> list[float]:  # Embed a single search query
        if "bge" in self.model_name.lower():  # BGE models need special instruction prefix for queries
            query = f"Represent this sentence for searching relevant passages: {query}"  # This improves search quality

        embedding = self.model.encode(query, normalize_embeddings=True)  # Generate single embedding
        return embedding.tolist()  # Convert numpy array to Python list
