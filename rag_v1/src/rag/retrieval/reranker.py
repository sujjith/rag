# src/rag/retrieval/reranker.py
from sentence_transformers import CrossEncoder  # Import CrossEncoder - a model that scores how well a query matches a text
from rag.config import settings  # Import our app settings (reranker model name, etc.)
from rag.models import Chunk  # Import our Chunk data model

class Reranker:
    """Rerank search results using cross-encoder."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model  # Use provided model or default from settings
        self.model = CrossEncoder(self.model_name)  # Load the cross-encoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(
        self,
        query: str,  # The user's question/search query
        chunks: list[tuple[Chunk, float]],  # List of (chunk, initial_score) pairs from vector search
        top_k: int = 5  # How many top results to return after reranking (default 5)
    ) -> list[tuple[Chunk, float]]:  # Returns list of (chunk, reranked_score) pairs sorted by relevance
        """Rerank chunks based on query relevance."""
        if not chunks:  # If we got no chunks to rerank
            return []  # Return empty list

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.content) for chunk, _ in chunks]  # Create (query, text) pairs for each chunk - cross-encoder compares these

        # Get reranking scores
        scores = self.model.predict(pairs)  # Cross-encoder scores each pair - higher score means more relevant to the query

        # Combine with chunks and sort
        reranked = [  # Build new list with chunks and their NEW reranking scores
            (chunk, float(score))  # Convert score to float and pair with chunk
            for (chunk, _), score in zip(chunks, scores)  # Zip chunks with new scores (ignore old scores with _)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)  # Sort by score (x[1]) in descending order (highest scores first)

        return reranked[:top_k]  # Return only the top K most relevant chunks