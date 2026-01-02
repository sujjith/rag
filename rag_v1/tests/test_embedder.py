# tests/test_embedder.py - Unit tests for Embedder
import pytest
from rag.embeddings.embedder import Embedder

# Note: These tests will download the embedding model on first run (~400MB)
# Use a smaller model for faster tests
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")  # Smaller model for testing

class TestEmbedder:

    def test_embed_returns_list(self):  # Basic test - does it return a list?
        texts = ["Hello world"]
        
        embeddings = embedder.embed(texts)
        
        assert isinstance(embeddings, list)  # Should return a list
        assert len(embeddings) == 1  # One embedding for one text

    def test_embed_returns_correct_dimensions(self):  # Check embedding size
        texts = ["Test sentence"]
        
        embeddings = embedder.embed(texts)
        
        assert len(embeddings[0]) == 384  # bge-small has 384 dimensions (bge-base has 768)

    def test_embed_multiple_texts(self):  # Batch embedding
        texts = ["First text", "Second text", "Third text"]
        
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 3  # One embedding per text
        assert all(len(e) == 384 for e in embeddings)  # All same dimension

    def test_embed_query_returns_list(self):  # Single query embedding
        query = "What is machine learning?"
        
        embedding = embedder.embed_query(query)
        
        assert isinstance(embedding, list)  # Should be a list of floats
        assert len(embedding) == 384  # Correct dimensions

    def test_embeddings_are_normalized(self):  # Check normalization (length ~= 1)
        import math
        texts = ["Test normalization"]
        
        embeddings = embedder.embed(texts)
        
        length = math.sqrt(sum(x**2 for x in embeddings[0]))  # Calculate vector length
        assert 0.99 < length < 1.01  # Should be approximately 1 (normalized)

    def test_similar_texts_have_similar_embeddings(self):  # Semantic similarity test
        texts = [
            "Machine learning is a type of artificial intelligence.",
            "ML is a subset of AI.",  # Similar meaning
            "I like to eat pizza."    # Different meaning
        ]
        
        embeddings = embedder.embed(texts)
        
        def cosine_sim(a, b):  # Calculate cosine similarity
            dot = sum(x*y for x, y in zip(a, b))
            return dot  # Already normalized, so dot product = cosine similarity
        
        sim_related = cosine_sim(embeddings[0], embeddings[1])  # ML texts
        sim_unrelated = cosine_sim(embeddings[0], embeddings[2])  # ML vs pizza
        
        print(f"\nSimilarity (ML texts): {sim_related:.3f}")
        print(f"Similarity (ML vs pizza): {sim_unrelated:.3f}")
        
        assert sim_related > sim_unrelated  # Related texts should be more similar

    def test_empty_text(self):  # Should handle empty text
        texts = [""]
        
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_embed_query_adds_instruction_for_bge(self):  # BGE instruction prefix
        query = "test query"
        
        embedding = embedder.embed_query(query)  # Should add instruction internally
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_different_texts_different_embeddings(self):  # Embeddings should differ
        texts = ["Hello", "Goodbye"]
        
        embeddings = embedder.embed(texts)
        
        assert embeddings[0] != embeddings[1]  # Different texts = different embeddings

    def test_same_text_same_embedding(self):  # Deterministic embeddings
        texts = ["Consistent text"]
        
        emb1 = embedder.embed(texts)
        emb2 = embedder.embed(texts)
        
        assert emb1[0][:10] == emb2[0][:10]  # Same text should give same embedding
