import pytest  # Testing framework
from unittest.mock import Mock, patch, MagicMock  # Tools for creating fake objects
import numpy as np  # For creating fake score arrays
from rag.retrieval.reranker import Reranker  # The class we're testing
from rag.models import Chunk  # Our Chunk data model


@pytest.fixture  # This creates reusable sample chunks for all tests
def sample_chunks():
    """Create test chunks with initial vector search scores."""
    return [  # Return list of (chunk, initial_score) tuples
        (
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="Python is a programming language created in 1991",  # Less relevant to installation
                metadata={"type": ".txt"},
                start_char=0,
                end_char=50,
            ),
            0.72,  # Initial vector search score
        ),
        (
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="Use pip install to add packages to your Python environment",  # Most relevant!
                metadata={"type": ".txt"},
                start_char=51,
                end_char=110,
            ),
            0.68,  # Initial vector search score (lower than chunk1 but actually more relevant)
        ),
        (
            Chunk(
                id="chunk3",
                document_id="doc1",
                content="Package managers help organize Python dependencies",  # Moderately relevant
                metadata={"type": ".txt"},
                start_char=111,
                end_char=162,
            ),
            0.65,  # Initial vector search score
        ),
    ]


class TestReranker:  # Group all Reranker tests in this class
    """Tests for the Reranker class."""

    @patch('rag.retrieval.reranker.settings')  # Mock the settings object
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock the CrossEncoder model
    def test_init_creates_model(self, mock_cross_encoder, mock_settings):
        """Test that __init__ creates a CrossEncoder model with correct name."""
        mock_settings.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Mock model name
        
        reranker = Reranker()  # Create reranker (should use settings)
        
        # Verify CrossEncoder was created with the model name from settings
        mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_init_with_custom_model(self, mock_cross_encoder, mock_settings):
        """Test that __init__ can use a custom model name instead of settings."""
        mock_settings.reranker_model = "default-model"  # This should be ignored
        
        custom_model = "my-custom-reranker"
        reranker = Reranker(model_name=custom_model)  # Provide custom model name
        
        # Verify it used our custom model, not the settings model
        mock_cross_encoder.assert_called_once_with("my-custom-reranker")
        assert reranker.model_name == "my-custom-reranker"

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_improves_order(self, mock_cross_encoder, mock_settings, sample_chunks):
        """Test that rerank correctly reorders chunks based on relevance."""
        mock_settings.reranker_model = "test-model"
        
        # Create mock model instance
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        # Mock the predict method to return reranking scores
        # These scores should reorder the chunks: chunk2 (0.92) > chunk3 (0.58) > chunk1 (0.15)
        mock_model_instance.predict.return_value = np.array([0.15, 0.92, 0.58])
        
        reranker = Reranker()
        
        query = "How do I install Python packages?"
        reranked = reranker.rerank(query, sample_chunks, top_k=3)
        
        # Verify predict was called with the correct (query, content) pairs
        expected_pairs = [
            (query, "Python is a programming language created in 1991"),
            (query, "Use pip install to add packages to your Python environment"),
            (query, "Package managers help organize Python dependencies"),
        ]
        mock_model_instance.predict.assert_called_once_with(expected_pairs)
        
        # Verify the chunks are now in the correct order (sorted by new scores)
        assert len(reranked) == 3
        assert reranked[0][0].id == "chunk2"  # Most relevant (score 0.92)
        assert reranked[0][1] == 0.92
        assert reranked[1][0].id == "chunk3"  # Second most relevant (score 0.58)
        assert reranked[1][1] == 0.58
        assert reranked[2][0].id == "chunk1"  # Least relevant (score 0.15)
        assert reranked[2][1] == 0.15

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_respects_top_k(self, mock_cross_encoder, mock_settings, sample_chunks):
        """Test that rerank returns only top_k results."""
        mock_settings.reranker_model = "test-model"
        
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        # Return scores that will sort chunks in order: chunk2, chunk3, chunk1
        mock_model_instance.predict.return_value = np.array([0.15, 0.92, 0.58])
        
        reranker = Reranker()
        
        # Request only top 2 results
        reranked = reranker.rerank("test query", sample_chunks, top_k=2)
        
        # Should only get 2 results back (the top 2 by score)
        assert len(reranked) == 2
        assert reranked[0][0].id == "chunk2"  # Highest score
        assert reranked[1][0].id == "chunk3"  # Second highest
        # chunk1 should NOT be included (it has the lowest score)

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_with_empty_chunks(self, mock_cross_encoder, mock_settings):
        """Test that rerank handles empty chunk list gracefully."""
        mock_settings.reranker_model = "test-model"
        
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        reranker = Reranker()
        
        # Call rerank with empty list
        reranked = reranker.rerank("test query", [], top_k=5)
        
        # Should return empty list without calling predict
        assert reranked == []
        mock_model_instance.predict.assert_not_called()  # Should not waste time calling the model

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_single_chunk(self, mock_cross_encoder, mock_settings):
        """Test that rerank works with a single chunk."""
        mock_settings.reranker_model = "test-model"
        
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        # Return a single score
        mock_model_instance.predict.return_value = np.array([0.85])
        
        reranker = Reranker()
        
        single_chunk = [
            (
                Chunk(
                    id="chunk1",
                    document_id="doc1",
                    content="Test content",
                    metadata={},
                    start_char=0,
                    end_char=12,
                ),
                0.70,  # Initial score
            )
        ]
        
        reranked = reranker.rerank("query", single_chunk, top_k=5)
        
        # Should return the single chunk with new score
        assert len(reranked) == 1
        assert reranked[0][0].id == "chunk1"
        assert reranked[0][1] == 0.85  # New score from reranking

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_preserves_chunk_data(self, mock_cross_encoder, mock_settings, sample_chunks):
        """Test that rerank doesn't modify the chunks themselves, only reorders them."""
        mock_settings.reranker_model = "test-model"
        
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        mock_model_instance.predict.return_value = np.array([0.9, 0.8, 0.7])
        
        reranker = Reranker()
        
        # Get original chunk content before reranking
        original_content = sample_chunks[1][0].content
        
        reranked = reranker.rerank("query", sample_chunks, top_k=3)
        
        # Find the same chunk in reranked results
        reranked_chunk2 = [c for c, s in reranked if c.id == "chunk2"][0]
        
        # Verify the chunk data is unchanged (same content, metadata, etc.)
        assert reranked_chunk2.content == original_content
        assert reranked_chunk2.id == "chunk2"
        assert reranked_chunk2.document_id == "doc1"

    @patch('rag.retrieval.reranker.settings')  # Mock settings
    @patch('rag.retrieval.reranker.CrossEncoder')  # Mock CrossEncoder
    def test_rerank_uses_float_scores(self, mock_cross_encoder, mock_settings, sample_chunks):
        """Test that rerank converts scores to float type."""
        mock_settings.reranker_model = "test-model"
        
        mock_model_instance = Mock()
        mock_cross_encoder.return_value = mock_model_instance
        
        # Return numpy array (which CrossEncoder actually returns)
        mock_model_instance.predict.return_value = np.array([0.1, 0.2, 0.3])
        
        reranker = Reranker()
        reranked = reranker.rerank("query", sample_chunks, top_k=3)
        
        # All scores should be Python floats, not numpy types
        for chunk, score in reranked:
            assert isinstance(score, float)  # Should be Python float
            assert not isinstance(score, np.floating)  # Should NOT be numpy float
