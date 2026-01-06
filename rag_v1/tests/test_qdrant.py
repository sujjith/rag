import pytest  # Testing framework
from unittest.mock import Mock, patch, MagicMock  # Tools for creating fake objects and replacing real ones
from qdrant_client.models import (  # Import Qdrant models we'll use
    VectorParams, Distance, PointStruct, ScoredPoint,  # Models for vectors, points, and search results
    Filter, FieldCondition, MatchValue  # Models for filtering
)
from rag.vectorstore.qdrant import QdrantStore  # The class we're testing
from rag.models import Chunk  # Our Chunk data model


@pytest.fixture  # This creates a reusable mock QdrantClient for all tests
def mock_qdrant_client():
    """Create a mock Qdrant client that doesn't connect to a real database."""
    with patch('rag.vectorstore.qdrant.QdrantClient') as mock_client:  # Replace the real QdrantClient with a fake one
        yield mock_client  # Provide the mock to tests that use this fixture


@pytest.fixture  # This creates sample chunks we can use in multiple tests
def sample_chunks():
    """Create test chunks with embeddings."""
    return [  # Return a list of 2 test chunks
        Chunk(
            id="chunk1",  # First chunk ID
            document_id="doc1",  # Document it belongs to
            content="This is the first chunk",  # Text content
            embedding=[0.1, 0.2, 0.3, 0.4],  # Mock embedding vector (4 dimensions for testing)
            metadata={"type": ".txt"},  # Some metadata
            start_char=0,  # Starts at character 0
            end_char=23,  # Ends at character 23
        ),
        Chunk(
            id="chunk2",  # Second chunk ID
            document_id="doc1",  # Same document
            content="This is the second chunk",  # Different text
            embedding=[0.5, 0.6, 0.7, 0.8],  # Different embedding
            metadata={"type": ".txt"},  # Same metadata
            start_char=24,  # Starts where first chunk ended
            end_char=48,  # Ends at character 48
        ),
    ]


class TestQdrantStore:  # Group all QdrantStore tests in this class
    """Tests for the QdrantStore class."""

    @patch('rag.vectorstore.qdrant.settings')  # Mock the settings object
    def test_init_creates_client_and_collection(self, mock_settings, mock_qdrant_client):
        """Test that __init__ creates a client and ensures collection exists."""
        # Configure our mock settings
        mock_settings.qdrant_host = "localhost"  # Fake host
        mock_settings.qdrant_port = 6333  # Fake port
        mock_settings.collection_name = "test_collection"  # Fake collection name
        mock_settings.embedding_dimension = 384  # Fake embedding size
        
        # Create a mock client instance that will be returned when QdrantClient() is called
        mock_client_instance = Mock()  # This represents the actual client object
        mock_qdrant_client.return_value = mock_client_instance  # When QdrantClient() is called, return our mock
        
        # Mock the get_collections method to return an empty list (no collections exist yet)
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        # Create the QdrantStore (this is what we're testing)
        store = QdrantStore()
        
        # Verify that QdrantClient was created with correct host and port
        mock_qdrant_client.assert_called_once_with(
            host="localhost",  # Should use our mock settings
            port=6333
        )
        
        # Verify that get_collections was called to check if collection exists
        mock_client_instance.get_collections.assert_called_once()
        
        # Verify that create_collection was called since the collection didn't exist
        mock_client_instance.create_collection.assert_called_once()

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_ensure_collection_creates_if_not_exists(self, mock_settings, mock_qdrant_client):
        """Test that _ensure_collection creates a collection when it doesn't exist."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        # Mock get_collections to return an empty list (collection doesn't exist)
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        store = QdrantStore()  # This will call _ensure_collection
        
        # Verify create_collection was called with correct parameters
        mock_client_instance.create_collection.assert_called_once_with(
            collection_name="test_collection",  # Correct collection name
            vectors_config=VectorParams(  # Correct vector configuration
                size=384,  # From our mock settings
                distance=Distance.COSINE  # Using cosine similarity
            )
        )

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_ensure_collection_skips_if_exists(self, mock_settings, mock_qdrant_client):
        """Test that _ensure_collection doesn't create a collection if it already exists."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        # Mock get_collections to return a collection with our name (it already exists)
        existing_collection = Mock()  # Create a fake collection object
        existing_collection.name = "test_collection"  # Give it the same name
        mock_client_instance.get_collections.return_value = Mock(collections=[existing_collection])
        
        store = QdrantStore()  # This will call _ensure_collection
        
        # Verify that create_collection was NOT called (since collection exists)
        mock_client_instance.create_collection.assert_not_called()

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_upsert_converts_chunks_to_points(self, mock_settings, mock_qdrant_client, sample_chunks):
        """Test that upsert correctly converts chunks to Qdrant points."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        store = QdrantStore()
        
        # Call upsert with our sample chunks
        store.upsert(sample_chunks)
        
        # Verify that upsert was called on the client
        mock_client_instance.upsert.assert_called_once()
        
        # Get the arguments that were passed to upsert
        call_args = mock_client_instance.upsert.call_args
        
        # Check that collection_name is correct
        assert call_args.kwargs['collection_name'] == "test_collection"
        
        # Check that we have 2 points (one for each chunk)
        points = call_args.kwargs['points']
        assert len(points) == 2
        
        # Verify the first point has correct data
        assert points[0].id == "chunk1"  # Correct ID
        assert points[0].vector == [0.1, 0.2, 0.3, 0.4]  # Correct embedding
        assert points[0].payload['content'] == "This is the first chunk"  # Correct content
        assert points[0].payload['document_id'] == "doc1"  # Correct document ID

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_upsert_skips_chunks_without_embeddings(self, mock_settings, mock_qdrant_client):
        """Test that upsert skips chunks that don't have embeddings."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        store = QdrantStore()
        
        # Create chunks where one has no embedding (embedding=None)
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="Has embedding",
                embedding=[0.1, 0.2, 0.3],  # This one has an embedding
                metadata={},
                start_char=0,
                end_char=10,
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="No embedding",
                embedding=None,  # This one doesn't have an embedding
                metadata={},
                start_char=11,
                end_char=20,
            ),
        ]
        
        store.upsert(chunks)
        
        # Get the points that were passed to upsert
        call_args = mock_client_instance.upsert.call_args
        points = call_args.kwargs['points']
        
        # Only 1 point should be created (the one with an embedding)
        assert len(points) == 1
        assert points[0].id == "chunk1"  # Should be the first chunk

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_search_returns_chunks_with_scores(self, mock_settings, mock_qdrant_client):
        """Test that search returns chunks with similarity scores."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        # Mock search results from Qdrant
        mock_result1 = ScoredPoint(  # Create a fake search result
            id="chunk1",  # Chunk ID
            version=1,  # Version (required by ScoredPoint)
            score=0.95,  # Similarity score (0.95 means very similar)
            payload={  # The stored data
                "document_id": "doc1",
                "content": "This is the first chunk",
                "metadata": {"type": ".txt"},
                "start_char": 0,
                "end_char": 23,
            },
            vector=None,  # We don't need the vector in the result
        )
        
        mock_result2 = ScoredPoint(  # Second search result
            id="chunk2",
            version=1,
            score=0.85,  # Lower score (less similar)
            payload={
                "document_id": "doc1",
                "content": "This is the second chunk",
                "metadata": {"type": ".txt"},
                "start_char": 24,
                "end_char": 48,
            },
            vector=None,
        )
        
        # Configure the mock client to return these results when search is called
        mock_client_instance.search.return_value = [mock_result1, mock_result2]
        
        store = QdrantStore()
        
        # Perform a search with a query vector
        query_vector = [0.1, 0.2, 0.3, 0.4]  # Fake query embedding
        results = store.search(query_vector, top_k=2)  # Get top 2 results
        
        # Verify search was called with correct parameters
        mock_client_instance.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=2,  # We asked for 2 results
            query_filter=None  # No filter specified
        )
        
        # Verify we got 2 results back
        assert len(results) == 2
        
        # Check the first result (should be a tuple of (Chunk, score))
        chunk1, score1 = results[0]
        assert chunk1.id == "chunk1"  # Correct chunk ID
        assert chunk1.content == "This is the first chunk"  # Correct content
        assert score1 == 0.95  # Correct similarity score
        
        # Check the second result
        chunk2, score2 = results[1]
        assert chunk2.id == "chunk2"
        assert score2 == 0.85

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_search_with_document_filter(self, mock_settings, mock_qdrant_client):
        """Test that search correctly applies document ID filter."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        # Mock empty search results (we only care about the filter being applied)
        mock_client_instance.search.return_value = []
        
        store = QdrantStore()
        
        # Search with a filter for a specific document
        query_vector = [0.1, 0.2, 0.3, 0.4]
        store.search(query_vector, top_k=5, filter_doc_id="doc1")  # Filter for doc1 only
        
        # Get the arguments passed to search
        call_args = mock_client_instance.search.call_args
        search_filter = call_args.kwargs['query_filter']
        
        # Verify that a filter was created
        assert search_filter is not None
        
        # Verify the filter has the correct structure
        assert isinstance(search_filter, Filter)  # Should be a Filter object
        assert len(search_filter.must) == 1  # Should have 1 condition
        
        # Verify the filter condition is for document_id matching "doc1"
        condition = search_filter.must[0]
        assert isinstance(condition, FieldCondition)
        assert condition.key == "document_id"  # Filtering on document_id field
        assert condition.match.value == "doc1"  # Must match "doc1"

    @patch('rag.vectorstore.qdrant.settings')  # Mock settings
    def test_search_without_filter(self, mock_settings, mock_qdrant_client):
        """Test that search works without a document filter."""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_dimension = 384
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        
        mock_client_instance.search.return_value = []
        
        store = QdrantStore()
        
        # Search without a filter
        query_vector = [0.1, 0.2, 0.3, 0.4]
        store.search(query_vector, top_k=5)  # No filter_doc_id specified
        
        # Get the arguments passed to search
        call_args = mock_client_instance.search.call_args
        search_filter = call_args.kwargs['query_filter']
        
        # Verify that no filter was applied (should be None)
        assert search_filter is None
