import pytest  # Testing framework
from unittest.mock import Mock, patch, MagicMock  # Tools for creating fake objects
from click.testing import CliRunner  # Click's built-in test runner for CLI commands
from pathlib import Path  # For working with file paths
import tempfile  # For creating temporary test files

from rag.cli import main, ingest, ask, status  # Import the CLI commands we're testing
from rag.models import Chunk, SearchResult, RAGResponse, Document  # Our data models


@pytest.fixture  # Create a CliRunner instance for all tests
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()  # This allows us to invoke CLI commands in tests


@pytest.fixture  # Create a temporary directory with test files
def temp_docs_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:  # Auto-cleanup after test
        tmpdir = Path(tmpdir)  # Convert to Path object
        
        # Create test files
        (tmpdir / "test1.txt").write_text("This is test document 1")  # Create a .txt file
        (tmpdir / "test2.md").write_text("# Test Document 2\n\nSome content")  # Create a .md file
        
        yield tmpdir  # Provide the directory to the test


class TestIngestCommand:  # Group all ingest command tests
    """Tests for the ingest CLI command."""
    
    @patch('rag.cli.QdrantStore')  # Mock QdrantStore so we don't need real Qdrant
    @patch('rag.cli.Embedder')  # Mock Embedder
    @patch('rag.cli.SentenceChunker')  # Mock SentenceChunker
    @patch('rag.cli.DocumentLoader')  # Mock DocumentLoader
    def test_ingest_single_file(
        self, 
        mock_loader_class, 
        mock_chunker_class, 
        mock_embedder_class,
        mock_store_class,
        runner,
        temp_docs_dir
    ):
        """Test ingesting a single file."""
        # Set up mocks
        mock_loader = Mock()  # Create mock loader instance
        mock_loader_class.return_value = mock_loader  # Return our mock when DocumentLoader() is called
        
        mock_doc = Document(  # Create a fake document
            id="doc1",
            content="Test content",
            source=str(temp_docs_dir / "test1.txt"),
            metadata={"filename": "test1.txt", "type": ".txt"}
        )
        mock_loader.load.return_value = mock_doc  # Mock the load method
        
        mock_chunker = Mock()  # Create mock chunker instance
        mock_chunker_class.return_value = mock_chunker
        
        mock_chunk = Chunk(  # Create a fake chunk
            id="chunk1",
            document_id="doc1",
            content="Test content",
            metadata={"type": ".txt"},
            start_char=0,
            end_char=12
        )
        mock_chunker.chunk.return_value = [mock_chunk]  # Return one chunk
        
        mock_embedder = Mock()  # Create mock embedder instance
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]  # Return fake embedding
        
        mock_store = Mock()  # Create mock store instance
        mock_store_class.return_value = mock_store
        
        # Run the command
        result = runner.invoke(ingest, [str(temp_docs_dir / "test1.txt")])  # Invoke 'ingest' with file path
        
        # Verify command succeeded
        assert result.exit_code == 0  # Should exit successfully (0 = success)
        
        # Verify the workflow was called correctly
        mock_loader.load.assert_called_once()  # Should load the file once
        mock_chunker.chunk.assert_called_once_with(mock_doc)  # Should chunk the document
        mock_embedder.embed.assert_called_once()  # Should generate embeddings
        mock_store.upsert.assert_called_once()  # Should store in Qdrant
        
        # Check output message
        assert "Ingested 1 documents" in result.output  # Should report 1 document ingested

    @patch('rag.cli.QdrantStore')
    @patch('rag.cli.Embedder')
    @patch('rag.cli.SentenceChunker')
    @patch('rag.cli.DocumentLoader')
    def test_ingest_directory(
        self,
        mock_loader_class,
        mock_chunker_class,
        mock_embedder_class,
        mock_store_class,
        runner,
        temp_docs_dir
    ):
        """Test ingesting a directory with multiple files."""
        # Set up mocks
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = Document(  # Return same document for both files
            id="doc1",
            content="Test",
            source="test",
            metadata={}
        )
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = [Mock(content="chunk", embedding=None)]  # Return a chunk
        
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.embed.return_value = [[0.1, 0.2]]  # Fake embedding
        
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        # Run command with directory
        result = runner.invoke(ingest, [str(temp_docs_dir)])
        
        # Should process 2 files (test1.txt and test2.md)
        assert result.exit_code == 0
        assert mock_loader.load.call_count == 2  # Should load 2 files
        assert "Ingested 2 documents" in result.output  # Should report 2 documents


class TestAskCommand:  # Group all ask command tests
    """Tests for the ask CLI command."""
    
    @patch('rag.cli.GroqLLM')  # Mock GroqLLM
    @patch('rag.cli.QdrantStore')  # Mock QdrantStore
    @patch('rag.cli.Embedder')  # Mock Embedder
    def test_ask_without_rerank(
        self,
        mock_embedder_class,
        mock_store_class,
        mock_llm_class,
        runner
    ):
        """Test asking a question without reranking."""
        # Set up mocks
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]  # Fake query vector
        
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        # Create fake search results
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Python packages can be installed with pip",
            metadata={"filename": "guide.txt"},
            start_char=0,
            end_char=42
        )
        mock_store.search.return_value = [(mock_chunk, 0.95)]  # Return one result
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock LLM response
        mock_response = RAGResponse(
            answer="Use pip install to add packages.",
            sources=[SearchResult(chunk=mock_chunk, score=0.95)],
            query="How to install packages?"
        )
        mock_llm.generate.return_value = mock_response
        
        # Run the command
        result = runner.invoke(ask, ["How to install packages?"])  # No --rerank flag
        
        # Verify
        assert result.exit_code == 0
        mock_embedder.embed_query.assert_called_once_with("How to install packages?")  # Should embed query
        mock_store.search.assert_called_once()  # Should search Qdrant
        
        # Should NOT use reranker since --rerank was not specified
        call_args = mock_store.search.call_args
        assert call_args.kwargs['top_k'] == 5  # Default top_k, not doubled
        
        mock_llm.generate.assert_called_once()  # Should generate answer
        assert "Use pip install" in result.output  # Should display answer

    @patch('rag.cli.Reranker')  # Mock Reranker
    @patch('rag.cli.GroqLLM')
    @patch('rag.cli.QdrantStore')
    @patch('rag.cli.Embedder')
    def test_ask_with_rerank(
        self,
        mock_embedder_class,
        mock_store_class,
        mock_llm_class,
        mock_reranker_class,
        runner
    ):
        """Test asking a question with reranking enabled."""
        # Set up mocks
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Test content",
            metadata={"filename": "test.txt"},
            start_char=0,
            end_char=12
        )
        mock_store.search.return_value = [(mock_chunk, 0.8)] * 6  # Return 6 results
        
        mock_reranker = Mock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.rerank.return_value = [(mock_chunk, 0.95)] * 3  # Return top 3 after reranking
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.generate.return_value = RAGResponse(
            answer="Test answer",
            sources=[SearchResult(chunk=mock_chunk, score=0.95)],
            query="test"
        )
        
        # Run with --rerank flag
        result = runner.invoke(ask, ["test query", "--rerank", "--top-k", "3"])
        
        # Verify
        assert result.exit_code == 0
        
        # Should search for 2x top_k when reranking (3 * 2 = 6)
        call_args = mock_store.search.call_args
        assert call_args.kwargs['top_k'] == 6
        
        # Should call reranker
        mock_reranker.rerank.assert_called_once()
        rerank_args = mock_reranker.rerank.call_args
        assert rerank_args.kwargs['top_k'] == 3  # Should filter down to top 3


class TestStatusCommand:  # Group all status command tests
    """Tests for the status CLI command."""
    
    @patch('rag.cli.settings')  # Mock settings
    @patch('rag.cli.QdrantStore')  # Mock QdrantStore
    def test_status_displays_info(self, mock_store_class, mock_settings, runner):
        """Test that status command displays system information."""
        # Set up mocks
        mock_settings.collection_name = "test_collection"
        mock_settings.embedding_model = "test-embedder"
        mock_settings.groq_model = "test-llm"
        mock_settings.use_reranker = True
        
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        # Mock collection info
        mock_info = Mock()
        mock_info.points_count = 42  # 42 vectors stored
        mock_store.client.get_collection.return_value = mock_info
        
        # Run command
        result = runner.invoke(status)
        
        # Verify
        assert result.exit_code == 0
        
        # Check that status table contains expected information
        assert "test_collection" in result.output  # Collection name
        assert "42" in result.output  # Vector count
        assert "test-embedder" in result.output  # Embedding model
        assert "test-llm" in result.output  # LLM model
        assert "Enabled" in result.output  # Reranker status
        
        # Verify get_collection was called
        mock_store.client.get_collection.assert_called_once_with("test_collection")
