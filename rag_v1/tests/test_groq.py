import pytest  # Testing framework
from unittest.mock import Mock, patch, MagicMock  # Tools for creating fake objects
from rag.llm.groq import GroqLLM  # The class we're testing
from rag.models import Chunk, SearchResult, RAGResponse  # Our data models


@pytest.fixture  # This creates reusable sample search results for all tests
def sample_search_results():
    """Create test search results with chunks."""
    return [  # Return list of SearchResult objects
        SearchResult(
            chunk=Chunk(
                id="chunk1",
                document_id="doc1",
                content="Use pip install to add packages to your Python environment",  # Relevant to installation
                metadata={"filename": "python_guide.txt", "type": ".txt"},
                start_char=100,
                end_char=200,
            ),
            score=0.92,  # High relevance score
        ),
        SearchResult(
            chunk=Chunk(
                id="chunk2",
                document_id="doc1",
                content="Package managers like pip and conda help you install libraries from PyPI",
                metadata={"filename": "python_guide.txt", "type": ".txt"},
                start_char=500,
                end_char=580,
            ),
            score=0.78,  # Medium relevance score
        ),
    ]


class TestGroqLLM:  # Group all GroqLLM tests in this class
    """Tests for the GroqLLM class."""

    @patch('rag.llm.groq.settings')  # Mock the settings object
    @patch('rag.llm.groq.Groq')  # Mock the Groq client
    def test_init_creates_client(self, mock_groq, mock_settings):
        """Test that __init__ creates a Groq client with correct API key."""
        mock_settings.groq_api_key = "test-api-key-12345"  # Mock API key
        mock_settings.groq_model = "mixtral-8x7b-32768"  # Mock model name
        
        llm = GroqLLM()  # Create GroqLLM instance
        
        # Verify Groq client was created with the API key from settings
        mock_groq.assert_called_once_with(api_key="test-api-key-12345")
        assert llm.model == "mixtral-8x7b-32768"  # Verify model name is stored

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_with_default_prompt(self, mock_groq, mock_settings, sample_search_results):
        """Test that generate creates correct API call with default system prompt."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        # Create mock client instance
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]  # Create a mock choice
        mock_response.choices[0].message.content = "You can install packages using pip install."  # Mock answer
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        
        query = "How do I install Python packages?"
        result = llm.generate(query, sample_search_results)
        
        # Verify chat.completions.create was called
        mock_client_instance.chat.completions.create.assert_called_once()
        
        # Get the arguments passed to create
        call_args = mock_client_instance.chat.completions.create.call_args
        
        # Verify model is correct
        assert call_args.kwargs['model'] == "test-model"
        
        # Verify temperature and max_tokens
        assert call_args.kwargs['temperature'] == 0.1  # Should be low for focused responses
        assert call_args.kwargs['max_tokens'] == 1024
        
        # Verify messages structure
        messages = call_args.kwargs['messages']
        assert len(messages) == 2  # Should have system + user message
        assert messages[0]['role'] == 'system'  # First message is system
        assert messages[1]['role'] == 'user'  # Second message is user
        
        # Verify system prompt contains the default rules
        assert "helpful assistant" in messages[0]['content']
        assert "Only use information from the provided context" in messages[0]['content']
        
        # Verify user message contains context and query
        user_content = messages[1]['content']
        assert "Use pip install to add packages" in user_content  # Should include chunk content
        assert "python_guide.txt" in user_content  # Should include filename
        assert query in user_content  # Should include the query
        
        # Verify the result is a RAGResponse with correct data
        assert isinstance(result, RAGResponse)
        assert result.answer == "You can install packages using pip install."
        assert result.query == query
        assert result.sources == sample_search_results

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_with_custom_prompt(self, mock_groq, mock_settings, sample_search_results):
        """Test that generate uses custom system prompt when provided."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        
        custom_prompt = "You are a Python expert. Answer briefly and technically."
        result = llm.generate(
            query="How do I install packages?",
            context=sample_search_results,
            system_prompt=custom_prompt  # Provide custom prompt
        )
        
        # Get the call arguments
        call_args = mock_client_instance.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        
        # Verify the custom system prompt was used instead of default
        assert messages[0]['content'] == custom_prompt
        assert "helpful assistant" not in messages[0]['content']  # Should NOT have default prompt

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_formats_context_correctly(self, mock_groq, mock_settings, sample_search_results):
        """Test that generate correctly formats context with source attribution."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        llm.generate("test query", sample_search_results)
        
        # Get the user message
        call_args = mock_client_instance.chat.completions.create.call_args
        user_message = call_args.kwargs['messages'][1]['content']
        
        # Verify context is formatted with source headers
        assert "[Source: python_guide.txt]" in user_message  # Should have source attribution
        assert "Use pip install to add packages" in user_message  # First chunk content
        assert "Package managers like pip and conda" in user_message  # Second chunk content
        
        # Verify chunks are separated (check for double newline between sources)
        assert user_message.count("[Source:") == 2  # Should have 2 source headers (one per chunk)

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_with_empty_context(self, mock_groq, mock_settings):
        """Test that generate handles empty context list."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I don't have enough information"
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        result = llm.generate("What is Python?", [])  # Empty context
        
        # Should still make API call but with empty context string
        call_args = mock_client_instance.chat.completions.create.call_args
        user_message = call_args.kwargs['messages'][1]['content']
        
        # Context section should be empty but structure should remain
        assert "Context:" in user_message
        assert "Question: What is Python?" in user_message

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_handles_missing_filename(self, mock_groq, mock_settings):
        """Test that generate handles chunks without filename in metadata."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        # Create search result with no filename in metadata
        search_result = SearchResult(
            chunk=Chunk(
                id="chunk1",
                document_id="doc1",
                content="Test content",
                metadata={},  # No filename!
                start_char=0,
                end_char=12,
            ),
            score=0.9,
        )
        
        llm = GroqLLM()
        llm.generate("test query", [search_result])
        
        # Get the user message
        call_args = mock_client_instance.chat.completions.create.call_args
        user_message = call_args.kwargs['messages'][1]['content']
        
        # Should use 'Unknown' as default filename
        assert "[Source: Unknown]" in user_message

    @patch('rag.llm.groq.settings')  # Mock settings
    @patch('rag.llm.groq.Groq')  # Mock Groq
    def test_generate_returns_correct_response_structure(self, mock_groq, mock_settings, sample_search_results):
        """Test that generate returns a properly structured RAGResponse."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "test-model"
        
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock response
        expected_answer = "Use pip install followed by the package name."
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = expected_answer
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        
        query = "How do I install packages?"
        result = llm.generate(query, sample_search_results)
        
        # Verify the RAGResponse structure
        assert isinstance(result, RAGResponse)  # Should return RAGResponse object
        assert result.answer == expected_answer  # Answer from LLM
        assert result.query == query  # Original query
        assert result.sources == sample_search_results  # Source chunks
        assert len(result.sources) == 2  # Should preserve all sources
