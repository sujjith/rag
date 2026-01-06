# src/rag/llm/groq.py
from groq import Groq  # Import the Groq API client to interact with Groq's LLM service
from rag.config import settings  # Import our app settings (API key, model name, etc.)
from rag.models import SearchResult, RAGResponse  # Import our data models for search results and responses

class GroqLLM:
    """Groq API client for LLM generation."""

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)  # Create Groq client with API key from settings
        self.model = settings.groq_model  # Store which model to use (e.g., "mixtral-8x7b-32768")

    def generate(
        self,
        query: str,  # The user's question (e.g., "How do I install Python packages?")
        context: list[SearchResult],  # List of relevant chunks retrieved from vector search + reranking
        system_prompt: str = None  # Optional custom instructions for how the LLM should behave
    ) -> RAGResponse:  # Returns a RAGResponse object containing the answer and sources
        """Generate answer using retrieved context."""

        if system_prompt is None:  # If no custom system prompt was provided
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.  # Default instructions for the LLM

Rules:
- Only use information from the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Cite your sources by mentioning the document name
- Be concise and accurate"""

        # Build context string
        context_str = "\n\n".join([  # Combine all retrieved chunks into one text block, separated by blank lines
            f"[Source: {r.chunk.metadata.get('filename', 'Unknown')}]\n{r.chunk.content}"  # For each chunk: add filename header + content
            for r in context  # Loop through all SearchResult objects
        ])

        user_message = f"""Context:  # Build the message we'll send to the LLM
{context_str}  # Insert all the relevant chunks here

Question: {query}  # Insert the user's question

Answer based on the context above:"""  # Instruct the LLM to answer using only the context

        response = self.client.chat.completions.create(  # Make API call to Groq to generate the answer
            model=self.model,  # Which LLM model to use (from settings)
            messages=[  # Conversation format: system message + user message
                {"role": "system", "content": system_prompt},  # System: sets the LLM's behavior rules
                {"role": "user", "content": user_message}  # User: provides context + question
            ],
            temperature=0.1,  # Low temperature (0.1) makes responses more focused and deterministic (less creative)
            max_tokens=1024  # Maximum length of the generated response (1024 tokens ≈ 750-1000 words)
        )

        return RAGResponse(  # Create and return a RAGResponse object
            answer=response.choices[0].message.content,  # Extract the LLM's answer from the API response
            sources=context,  # Include the source chunks that were used to generate the answer
            query=query  # Include the original question
        )