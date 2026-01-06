# src/rag/cli.py
import click  # CLI framework for creating command-line interfaces with decorators
from rich.console import Console  # For pretty terminal output with colors and formatting
from rich.panel import Panel  # For displaying content in bordered boxes
from rich.table import Table  # For displaying data in formatted tables
from pathlib import Path  # For working with file paths in an object-oriented way

from rag.config import settings  # Import our app settings (API keys, model names, etc.)
from rag.ingestion.loader import DocumentLoader  # Loads documents from files
from rag.ingestion.chunker import SentenceChunker  # Splits documents into smaller chunks
from rag.embeddings.embedder import Embedder  # Converts text to vector embeddings
from rag.vectorstore.qdrant import QdrantStore  # Stores and searches vectors in Qdrant
from rag.retrieval.reranker import Reranker  # Reranks search results for better accuracy
from rag.llm.groq import GroqLLM  # Generates answers using Groq's LLM
from rag.models import SearchResult  # Data model for search results

console = Console()  # Create a Rich console for pretty terminal output

@click.group()  # Creates a command group - allows multiple subcommands (ingest, ask, status)
def main():
    """RAG System CLI"""
    pass  # This is the main entry point, actual commands are defined below

@main.command()  # Register 'ingest' as a subcommand of main (run with: rag ingest <path>)
@click.argument("path", type=click.Path(exists=True))  # Require a path argument that must exist
def ingest(path: str):
    """Ingest documents from a file or directory."""
    path = Path(path)  # Convert string path to Path object for easier manipulation
    loader = DocumentLoader()  # Create document loader instance
    chunker = SentenceChunker()  # Create chunker instance
    embedder = Embedder()  # Create embedder instance
    store = QdrantStore()  # Create Qdrant store instance

    files = [path] if path.is_file() else list(path.glob("**/*"))  # If single file, use it; if directory, find all files recursively
    files = [f for f in files if f.suffix.lower() in [".pdf", ".txt", ".md", ".docx"]]  # Filter to only supported file types

    with console.status("[bold green]Ingesting documents..."):  # Show a spinner with status message while processing
        for file_path in files:  # Process each file one by one
            console.print(f"Processing: {file_path.name}")  # Show which file is being processed

            # Load document
            doc = loader.load(file_path)  # Load the file content into a Document object

            # Chunk document
            chunks = list(chunker.chunk(doc))  # Split the document into smaller chunks
            console.print(f"  Created {len(chunks)} chunks")  # Show how many chunks were created

            # Generate embeddings
            texts = [c.content for c in chunks]  # Extract just the text content from each chunk
            embeddings = embedder.embed(texts)  # Convert all chunk texts to vector embeddings (batch processing)

            for chunk, embedding in zip(chunks, embeddings):  # Pair each chunk with its embedding
                chunk.embedding = embedding  # Add the embedding to the chunk object

            # Store in Qdrant
            store.upsert(chunks)  # Upload chunks with embeddings to Qdrant vector database

    console.print(f"[bold green]✓ Ingested {len(files)} documents")  # Show success message with count

@main.command()  # Register 'ask' as a subcommand (run with: rag ask "your question")
@click.argument("query")  # Require a query string as the first argument
@click.option("--top-k", default=5, help="Number of results")  # Optional flag to specify how many results to return (default 5)
@click.option("--rerank", is_flag=True, help="Use reranker")  # Optional flag to enable reranking (--rerank)
def ask(query: str, top_k: int, rerank: bool):
    """Ask a question about your documents."""
    embedder = Embedder()  # Create embedder to convert query to vector
    store = QdrantStore()  # Create store to search vectors
    llm = GroqLLM()  # Create LLM client to generate answers

    with console.status("[bold blue]Searching..."):  # Show spinner while processing
        # Embed query
        query_vector = embedder.embed_query(query)  # Convert the user's question to a vector embedding

        # Search
        results = store.search(query_vector, top_k=top_k * 2 if rerank else top_k)  # Search Qdrant for similar chunks (get 2x if reranking, since we'll filter down)

        # Rerank if enabled
        if rerank and results:  # If --rerank flag was used and we have results
            reranker = Reranker()  # Create reranker instance
            results = reranker.rerank(query, results, top_k=top_k)  # Rerank results and keep only top K
        else:  # If no reranking requested
            results = results[:top_k]  # Just take the first top_k results

        # Convert to SearchResult
        search_results = [SearchResult(chunk=chunk, score=score) for chunk, score in results]  # Wrap results in SearchResult objects for LLM

        # Generate answer
        response = llm.generate(query, search_results)  # Send query + context to LLM to generate an answer

    # Display answer
    console.print(Panel(response.answer, title="Answer", border_style="green"))  # Show the LLM's answer in a nice green box

    # Display sources
    table = Table(title="Sources")  # Create a table to show source chunks
    table.add_column("File", style="cyan")  # Add column for filename
    table.add_column("Score", style="magenta")  # Add column for relevance score
    table.add_column("Preview", style="dim")  # Add column for content preview

    for result in response.sources:  # Loop through each source chunk used to generate the answer
        table.add_row(
            result.chunk.metadata.get("filename", "Unknown"),  # Get filename from metadata
            f"{result.score:.3f}",  # Format score to 3 decimal places
            result.chunk.content[:100] + "..."  # Show first 100 characters of chunk content
        )

    console.print(table)  # Display the sources table

@main.command()  # Register 'status' as a subcommand (run with: rag status)
def status():
    """Show system status."""
    store = QdrantStore()  # Create store to query Qdrant

    info = store.client.get_collection(settings.collection_name)  # Get collection information from Qdrant

    table = Table(title="RAG System Status")  # Create a status table
    table.add_column("Component", style="cyan")  # Add column for component names
    table.add_column("Status", style="green")  # Add column for status values

    table.add_row("Qdrant", "✓ Connected")  # Show that Qdrant is connected
    table.add_row("Collection", settings.collection_name)  # Show the collection name
    table.add_row("Vectors", str(info.points_count))  # Show how many vectors are stored
    table.add_row("Embedding Model", settings.embedding_model)  # Show which embedding model is configured
    table.add_row("LLM Model", settings.groq_model)  # Show which LLM model is configured
    table.add_row("Reranker", "Enabled" if settings.use_reranker else "Disabled")  # Show if reranker is enabled

    console.print(table)  # Display the status table

if __name__ == "__main__":  # If this file is run directly (not imported)
    main()  # Start the CLI application