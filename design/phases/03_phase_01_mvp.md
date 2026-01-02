# Phase 1: MVP - Basic RAG System

> **Goal**: Build a working CLI that can ingest documents and answer questions using RAG.

---

## Overview

This phase focuses on building the core RAG pipeline with minimal dependencies. By the end, you'll have a functional system that can:

1. Ingest documents (PDF, TXT, MD)
2. Chunk and embed text
3. Store vectors in Qdrant
4. Answer questions with source citations

---

## Technology Stack (Phase 1)

| Component | Choice | Why |
|-----------|--------|-----|
| **Embedding Model** | `BAAI/bge-base-en-v1.5` | Best balance of speed/quality, 768 dimensions |
| **Vector Database** | Qdrant | Easy setup, excellent Python SDK |
| **LLM** | Groq API | Fast, free tier, easy to start |
| **Document Processing** | PyMuPDF, python-docx | Lightweight, no Java dependency |
| **Chunking** | Custom (sentence-aware) | Learn the fundamentals |
| **Reranker** (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Improves retrieval accuracy |

### Alternative Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `BAAI/bge-base-en-v1.5` | 768 | Fast | High | **Recommended for MVP** |
| `BAAI/bge-small-en-v1.5` | 384 | Faster | Good | Low memory environments |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Fast | High | Longer context (8192 tokens) |
| `intfloat/e5-base-v2` | 768 | Fast | High | Alternative to BGE |

---

## Architecture (Phase 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 1: MVP Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Documents (PDF, TXT, MD)                                      │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│   │  PyMuPDF /  │ →  │  Sentence   │ →  │  BGE Embeddings     │ │
│   │  python-docx│    │  Chunker    │    │  (bge-base-en-v1.5) │ │
│   └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                     │            │
│                                                     ▼            │
│                                          ┌─────────────────────┐ │
│                                          │       Qdrant        │ │
│                                          │   (Vector Store)    │ │
│                                          └──────────┬──────────┘ │
│                                                     │            │
│   User Question                                     │            │
│       │                                             │            │
│       ▼                                             ▼            │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│   │   Embed     │ →  │   Vector    │ →  │   Rerank (optional) │ │
│   │   Query     │    │   Search    │    │   cross-encoder     │ │
│   └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                     │            │
│                                                     ▼            │
│                                          ┌─────────────────────┐ │
│                                          │     Groq API        │ │
│                                          │  (llama-3.1-8b)     │ │
│                                          └──────────┬──────────┘ │
│                                                     │            │
│                                                     ▼            │
│                                          ┌─────────────────────┐ │
│                                          │  Answer + Sources   │ │
│                                          └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Python Concepts You'll Learn

| Concept | Where Used |
|---------|------------|
| Type hints and annotations | All modules |
| Dataclasses for data models | `Document`, `Chunk`, `SearchResult` |
| File I/O and context managers | Document ingestion |
| Generators for memory efficiency | Chunking large documents |
| JSON serialization | Config, metadata |
| Basic OOP (classes, methods) | All components |
| List comprehensions | Data transformations |
| Environment variables | API keys, configuration |
| Async/await basics | Qdrant client, Groq API |

---

## Project Structure

```
rag_v1/
├── pyproject.toml           # Project config (uv/poetry)
├── .env                     # API keys (GROQ_API_KEY)
├── .env.example             # Template for .env
│
├── src/
│   └── rag/
│       ├── __init__.py      # Only one needed (package root)
│       ├── config.py        # Configuration management
│       ├── models.py        # Data models (Document, Chunk, etc.)
│       ├── cli.py           # CLI interface
│       │
│       ├── ingestion/       # No __init__.py needed (Python 3.3+)
│       │   ├── loader.py    # Document loaders (PDF, TXT, MD)
│       │   └── chunker.py   # Text chunking strategies
│       │
│       ├── embeddings/
│       │   └── embedder.py  # Embedding generation (BGE)
│       │
│       ├── vectorstore/
│       │   └── qdrant.py    # Qdrant integration
│       │
│       ├── retrieval/
│       │   ├── search.py    # Vector search
│       │   └── reranker.py  # Optional reranking
│       │
│       └── llm/
│           └── groq.py      # Groq API client
│
├── data/
│   └── documents/           # Place documents here
│
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    └── test_search.py
```

> **Note**: Python 3.3+ supports implicit namespace packages. Only `src/rag/__init__.py` is required to mark the package root.

---

## Implementation Tasks

| # | Task | Priority | Python Concepts | Files |
|---|------|----------|-----------------|-------|
| 1 | Project setup | High | Package structure, `pyproject.toml` | `pyproject.toml`, `src/rag/__init__.py` |
| 2 | Configuration | High | `os.getenv`, dataclasses, Pydantic | `config.py` |
| 3 | Data models | High | Dataclasses, type hints | `models.py` |
| 4 | Document loaders | High | File I/O, context managers, pathlib | `ingestion/loader.py` |
| 5 | Text chunking | High | Generators, iterators, regex | `ingestion/chunker.py` |
| 6 | Embedding generation | High | NumPy, sentence-transformers | `embeddings/embedder.py` |
| 7 | Qdrant integration | High | HTTP clients, async | `vectorstore/qdrant.py` |
| 8 | Vector search | High | List operations, sorting | `retrieval/search.py` |
| 9 | Reranker (optional) | Medium | Cross-encoder models | `retrieval/reranker.py` |
| 10 | Groq LLM client | High | API clients, error handling | `llm/groq.py` |
| 11 | CLI interface | Medium | `click`, `rich` for output | `cli.py` |
| 12 | Basic tests | Medium | `pytest`, fixtures | `tests/` |

---

## Step-by-Step Implementation

### Step 1: Project Setup

```bash
# Create project directory
mkdir rag-system && cd rag-system

# Initialize with uv (recommended) or poetry
uv init
# OR: poetry init

# Install dependencies
uv add sentence-transformers qdrant-client groq pymupdf python-docx
uv add click rich python-dotenv pydantic

# Optional: reranker
uv add sentence-transformers  # Already includes cross-encoder

# Development dependencies
uv add --dev pytest pytest-asyncio ruff mypy
```

**pyproject.toml:**
```toml
[project]
name = "rag-system"
version = "0.1.0"
description = "Open source RAG system"
requires-python = ">=3.11"

dependencies = [
    "sentence-transformers>=2.2.0",
    "qdrant-client>=1.7.0",
    "groq>=0.4.0",
    "pymupdf>=1.23.0",
    "python-docx>=1.1.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

[project.scripts]
rag = "rag.cli:main"
```

### Step 2: Configuration

```python
# src/rag/config.py
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
```

### Step 3: Data Models

```python
# src/rag/models.py
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Document:
    """Represents a source document."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Chunk:
    """A chunk of text from a document."""
    id: str
    document_id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

@dataclass
class SearchResult:
    """Result from vector search."""
    chunk: Chunk
    score: float

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: list[SearchResult]
    query: str
```

### Step 4: Document Loaders

```python
# src/rag/ingestion/loader.py
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from rag.models import Document
import hashlib

class DocumentLoader:
    """Load documents from various file formats."""

    def load(self, file_path: Path) -> Document:
        """Load a document based on file extension."""
        suffix = file_path.suffix.lower()

        loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".docx": self._load_docx,
        }

        loader = loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        content = loader(file_path)
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]

        return Document(
            id=doc_id,
            content=content,
            source=str(file_path),
            metadata={"filename": file_path.name, "type": suffix}
        )

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)

    def _load_text(self, path: Path) -> str:
        """Load plain text file."""
        return path.read_text(encoding="utf-8")

    def _load_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(path)
        return "\n".join(para.text for para in doc.paragraphs)
```

### Step 5: Text Chunking

```python
# src/rag/ingestion/chunker.py
import re
from typing import Generator
from rag.models import Document, Chunk
import hashlib

class SentenceChunker:
    """Chunk text by sentences with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, document: Document) -> Generator[Chunk, None, None]:
        """Split document into chunks."""
        sentences = self.sentence_pattern.split(document.content)

        current_chunk = []
        current_length = 0
        start_char = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Yield current chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = hashlib.md5(
                    f"{document.id}:{start_char}".encode()
                ).hexdigest()[:12]

                yield Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=document.metadata.copy()
                )

                # Keep overlap
                overlap_text = chunk_text[-self.overlap:] if len(chunk_text) > self.overlap else chunk_text
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
                start_char += len(chunk_text) - len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_len

        # Yield remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = hashlib.md5(
                f"{document.id}:{start_char}".encode()
            ).hexdigest()[:12]

            yield Chunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=document.metadata.copy()
            )
```

### Step 6: Embedding Generation

```python
# src/rag/embeddings/embedder.py
from sentence_transformers import SentenceTransformer
from rag.config import settings

class Embedder:
    """Generate embeddings using BGE model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # BGE models work better with instruction prefix for queries
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (with instruction prefix for BGE)."""
        # BGE recommends adding instruction for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding.tolist()
```

### Step 7: Qdrant Integration

```python
# src/rag/vectorstore/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from rag.config import settings
from rag.models import Chunk

class QdrantStore:
    """Qdrant vector store wrapper."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )

    def upsert(self, chunks: list[Chunk]):
        """Insert or update chunks in Qdrant."""
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )
            for chunk in chunks
            if chunk.embedding is not None
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_doc_id: str = None
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        search_filter = None
        if filter_doc_id:
            search_filter = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=filter_doc_id)
                )]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter
        )

        chunks = []
        for result in results:
            chunk = Chunk(
                id=result.id,
                document_id=result.payload["document_id"],
                content=result.payload["content"],
                metadata=result.payload["metadata"],
                start_char=result.payload["start_char"],
                end_char=result.payload["end_char"],
            )
            chunks.append((chunk, result.score))

        return chunks
```

### Step 8: Reranker (Optional)

```python
# src/rag/retrieval/reranker.py
from sentence_transformers import CrossEncoder
from rag.config import settings
from rag.models import Chunk

class Reranker:
    """Rerank search results using cross-encoder."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks based on query relevance."""
        if not chunks:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.content) for chunk, _ in chunks]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Combine with chunks and sort
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(chunks, scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]
```

### Step 9: Groq LLM Client

```python
# src/rag/llm/groq.py
from groq import Groq
from rag.config import settings
from rag.models import SearchResult, RAGResponse

class GroqLLM:
    """Groq API client for LLM generation."""

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str = None
    ) -> RAGResponse:
        """Generate answer using retrieved context."""

        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use information from the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Cite your sources by mentioning the document name
- Be concise and accurate"""

        # Build context string
        context_str = "\n\n".join([
            f"[Source: {r.chunk.metadata.get('filename', 'Unknown')}]\n{r.chunk.content}"
            for r in context
        ])

        user_message = f"""Context:
{context_str}

Question: {query}

Answer based on the context above:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=context,
            query=query
        )
```

### Step 10: CLI Interface

```python
# src/rag/cli.py
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

from rag.config import settings
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import SentenceChunker
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.retrieval.reranker import Reranker
from rag.llm.groq import GroqLLM
from rag.models import SearchResult

console = Console()

@click.group()
def main():
    """RAG System CLI"""
    pass

@main.command()
@click.argument("path", type=click.Path(exists=True))
def ingest(path: str):
    """Ingest documents from a file or directory."""
    path = Path(path)
    loader = DocumentLoader()
    chunker = SentenceChunker()
    embedder = Embedder()
    store = QdrantStore()

    files = [path] if path.is_file() else list(path.glob("**/*"))
    files = [f for f in files if f.suffix.lower() in [".pdf", ".txt", ".md", ".docx"]]

    with console.status("[bold green]Ingesting documents..."):
        for file_path in files:
            console.print(f"Processing: {file_path.name}")

            # Load document
            doc = loader.load(file_path)

            # Chunk document
            chunks = list(chunker.chunk(doc))
            console.print(f"  Created {len(chunks)} chunks")

            # Generate embeddings
            texts = [c.content for c in chunks]
            embeddings = embedder.embed(texts)

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # Store in Qdrant
            store.upsert(chunks)

    console.print(f"[bold green]✓ Ingested {len(files)} documents")

@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results")
@click.option("--rerank", is_flag=True, help="Use reranker")
def ask(query: str, top_k: int, rerank: bool):
    """Ask a question about your documents."""
    embedder = Embedder()
    store = QdrantStore()
    llm = GroqLLM()

    with console.status("[bold blue]Searching..."):
        # Embed query
        query_vector = embedder.embed_query(query)

        # Search
        results = store.search(query_vector, top_k=top_k * 2 if rerank else top_k)

        # Rerank if enabled
        if rerank and results:
            reranker = Reranker()
            results = reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Convert to SearchResult
        search_results = [SearchResult(chunk=chunk, score=score) for chunk, score in results]

        # Generate answer
        response = llm.generate(query, search_results)

    # Display answer
    console.print(Panel(response.answer, title="Answer", border_style="green"))

    # Display sources
    table = Table(title="Sources")
    table.add_column("File", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Preview", style="dim")

    for result in response.sources:
        table.add_row(
            result.chunk.metadata.get("filename", "Unknown"),
            f"{result.score:.3f}",
            result.chunk.content[:100] + "..."
        )

    console.print(table)

@main.command()
def status():
    """Show system status."""
    store = QdrantStore()

    info = store.client.get_collection(settings.collection_name)

    table = Table(title="RAG System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Qdrant", "✓ Connected")
    table.add_row("Collection", settings.collection_name)
    table.add_row("Vectors", str(info.points_count))
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("LLM Model", settings.groq_model)
    table.add_row("Reranker", "Enabled" if settings.use_reranker else "Disabled")

    console.print(table)

if __name__ == "__main__":
    main()
```

---

## Running the MVP

### 1. Start Qdrant

```bash
# Using Docker
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

### 2. Set Environment Variables

```bash
# .env file
GROQ_API_KEY=your-groq-api-key-here
```

### 3. Ingest Documents

```bash
# Single file
rag ingest ./documents/manual.pdf

# Directory
rag ingest ./documents/
```

### 4. Ask Questions

```bash
# Basic query
rag ask "What is the main topic of the documents?"

# With reranking
rag ask "How do I configure the system?" --rerank --top-k 3

# Check status
rag status
```

---

## Milestone Checklist

- [ ] Project structure created
- [ ] Configuration with environment variables
- [ ] PDF, TXT, MD, DOCX ingestion working
- [ ] Sentence-aware chunking implemented
- [ ] BGE embeddings generating correctly
- [ ] Qdrant storing and retrieving vectors
- [ ] Groq API generating answers
- [ ] CLI commands working (ingest, ask, status)
- [ ] Optional: Reranking improving results

---

## Next Steps

After completing Phase 1:

1. **Phase 2**: Add FastAPI REST endpoints
2. **Phase 3**: Add PostgreSQL for metadata and multi-tenancy
3. **Phase 4**: Add Kong API gateway and authentication

---

**Ready to start?** Begin with Step 1: Project Setup!
