# Enterprise RAG System - Design Document

> **Project**: Enterprise-grade Retrieval Augmented Generation (RAG) System
> **Version**: 1.0
> **Status**: Design Phase
> **Repository**: `/home/sujith/github/rag`
> **Purpose**: Python Learning Project + Production RAG System

---


#### Books
| Book | Level | Topics |
|------|-------|--------|
| *Python Crash Course* | Beginner | Fundamentals |
| *Fluent Python* | Intermediate | Pythonic code |
| *Architecture Patterns with Python* | Advanced | Design patterns, DDD |
| *High Performance Python* | Advanced | Optimization |

#### Online Resources
- [Real Python](https://realpython.com/) - Tutorials and articles
- [Python Documentation](https://docs.python.org/3/) - Official docs
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) - API development
- [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/) - ORM
- [pytest Documentation](https://docs.pytest.org/) - Testing

#### Practice Exercises

Each phase includes exercises to reinforce learning:

**Phase 1 Exercises:**
1. Create a `Chunk` dataclass with validation
2. Write a generator for processing large files
3. Implement a `@timed` decorator
4. Create a context manager for database connections

**Phase 2 Exercises:**
1. Build Pydantic models for API request/response
2. Write unit tests with pytest fixtures
3. Implement dependency injection for services
4. Create an abstract base class for connectors

**Phase 3 Exercises:**
1. Convert synchronous code to async
2. Design a repository pattern with SQLAlchemy
3. Implement the Strategy pattern for chunking
4. Build an event-driven processing pipeline

---

## Goals & Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Ingest documents (PDF, TXT, MD, DOCX) | Must Have |
| FR-2 | Semantic search across documents | Must Have |
| FR-3 | Question answering with source citations | Must Have |
| FR-4 | REST API for all operations | Must Have |
| FR-5 | Multi-tenant data isolation | Must Have |
| FR-6 | Document-level access control | Should Have |
| FR-7 | Conversation history / multi-turn | Should Have |
| FR-8 | 50+ data source connectors | Nice to Have |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Query latency (P95) | < 2 seconds |
| NFR-2 | Availability | 99.9% uptime |
| NFR-3 | Concurrent users | 1000+ |
| NFR-4 | Document scale | 10M+ chunks |
| NFR-5 | Security | SOC2, GDPR compliant |

---

## System Architecture

### Component Overview

```
                                    ┌─────────────────┐
                                    │   CDN / WAF     │
                                    │  (Cloudflare)   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Load Balancer  │
                                    │   (Traefik)     │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   RAG API #1    │     │   RAG API #2    │     │   RAG API #3    │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
        ┌──────────────┬─────────────────────┼─────────────────────┬──────────────┐
        │              │                     │                     │              │
┌───────▼───────┐ ┌────▼────┐ ┌──────────────▼──────────────┐ ┌────▼────┐ ┌───────▼───────┐
│    Redis      │ │  Kafka  │ │      Qdrant Cluster         │ │Postgres │ │  Object Store │
│ (Cache/Queue) │ │ (Events)│ │  ┌─────┐ ┌─────┐ ┌─────┐   │ │(Metadata│ │     (S3)      │
└───────────────┘ └─────────┘ │  │Node1│ │Node2│ │Node3│   │ │ & Auth) │ │  (Documents)  │
                              │  └─────┘ └─────┘ └─────┘   │ └─────────┘ └───────────────┘
                              └────────────────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                     ┌────────▼────────┐          ┌────────▼────────┐
                     │  Celery Worker  │          │  Celery Worker  │
                     │ (Doc Processing)│          │ (Doc Processing)│
                     └─────────────────┘          └─────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                    Observability Stack                       │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
        │  │Prometheus│  │ Grafana  │  │  Jaeger  │  │   Loki   │    │
        │  │ (Metrics)│  │(Dashboard│  │ (Traces) │  │  (Logs)  │    │
        │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
        └─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
/home/sujith/github/rag/
├── design/                      # Design documents
│   ├── 01_rag_design.md         # This document
│   ├── 02_api_spec.md           # API specification
│   └── 03_database_schema.md    # Database design
│
├── src/                         # Source code
│   └── rag/                     # Main package
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── models.py            # Pydantic models
│       ├── api/                 # FastAPI routes
│       │   ├── __init__.py
│       │   ├── routes.py
│       │   ├── auth.py
│       │   └── deps.py          # Dependencies
│       ├── core/                # Core RAG logic
│       │   ├── __init__.py
│       │   ├── ingest.py        # Document ingestion
│       │   ├── chunking.py      # Text chunking strategies
│       │   ├── embedding.py     # Embedding generation
│       │   ├── search.py        # Vector search
│       │   ├── rerank.py        # Re-ranking
│       │   └── llm.py           # LLM integration
│       ├── connectors/          # Data source connectors
│       │   ├── __init__.py
│       │   ├── base.py          # Base connector interface
│       │   ├── local.py         # Local files
│       │   ├── s3.py            # AWS S3
│       │   ├── confluence.py    # Confluence
│       │   └── google_drive.py  # Google Drive
│       ├── db/                  # Database layer
│       │   ├── __init__.py
│       │   ├── postgres.py      # PostgreSQL operations
│       │   ├── qdrant.py        # Qdrant operations
│       │   └── redis.py         # Redis cache
│       ├── security/            # Security utilities
│       │   ├── __init__.py
│       │   ├── auth.py          # Authentication
│       │   ├── acl.py           # Access control
│       │   └── pii.py           # PII detection
│       └── utils/               # Utilities
│           ├── __init__.py
│           ├── logging.py
│           └── metrics.py
│
├── workers/                     # Background workers
│   └── tasks.py                 # Celery tasks
│
├── tests/                       # Tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── kubernetes/                  # K8s manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── hpa.yaml
│
├── docker/                      # Docker files
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   └── docker-compose.yml       # Local development
│
├── scripts/                     # Utility scripts
│   ├── setup.sh
│   ├── migrate.sh
│   └── seed.sh
│
├── .github/                     # GitHub Actions
│   └── workflows/
│       └── ci-cd.yaml
│
├── pyproject.toml               # Python project config
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── Makefile                     # Common commands
└── README.md                    # Project documentation
```

### Module Responsibilities

#### `config.py`
```python
"""Configuration and environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Paths
DOCS_FOLDER = os.getenv("RAG_DOCS_FOLDER", "./docs")
TRACKING_FILE = os.getenv("RAG_TRACKING_FILE", "./processed_docs.json")

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
COLLECTION_NAME = "knowledge_base"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
RETRIEVAL_LIMIT = 10
RERANK_TOP_K = 3
```

#### `ingest.py`
```python
"""Document ingestion and processing."""
from .config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_FOLDER
from .utils import calculate_file_hash, load_processed_docs, save_processed_doc

def extract_text(file_path: str) -> str: ...
def split_into_chunks(text: str) -> list[str]: ...
def process_file(file_path: str, model, client) -> int: ...
def add_documents(docs_folder: str = DOCS_FOLDER) -> dict: ...
```

#### `search.py`
```python
"""Search and retrieval functionality."""
from .config import COLLECTION_NAME, RETRIEVAL_LIMIT, RERANK_TOP_K

def embed_query(query: str, model) -> list[float]: ...
def search_documents(query_vector: list, client, limit: int = RETRIEVAL_LIMIT): ...
def rerank_results(query: str, results: list, reranker, top_k: int = RERANK_TOP_K): ...
def build_context(results: list, max_tokens: int = 2000) -> str: ...
```

#### `llm.py`
```python
"""LLM integration and prompting."""
from .config import GROQ_API_KEY

SYSTEM_PROMPT = """You are a helpful assistant..."""

def create_prompt(question: str, context: str) -> list[dict]: ...
def generate_answer(question: str, context: str, groq_client) -> str: ...
def generate_with_citations(question: str, results: list, groq_client) -> dict: ...
```

#### `cli.py`
```python
"""Command-line interface."""
import sys
from .ingest import add_documents
from .search import search_documents, rerank_results, build_context
from .llm import generate_answer

def cmd_add(): ...
def cmd_ask(): ...
def cmd_status(): ...
def interactive_menu(): ...

def main():
    if len(sys.argv) < 2:
        interactive_menu()
    else:
        command = sys.argv[1].lower()
        commands = {"add": cmd_add, "ask": cmd_ask, "status": cmd_status}
        commands.get(command, print_help)()
```

#### `__init__.py`
```python
"""RAG System - Document Q&A with Vector Search."""
from .config import COLLECTION_NAME, DOCS_FOLDER
from .ingest import add_documents, process_file
from .search import search_documents, build_context
from .llm import generate_answer
from .cli import main

__all__ = [
    "add_documents",
    "search_documents",
    "generate_answer",
    "main"
]
```

---

## 5. Core Components

This section details the design of each core RAG component.

### 5.1 Document Chunking (`src/rag/core/chunking.py`)

Text chunking is critical for retrieval quality. The system will support multiple strategies:

| Strategy | Use Case | Quality | Speed |
|----------|----------|---------|-------|
| Fixed-size | Simple documents | Low | Fast |
| Sentence-aware | General text | Medium | Medium |
| Semantic | Complex documents | High | Slow |
| Structure-aware | PDFs, Markdown | High | Medium |

#### 5.1.1 Sentence-Aware Chunking (Default)
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_into_chunks_v2(text: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[str]:
    """Split text while respecting sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

#### 5.1.2 Semantic Chunking (Advanced)
Split based on semantic similarity between sentences:
```python
def semantic_chunk(text: str, model, threshold: float = 0.5) -> list[str]:
    """Split when semantic similarity between consecutive sentences drops."""
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

#### 5.1.3 Document Structure-Aware Chunking
For PDFs and Markdown, respect document structure:
- Split at headers/sections first
- Then apply sentence-aware chunking within sections
- Preserve hierarchy metadata (chapter > section > subsection)

---

### 5.2 Embedding Models (`src/rag/core/embedding.py`)

The embedding model converts text to vectors for semantic search.

#### 5.2.1 Supported Models

| Model | Dimensions | Quality | Speed | Use Case |
|-------|------------|---------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Good | Fast | Development/Testing |
| `all-mpnet-base-v2` | 768 | Better | Medium | Production |
| `BAAI/bge-large-en-v1.5` | 1024 | Excellent | Slower | High quality |
| `intfloat/e5-large-v2` | 1024 | Excellent | Slower | High quality |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Excellent | Medium | Good balance |

```python
# Recommended upgrade
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # or "intfloat/e5-large-v2"
model = SentenceTransformer(MODEL_NAME)

# Update collection vector size accordingly
VECTOR_SIZE = 1024  # Was 384
```

#### 5.2.2 Instruction-Tuned Embeddings
Some models perform better with prefixes:
```python
# For BGE models
def embed_query(text: str, model) -> list[float]:
    """Embed query with instruction prefix."""
    return model.encode(f"Represent this sentence for searching relevant passages: {text}")

def embed_document(text: str, model) -> list[float]:
    """Embed document without prefix."""
    return model.encode(text)
```

---

### 5.3 Search & Retrieval (`src/rag/core/search.py`)

The retrieval system finds relevant documents for a given query.

#### 5.3.1 Search Pipeline
```python
# Retrieve more, then filter/rerank
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=10  # Was 3, retrieve more candidates
)
```

#### 5.3.2 Re-ranking
Use a cross-encoder for more accurate relevance scoring:
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, results: list, top_k: int = 3) -> list:
    """Re-rank results using cross-encoder."""
    pairs = [(query, hit.payload['text']) for hit in results]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [hit for hit, _ in ranked[:top_k]]
```

#### 5.3.3 Maximal Marginal Relevance (MMR)
Reduce redundancy in retrieved chunks:
```python
def mmr_rerank(query_embedding, results, lambda_param: float = 0.5, top_k: int = 3):
    """Select diverse, relevant results using MMR."""
    selected = []
    remaining = list(results)

    while len(selected) < top_k and remaining:
        best_score = float('-inf')
        best_idx = 0

        for i, doc in enumerate(remaining):
            # Relevance to query
            relevance = cosine_similarity(query_embedding, doc.vector)

            # Diversity from already selected
            diversity = min(
                1 - cosine_similarity(doc.vector, s.vector)
                for s in selected
            ) if selected else 1.0

            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

#### 5.3.4 Hybrid Search (Dense + Sparse)
Combine semantic search with keyword matching:
```python
from qdrant_client.models import SparseVector

# Create collection with both vector types
client.create_collection(
    collection_name="hybrid_kb",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

# Use BM25 or TF-IDF for sparse vectors
from sklearn.feature_extraction.text import TfidfVectorizer

def create_sparse_vector(text: str, vectorizer) -> SparseVector:
    sparse = vectorizer.transform([text])
    indices = sparse.indices.tolist()
    values = sparse.data.tolist()
    return SparseVector(indices=indices, values=values)
```

---

### 5.4 Query Enhancement (`src/rag/core/search.py`)

Techniques to improve query understanding and retrieval.

#### 5.4.1 Query Expansion
Generate multiple query variants:
```python
def expand_query(query: str, groq_client) -> list[str]:
    """Generate query variants for better recall."""
    prompt = f"""Generate 3 different ways to ask this question.
Keep the same meaning but use different words.

Original: {query}

Variants:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    variants = response.choices[0].message.content.strip().split('\n')
    return [query] + [v.strip() for v in variants if v.strip()]
```

#### 5.4.2 HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer, then search:
```python
def hyde_search(query: str, groq_client, model, client):
    """Search using hypothetical document embedding."""
    # Generate hypothetical answer
    prompt = f"Write a short passage that would answer this question: {query}"

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    hypothetical_doc = response.choices[0].message.content

    # Embed the hypothetical document (not the query)
    hyde_vector = model.encode(hypothetical_doc).tolist()

    # Search with hypothetical document embedding
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=hyde_vector,
        limit=5
    )
```

---

### 5.5 Context Management

Building effective context for LLM generation.

#### 5.5.1 Structured Context Building
```python
def build_context(results: list, max_tokens: int = 2000) -> str:
    """Build structured context with source attribution."""
    context_parts = []
    total_tokens = 0

    for i, hit in enumerate(results, 1):
        text = hit.payload['text']
        source = hit.payload.get('source', 'Unknown')
        score = hit.score

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        chunk_tokens = len(text) // 4

        if total_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(
            f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}"
        )
        total_tokens += chunk_tokens

    return "\n\n".join(context_parts)
```

#### 5.5.2 Context Ordering
Order chunks by:
1. Relevance score (highest first)
2. Document structure (maintain reading order within same document)
3. Recency (if timestamps available)

---

### 5.6 LLM Integration (`src/rag/core/llm.py`)

LLM prompting and response generation.

#### 5.6.1 System Prompt
```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Cite your sources using [Source N] notation
4. Be concise but thorough
5. If multiple sources provide different information, acknowledge the differences"""

def build_prompt(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
```

#### 5.6.2 Citation Handling
```python
def generate_answer_with_citations(question: str, results: list, groq_client):
    """Generate answer with inline citations."""
    # Build context with numbered sources
    sources = []
    context_parts = []

    for i, hit in enumerate(results, 1):
        source = hit.payload.get('source', 'Unknown')
        sources.append(f"[{i}] {source}")
        context_parts.append(f"[Source {i}]: {hit.payload['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the question using the context below.
Cite sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer with citations:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }
```

---

### 5.7 Metadata Management

Rich metadata enables filtering and source tracking.

#### 5.7.1 Document Metadata Schema
```python
def create_chunk_metadata(
    text: str,
    filename: str,
    file_path: str,
    chunk_index: int,
    total_chunks: int,
    page_number: int = None
) -> dict:
    return {
        "text": text,
        "source": filename,
        "file_path": file_path,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "page_number": page_number,
        "char_count": len(text),
        "word_count": len(text.split()),
        "processed_at": datetime.now().isoformat(),
        "file_type": get_file_type(filename),
    }
```

#### 5.7.2 PDF Page Numbers
```python
def extract_text_with_pages(file_path: str) -> list[tuple[str, int]]:
    """Extract text with page numbers from PDF."""
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append((text, page_num))

    doc.close()
    return pages
```

#### 5.7.3 Filtering by Metadata
```python
# Search only in specific document
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="important_doc.pdf")
            )
        ]
    ),
    limit=5
)
```

---

## 6. Error Handling & Robustness

Production-grade error handling for reliability.

### 6.1 Retry with Backoff
```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def query_llm(prompt: str, groq_client):
    """Query LLM with retry logic."""
    return groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
```

### 6.2 Health Checks
```python
def check_services():
    """Verify all services are available."""
    issues = []

    # Check Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
    except Exception as e:
        issues.append(f"Qdrant: {e}")

    # Check Groq API
    try:
        groq = Groq(api_key=GROQ_API_KEY)
        # Simple test
    except Exception as e:
        issues.append(f"Groq: {e}")

    return issues
```

---

## 7. Security & Compliance

Security is critical for enterprise deployment.

### 7.1 Configuration Management
```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
```

### 7.2 Input Validation
```python
def validate_question(question: str) -> str:
    """Validate and sanitize user input."""
    question = question.strip()

    if len(question) < 3:
        raise ValueError("Question too short")

    if len(question) > 1000:
        raise ValueError("Question too long (max 1000 characters)")

    return question
```

---

## 8. Performance Optimizations

Optimizations for production scale.

### 8.1 Batch Embedding
```python
def add_documents_batch(files: list, batch_size: int = 32):
    """Process documents in batches for efficiency."""
    all_chunks = []

    # Collect all chunks
    for file in files:
        text = extract_text(file)
        chunks = split_into_chunks(text)
        all_chunks.extend([(chunk, file) for chunk in chunks])

    # Batch embed
    texts = [c[0] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Batch upsert to Qdrant
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"text": texts[i], "source": all_chunks[i][1]}
        )
        for i in range(len(all_chunks))
    ]

    # Upsert in batches
    for i in range(0, len(points), 100):
        batch = points[i:i+100]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
```

### 8.2 Query Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_embed(text: str) -> tuple:
    """Cache embeddings for repeated queries."""
    return tuple(model.encode(text).tolist())

def search_with_cache(query: str):
    """Search using cached embeddings."""
    query_vector = list(cached_embed(query))
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    )
```

### 8.3 Payload Indexing
```python
# Create indexes for frequently filtered fields
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="source",
    field_schema="keyword"
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="file_type",
    field_schema="keyword"
)
```

---

## 9. Evaluation & Monitoring

Continuous quality monitoring for production systems.

### 9.1 Retrieval Metrics
```python
def evaluate_retrieval(queries: list[dict]) -> dict:
    """
    Evaluate retrieval quality.
    queries = [{"query": "...", "relevant_docs": ["doc1.pdf", "doc2.pdf"]}]
    """
    metrics = {"mrr": [], "recall@3": [], "precision@3": []}

    for q in queries:
        query_vector = model.encode(q["query"]).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10
        )

        retrieved = [r.payload["source"] for r in results.points]
        relevant = set(q["relevant_docs"])

        # MRR (Mean Reciprocal Rank)
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                metrics["mrr"].append(1 / (i + 1))
                break
        else:
            metrics["mrr"].append(0)

        # Recall@3 and Precision@3
        top3 = set(retrieved[:3])
        hits = len(top3 & relevant)
        metrics["recall@3"].append(hits / len(relevant) if relevant else 0)
        metrics["precision@3"].append(hits / 3)

    return {k: sum(v) / len(v) for k, v in metrics.items()}
```

### 9.2 Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('RAG')

def cmd_ask():
    logger.info(f"Question: {question}")
    logger.info(f"Retrieved {len(results.points)} documents")
    logger.info(f"Top scores: {[r.score for r in results.points]}")
```


## 9.3 RAG-Specific Testing Strategy

Testing RAG systems requires evaluating both retrieval quality and generation quality.

### 9.3.1 Retrieval Tests

```python
import pytest
from typing import List, Dict

def test_retrieval_relevance():
    """Test that retrieved chunks are relevant to query."""
    query = "What is vector similarity?"
    results = search_documents(query)
    
    # All results should have minimum relevance score
    assert all(r.score > 0.3 for r in results), "Low relevance scores"
    
    # Top result should be highly relevant
    assert results[0].score > 0.7, "Top result not highly relevant"

def test_retrieval_coverage():
    """Test that all relevant documents are retrieved."""
    test_cases = [
        {
            "query": "How to deploy RAG system?",
            "expected_docs": ["deployment.md", "kubernetes.md"],
        },
        {
            "query": "What is chunking?",
            "expected_docs": ["chunking.md", "preprocessing.md"],
        }
    ]
    
    for case in test_cases:
        results = search_documents(case["query"], limit=10)
        retrieved_docs = {r.payload["source"] for r in results}
        expected_docs = set(case["expected_docs"])
        
        # Check if at least half of expected docs are retrieved
        overlap = retrieved_docs & expected_docs
        coverage = len(overlap) / len(expected_docs)
        assert coverage >= 0.5, f"Low coverage: {coverage}"

def test_retrieval_diversity():
    """Test that retrieved chunks are diverse (not redundant)."""
    query = "Explain RAG architecture"
    results = search_documents(query, limit=5)
    
    # Check cosine similarity between consecutive results
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    vectors = np.array([r.vector for r in results])
    
    for i in range(len(vectors) - 1):
        similarity = cosine_similarity([vectors[i]], [vectors[i+1]])[0][0]
        # Consecutive results shouldn't be too similar (redundant)
        assert similarity < 0.95, f"Results {i} and {i+1} too similar: {similarity}"
```

### 9.3.2 Generation Tests

```python
def test_answer_faithfulness():
    """Test that answer is grounded in retrieved context."""
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France."
    answer = generate_answer(question, context)
    
    # Answer should mention Paris
    assert "Paris" in answer or "paris" in answer.lower()
    
    # Should not hallucinate facts not in context
    assert "London" not in answer  # Wrong capital

def test_answer_relevance():
    """Test that answer addresses the question."""
    question = "How do I deploy RAG to Kubernetes?"
    context = "Use kubectl apply -f deployment.yaml to deploy."
    answer = generate_answer(question, context)
    
    # Answer should contain relevant keywords
    relevant_keywords = ["kubectl", "deploy", "kubernetes"]
    assert any(kw in answer.lower() for kw in relevant_keywords)

def test_citation_accuracy():
    """Test that citations are accurate."""
    question = "What is chunking?"
    results = search_documents(question, limit=3)
    response = generate_answer_with_citations(question, results)
    
    # Check that citations are present
    assert "[1]" in response["answer"] or "[2]" in response["answer"]
    
    # Check that sources match citations
    assert len(response["sources"]) >= 1

@pytest.mark.parametrize("question,expected_keywords", [
    ("What is RAG?", ["retrieval", "generation", "augmented"]),
    ("How to chunk documents?", ["chunk", "split", "text"]),
    ("Best embedding models?", ["embedding", "model", "vector"]),
])
def test_answer_completeness(question: str, expected_keywords: List[str]):
    """Test that answers cover key concepts."""
    results = search_documents(question, limit=5)
    answer = generate_answer(question, build_context(results))
    
    answer_lower = answer.lower()
    keyword_coverage = sum(1 for kw in expected_keywords if kw in answer_lower)
    coverage_ratio = keyword_coverage / len(expected_keywords)
    
    assert coverage_ratio >= 0.5, f"Answer missing key concepts: {coverage_ratio}"
```

### 9.3.3 End-to-End Tests

```python
def test_full_rag_pipeline():
    """Test complete RAG flow from query to answer."""
    # Ingest test document
    test_doc = "RAG combines retrieval with generation for better QA."
    doc_id = ingest_document(test_doc, "test.txt")
    
    try:
        # Query the system
        question = "What does RAG combine?"
        response = rag_query(question)
        
        # Verify response structure
        assert "answer" in response
        assert "sources" in response
        assert len(response["sources"]) > 0
        
        # Verify answer quality
        answer = response["answer"].lower()
        assert "retrieval" in answer or "generation" in answer
        
    finally:
        # Cleanup
        delete_document(doc_id)

def test_conversation_history():
    """Test multi-turn conversation."""
    rag = RAGWithHistory()
    
    # First question
    answer1 = rag.ask("What is RAG?")
    assert len(answer1) > 0
    
    # Follow-up question (requires history)
    answer2 = rag.ask("How does it work?")
    assert len(answer2) > 0
    
    # History should be maintained
    assert len(rag.history) == 2
```

### 9.3.4 Evaluation Framework

```python
from typing import List, Dict
import numpy as np

class RAGEvaluator:
    """Comprehensive RAG evaluation framework."""
    
    def __init__(self, test_set: List[Dict]):
        """
        test_set format:
        [
            {
                "question": "What is RAG?",
                "expected_answer": "RAG combines retrieval...",
                "relevant_docs": ["rag_intro.pdf", "architecture.md"],
                "context_relevance": 0.9  # Optional ground truth
            }
        ]
        """
        self.test_set = test_set
    
    def evaluate_retrieval(self, rag_system) -> Dict[str, float]:
        """Evaluate retrieval quality."""
        metrics = {
            "recall@3": [],
            "recall@5": [],
            "precision@3": [],
            "mrr": [],  # Mean Reciprocal Rank
            "ndcg@5": []  # Normalized Discounted Cumulative Gain
        }
        
        for test in self.test_set:
            query_vector = rag_system.embed(test["question"])
            results = rag_system.search(query_vector, limit=10)
            
            retrieved_docs = [r.payload["source"] for r in results]
            relevant_docs = set(test["relevant_docs"])
            
            # Recall@k
            for k in [3, 5]:
                top_k = set(retrieved_docs[:k])
                recall = len(top_k & relevant_docs) / len(relevant_docs)
                metrics[f"recall@{k}"].append(recall)
            
            # Precision@3
            top_3 = set(retrieved_docs[:3])
            precision = len(top_3 & relevant_docs) / 3
            metrics["precision@3"].append(precision)
            
            # MRR
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    metrics["mrr"].append(1 / (i + 1))
                    break
            else:
                metrics["mrr"].append(0)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def evaluate_generation(self, rag_system) -> Dict[str, float]:
        """Evaluate generation quality using LLM-as-judge."""
        metrics = {
            "relevance": [],
            "faithfulness": [],
            "conciseness": []
        }
        
        for test in self.test_set:
            response = rag_system.query(test["question"])
            
            # Use LLM to judge quality
            relevance_score = self._judge_relevance(
                test["question"], 
                response["answer"]
            )
            metrics["relevance"].append(relevance_score)
            
            faithfulness_score = self._judge_faithfulness(
                response["answer"],
                response["context"]
            )
            metrics["faithfulness"].append(faithfulness_score)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _judge_relevance(self, question: str, answer: str) -> float:
        """Use LLM to judge if answer is relevant to question."""
        prompt = f"""Rate the relevance of this answer to the question on a scale of 0-1.
        
Question: {question}
Answer: {answer}

Score (0-1):"""
        
        # Call LLM and parse score
        response = llm_client.generate(prompt, max_tokens=10)
        try:
            score = float(response.strip())
            return max(0, min(1, score))  # Clamp to [0, 1]
        except:
            return 0.5  # Default if parsing fails
    
    def _judge_faithfulness(self, answer: str, context: str) -> float:
        """Use LLM to judge if answer is grounded in context."""
        prompt = f"""Is this answer fully grounded in the context? Rate 0-1.
        
Context: {context}
Answer: {answer}

Score (0-1):"""
        
        response = llm_client.generate(prompt, max_tokens=10)
        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except:
            return 0.5

# Usage
test_set = [
    {
        "question": "What is RAG?",
        "expected_answer": "RAG is Retrieval Augmented Generation...",
        "relevant_docs": ["rag_intro.md"]
    }
]

evaluator = RAGEvaluator(test_set)
retrieval_metrics = evaluator.evaluate_retrieval(rag_system)
generation_metrics = evaluator.evaluate_generation(rag_system)

print(f"Retrieval Metrics: {retrieval_metrics}")
print(f"Generation Metrics: {generation_metrics}")
```

---

### Python Debugging for RAG Development

> **For Beginners**: Debugging is like being a detective - you follow clues to find bugs!

#### Using the VS Code Debugger

The graphical debugger is your best friend:

**1. Set Breakpoints**
- Click in the left margin (gutter) next to any line number
- A red dot appears = breakpoint set
- Code will PAUSE when it hits this line

**2. Start Debugging**
- Press `F5` OR click "Run and Debug" in sidebar
- Choose "Python File" when asked
- Code runs and stops at your breakpoint

**3. Inspect Variables**
- Hover over any variable to see its value
- Check the "Variables" panel on the left
- Expand lists/dicts to see contents

**4. Step Through Code**
- `F10` (Step Over): Execute current line, move to next
- `F11` (Step Into): Go inside function calls
- `Shift+F11` (Step Out): Finish current function
- `F5` (Continue): Run until next breakpoint

**Example Debugging Session:**

```python
# File: rag.py
def search_documents(query: str, limit: int = 5):
    # Set breakpoint here (click line number)
    query_vector = model.encode(query).tolist()
    
    # When paused, hover over query_vector to see the values
    results = client.search(query_vector, limit=limit)
    
    # Check results count
    return results

# Run with F5, code pauses at breakpoint
# Check query_vector length = should be 384
# Step through with F10
```

---

#### Common Debugging Scenarios for RAG

**Scenario 1: Empty Search Results**

```python
# Problem: search_documents() returns empty listdef search_documents(query: str):
    # Add strategic debug prints
    print(f"🔍 DEBUG: Query = '{query}'")  # Check query
    
    query_vector = model.encode(query).tolist()
    print(f"🔍 DEBUG: Vector length = {len(query_vector)}")  # Should be 384/1024
    print(f"🔍 DEBUG: First 3 values = {query_vector[:3]}")  # Should be floats
    
    results = client.search(query_vector, limit=5)
    print(f"🔍 DEBUG: Results count = {len(results)}")  # Should be > 0
    
    if len(results) == 0:
        print("⚠️  WARNING: No results found!")
        # Check if collection has documents
        collection_info = client.get_collection("knowledge_base")
        print(f"🔍 DEBUG: Total docs in collection = {collection_info.vectors_count}")
    
    return results

# Common causes:
# - Collection is empty (forgot to ingest docs)
# - Wrong collection name
# - Vector dimension mismatch
# - Query too specific
```

**Scenario 2: Import Errors**

```bash
# Error: ModuleNotFoundError: No module named 'qdrant_client'

# Debugging steps:
# 1. Check if you're in virtual environment
which python  # Should show .venv/bin/python
# NOT /usr/bin/python (system Python)

# 2. Check installed packages
pip list | grep qdrant
# If not listed, install it

# 3. Install and verify
pip install qdrant-client
python -c "import qdrant_client; print('OK')"

# 4. If still failing, recreate venv
deactivate
rm -rf .venv
python -m venv .venv
source.venv/bin/activate
pip install -r requirements.txt
```

**Scenario 3: Type Errors (Common for Beginners)**

```python
# Error: TypeError: 'NoneType' object is not subscriptable

# Bad code:
def get_first_result(results):
    return results[0]  # Crashes if results is None!

# Better code with defensive checks:
def get_first_result(results):
    # Add type checking
    if results is None:
        print("⚠️  WARNING: Results is None!")
        return None
    
    if not isinstance(results, list):
        print(f"⚠️  WARNING: Expected list, got {type(results)}")
        return None
    
    if len(results) == 0:
        print("⚠️  WARNING: Results list is empty!")
        return None
    
    return results[0]

# Even better: use type hints
from typing import Optional

def get_first_result(results: Optional[list]) -> Optional[dict]:
    """Get first result with proper error handling."""
    if not results:  # Handles None and empty list
        return None
    return results[0]
```

**Scenario 4: File Not Found Errors**

```python
# Error: FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'

import os

def load_document(file_path: str):
    # Debug: Check if file exists
    print(f"🔍 Looking for: {file_path}")
    print(f"🔍 Current dir: {os.getcwd()}")
    print(f"🔍 File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        # List files in current directory
        print(f"🔍 Files in {os.getcwd()}:")
        for f in os.listdir('.'):
            print(f"  - {f}")
        raise FileNotFoundError(f"Cannot find {file_path}")
    
    with open(file_path) as f:
        return f.read()

# Common causes:
# - Wrong working directory (check with os.getcwd())
# - Typo in filename
# - File in different folder
# - Using relative path when need absolute
```

---

#### Using pdb (Python Debugger)

For when you can't use VS Code debugger:

```python
import pdb

def process_document(file_path: str):
    text = extract_text(file_path)
    
    # Drop into debugger here
    pdb.set_trace()  # Code will pause here
    
    chunks = split_into_chunks(text)
    return chunks
```

**When code pauses, you can type:**

```python
# In pdb prompt:
p text              # Print variable
len(text)           # Check length
type(chunks)        # Check type
l                   # List code around current line
n                   # Next line
c                   # Continue execution
q                   # Quit debugger
h                   # Help
```

**Better: Use breakpoint() (Python 3.7+)**

```python
def process_document(file_path: str):
    text = extract_text(file_path)
    
    breakpoint()  # Modern way - easier to type!
    
    chunks = split_into_chunks(text)
    return chunks
```

---

### When You Get Stuck 🆘

> **Remember**: Every programmer gets stuck. It's normal! Here's how to get unstuck.

#### Step-by-Step Debugging Process

**1. Read the Error Message CAREFULLY**

```
Traceback (most recent call last):
  File "rag.py", line 42, in process_document
    chunks = split_text(text, size=500)
TypeError: split_text() missing 1 required positional argument: 'text'
```

**What to note:**
- ✅ File name: `rag.py`
- ✅ Line number: `42`
- ✅ Function: `process_document`
- ✅ Problem: Missing argument `text`
- ✅ Error type: `TypeError`

**2. Check the Basics First**

```markdown
- [ ] Did you save the file? (Check file tab for dot)
- [ ] Are you in the virtual environment? (see (.venv) in prompt)
- [ ] Did you install all dependencies? (pip list)
- [ ] Are you running the right file/script?
- [ ] Is your syntax correct? (check for typos)
```

**3. Add Print Statements** (Most Effective!)

```python
def problematic_function(data):
    print(f"===== DEBUG START =====")
    print(f"Type of data: {type(data)}")
    print(f"Data value: {data}")
    
    if isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First item: {data[0]}")
    
    # Your actual code here
    result = do_something(data)
    
    print(f"Type of result: {type(result)}")
    print(f"Result value: {result}")
    print(f"===== DEBUG END =====")
    
    return result
```

**4. Simplify to Find the Problem**

```python
# Original complex code
def complex_rag_query(question: str):
    query_vec = model.encode(rephrase(expand(question)))
    results = client.search(query_vec, filter=build_filter())
    reranked = reranker.predict([(question, r.text) for r in results])
    answer = llm.generate(build_prompt(question, reranked[:5]))
    return answer

# Simplified for debugging
def complex_rag_query(question: str):
    # Test each step individually
    print("1. Encoding...")
    query_vec = model.encode(question)  # Removed rephrase/expand
    print("2. Searching...")
    results = client.search(query_vec, limit=5)  # Removed filter
    print("3. Generating...")
    answer = llm.generate(f"Answer: {question}")  # Removed reranking
    return answer
```

**5. Google the Error Message**

```
Search term: "python TypeError split_text() missing 1 required positional argument"

Tips:
- Include "python" in search
- Copy exact error message
- Remove file-specific names
- Check Stack Overflow first
```

**6. Ask for Help** (After trying above!)

**Good Question Format:**

```markdown
**Problem**: Search returns empty results

**What I'm trying to do**:
Implementing RAG document search, but getting 0 results even though I ingested 10 documents.

**What I tried**:
1. Checked collection has docs: collection.vectors_count = 10 ✓
2. Verified query vector: len = 384 ✓
3. Printed search results: empty list []

**Code**:
```python
results = client.search(
    collection_name="knowledge_base",
    query_vector=query_vec,
    limit=5
)
print(results)  # Prints: []
```

**Error/Output**:
No error, just empty list.

**Environment**:
- Python 3.11
- Qdrant 1.7.0
- Ubuntu 22.04
```

**Where to Ask:**
- [r/learnpython](https://reddit.com/r/learnpython) - Very beginnerriendly
- [Python Discord](https://discord.gg/python) - Fast responses
- Stack Overflow - Search first, then ask

---

#### Debugging Checklist

Before asking for help, check:

```markdown
- [ ] Read error message completely
- [ ] Checked I'm in virtual environment (see `.venv` in prompt)
- [ ] Verified file is saved (no dot on file tab)
- [ ] Added print statements to see values
- [ ] Checked variable types with `print(type(x))`
- [ ] Tried simplest possible example
- [ ] Googled the exact error message
- [ ] Checked if imports are correct
- [ ] Verified I'm using right Python (python --version)
- [ ] Looked at relevant documentation
- [ ] Tried running in Python REPL
```

---

#### Common "I'm Stuck" Situations

**"My code doesn't run at all"**
- Missing `:` after if/for/def?
- Indentation wrong? (use spaces, not tabs)
- Syntax error? Check line mentioned in error

**"It runs but gives wrong answer"**
- Add print statements everywhere
- Use debugger to step through
- Check assumptions with assertions

**"It works in REPL but not in file"**
- Check working directory (os.getcwd())
- Verify imports are in file
- Make sure indentation is correct

**"It worked yesterday, now it doesn't"**
- What did you change?
- Check git diff if using version control
- Try reverting recent changes

---

###Recommended Daily Learning Routine

> **Success = Consistency, not marathons!**

#### For Complete Beginners (2-3 hours/day)

**Morning (1 hour) - Learn Concepts**
```markdown
Time: 8:00 AM - 9:00 AM

- 30 min: Read Python tutorial OR watch video
  - Week 1: Python basics (variables, types, functions)
  - Week 2: Control flow (if/else, loops)
  - Week 3: File I/O and error handling
  - Week 4: Modules and packages

- 30 min: Practice in Python REPL or Jupyter
  - Type out examples (don't copy-paste!)
  - Experiment: change values, break things on purpose
  - See what error messages look like
```

**Afternoon (1-2 hours) - Build RAG**
```markdown
Time: 2:00 PM - 4:00 PM

- Work through ONE section at a time
- Read code, type it out manually
- Run it, see if it works
- If stuck >15 min, take a break or ask for help

Weekly Focus:
- Week 1-2: Set up environment, learn Python basics
- Week 3: Start Phase 1 - Document ingestion
- Week 4: Complete Phase 1 - Simple CLI working
```

**Evening (30 min) - Review & Reflect**
```markdown
Time: 8:00 PM - 8:30 PM

Journal questions:
- What did I learn today?
- What's still confusing?
- What do I want to focus on tomorrow?
- Any "aha!" moments?

Quick review:
- Re-read code you wrote
- Explain it to yourself out loud
- Plan next day's work
```

---

#### Weekly Milestones

**Week 1: Python Basics + Setup**
- [ ] Python installed, IDE configured
- [ ] Can create & activatevirtual environment
- [ ] Understand variables, types, functions
- [ ] Can write simple loops and if/else

**Week 2: File Operations**
- [ ] Can read/write files
- [ ] Understand with statements
- [ ] Know basic string operations
- [ ] Can use lists and dictionaries

**Week 3: Start Phase 1**
- [ ] Can import libraries
- [ ] Understand function parameters
- [ ] Start document ingestion
- [ ] Can debug basic errors

**Week 4: Complete Phase 1**
- [ ] Working CLI application
- [ ] Can ingest documents
- [ ] Can query and get answers
- [ ] All Phase 1 acceptance tests pass

---

#### Learning Tips for Success

**1. Type, Don't Copy-Paste**
```python
# When you see code like this:
def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]

# Type it out manually (builds muscle memory)
# Then experiment:
# - What if size is 0? (Try it!)
# - What if text is empty? (Try it!)
# - What does range(0, len(text), size) return? (Print it!)
```

**2. Break When Frustrated**
- Stuck for >30 min? Take a 10 min walk
- Come back with fresh eyes
- Sometimes solutions appear in the shower!

**3. Celebrate Small Wins**
```markdown
✅ Got venv working? Celebrate!
✅ Fixed first bug? Celebrate!
✅ Code runs without errors? Celebrate!
✅ Understood type hints? Celebrate!

Progress > perfection
```

**4. Join a Community**
- Find a study buddy
- Join Python Discord
- Share your progress on Twitter/LinkedIn
- Teaching others helps you learn

**5. Keep a Learning Log**
```markdown
## Day 12 - Dec 15

### What I learned:
- Type hints make code clearer
- Generators save memory
- With statements auto-close files

### Problems faced:
- Import error -> fixed by activating venv
- Empty results -> forgot to ingest docs first

### Tomorrow:
- Implement chunking function
- Add error handling
```

---

---

---

## 10. Advanced Features

Optional features for enhanced functionality.

### 10.1 Conversation History
```python
class RAGWithHistory:
    def __init__(self):
        self.history = []

    def ask(self, question: str) -> str:
        # Include conversation history in context
        history_context = "\n".join([
            f"Q: {h['q']}\nA: {h['a']}"
            for h in self.history[-3:]  # Last 3 exchanges
        ])

        # Retrieve relevant documents
        results = self.search(question)
        doc_context = self.build_context(results)

        # Build prompt with history
        prompt = f"""Conversation history:
{history_context}

Relevant documents:
{doc_context}

Current question: {question}

Answer:"""

        answer = self.generate(prompt)
        self.history.append({"q": question, "a": answer})
        return answer
```

### 10.2 Multi-Document Synthesis
```python
def synthesize_from_multiple_sources(question: str, results: list) -> str:
    """Generate answer that synthesizes information from multiple sources."""
    sources_by_doc = {}
    for r in results:
        source = r.payload["source"]
        if source not in sources_by_doc:
            sources_by_doc[source] = []
        sources_by_doc[source].append(r.payload["text"])

    prompt = f"""Synthesize an answer from these sources:

"""
    for source, texts in sources_by_doc.items():
        prompt += f"\n[{source}]:\n" + "\n".join(texts) + "\n"

    prompt += f"""
Question: {question}

Provide a comprehensive answer that:
1. Combines information from all relevant sources
2. Notes any contradictions between sources
3. Cites which source provided which information"""

    return generate(prompt)
```

### 10.3 Document Update Detection
```python
def should_reprocess(file_path: str) -> bool:
    """Check if file has been modified since last processing."""
    file_hash = calculate_file_hash(file_path)
    file_mtime = os.path.getmtime(file_path)

    processed = load_processed_docs()

    if file_hash not in processed:
        return True

    stored_mtime = processed[file_hash].get("mtime", 0)
    return file_mtime > stored_mtime
```

## 10.4 Common Pitfalls & Troubleshooting

### 10.4.1 Retrieval Issues

#### Problem: Low Relevance Scores

**Symptoms:**
- All search results have scores < 0.5
- Retrieved chunks don't match query intent
- Users complain about irrelevant answers

**Causes & Solutions:**

```python
# ❌ Problem: Using wrong embedding model for queries
query_vector = model.encode(query)  # Uses document encoding

# ✅ Solution: Use instruction prefix for BGE models
query_text = f"Represent this sentence for searching relevant passages: {query}"
query_vector = model.encode(query_text)
```

**Solutions:**
1. **Upgrade embedding model**: Switch from `all-MiniLM-L6-v2` (384d) to `BAAI/bge-large-en-v1.5` (1024d)
2. **Add query preprocessing**: Expand acronyms, fix typos
3. **Use hybrid search**: Combine dense + sparse vectors
4. **Check vector normalization**: Ensure consistent normalization

#### Problem: Missing Relevant Documents

**Symptoms:**
- Known relevant docs don't appear in results
- Recall@5 metric < 0.5

**Debug Steps:**
```python
def debug_missing_document(query: str, expected_doc: str):
    """Debug why a document isn't being retrieved."""
    
    # 1. Check if document exists in collection
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=expected_doc))]
        ),
        limit=10
    )
    
    if not results[0]:
        print(f"❌ Document '{expected_doc}' not in collection!")
        print("   → Re-ingest the document")
        return
    
    print(f"✓ Found {len(results[0])} chunks from '{expected_doc}'")
    
    # 2. Check similarity scores for chunks from this doc
    query_vector = model.encode(query).tolist()
    all_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=100  # Get many results
    )
    
    doc_results = [r for r in all_results if r.payload.get("source") == expected_doc]
    
    if doc_results:
        best_score = max(r.score for r in doc_results)
        best_rank = min(i for i, r in enumerate(all_results) if r.payload.get("source") == expected_doc)
        print(f"✓ Best score: {best_score:.3f}, Best rank: {best_rank}")
        
        if best_rank > 10:
            print(f"⚠ Document ranked low (#{best_rank})")
            print("   → Consider re-ranking or query expansion")
    else:
        print(f"❌ Document not in top 100 results")
        print("   → Check chunking strategy or embedding quality")
```

**Solutions:**
1. **Chunking too large/small**: Adjust `CHUNK_SIZE` (try 300-800 chars)
2. **Query too vague**: Use query expansion or HyDE
3. **Increase retrieval limit**: Retrieve 20 candidates, then rerank to top 5

#### Problem: Redundant Results

**Symptoms:**
- Same information appears multiple times
- Results are too similar to each other

**Solution: Use MMR (Maximal Marginal Relevance)**

```python
def mmr_rerank(query_embedding, results, lambda_param: float = 0.7, top_k: int = 5):
    """
    Rerank results for diversity.
    lambda_param: 1.0 = only relevance, 0.0 = only diversity
    """
    selected = []
    remaining = list(results)
    
    while len(selected) < top_k and remaining:
        best_score = float('-inf')
        best_idx = 0
        
        for i, doc in enumerate(remaining):
            # Relevance to query
            relevance = cosine_similarity(query_embedding, doc.vector)
            
            # Maximum similarity to already selected docs
            if selected:
                max_sim = max(
                    cosine_similarity(doc.vector, s.vector)
                    for s in selected
                )
                diversity = 1 - max_sim
            else:
                diversity = 1.0
            
            # MMR score = λ * relevance + (1-λ) * diversity
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected
```

---

### 10.4.2 Generation Issues

#### Problem: Hallucinations (Answer Not Grounded)

**Symptoms:**
- LLM provides information not in retrieved context
- Citations point to wrong sources
- Factually incorrect answers

**Solutions:**

```python
# ❌ Problem: Weak system prompt
WEAK_PROMPT = "Answer the question based on context."

# ✅ Solution: Strict grounding prompt
STRICT_PROMPT = """You are a helpful assistant that ONLY answers questions based on the provided context.

CRITICAL RULES:
1. ONLY use information from the context below
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the provided documents."
3. Do NOT use your general knowledge
4. ALWAYS cite sources using [Source N] notation
5. If the context contradicts itself, mention both viewpoints

Context:
{context}

Question: {question}

Answer:"""

# Additional safety: Lower temperature
response = groq.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,  # Lower = more deterministic, less creative
    max_tokens=500
)
```

**Verification:**
```python
def verify_no_hallucination(answer: str, context: str) -> bool:
    """Check if answer facts appear in context."""
    # Use LLM-as-judge
    verification_prompt = f"""Does the answer contain information NOT present in the context?
    
Context: {context}
Answer: {answer}

Respond with ONLY 'yes' or 'no'."""
    
    response = llm.generate(verification_prompt, max_tokens=5)
    return response.strip().lower() == "no"
```

#### Problem: Incomplete or Too Brief Answers

**Symptoms:**
- Answers are 1-2 sentences when more detail exists
- Missing important context from retrieved chunks

**Solutions:**

```python
# Adjust prompt to encourage completeness
DETAILED_PROMPT = """Answer the question thoroughly using the context below.

Guidelines:
- Provide a complete, well-explained answer
- Include relevant details and examples from the context
- Aim for 3-5 sentences minimum
- Use bullet points for lists
- Cite sources using [Source N]

Context:
{context}

Question: {question}

Detailed Answer:"""

# Increase max_tokens if answers are cut off
max_tokens=1000  # Was 500
```

#### Problem: Ignoring Retrieved Context

**Symptoms:**
- Answer doesn't use retrieved chunks
- Generic answers that could apply to anything

**Debug:**
```python
def check_context_usage(answer: str, context_chunks: List[str]) -> float:
    """Measure how much of the context is used in the answer."""
    
    # Extract key phrases from context (3-5 word n-grams)
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(ngram_range=(3, 5), max_features=50)
    context_text = " ".join(context_chunks)
    
    try:
        context_phrases = vectorizer.fit_transform([context_text])
        context_vocab = set(vectorizer.get_feature_names_out())
        
        # Check which phrases appear in answer
        answer_lower = answer.lower()
        used_phrases = sum(1 for phrase in context_vocab if phrase in answer_lower)
        
        usage_ratio = used_phrases / len(context_vocab) if context_vocab else 0
        
        print(f"Context usage: {usage_ratio:.1%}")
        if usage_ratio < 0.1:
            print("⚠ Warning: Answer may not be using retrieved context")
        
        return usage_ratio
    except:
        return 0.0
```

**Solution: Few-shot prompting**
```python
FEW_SHOT_PROMPT = """Answer questions using ONLY the provided context. See examples:

Example 1:
Context: RAG combines retrieval with generation. It retrieves relevant docs from a knowledge base.
Question: What is RAG?
Answer: RAG (Retrieval Augmented Generation) is a technique that combines retrieval with generation. It works by retrieving relevant documents from a knowledge base. [Source 1]

Now answer this question:

Context:
{context}

Question: {question}

Answer:"""
```

---

### 10.4.3 Performance Issues

#### Problem: Slow Query Latency (>5 seconds)

**Symptoms:**
- P95 latency > 3 seconds
- Users complain about slow responses

**Profiling:**
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{name}: {elapsed:.1f}ms")

def profile_rag_query(question: str):
    """Profile each stage of RAG pipeline."""
    
    with timer("Total"):
        with timer("  1. Embed query"):
            query_vector = model.encode(question).tolist()
        
        with timer("  2. Vector search"):
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=10
            )
        
        with timer("  3. Rerank"):
            reranked = reranker.predict([
                (question, r.payload['text']) for r in results
            ])
        
        with timer("  4. Build context"):
            context = build_context(results[:5])
        
        with timer("  5. LLM generation"):
            answer = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}],
                max_tokens=500
            )
    
    return answer

# Example output:
#   1. Embed query: 45.2ms
#   2. Vector search: 123.5ms
#   3. Rerank: 456.8ms  ← Bottleneck!
#   4. Build context: 2.1ms
#   5. LLM generation: 1234.5ms
# Total: 1862.1ms
```

**Solutions:**

1. **Caching** (biggest impact):
```python
# Cache embeddings
@lru_cache(maxsize=1000)
def cached_embed(text: str) -> tuple:
    return tuple(model.encode(text).tolist())

# Cache full query results (Redis)
def query_with_cache(question: str, ttl: int = 3600):
    cache_key = f"rag:query:{hashlib.md5(question.encode()).hexdigest()}"
    
    # Check cache
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Execute query
    result = execute_rag_query(question)
    
    # Cache for 1 hour
    redis.setex(cache_key, ttl, json.dumps(result))
    return result
```

2. **Skip reranking for simple queries**:
```python
def should_rerank(query: str, initial_scores: List[float]) -> bool:
    """Skip reranking if top results are already good."""
    # If top result is very confident, skip reranking
    if initial_scores[0] > 0.9:
        return False
    
    # If query is short/simple, skip
    if len(query.split()) < 5:
        return False
    
    return True
```

3. **Reduce retrieval limit**:
```python
# Instead of retrieving 20 and reranking to 5
results = search(query, limit=20)  # Slow
reranked = rerank(results)[:5]

# Try retrieving fewer with better initial ranking
results = search(query, limit=8)  # Faster
reranked = rerank(results)[:5]
```

4. **Use faster LLM**:
```python
# Groq is ~10x faster than OpenAI for inference
# llama-3.1-8b-instant on Groq: ~500ms
# gpt-3.5-turbo on OpenAI: ~2000ms
```

#### Problem: High Memory Usage

**Symptoms:**
- OOM (Out of Memory) errors
- Process killed by OS
- Slowdowns over time

**Solutions:**

```python
# ❌ Problem: Loading all embeddings at once
all_texts = [chunk for doc in documents for chunk in doc.chunks]
all_embeddings = model.encode(all_texts)  # May exceed memory!

# ✅ Solution: Batch processing
def embed_documents_batched(documents, batch_size=32):
    """Process documents in batches to limit memory."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        
        # Process/store immediately
        store_embeddings(batch, embeddings)
        
        # Free memory
        del embeddings
        gc.collect()
```

#### Problem: High Costs

**Symptoms:**
- Monthly LLM API bill too high
- Cost per query > target  

**Monitoring:**
```python
class CostTracker:
    def track_query(self, tokens_in: int, tokens_out: int, model: str):
        # Groq pricing (Jan 2024)
        costs = {
            "llama-3.1-8b-instant": {"in": 0.05/1M, "out": 0.08/1M},
            "llama-3.1-70b-versatile": {"in": 0.59/1M, "out": 0.79/1M},
        }
        
        cost = (
            tokens_in * costs[model]["in"] +
            tokens_out * costs[model]["out"]
        )
        
        self.total_cost += cost
        return cost

# Per-query cost logging
logger.info(f"Query cost: ${cost:.4f}, Total today: ${tracker.total_cost:.2f}")
```

**Solutions:**
1. **Cache aggressively** (70-90% cache hit rate)
2. **Use smaller models** for simple queries
3. **Reduce `max_tokens`**: 300 instead of 500
4. **Batch similar queries**: Detect duplicates
5. **Rate limiting**: Prevent abuse

---

### 10.4.4 Data Quality Issues

#### Problem: Poor Chunking

**Symptoms:**
- Chunks end mid-sentence
- Important context split across chunks
- Tables/code split incorrectly

**Debug:**
```python
def inspect_chunks(document_id: str):
    """Inspect how a document was chunked."""
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=document_id))]
        ),
        limit=100
    )
    
    for i, point in enumerate(results[0]):
        chunk_text = point.payload['text']
        print(f"\n--- Chunk {i+1} ({len(chunk_text)} chars) ---")
        print(chunk_text[:200] + "...")
        
        # Check for issues
        if not chunk_text.endswith(('.', '!', '?', '\n')):
            print("⚠ Warning: Chunk doesn't end at sentence boundary")
        
        if len(chunk_text) < 50:
            print("⚠ Warning: Chunk very short")
        
        if len(chunk_text) > 1000:
            print("⚠ Warning: Chunk very long")
```

**Solutions:**
1. Use sentence-aware chunking (section 5.1.1)
2. Add overlap between chunks (50-100 chars)
3. Preserve tables/code blocks as single chunks

#### Problem: Duplicate Documents

**Symptoms:**
- Same document appears multiple times
- Redundant chunks in search results

**Detection & Deduplication:**
```python
import hashlib

def calculate_content_hash(text: str) -> str:
    """Generate hash for deduplication."""
    # Normalize: lowercase, remove extra whitespace
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()

def ingest_with_dedup(file_path: str):
    """Ingest only if document is new."""
    text = extract_text(file_path)
    content_hash = calculate_content_hash(text)
    
    # Check if already processed
    existing = db.execute(
        "SELECT id FROM documents WHERE content_hash = ?",
        (content_hash,)
    ).fetchone()
    
    if existing:
        print(f"⏭ Skipping duplicate: {file_path}")
        return None
    
    # Process new document
    doc_id = process_document(text)
    
    # Store hash
    db.execute(
        "INSERT INTO documents (id, file_path, content_hash) VALUES (?, ?, ?)",
        (doc_id, file_path, content_hash)
    )
    
    return doc_id
```

---

### 10.4.5 Quick Troubleshooting Checklist

```markdown
## RAG System Troubleshooting Checklist

### Symptoms: No/Poor Results
- [ ] Check if documents are in Qdrant collection
- [ ] Verify embedding model is loaded correctly
- [ ] Test with a known-good query
- [ ] Check vector dimensions match (384 vs 1024)
- [ ] Verify collection has proper distance metric (Cosine)

### Symptoms: Slow Performance
- [ ] Profile query to find bottleneck
- [ ] Check cache hit rate (should be >50%)
- [ ] Verify Qdrant index status
- [ ] Check LLM API rate limits
- [ ] Monitor memory usage

### Symptoms: Poor Answer Quality
- [ ] Review retrieved chunks (are they relevant?)
- [ ] Check if reranking is enabled
- [ ] Verify system prompt is strict about grounding
- [ ] Test with lower LLM temperature (0.1-0.3)
- [ ] Check if enough context is provided (token limit)

### Symptoms: High Costs
- [ ] Enable query result caching
- [ ] Reduce max_tokens in LLM calls
- [ ] Use smaller/faster model for simple queries
- [ ] Check for duplicate/spam queries
- [ ] Implement rate limiting per user
```

---


---

## 10.5 Advanced RAG Techniques

This section covers next-generation RAG approaches that go beyond traditional retrieve-and-generate.

---

### 10.5.1 Agentic RAG (ReAct Pattern)

**Concept:** The LLM decides *when* to retrieve and can use tools/functions, rather than always retrieving upfront.

**ReAct (Reasoning + Acting) Pattern:**

```python
from typing import List, Dict, Callable
import re

class AgenticRAG:
    """LLM agent that decides when to retrieve documents."""
    
    def __init__(self, llm_client, search_function: Callable, tools: Dict[str, Callable]):
        self.llm = llm_client
        self.search = search_function
        self.tools = tools
        self.max_iterations = 5
    
    def query(self, question: str) -> str:
        """Run agentic RAG with ReAct loop."""
        
        conversation_history = []
        
        # System prompt with tool descriptions
        system_prompt = """You are a helpful assistant with access to tools.

Available tools:
- search(query: str) -> List[str]: Search the knowledge base
- calculator(expression: str) -> float: Evaluate math expressions
- get_current_date() -> str: Get today's date

To use a tool, write:
Thought: I need to search for information about X
Action: search("information about X")

After seeing the Observation, continue reasoning or provide Final Answer.

Always end with:
Final Answer: <your answer>
"""
        
        for iteration in range(self.max_iterations):
            # Build prompt with history
            prompt = system_prompt + "\n\nQuestion: " + question + "\n\n"
            for entry in conversation_history:
                prompt += entry + "\n"
            
            # Get LLM response
            response = self.llm.generate(prompt, max_tokens=300)
            conversation_history.append(response)
            
            # Check if final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Parse action
            action_match = re.search(r'Action: (\w+)\((.*?)\)', response)
            if not action_match:
                continue
            
            tool_name = action_match.group(1)
            tool_args = action_match.group(2).strip('"\'')
            
            # Execute tool
            if tool_name == "search":
                results = self.search(tool_args)
                observation = "Observation: Found:\n" + "\n".join(results[:3])
            elif tool_name in self.tools:
                observation = f"Observation: {self.tools[tool_name](tool_args)}"
            else:
                observation = f"Observation: Unknown tool {tool_name}"
            
            conversation_history.append(observation)
        
        return "Could not find answer within iteration limit."

# Usage
def search_kb(query: str) -> List[str]:
    """Search knowledge base (simplified)."""
    results = qdrant_client.search(query, limit=5)
    return [r.payload['text'] for r in results]

tools = {
    "calculator": lambda expr: eval(expr),  # Be careful with eval in production!
    "get_current_date": lambda _: "2024-01-15"
}

agent = AgenticRAG(llm_client, search_kb, tools)

# The agent will decide whether to search or not
answer = agent.query("What is 50% of the RAG system's latency target?")
# Agent thinks:
# Thought: I need to search for the latency target
# Action: search("RAG system latency target")
# Observation: Found: "Target latency is 2 seconds"
# Thought: Now I need to calculate 50% of 2 seconds
# Action: calculator("2 * 0.5")
# Observation: 1.0
# Final Answer: 50% of the RAG system's latency target (2 seconds) is 1 second.
```

**Function Calling (OpenAI/Anthropic Style):**

```python
from openai import OpenAI

class FunctionCallingRAG:
    """RAG using OpenAI function calling."""
    
    def __init__(self, openai_client: OpenAI, qdrant_client):
        self.openai = openai_client
        self.qdrant = qdrant_client
    
    def search_documents(self, query: str) -> str:
        """Search knowledge base - exposed as function."""
        results = self.qdrant.search(query, limit=5)
        return "\n".join([r.payload['text'] for r in results])
    
    def query(self, question: str) -> str:
        """Query with function calling."""
        
        # Define tools/functions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search the knowledge base for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        messages = [{"role": "user", "content": question}]
        
        # First call - LLM decides whether to search
        response = self.openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check if LLM wants to call function
        if response_message.tool_calls:
            # Execute search
            tool_call = response_message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            search_results = self.search_documents(function_args["query"])
            
            # Add function result to conversation
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_results
            })
            
            # Second call - LLM generates answer with context
            final_response = self.openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        # No search needed
        return response_message.content
```

---

### 10.5.2 Self-RAG (Adaptive Retrieval)

**Concept:** The system decides whether retrieval is needed, retrieves if necessary, and self-critiques the generated answer.

**Implementation:**

```python
class SelfRAG:
    """Self-reflective RAG that adapts retrieval strategy."""
    
    def __init__(self, llm_client, qdrant_client, embedding_model):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.model = embedding_model
    
    def needs_retrieval(self, question: str) -> bool:
        """Decide if question requires external knowledge."""
        
        prompt = f"""Does this question require looking up external information or can it be answered from general knowledge?

Question: {question}

Answer only 'NEEDS_RETRIEVAL' or 'NO_RETRIEVAL'."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        return "NEEDS_RETRIEVAL" in response.upper()
    
    def is_answer_supported(self, answer: str, context: str) -> tuple[bool, float]:
        """Check if answer is supported by context."""
        
        prompt = f"""Is this answer fully supported by the context?

Context: {context}

Answer: {answer}

Rate support from 0.0 (not supported) to 1.0 (fully supported).
Respond with just a number."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            support_score = float(response.strip())
            is_supported = support_score >= 0.7
            return is_supported, support_score
        except:
            return False, 0.0
    
    def is_answer_useful(self, question: str, answer: str) -> tuple[bool, float]:
        """Check if answer actually addresses the question."""
        
        prompt = f"""Does this answer properly address the question?

Question: {question}

Answer: {answer}

Rate usefulness from 0.0 (not useful) to 1.0 (very useful).
Respond with just a number."""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            usefulness_score = float(response.strip())
            is_useful = usefulness_score >= 0.7
            return is_useful, usefulness_score
        except:
            return False, 0.0
    
    def query(self, question: str) -> Dict:
        """Self-RAG query with adaptive retrieval."""
        
        # Step 1: Decide if retrieval is needed
        needs_docs = self.needs_retrieval(question)
        
        if not needs_docs:
            # Answer from parametric knowledge
            answer = self.llm.generate(
                f"Answer this question: {question}",
                max_tokens=200
            )
            return {
                "answer": answer,
                "retrieval_used": False,
                "confidence": "high"
            }
        
        # Step 2: Retrieve documents
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=5)
        context = "\n\n".join([r.payload['text'] for r in results])
        
        # Step 3: Generate answer with context
        prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        answer = self.llm.generate(prompt, max_tokens=300)
        
        # Step 4: Self-critique
        is_supported, support_score = self.is_answer_supported(answer, context)
        is_useful, usefulness_score = self.is_answer_useful(question, answer)
        
        # Step 5: Decide on final answer
        if is_supported and is_useful:
            return {
                "answer": answer,
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "high"
            }
        elif is_useful but not is_supported:
            # Answer is useful but not grounded - add disclaimer
            return {
                "answer": f"{answer}\n\n⚠️ Note: This answer may not be fully supported by the retrieved documents.",
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "medium"
            }
        else:
            # Answer not good - try parallel retrieval or admit uncertainty
            return {
                "answer": "I don't have sufficient information to answer this question confidently based on the available documents.",
                "retrieval_used": True,
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "confidence": "low"
            }

# Usage
self_rag = SelfRAG(llm_client, qdrant_client, embedding_model)
result = self_rag.query("What is the capital of France?")
# Result: No retrieval needed (general knowledge)

result = self_rag.query("What is our company's RAG latency target?")
# Result: Retrieves documents, generates answer, critiques itself
```

---

### 10.5.3 Corrective RAG (CRAG)

**Concept:** Detect when retrieval fails and take corrective action (web search, different query, etc.).

```python
from typing import List, Optional
import requests

class CorrectiveRAG:
    """RAG with fallback strategies when retrieval fails."""
    
    def __init__(self, llm_client, qdrant_client, embedding_model, web_search_api_key: str):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.model = embedding_model
        self.web_search_key = web_search_api_key
    
    def assess_retrieval_quality(self, question: str, retrieved_docs: List[str]) -> float:
        """Assess if retrieved documents are relevant to the question."""
        
        if not retrieved_docs:
            return 0.0
        
        # Use LLM to judge relevance
        docs_sample = "\n".join(retrieved_docs[:3])
        prompt = f"""Rate how relevant these documents are to the question on a scale of 0.0 to 1.0.

Question: {question}

Documents:
{docs_sample}

Relevance score (0.0-1.0):"""
        
        response = self.llm.generate(prompt, max_tokens=10)
        try:
            return float(response.strip())
        except:
            return 0.5  # Default to medium relevance
    
    def web_search(self, query: str) -> List[str]:
        """Fallback to web search (e.g., Serper API, Brave Search)."""
        
        # Example using Serper API
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.web_search_key,
            "Content-Type": "application/json"
        }
        data = {"q": query, "num": 5}
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            results = response.json().get("organic", [])
            return [f"{r['title']}: {r['snippet']}" for r in results]
        return []
    
    def decompose_query(self, question: str) -> List[str]:
        """Break complex question into sub-questions."""
        
        prompt = f"""Break this complex question into 2-3 simpler sub-questions.

Question: {question}

Sub-questions (one per line):"""
        
        response = self.llm.generate(prompt, max_tokens=150)
        sub_questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return sub_questions[:3]
    
    def query(self, question: str) -> Dict:
        """Corrective RAG with multiple fallback strategies."""
        
        # Strategy 1: Try normal RAG retrieval
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=10)
        retrieved_docs = [r.payload['text'] for r in results]
        
        # Assess retrieval quality
        quality_score = self.assess_retrieval_quality(question, retrieved_docs)
        
        # Strategy 2: If quality low, try query decomposition
        if quality_score < 0.5:
            print("⚠️ Low retrieval quality. Trying query decomposition...")
            sub_questions = self.decompose_query(question)
            
            all_docs = []
            for sub_q in sub_questions:
                sub_vector = self.model.encode(sub_q).tolist()
                sub_results = self.qdrant.search(sub_vector, limit=5)
                all_docs.extend([r.payload['text'] for r in sub_results])
            
            # Deduplicate
            retrieved_docs = list(set(all_docs))
            quality_score = self.assess_retrieval_quality(question, retrieved_docs)
        
        # Strategy 3: If still poor, fall back to web search
        if quality_score < 0.4:
            print("⚠️ Still low quality. Falling back to web search...")
            web_results = self.web_search(question)
            
            if web_results:
                retrieved_docs = web_results
                quality_score = 0.7  # Assume web is decent quality
        
        # Strategy 4: If all else fails, admit uncertainty
        if quality_score < 0.3:
            return {
                "answer": "I don't have sufficient information to answer this question reliably. The available documents and web search did not provide relevant information.",
                "strategy_used": "none",
                "confidence": "very_low"
            }
        
        # Generate answer with best available context
        context = "\n\n".join(retrieved_docs[:5])
        prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        answer = self.llm.generate(prompt, max_tokens=400)
        
        strategy = "web_search" if quality_score == 0.7 else "decomposition" if len(retrieved_docs) > 10 else "standard"
        
        return {
            "answer": answer,
            "strategy_used": strategy,
            "quality_score": quality_score,
            "confidence": "high" if quality_score > 0.7 else "medium"
        }

# Usage
crag = CorrectiveRAG(llm_client, qdrant_client, embedding_model, web_search_key)
result = crag.query("What happened in the news today?")
# Falls back to web search since internal docs won't have today's news
```

---

### 10.5.4 Graph RAG (Knowledge Graph Integration)

**Concept:** Combine vector search with knowledge graph traversal for better relational understanding.

```python
from neo4j import GraphDatabase
from typing import List, Tuple

class GraphRAG:
    """RAG with knowledge graph for entity relationships."""
    
    def __init__(self, llm_client, qdrant_client, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.graph_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using LLM."""
        
        prompt = f"""Extract the main entities (people, organizations, technologies) from this text.

Text: {text}

Entities (comma-separated):"""
        
        response = self.llm.generate(prompt, max_tokens=100)
        entities = [e.strip() for e in response.split(',')]
        return entities[:5]  # Limit to 5 entities
    
    def get_entity_relationships(self, entity: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        """Get relationships from knowledge graph."""
        
        query = """
        MATCH path = (e:Entity {name: $entity})-[r*1..depth]-(related)
        RETURN e.name as source, type(r[0]) as relationship, related.name as target
        LIMIT 20
        """
        
        with self.graph_driver.session() as session:
            result = session.run(query, entity=entity, depth=depth)
            relationships = [(record["source"], record["relationship"], record["target"]) 
                           for record in result]
        
        return relationships
    
    def expand_context_with_graph(self, chunks: List[str], question: str) -> str:
        """Expand retrieved chunks with graph knowledge."""
        
        # Extract entities from question
        entities = self.extract_entities(question)
        
        # Get relationships for each entity
        graph_context = []
        for entity in entities:
            rels = self.get_entity_relationships(entity, depth=2)
            if rels:
                entity_info = f"\n{entity} relationships:"
                for source, rel, target in rels[:5]:
                    entity_info += f"\n  - {source} {rel} {target}"
                graph_context.append(entity_info)
        
        # Combine vector-retrieved chunks with graph context
        combined_context = "Retrieved Documents:\n" + "\n\n".join(chunks)
        
        if graph_context:
            combined_context += "\n\nKnowledge Graph Information:" + "".join(graph_context)
        
        return combined_context
    
    def query(self, question: str) -> Dict:
        """Query with vector search + knowledge graph."""
        
        # Step 1: Vector search
        query_vector = self.model.encode(question).tolist()
        results = self.qdrant.search(query_vector, limit=5)
        chunks = [r.payload['text'] for r in results]
        
        # Step 2: Expand with knowledge graph
        enhanced_context = self.expand_context_with_graph(chunks, question)
        
        # Step 3: Generate answer
        prompt = f"""Use both the documents and knowledge graph to answer.

{enhanced_context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=500)
        
        return {
            "answer": answer,
            "used_graph": len(self.extract_entities(question)) > 0,
            "sources": {"vector_chunks": len(chunks), "graph_entities": len(self.extract_entities(question))}
        }

# Setup: First populate knowledge graph
def build_knowledge_graph_from_documents(documents: List[str], graph_driver):
    """Extract entities and relationships from documents to build graph."""
    
    for doc in documents:
        # Use LLM to extract triples (subject, relation, object)
        prompt = f"""Extract knowledge triples from this text.
        
Text: {doc}

Format: subject | relation | object
One per line."""
        
        response = llm.generate(prompt)
        triples = [line.split('|') for line in response.strip().split('\n')]
        
        # Insert into Neo4j
        with graph_driver.session() as session:
            for triple in triples:
                if len(triple) == 3:
                    subject, relation, obj = [t.strip() for t in triple]
                    query = """
                    MERGE (s:Entity {name: $subject})
                    MERGE (o:Entity {name: $object})
                   MERGE (s)-[r:RELATED {type: $relation}]->(o)
                    """
                    session.run(query, subject=subject, relation=relation, object=obj)

# Usage
graph_rag = GraphRAG(llm_client, qdrant_client, "bolt://localhost:7687", "neo4j", "password")
result = graph_rag.query("How is FastAPI related to Pydantic?")
# Combines vector search with graph traversal to find relationships
```

---

### 10.5.5 Multi-Modal RAG

**Concept:** Handle images, tables, charts, and other non-text content in RAG.

```python
from PIL import Image
import pytesseract
import pandas as pd
from typing import Union
import base64

class MultiModalRAG:
    """RAG that handles text, images, tables, and charts."""
    
    def __init__(self, llm_client, qdrant_client, vision_model):
        self.llm = llm_client
        self.qdrant = qdrant_client
        self.vision_model = vision_model  # e.g., GPT-4 Vision, Claude 3
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from images using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    def describe_image(self, image_path: str) -> str:
        """Get semantic description of image using vision model."""
        
        # Encode image to base64
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Use vision model (GPT-4 Vision example)
        prompt = "Describe this image in detail, including any text, charts, or diagrams."
        
        response = self.vision_model.generate(
            prompt=prompt,
            image_base64=image_data,
            max_tokens=300
        )
        
        return response
    
    def extract_table_data(self, table_html: str) -> str:
        """Convert HTML table to searchable text."""
        
        # Parse table
        df = pd.read_html(table_html)[0]
        
        # Convert to natural language
        table_description = f"Table with {len(df)} rows and {len(df.columns)} columns.\n"
        table_description += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add sample rows
        for idx, row in df.head(5).iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            table_description += f"Row {idx}: {row_text}\n"
        
        return table_description
    
    def ingest_multimodal_document(self, file_path: str, file_type: str) -> List[Dict]:
        """Ingest document with mixed content types."""
        
        chunks = []
        
        if file_type == "pdf_with_images":
            # Extract text chunks
            text = extract_text_from_pdf(file_path)
            text_chunks = split_into_chunks(text)
            
            # Extract images
            images = extract_images_from_pdf(file_path)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "type": "text",
                    "content": chunk,
                    "metadata": {"page": i, "modality": "text"}
                })
            
            for i, img_path in enumerate(images):
                # OCR text
                ocr_text = self.extract_text_from_image(img_path)
                
                # Vision description
                description = self.describe_image(img_path)
                
                # Combined representation
                image_content = f"[IMAGE]\nOCR Text: {ocr_text}\nDescription: {description}"
                
                chunks.append({
                    "type": "image",
                    "content": image_content,
                    "metadata": {"image_path": img_path, "modality": "image"}
                })
        
        return chunks
    
    def query(self, question: str, include_images: bool = True) -> Dict:
        """Multi-modal RAG query."""
        
        # Embed question
        query_vector = self.model.encode(question).tolist()
        
        # Search across all modalities
        results = self.qdrant.search(query_vector, limit=10)
        
        # Separate by modality
        text_chunks = []
        image_chunks = []
        
        for r in results:
            if r.payload.get('modality') == 'text':
                text_chunks.append(r.payload['content'])
            elif r.payload.get('modality') == 'image' and include_images:
                image_chunks.append(r.payload['content'])
        
        # Build multimodal context
        context = "Text Sources:\n" + "\n\n".join(text_chunks[:5])
        
        if image_chunks:
            context += "\n\nImage Sources:\n" + "\n\n".join(image_chunks[:3])
        
        # Generate answer
        prompt = f"""Answer using both text and image information.

{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=500)
        
        return {
            "answer": answer,
            "sources": {
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks)
            }
        }

# Usage
multimodal_rag = MultiModalRAG(llm_client, qdrant_client, vision_model)

# Ingest PDF with charts
chunks = multimodal_rag.ingest_multimodal_document("financial_report.pdf", "pdf_with_images")
# Stores both text and image descriptions in Qdrant

# Query can retrieve both text and images
result = multimodal_rag.query("What does the revenue chart show for Q4?")
# Answer: "According to the chart (from image on page 5), Q4 revenue was $2.5M, 
# a 15% increase from Q3..."
```

---

### 10.5.6 Comparison of Advanced Techniques

| Technique | Best For | Complexity | When to Use |
|-----------|----------|------------|-------------|
| **Agentic RAG** | Complex multi-step queries | High | Questions requiring reasoning + tool use |
| **Self-RAG** | Variable question complexity | Medium | When some questions don't need retrieval |
| **Corrective RAG** | Handling retrieval failures | Medium | When internal KB may be incomplete |
| **Graph RAG** | Relational/entity queries | High | Questions about relationships, hierarchies |
| **Multi-Modal** | Documents with images/charts | High | PDFs, presentations, reports with visuals |

---

### 10.5.7 Hybrid Approach Example

```python
class AdvancedRAG:
    """Combines multiple advanced techniques."""
    
    def __init__(self, components: Dict):
        self.self_rag = components['self_rag']
        self.crag = components['crag']
        self.graph_rag = components.get('graph_rag')
    
    def query(self, question: str) -> Dict:
        """Adaptive RAG that chooses the best strategy."""
        
        # Step 1: Self-RAG decides if retrieval needed
        needs_retrieval = self.self_rag.needs_retrieval(question)
        
        if not needs_retrieval:
            return self.self_rag.query(question)
        
        # Step 2: Try Corrective RAG with fallbacks
        result = self.crag.query(question)
        
        # Step 3: If question involves entities and graph available, use Graph RAG
        if self.graph_rag and self._has_entities(question):
            graph_result = self.graph_rag.query(question)
            
            # Combine results
            result['answer'] += f"\n\nAdditional context from knowledge graph: {graph_result['answer']}"
        
        return result
    
    def _has_entities(self, question: str) -> bool:
        """Check if question involves named entities."""
        # Simple heuristic: check for capitalized words
        words = question.split()
        capitalized = sum(1 for w in words if w[0].isupper())
        return capitalized >= 2

# Usage: Automatically adapts to question type
advanced_rag = AdvancedRAG({
    'self_rag': SelfRAG(...),
    'crag': CorrectiveRAG(...),
    'graph_rag': GraphRAG(...)
})

result = advanced_rag.query("How does our system handle errors?")
# Uses Self-RAG → determines retrieval needed → CRAG retrieves docs

result = advanced_rag.query("What is the relationship between microservices and containers?")
# Uses Graph RAG for entity relationships
```

---

### Resources for Advanced RAG

**Papers:**
- Self-RAG: [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
- Corrective RAG (CRAG): [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)
- Graph RAG: [Microsoft Research](https://www.microsoft.com/en-us/research/blog/graphrag/)
- ReAct: [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

**Frameworks:**
- LangChain: Agents with tools
- LlamaIndex: Multi-modal RAG
- Neo4j + LLM: Graph RAG
- Hayden: Multi-modal pipelines

---


---


---

## 11. Development Roadmap

### Phase 1: MVP - Python Fundamentals

**Python Concepts You'll Learn:**
- Type hints and annotations
- Dataclasses for data models
- File I/O and context managers
- Generators for memory efficiency
- JSON serialization
- Basic OOP (classes, methods)
- List comprehensions
- Environment variables with `python-dotenv`

| Task | Priority | Python Concepts | Status |
|------|----------|-----------------|--------|
| Project setup (pyproject.toml) | High | Package structure, `__init__.py` | Pending |
| Configuration management | High | `os.getenv`, dataclasses, type hints | Pending |
| Document ingestion (PDF, TXT, MD) | High | File I/O, context managers, pathlib | Pending |
| Sentence-aware chunking | High | Generators, iterators, string methods | Pending |
| Embedding generation | High | NumPy arrays, list operations | Pending |
| Qdrant integration | High | HTTP clients, context managers | Pending |
| Basic vector search | High | List comprehensions, sorting | Pending |
| LLM integration (Groq) | High | API clients, error handling | Pending |
| CLI interface | Medium | `argparse` or `click`, `if __name__` | Pending |

**Milestone Project**: Working CLI that can ingest documents and answer questions.

### Phase 2: API & Quality - Intermediate Python

**Python Concepts You'll Learn:**
- FastAPI and Pydantic models
- Decorators (custom and built-in)
- Abstract Base Classes (ABC)
- Dependency injection
- Unit testing with pytest
- Mocking and fixtures
- Logging module
- HTTP status codes and exceptions

| Task | Priority | Python Concepts | Status |
|------|----------|-----------------|--------|
| FastAPI REST endpoints | High | Pydantic, async def, decorators | Pending |
| Re-ranking with cross-encoder | High | Callable types, sorting algorithms | Pending |
| Citation handling | High | String formatting, dataclasses | Pending |
| Error handling & retries | High | Custom exceptions, decorators | Pending |
| Basic logging | Medium | `logging` module, formatters | Pending |
| Input validation | Medium | Pydantic validators, regex | Pending |
| Docker setup | Medium | Environment variables, configs | Pending |
| Unit tests | High | pytest, fixtures, mocking | Pending |

**Milestone Project**: REST API with tests and documentation.

### Phase 3: Enterprise Features - Advanced Python

**Python Concepts You'll Learn:**
- Async/await and asyncio
- SQLAlchemy ORM and relationships
- Alembic migrations
- Celery for background tasks
- Redis integration
- Design patterns (Factory, Strategy)
- Connection pooling
- Multi-threading vs multi-processing

| Task | Priority | Python Concepts | Status |
|------|----------|-----------------|--------|
| PostgreSQL metadata store | High | SQLAlchemy ORM, sessions, relationships | Pending |
| Authentication (API keys) | High | Hashing, middleware, dependencies | Pending |
| Multi-tenancy | High | Class composition, filtering | Pending |
| Redis caching | Medium | Decorators, serialization, TTL | Pending |
| Async document processing | Medium | Celery, async/await, queues | Pending |
| Prometheus metrics | Medium | Decorators, counters, histograms | Pending |
| Kubernetes manifests | Medium | YAML, environment configs | Pending |
| CI/CD pipeline | Medium | GitHub Actions, pytest | Pending |

**Milestone Project**: Production-ready multi-tenant API.

### Phase 4: Advanced - Expert Python

**Python Concepts You'll Learn:**
- Metaclasses and descriptors
- Custom decorators with arguments
- Performance profiling (cProfile)
- Memory optimization
- Plugin architecture
- Protocol classes (structural typing)
- Concurrent.futures
- Context variables

| Task | Priority | Python Concepts | Status |
|------|----------|-----------------|--------|
| Hybrid search | Medium | Protocol classes, composition | Pending |
| Document-level ACL | Medium | Permission decorators, middleware | Pending |
| Conversation history | Low | State management, sessions | Pending |
| Additional connectors | Low | ABC, Factory pattern, plugins | Pending |
| PII detection | Low | Regex, streaming, generators | Pending |
| A/B testing framework | Low | Random, statistics, decorators | Pending |
| Performance optimization | Low | cProfile, memory_profiler | Pending |

**Milestone Project**: Extensible, high-performance RAG system.

---

### Learning Checkpoints

After each phase, you should be able to answer these questions:

**Phase 1 Checkpoint:**
- [ ] What's the difference between `list` and `generator`?
- [ ] Why use type hints?
- [ ] How do context managers work?
- [ ] What is `__init__.py` for?

**Phase 2 Checkpoint:**
- [ ] How does dependency injection improve testability?
- [ ] What's the difference between `@staticmethod` and `@classmethod`?
- [ ] How do you mock external services in tests?
- [ ] What's the purpose of Pydantic validators?

**Phase 3 Checkpoint:**
- [ ] When should you use async vs threading vs multiprocessing?
- [ ] How does SQLAlchemy manage database sessions?
- [ ] What's the difference between Factory and Strategy patterns?
- [ ] How does Redis caching improve performance?

**Phase 4 Checkpoint:**
- [ ] When would you use metaclasses?
- [ ] How do you profile Python code for bottlenecks?
- [ ] What's structural typing (Protocols)?
- [ ] How do you design a plugin system?

---

## 12. Quick Start Guide

### Prerequisites

```bash
# Required software
- Python 3.11+
- Docker & Docker Compose
- Git
```

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/sujith/rag.git
cd rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure (Qdrant, Redis, PostgreSQL)
docker-compose -f docker/docker-compose.yml up -d

# Run database migrations
python scripts/migrate.py

# Start the API server
uvicorn src.rag.api:app --reload --port 8000

# Or use the CLI
python -m rag.cli add ./documents/
python -m rag.cli ask "What is RAG?"
```

### Example API Usage

```python
import requests

# Ingest a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/ingest",
        files={"file": f},
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
print(response.json())

# Query the knowledge base
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"question": "What is RAG?", "top_k": 5},
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
print(response.json())
```

### Sample RAG Query Implementation

```python
def query_rag(question: str) -> dict:
    """Complete RAG query with re-ranking and citations."""
    client, model, groq = get_clients()
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        # 1. Embed query
        query_vector = model.encode(question).tolist()

        # 2. Initial retrieval (more candidates)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10  # Retrieve more
        )

        if not results.points:
            print("\nNo relevant documents found.")
            continue

        # 3. Re-rank with cross-encoder
        pairs = [(question, hit.payload['text']) for hit in results.points]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(results.points, scores), key=lambda x: x[1], reverse=True)
        top_results = [hit for hit, _ in ranked[:3]]

        # 4. Build context with citations
        context_parts = []
        sources = []
        for i, hit in enumerate(top_results, 1):
            source = hit.payload.get('source', 'Unknown')
            sources.append(f"[{i}] {source}")
            context_parts.append(f"[Source {i}]: {hit.payload['text']}")

        context = "\n\n".join(context_parts)

        # 5. Enhanced prompt
        prompt = f"""You are a helpful assistant. Answer based on the provided sources.
Cite sources using [1], [2], etc. If information is insufficient, say so.

Sources:
{context}

Question: {question}

Answer:"""

        # 6. Generate response
        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3  # Lower for more focused answers
        )

        answer = response.choices[0].message.content

        # 7. Display with sources
        print(f"\nAssistant: {answer}")
        print("\n  Sources:")
        for s in sources:
            print(f"    {s}")
```

---

## 13. Enterprise Infrastructure Details

Detailed specifications for enterprise-grade deployment.

### 13.1 Security & Compliance

**Security Requirements:**
- **Authentication**: OAuth2/OIDC, API keys with rotation, SSO integration
- **Authorization**: Role-based access control (RBAC), document-level permissions
- **Encryption**: Data at rest (AES-256), in transit (TLS 1.3), key management
- **Audit Logging**: Who accessed what, when, immutable logs
- **Compliance**: GDPR (right to deletion), SOC2, HIPAA (if healthcare)
- **PII Detection**: Automatically detect and mask sensitive data
- **Data Residency**: Control where data is stored geographically

```python
# Example: Document-level access control
class DocumentACL:
    def __init__(self):
        self.permissions = {}  # doc_id -> {user_ids, group_ids, roles}

    def can_access(self, user_id: str, doc_id: str) -> bool:
        """Check if user can access document."""
        user_groups = self.get_user_groups(user_id)
        doc_perms = self.permissions.get(doc_id, {})

        return (
            user_id in doc_perms.get("users", []) or
            any(g in doc_perms.get("groups", []) for g in user_groups) or
            self.user_has_role(user_id, doc_perms.get("roles", []))
        )

    def filter_search_results(self, user_id: str, results: list) -> list:
        """Filter results to only accessible documents."""
        return [r for r in results if self.can_access(user_id, r.payload["doc_id"])]
```

### 13.2 Multi-Tenancy

The system supports isolated data per organization/customer.

**Architecture Options:**

| Approach | Isolation | Cost | Complexity |
|----------|-----------|------|------------|
| Separate Qdrant instances | Highest | High | Medium |
| Separate collections per tenant | High | Medium | Low |
| Shared collection with tenant_id filter | Lower | Low | Low |

```python
# Collection-per-tenant approach
def get_collection_name(tenant_id: str) -> str:
    return f"kb_{tenant_id}"

def search_tenant_docs(tenant_id: str, query: str, client, model):
    """Search within tenant's isolated collection."""
    collection = get_collection_name(tenant_id)
    query_vector = model.encode(query).tolist()

    return client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10
    )
```

### 13.3 API Layer

REST API with authentication and async processing.

**FastAPI Implementation:**
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uuid

app = FastAPI(title="Enterprise RAG API", version="1.0.0")
security = HTTPBearer()

class QueryRequest(BaseModel):
    question: str
    filters: dict = None
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query_id: str
    latency_ms: float
    tokens_used: int

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks
):
    """Query the knowledge base with authentication."""
    # Validate token and get user/tenant
    user = await validate_token(credentials.credentials)

    query_id = str(uuid.uuid4())
    start_time = time.time()

    # Execute RAG pipeline
    results = await rag_query(
        question=request.question,
        tenant_id=user.tenant_id,
        user_id=user.id,
        filters=request.filters,
        top_k=request.top_k
    )

    latency_ms = (time.time() - start_time) * 1000

    # Log query asynchronously
    background_tasks.add_task(
        log_query, query_id, user.id, request.question, latency_ms
    )

    return QueryResponse(
        answer=results["answer"],
        sources=results["sources"] if request.include_sources else [],
        query_id=query_id,
        latency_ms=latency_ms,
        tokens_used=results["tokens_used"]
    )

@app.post("/api/v1/documents/ingest")
async def ingest_document(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks
):
    """Ingest document asynchronously."""
    user = await validate_token(credentials.credentials)
    job_id = str(uuid.uuid4())

    # Queue for async processing
    background_tasks.add_task(
        process_document_async, job_id, file, user.tenant_id
    )

    return {"job_id": job_id, "status": "processing"}

@app.get("/api/v1/documents/ingest/{job_id}/status")
async def get_ingest_status(job_id: str):
    """Check document ingestion status."""
    return await get_job_status(job_id)
```

### 13.4 Scalability & High Availability

Clustered infrastructure with auto-scaling.

**Infrastructure (docker-compose.enterprise.yml):**
```yaml
# docker-compose.enterprise.yml
services:
  # Load Balancer
  traefik:
    image: traefik:v2.10
    ports:
      - "443:443"
    volumes:
      - ./traefik.yml:/etc/traefik/traefik.yml

  # RAG API (multiple replicas)
  rag-api:
    image: rag-api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
    environment:
      - QDRANT_URL=qdrant-cluster:6333
      - REDIS_URL=redis:6379

  # Qdrant Cluster
  qdrant-node-1:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    volumes:
      - qdrant-data-1:/qdrant/storage

  qdrant-node-2:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    volumes:
      - qdrant-data-2:/qdrant/storage

  # Redis for caching & queues
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  # Async worker for document processing
  celery-worker:
    image: rag-api:latest
    command: celery -A tasks worker -l info
    deploy:
      replicas: 2
```

### 13.5 Observability Stack

Full observability with logs, metrics, and traces.

**Implementation:**
```python
import logging
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total queries', ['tenant', 'status'])
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Query latency', ['tenant'])
ACTIVE_QUERIES = Gauge('rag_active_queries', 'Active queries')
RETRIEVAL_SCORE = Histogram('rag_retrieval_score', 'Top retrieval scores')
TOKEN_USAGE = Counter('rag_tokens_total', 'Tokens used', ['tenant', 'type'])

# Tracing
tracer = trace.get_tracer(__name__)

# Structured logging
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

class ObservableRAG:
    def query(self, question: str, tenant_id: str) -> dict:
        ACTIVE_QUERIES.inc()

        with tracer.start_as_current_span("rag_query") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("question_length", len(question))

            start_time = time.time()

            try:
                # Embedding span
                with tracer.start_as_current_span("embed_query"):
                    query_vector = self.embed(question)

                # Retrieval span
                with tracer.start_as_current_span("retrieve_documents"):
                    results = self.search(query_vector, tenant_id)
                    RETRIEVAL_SCORE.observe(results[0].score if results else 0)

                # Generation span
                with tracer.start_as_current_span("generate_answer"):
                    answer, tokens = self.generate(question, results)
                    TOKEN_USAGE.labels(tenant=tenant_id, type="completion").inc(tokens)

                latency = time.time() - start_time
                QUERY_LATENCY.labels(tenant=tenant_id).observe(latency)
                QUERY_COUNTER.labels(tenant=tenant_id, status="success").inc()

                logger.info("Query completed", extra={
                    "tenant_id": tenant_id,
                    "latency_ms": latency * 1000,
                    "num_results": len(results),
                    "tokens_used": tokens
                })

                span.set_status(Status(StatusCode.OK))
                return {"answer": answer, "sources": results}

            except Exception as e:
                QUERY_COUNTER.labels(tenant=tenant_id, status="error").inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("Query failed", extra={"error": str(e), "tenant_id": tenant_id})
                raise

            finally:
                ACTIVE_QUERIES.dec()
```

**Grafana Dashboard Metrics:**
- Query rate (queries/sec by tenant)
- P50/P95/P99 latency
- Error rate
- Retrieval quality scores
- Token usage and costs
- Active queries gauge
- Document count per tenant

### 13.6 Data Source Connectors

Support for 50+ data sources.

**Supported Connectors:**
| Category | Sources |
|----------|---------|
| Cloud Storage | S3, GCS, Azure Blob, Dropbox |
| Documents | SharePoint, Google Drive, Confluence, Notion |
| Databases | PostgreSQL, MongoDB, Snowflake |
| Communication | Slack, Teams, Email (IMAP) |
| CRM/Support | Salesforce, Zendesk, Intercom |
| Code | GitHub, GitLab, Bitbucket |
| Web | Website crawler, Sitemap |

```python
# Connector interface
from abc import ABC, abstractmethod

class DataConnector(ABC):
    @abstractmethod
    async def list_documents(self) -> list[str]:
        """List available documents."""
        pass

    @abstractmethod
    async def fetch_document(self, doc_id: str) -> Document:
        """Fetch document content."""
        pass

    @abstractmethod
    async def get_changes_since(self, timestamp: datetime) -> list[str]:
        """Get documents changed since timestamp (for incremental sync)."""
        pass

# Example: Confluence connector
class ConfluenceConnector(DataConnector):
    def __init__(self, base_url: str, api_token: str, space_key: str):
        self.client = ConfluenceClient(base_url, api_token)
        self.space_key = space_key

    async def list_documents(self) -> list[str]:
        pages = await self.client.get_all_pages(self.space_key)
        return [p["id"] for p in pages]

    async def fetch_document(self, doc_id: str) -> Document:
        page = await self.client.get_page(doc_id, expand="body.storage")
        return Document(
            id=doc_id,
            title=page["title"],
            content=html_to_text(page["body"]["storage"]["value"]),
            metadata={
                "source": "confluence",
                "space": self.space_key,
                "url": page["_links"]["webui"],
                "last_modified": page["version"]["when"],
                "author": page["version"]["by"]["displayName"]
            }
        )
```

### 13.7 Evaluation & Feedback Loop

Continuous quality monitoring and improvement.

**Components:**
```python
# 1. Automated evaluation
class RAGEvaluator:
    def __init__(self, test_set: list[dict]):
        """
        test_set = [
            {"question": "...", "expected_answer": "...", "relevant_docs": [...]}
        ]
        """
        self.test_set = test_set

    def evaluate(self, rag_system) -> dict:
        metrics = {
            "retrieval_recall@5": [],
            "retrieval_mrr": [],
            "answer_relevance": [],  # LLM-as-judge
            "answer_faithfulness": [],  # Grounded in context?
            "answer_correctness": [],  # Matches expected?
        }

        for test in self.test_set:
            result = rag_system.query(test["question"])

            # Retrieval metrics
            retrieved_docs = [r.payload["source"] for r in result["sources"]]
            relevant = set(test["relevant_docs"])
            metrics["retrieval_recall@5"].append(
                len(set(retrieved_docs[:5]) & relevant) / len(relevant)
            )

            # Answer quality (using LLM-as-judge)
            metrics["answer_relevance"].append(
                self.judge_relevance(test["question"], result["answer"])
            )
            metrics["answer_faithfulness"].append(
                self.judge_faithfulness(result["answer"], result["context"])
            )

        return {k: sum(v)/len(v) for k, v in metrics.items()}

# 2. User feedback collection
@app.post("/api/v1/feedback")
async def submit_feedback(
    query_id: str,
    rating: int,  # 1-5
    feedback_type: str,  # "helpful", "incorrect", "incomplete", "outdated"
    comment: str = None
):
    """Collect user feedback for continuous improvement."""
    await store_feedback(query_id, rating, feedback_type, comment)

    # If negative feedback, queue for review
    if rating <= 2:
        await queue_for_human_review(query_id)

# 3. A/B testing
class ABTestManager:
    def __init__(self):
        self.experiments = {}

    def get_variant(self, user_id: str, experiment: str) -> str:
        """Deterministically assign user to variant."""
        hash_val = hash(f"{user_id}:{experiment}") % 100

        variants = self.experiments[experiment]["variants"]
        cumulative = 0
        for variant, percentage in variants.items():
            cumulative += percentage
            if hash_val < cumulative:
                return variant

        return "control"

# Usage
ab_manager = ABTestManager()
ab_manager.experiments["reranker"] = {
    "variants": {"control": 50, "cross_encoder": 50}
}

variant = ab_manager.get_variant(user_id, "reranker")
if variant == "cross_encoder":
    results = rerank_with_cross_encoder(results)
```

### 13.8 Cost Management

Full cost attribution and optimization.

```python
class CostTracker:
    # Approximate costs (as of 2024)
    COSTS = {
        "embedding": {
            "openai-ada-002": 0.0001 / 1000,  # per token
            "local": 0  # self-hosted
        },
        "llm": {
            "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
            "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
            "claude-3-sonnet": {"input": 0.003/1000, "output": 0.015/1000},
            "llama-3.1-8b": {"input": 0.0001/1000, "output": 0.0001/1000},  # Groq
        },
        "qdrant": {
            "cloud": 0.025 / 1000000,  # per vector per month
            "self_hosted": 0
        }
    }

    def track_query_cost(self, tenant_id: str, query_stats: dict):
        embedding_cost = query_stats["embedding_tokens"] * self.COSTS["embedding"]["local"]
        llm_cost = (
            query_stats["input_tokens"] * self.COSTS["llm"]["llama-3.1-8b"]["input"] +
            query_stats["output_tokens"] * self.COSTS["llm"]["llama-3.1-8b"]["output"]
        )

        total_cost = embedding_cost + llm_cost

        # Store for billing
        self.store_cost(tenant_id, total_cost, query_stats)

        return total_cost

    def get_monthly_cost(self, tenant_id: str) -> dict:
        """Get cost breakdown for tenant."""
        return {
            "total": self.get_total_cost(tenant_id),
            "breakdown": {
                "embeddings": self.get_embedding_cost(tenant_id),
                "llm_queries": self.get_llm_cost(tenant_id),
                "storage": self.get_storage_cost(tenant_id)
            },
            "query_count": self.get_query_count(tenant_id)
        }
```

### 13.9 Enterprise Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   CDN/WAF       │
                                    │   (Cloudflare)  │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Load Balancer  │
                                    │    (Traefik)    │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   RAG API #1    │     │   RAG API #2    │     │   RAG API #3    │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
        ┌──────────────┬─────────────────────┼─────────────────────┬──────────────┐
        │              │                     │                     │              │
┌───────▼───────┐ ┌────▼────┐ ┌──────────────▼──────────────┐ ┌────▼────┐ ┌───────▼───────┐
│    Redis      │ │  Kafka  │ │      Qdrant Cluster         │ │ Postgres│ │  Object Store │
│ (Cache/Queue) │ │ (Events)│ │  ┌─────┐ ┌─────┐ ┌─────┐   │ │(Metadata│ │    (S3)       │
└───────────────┘ └─────────┘ │  │Node1│ │Node2│ │Node3│   │ │ & Auth) │ │  (Documents)  │
                              │  └─────┘ └─────┘ └─────┘   │ └─────────┘ └───────────────┘
                              └────────────────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                     ┌────────▼────────┐          ┌────────▼────────┐
                     │  Celery Worker  │          │  Celery Worker  │
                     │ (Doc Processing)│          │ (Doc Processing)│
                     └─────────────────┘          └─────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                    Observability Stack                       │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
        │  │Prometheus│  │  Grafana │  │  Jaeger  │  │   Loki   │    │
        │  │ (Metrics)│  │(Dashboard)│  │ (Traces) │  │  (Logs)  │    │
        │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
        └─────────────────────────────────────────────────────────────┘
```

### 13.10 Infrastructure Components Explained

#### Redis - Caching & Job Queues
```python
import redis
from functools import lru_cache
import json
import hashlib

class RAGCache:
    """Redis-based caching for RAG system."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.DEFAULT_TTL = 3600  # 1 hour

    def _query_key(self, query: str, tenant_id: str) -> str:
        """Generate cache key for query."""
        hash_input = f"{tenant_id}:{query}"
        return f"rag:query:{hashlib.md5(hash_input.encode()).hexdigest()}"

    def get_cached_answer(self, query: str, tenant_id: str) -> dict | None:
        """Get cached answer if exists."""
        key = self._query_key(query, tenant_id)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def cache_answer(self, query: str, tenant_id: str, result: dict, ttl: int = None):
        """Cache query result."""
        key = self._query_key(query, tenant_id)
        self.redis.setex(key, ttl or self.DEFAULT_TTL, json.dumps(result))

    def invalidate_tenant_cache(self, tenant_id: str):
        """Invalidate all cached queries for a tenant (after doc update)."""
        pattern = f"rag:query:*"  # In production, use tenant-specific pattern
        for key in self.redis.scan_iter(pattern):
            self.redis.delete(key)

# Usage in RAG query
cache = RAGCache()

def query_with_cache(question: str, tenant_id: str) -> dict:
    # Check cache first
    cached = cache.get_cached_answer(question, tenant_id)
    if cached:
        return {**cached, "cached": True}

    # Execute RAG pipeline
    result = execute_rag_pipeline(question, tenant_id)

    # Cache result
    cache.cache_answer(question, tenant_id, result)
    return {**result, "cached": False}
```

**Redis Use Cases in RAG:**
- Query result caching (avoid repeated LLM calls)
- Rate limiting per tenant/user
- Session storage for conversation history
- Job queue backend (Celery/RQ)
- Distributed locking for document processing

#### Kafka - Event Streaming
```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer - Publish events
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Events to publish
class RAGEventPublisher:
    def __init__(self, producer: KafkaProducer):
        self.producer = producer

    def document_ingested(self, tenant_id: str, doc_id: str, metadata: dict):
        """Publish when document is ingested."""
        self.producer.send('rag.documents.ingested', {
            'event': 'document_ingested',
            'tenant_id': tenant_id,
            'doc_id': doc_id,
            'metadata': metadata,
            'timestamp': datetime.utcnow().isoformat()
        })

    def query_executed(self, tenant_id: str, query_id: str, latency_ms: float):
        """Publish query metrics for analytics."""
        self.producer.send('rag.queries.executed', {
            'event': 'query_executed',
            'tenant_id': tenant_id,
            'query_id': query_id,
            'latency_ms': latency_ms,
            'timestamp': datetime.utcnow().isoformat()
        })

    def feedback_received(self, query_id: str, rating: int, feedback_type: str):
        """Publish user feedback."""
        self.producer.send('rag.feedback.received', {
            'event': 'feedback_received',
            'query_id': query_id,
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': datetime.utcnow().isoformat()
        })

# Consumer - Process events (separate service)
def analytics_consumer():
    consumer = KafkaConsumer(
        'rag.queries.executed',
        bootstrap_servers=['localhost:9092'],
        group_id='analytics-service',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    for message in consumer:
        event = message.value
        # Store in analytics database, update dashboards, etc.
        process_analytics_event(event)
```

**Kafka Use Cases in RAG:**
- Async document ingestion pipeline
- Query analytics and metrics streaming
- Cross-service communication
- Audit log streaming
- Real-time feedback processing

#### PostgreSQL - Metadata & Auth
```sql
-- Database schema for RAG metadata

-- Tenants/Organizations
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) DEFAULT 'free',  -- free, pro, enterprise
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',  -- admin, user, viewer
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,  -- Store hashed, not plain
    name VARCHAR(255),
    permissions JSONB DEFAULT '["read"]',
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents metadata (vectors stored in Qdrant)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    qdrant_collection VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000),
    file_hash VARCHAR(64),  -- For deduplication
    file_size_bytes BIGINT,
    chunk_count INTEGER,
    status VARCHAR(50) DEFAULT 'processing',  -- processing, ready, failed
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document access control
CREATE TABLE document_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permission VARCHAR(50) DEFAULT 'read',  -- read, write, admin
    granted_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, user_id)
);

-- Query history & analytics
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    query_text TEXT NOT NULL,
    response_text TEXT,
    sources JSONB,  -- [{doc_id, score, chunk_text}]
    latency_ms FLOAT,
    tokens_used INTEGER,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_log_id UUID REFERENCES query_logs(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_type VARCHAR(50),  -- helpful, incorrect, incomplete
    comment TEXT,
    reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_query_logs_tenant ON query_logs(tenant_id);
CREATE INDEX idx_query_logs_created ON query_logs(created_at);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
```

### 13.11 Kubernetes Deployment

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system

---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  QDRANT_HOST: "qdrant-service"
  QDRANT_PORT: "6333"
  REDIS_URL: "redis://redis-service:6379"
  LOG_LEVEL: "INFO"

---
# kubernetes/secrets.yaml (use sealed-secrets or external-secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
stringData:
  GROQ_API_KEY: "your-api-key-here"  # Use external secret manager!
  DATABASE_URL: "postgresql://user:pass@postgres-service:5432/rag"

---
# kubernetes/rag-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: rag-config
        - secretRef:
            name: rag-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
# kubernetes/rag-api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# kubernetes/rag-api-hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.rag.yourcompany.com
    secretName: rag-tls
  rules:
  - host: api.rag.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80

---
# kubernetes/celery-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
  namespace: rag-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
      - name: celery-worker
        image: your-registry/rag-api:latest
        command: ["celery", "-A", "tasks", "worker", "-l", "info", "-c", "4"]
        envFrom:
        - configMapRef:
            name: rag-config
        - secretRef:
            name: rag-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
```

### 13.12 CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yaml
name: RAG CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/rag-api

jobs:
  # ============================================
  # STAGE 1: Code Quality & Testing
  # ============================================
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_rag
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov ruff mypy

    - name: Lint with Ruff
      run: ruff check .

    - name: Type check with mypy
      run: mypy rag/ --ignore-missing-imports

    - name: Run unit tests
      run: pytest tests/unit -v --cov=rag --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_rag
        REDIS_URL: redis://localhost:6379
        QDRANT_HOST: localhost

    - name: Run integration tests
      run: pytest tests/integration -v
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_rag
        REDIS_URL: redis://localhost:6379
        QDRANT_HOST: localhost

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: coverage.xml

  # ============================================
  # STAGE 2: Security Scanning
  # ============================================
  security:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        severity: 'CRITICAL,HIGH'

    - name: Run Bandit security linter
      run: |
        pip install bandit
        bandit -r rag/ -ll

    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: ${{ github.event.repository.default_branch }}
        head: HEAD

  # ============================================
  # STAGE 3: Build & Push Docker Image
  # ============================================
  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=sha,prefix=
          type=ref,event=branch
          type=semver,pattern={{version}}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # ============================================
  # STAGE 4: Deploy to Staging
  # ============================================
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy to staging
      run: |
        kubectl set image deployment/rag-api \
          rag-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n rag-staging
        kubectl rollout status deployment/rag-api -n rag-staging

    - name: Run smoke tests
      run: |
        ./scripts/smoke-tests.sh https://staging-api.rag.yourcompany.com

  # ============================================
  # STAGE 5: Deploy to Production
  # ============================================
  deploy-production:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy to production (canary)
      run: |
        # Deploy to 10% of traffic first
        kubectl set image deployment/rag-api-canary \
          rag-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n rag-production
        kubectl rollout status deployment/rag-api-canary -n rag-production

    - name: Run production smoke tests
      run: |
        ./scripts/smoke-tests.sh https://api.rag.yourcompany.com

    - name: Full rollout
      run: |
        kubectl set image deployment/rag-api \
          rag-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n rag-production
        kubectl rollout status deployment/rag-api -n rag-production

    - name: Notify Slack
      uses: slackapi/slack-github-action@v1
      with:
        payload: |
          {
            "text": "RAG API deployed to production: ${{ github.sha }}"
          }
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 13.13 Security Review Process

#### Pre-Deployment Security Checklist

```markdown
## Security Review Checklist

### Code Security
- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All secrets loaded from environment/secret manager
- [ ] Input validation on all user inputs
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CSRF protection enabled
- [ ] Rate limiting implemented
- [ ] Authentication on all endpoints

### Infrastructure Security
- [ ] TLS 1.3 enforced
- [ ] Network policies restrict pod-to-pod communication
- [ ] Secrets encrypted at rest (Kubernetes secrets/Vault)
- [ ] Container images scanned for vulnerabilities
- [ ] Non-root container user
- [ ] Read-only root filesystem where possible
- [ ] Resource limits set (prevent DoS)

### Data Security
- [ ] PII detection implemented
- [ ] Data encryption at rest (Qdrant, PostgreSQL)
- [ ] Data encryption in transit (TLS)
- [ ] Audit logging enabled
- [ ] Data retention policies defined
- [ ] Backup encryption enabled

### Access Control
- [ ] RBAC implemented
- [ ] Document-level permissions working
- [ ] API key rotation mechanism
- [ ] Session timeout configured
- [ ] Failed login attempt limiting
```

#### Security Tools Integration

```python
# security/pii_detector.py
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIDetector:
    """Detect and optionally mask PII in documents."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def detect_pii(self, text: str) -> list[dict]:
        """Detect PII entities in text."""
        results = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "CREDIT_CARD", "US_SSN", "US_PASSPORT",
                "IP_ADDRESS", "LOCATION"
            ],
            language="en"
        )

        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end]
            }
            for r in results
            if r.score > 0.7  # Confidence threshold
        ]

    def mask_pii(self, text: str) -> str:
        """Mask PII in text before storage."""
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

# Usage in document ingestion
pii_detector = PIIDetector()

def ingest_document_secure(text: str, tenant_id: str):
    # Check for PII
    pii_found = pii_detector.detect_pii(text)

    if pii_found:
        # Log warning
        logger.warning(f"PII detected in document", extra={
            "tenant_id": tenant_id,
            "pii_types": [p["type"] for p in pii_found],
            "count": len(pii_found)
        })

        # Option 1: Reject document
        # raise ValueError("Document contains PII")

        # Option 2: Mask PII before storage
        text = pii_detector.mask_pii(text)

    # Continue with ingestion
    return process_document(text, tenant_id)
```

#### Compliance Automation

```python
# compliance/gdpr.py
from datetime import datetime, timedelta

class GDPRCompliance:
    """GDPR compliance utilities."""

    def __init__(self, db, qdrant_client):
        self.db = db
        self.qdrant = qdrant_client

    async def handle_deletion_request(self, user_id: str, tenant_id: str):
        """
        Handle GDPR Article 17 - Right to Erasure.
        Delete all user data within 30 days.
        """
        deletion_record = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "requested_at": datetime.utcnow(),
            "deadline": datetime.utcnow() + timedelta(days=30),
            "status": "pending"
        }

        # 1. Delete from PostgreSQL
        await self.db.execute(
            "DELETE FROM query_logs WHERE user_id = $1", user_id
        )
        await self.db.execute(
            "DELETE FROM feedback WHERE query_log_id IN "
            "(SELECT id FROM query_logs WHERE user_id = $1)", user_id
        )

        # 2. Delete user-uploaded documents from Qdrant
        user_docs = await self.db.fetch_all(
            "SELECT id, qdrant_collection FROM documents "
            "WHERE uploaded_by = $1", user_id
        )

        for doc in user_docs:
            # Delete vectors from Qdrant
            self.qdrant.delete(
                collection_name=doc["qdrant_collection"],
                points_selector={"filter": {"doc_id": doc["id"]}}
            )

        # 3. Delete document metadata
        await self.db.execute(
            "DELETE FROM documents WHERE uploaded_by = $1", user_id
        )

        # 4. Delete user account
        await self.db.execute("DELETE FROM users WHERE id = $1", user_id)

        # 5. Log for compliance audit
        await self.log_deletion(deletion_record)

        return {"status": "completed", "user_id": user_id}

    async def export_user_data(self, user_id: str) -> dict:
        """
        Handle GDPR Article 20 - Right to Data Portability.
        Export all user data in machine-readable format.
        """
        user = await self.db.fetch_one(
            "SELECT * FROM users WHERE id = $1", user_id
        )

        queries = await self.db.fetch_all(
            "SELECT * FROM query_logs WHERE user_id = $1", user_id
        )

        documents = await self.db.fetch_all(
            "SELECT * FROM documents WHERE uploaded_by = $1", user_id
        )

        return {
            "user": dict(user) if user else None,
            "queries": [dict(q) for q in queries],
            "documents": [dict(d) for d in documents],
            "exported_at": datetime.utcnow().isoformat()
        }
```

### 13.14 Enterprise RAG Checklist

| Category | Requirement | Priority |
|----------|-------------|----------|
| **Security** | | |
| | API authentication (OAuth2/API keys) | Critical |
| | Document-level access control | Critical |
| | Encryption at rest & in transit | Critical |
| | Audit logging | Critical |
| | PII detection & masking | High |
| **Scalability** | | |
| | Horizontal API scaling | High |
| | Qdrant clustering | High |
| | Async document processing | High |
| | Query result caching | Medium |
| **Reliability** | | |
| | Health checks & readiness probes | Critical |
| | Automated backups | Critical |
| | Disaster recovery plan | High |
| | Circuit breakers for external services | Medium |
| **Observability** | | |
| | Structured logging | High |
| | Metrics & dashboards | High |
| | Distributed tracing | Medium |
| | Alerting | High |
| **Multi-tenancy** | | |
| | Tenant isolation | Critical |
| | Per-tenant configuration | Medium |
| | Usage quotas & rate limiting | High |
| **Data Management** | | |
| | Incremental sync | High |
| | Document versioning | Medium |
| | Data retention policies | High |
| **Quality** | | |
| | Automated evaluation pipeline | High |
| | User feedback collection | High |
| | A/B testing framework | Medium |
| **Operations** | | |
| | CI/CD pipeline | High |
| | Infrastructure as Code | High |
| | Runbooks & documentation | Medium |


## 13.15 Production Runbook

### Common Issues & Solutions

This runbook provides step-by-step troubleshooting for production incidents.

---

#### Issue #1: Query Latency Spike

**Symptoms:**
- P95 latency > 3 seconds (target: < 2s)
- User complaints about slow responses
-Dashboard shows latency spike

**Diagnosis:**

```bash
# 1. Check current latency
kubectl logs -f deployment/rag-api | grep "latency_ms"

# 2. Check Qdrant status
curl http://qdrant:6333/health
curl http://qdrant:6333/metrics

# 3. Check LLM API rate limits
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/v1/status

# 4. Check cache hit rate
redis-cli INFO stats | grep keyspace_hits
```

**Common Causes & Fixes:**

| Cause | Check | Fix |
|-------|-------|-----|
| Qdrant index degraded | Check collection status | Rebuild index |
| LLM API rate limited | Check API logs | Scale down QPS or upgrade plan |
| Low cache hit rate | Redis stats | Increase TTL, warm cache |
| Reranker bottleneck | Profile query | Skip reranking for simple queries |
| Memory pressure | `kubectl top pods` | Scale up memory limits |

**Quick Fix:**
```bash
# Temporarily increase cache TTL to reduce load
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Scale up API replicas
kubectl scale deployment/rag-api --replicas=5

# If Qdrant is slow, restart to clear memory
kubectl rollout restart deployment/qdrant
```

---

#### Issue #2: Low Retrieval Quality

**Symptoms:**
- Users report irrelevant answers
- Feedback ratings drop below 3.5/5
- Retrieval recall@5 < 0.5

**Diagnosis:**

```python
# Run evaluation suite
python scripts/evaluate.py --test-set data/eval/golden_set.json

# Check recent queries with low scores
SELECT query_text, rating, sources 
FROM query_logs 
JOIN feedback ON query_logs.id = feedback.query_log_id
WHERE rating <= 2 
ORDER BY created_at DESC 
LIMIT 20;
```

**Common Causes & Fixes:**

1. **Outdated/Missing Documents**
   ```bash
   # Check when documents were last updated
   SELECT source, MAX(created_at) as last_update
   FROM documents
   GROUP BY source
   ORDER BY last_update ASC;
   
   # Re-index old documents
   python scripts/reindex_old_docs.py --older-than 30d
   ```

2. **Poor Chunking**
   ```python
   # Inspect problematic chunks
   python scripts/inspect_chunks.py --source "problematic_doc.pdf"
   
   # Re-chunk with better strategy
   python scripts/rechunk.py --strategy sentence-aware --chunk-size 500
   ```

3. **Embedding Model Mismatch**
   ```python
   # Check current model version
   curl http://rag-api:8000/health | jq '.embedding_model'
   
   # If upgraded model, reindex all documents
   python scripts/migrate_embeddings.py --new-model "bge-large-en-v1.5"
   ```

**Rollback Plan:**
```bash
# Rollback to previous deployment
kubectl rollout undo deployment/rag-api

# Restore previous Qdrant snapshot
./scripts/restore_qdrant_snapshot.sh latest-good-snapshot.tar.gz
```

---

#### Issue #3: High Error Rate

**Symptoms:**
- 5xx errors > 1%
- Alerts firing for error rate
- Sentry showing exceptions

**Diagnosis:**

```bash
# Check error logs
kubectl logs -f deployment/rag-api --tail=100 | grep ERROR

# Check error breakdown
curl http://rag-api:8000/metrics | grep rag_errors_total

# Check dependencies
curl http://qdrant:6333/health      # Should return 200
curl http://postgres:5432           # Should connect
redis-cli PING                       # Should return PONG
```

**Common Errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `QdrantException: Connection timeout` | Qdrant unreachable | Check network, restart Qdrant |
| `GroqAPIError: Rate limit exceeded` | Too many LLM calls | Enable caching, reduce QPS |
| `DatabaseError: Too many connections` | Connection pool exhausted | Increase pool size |
| `ValidationError: Question too long` | Input validation | Check client-side validation |

**Emergency Fix:**
```bash
# Enable circuit breaker to fail fast
kubectl set env deployment/rag-api CIRCUIT_BREAKER_ENABLED=true

# Redirect traffic to healthy pods only
kubectl set env deployment/rag-api HEALTH_CHECK_STRICT=true
```

---

#### Issue #4: High Memory Usage / OOM Kills

**Symptoms:**
- Pods restarting with `OOMKilled` status
- Memory usage > 90%
- Queries fail intermittently

**Diagnosis:**

```bash
# Check memory usage
kubectl top pods -n rag-system

# Check OOM kills
kubectl get pods -n rag-system | grep OOMKilled

# Check memory leaks
kubectl exec -it rag-api-xxx -- python scripts/memory_profile.py
```

**Fixes:**

1. **Increase Memory Limits**
   ```bash
   kubectl set resources deployment/rag-api \
     --limits=memory=8Gi \
     --requests=memory=4Gi
   ```

2. **Enable Batch Processing**
   ```yaml
   # In deployment.yaml
   env:
   - name: BATCH_SIZE
     value: "16"  # Reduce from 32
   - name: MAX_CONCURRENT_QUERIES
     value: "10"  # Limit concurrency
   ```

3. **Clear Model Cache**
   ```python
   # In application
   import gc
   import torch
   
   # After batch processing
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

---

#### Issue #5: Database Connection Issues

**Symptoms:**
- `FATAL: too many clients already`
- `OperationalError: could not connect to server`
- Query logs not being saved

**Diagnosis:**

```bash
# Check PostgreSQL connections
kubectl exec -it postgres-0 -- psql -U rag -c \
  "SELECT count(*) FROM pg_stat_activity;"

# Check connection pool status
curl http://rag-api:8000/debug/db_pool
```

**Fixes:**

1. **Increase PostgreSQL max_connections**
   ```sql
   ALTER SYSTEM SET max_connections = 200;  -- Was 100
   SELECT pg_reload_conf();
   ```

2. **Adjust Connection Pool**
   ```python
   # In database.py
   engine = create_engine(
       DATABASE_URL,
       pool_size=10,        # Was 20
       max_overflow=5,      # Was 10
       pool_pre_ping=True,  # Verify connections
       pool_recycle=3600    # Recycle after 1 hour
   )
   ```

3. **Close Idle Connections**
   ```sql
   -- Kill idle connections older than 10 minutes
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE state = 'idle'
     AND state_change < NOW() - INTERVAL '10 minutes';
   ```

---

### Monitoring Dashboards

#### Key Metrics to Watch

**Grafana Dashboard: RAG System Health**

```promql
# Query Rate
rate(rag_queries_total[5m])

# Latency (P50, P95, P99)
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))

# Error Rate
rate(rag_errors_total[5m]) / rate(rag_queries_total[5m]) * 100

# Cache Hit Rate
rate(rag_cache_hits_total[5m]) / rate(rag_cache_requests_total[5m]) * 100

# Retrieval Quality (average score)
avg(rag_retrieval_score)

# Active Queries
rag_active_queries
```

**Alert Rules:**

```yaml
# prometheus/alerts.yaml
groups:
- name: rag_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "RAG P95 latency above 3s"
  
  - alert: HighErrorRate
    expr: rate(rag_errors_total[5m]) / rate(rag_queries_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "RAG error rate above 5%"
  
  - alert: LowRetrievalQuality
    expr: avg(rag_retrieval_score) < 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "RAG retrieval scores below 0.5"
```

---

### Maintenance Procedures

#### Weekly Maintenance

```bash
#!/bin/bash
# scripts/weekly_maintenance.sh

# 1. Clear old logs (older than 30 days)
kubectl exec -it postgres-0 -- psql -U rag -c \
  "DELETE FROM query_logs WHERE created_at < NOW() - INTERVAL '30 days';"

# 2. Vacuum PostgreSQL
kubectl exec -it postgres-0 -- psql -U rag -c "VACUUM ANALYZE;"

# 3. Check Qdrant disk usage
kubectl exec -it qdrant-0 -- df -h /qdrant/storage

# 4. Backup databases
./scripts/backup_postgres.sh
./scripts/backup_qdrant.sh

# 5. Review low-rated queries
python scripts/review_feedback.py --rating-below 3
```

#### Monthly Maintenance

```bash
# 1. Update embedding models (if new version available)
python scripts/check_model_updates.py

# 2. Re-evaluate with test set
python scripts/evaluate.py --test-set data/eval/golden_set.json --report

# 3. Analyze cost trends
python scripts/cost_analysis.py --month $(date +%Y-%m)

# 4. Review and rotate API keys
python scripts/rotate_api_keys.py --older-than 90d

# 5. Update dependencies
pip install -U sentence-transformers groq qdrant-client
pytest  # Run full test suite
```

---

### Disaster Recovery

#### Scenario: Complete Qdrant Data Loss

```bash
#!/bin/bash
# scripts/restore_from_disaster.sh

echo "⚠️  DISASTER RECOVERY: Restoring RAG system from backups"

# 1. Restore PostgreSQL (has document metadata)
./scripts/restore_postgres_backup.sh latest

# 2. Get list of all documents from PostgreSQL
psql -U rag -c "COPY (SELECT id, file_path FROM documents) TO '/tmp/docs.csv' CSV;"

# 3. Recreate Qdrant collection
python scripts/init_qdrant_collection.py --force

# 4. Re-process all documents (slow but comprehensive)
python scripts/bulk_reindex.py \
  --input /tmp/docs.csv \
  --parallel 4 \
  --batch-size 16

echo "✓ Recovery complete. Run evaluation to verify quality."
```

**Recovery Time Objective (RTO):** 4 hours for 1M documents  
**Recovery Point Objective (RPO):** 24 hours (daily backups)

---

### Escalation Matrix

| Severity | Response Time | Escalate To |
|----------|---------------|-------------|
| **P0 - Critical** (Service down) | 15 minutes | Engineering Lead + On-call |
| **P1 - High** (Major degradation) | 1 hour | On-call engineer |
| **P2 - Medium** (Minor issues) | 4 hours | Engineering team |
| **P3 - Low** (Feature request) | Next sprint | Product manager |

**Communication Channels:**
- PagerDuty: Critical alerts
- Slack #rag-alerts: All alerts
- Status page: Public incidents

---


---

---


## 14.1 Cost Estimation Examples

### Monthly Cost Breakdown

This section provides realistic cost estimates for running a RAG system at different scales.

---

### Scenario 1: Small Team (100 queries/day)

**Usage:**
- 100 queries/day × 30 days = 3,000 queries/month
- Average 10 chunks per query
- Average 500 tokens per LLM response

**Infrastructure Costs:**

| Component | Option | Monthly Cost |
|-----------|--------|--------------|
| **Embedding** | Self-hosted (all-MiniLM-L6-v2) | $0 |
| | OpenAI ada-002 | ~$0.30 |
| **LLM** | Groq (llama-3.1-8b-instant) | ~$0.50 |
| | OpenAI gpt-3.5-turbo | ~$15 |
| | OpenAI gpt-4-turbo | ~$90 |
| **Vector DB** | Qdrant Cloud (1M vectors) | $25 |
| | Self-hosted Qdrant (1 vCPU, 2GB RAM) | ~$10 |
| **PostgreSQL** | Managed (small instance) | $15 |
| **Redis** | Managed (256MB) | $5 |
| **Total (Budget)** | Self-hosted + Groq | **~$30/month** |
| **Total (Premium)** | Managed + GPT-4 | **~$135/month** |

**Cost Per Query:**
- Budget: $0.01/query
- Premium: $0.045/query

---

### Scenario 2: Startup (1,000 queries/day)

**Usage:**
- 1,000 queries/day × 30 days = 30,000 queries/month
- 50% cache hit rate (effective: 15,000 LLM calls)
- Average 800 tokens per response

**Infrastructure Costs:**

| Component | Option | Monthly Cost |
|-----------|--------|--------------|
| **Embedding** | Self-hosted (bge-large-en-v1.5) | $0 |
| **LLM** | Groq (llama-3.1-70b-versatile) | ~$50 |
| | OpenAI gpt-4-turbo | ~$900 |
| **Vector DB** | Qdrant Cloud (5M vectors, replicated) | $95 |
| **PostgreSQL** | Managed (medium instance) | $50 |
| **Redis** | Managed (2GB, for caching) | $30 |
| **Monitoring** | Datadog / Grafana Cloud | $50 |
| **Total (Budget)** | Groq + Managed services | **~$275/month** |
| **Total (Premium)** | GPT-4 + Full observability | **~$1,125/month** |

**Cost Per Query:**
- Budget: $0.009/query (with caching)
- Premium: $0.038/query

**Cost Optimization Tips:**
```python
# 1. Aggressive caching
cache_ttl = 3600  # 1 hour
# Savings: ~50% reduction in LLM costs

# 2. Deduplication
if query_hash in recent_queries:  # Within 5 minutes
    return cached_result
# Savings: ~10-20% reduction

# 3. Use smaller model for simple queries
if len(query.split()) < 10 and not requires_reasoning(query):
    model = "llama-3.1-8b-instant"  # 10x cheaper
else:
    model = "llama-3.1-70b-versatile"
# Savings: ~30% reduction
```

---

### Scenario 3: Enterprise (10,000 queries/day)

**Usage:**
- 10,000 queries/day × 30 days = 300,000 queries/month
- 70% cache hit rate (effective: 90,000 LLM calls)
- Multi-tenant (50 organizations)
- 99.9% SLA requirement

**Infrastructure Costs:**

| Component | Specification | Monthly Cost |
|-----------|---------------|--------------|
| **Kubernetes Cluster** | 3 nodes (4 vCPU, 16GB each) | $300 |
| **Embedding service** | 2 GPU instances (T4) | $400 |
| **LLM API** | Groq (mixed models) | $400 |
| | OpenAI GPT-4 fallback | $600 |
| **Qdrant Cluster** | 3 nodes (50M vectors, HA) | $600 |
| **PostgreSQL** | Multi-AZ, 4 vCPU, 32GB | $200 |
| **Redis Cluster** | 3 nodes, 16GB total | $150 |
| **Load Balancer** | Managed ALB | $25 |
| **Monitoring & Logging** | Datadog | $200 |
| **Backups & Storage** | S3 + snapshots | $100 |
| **WAF & Security** | Cloudflare Enterprise | $200 |
| **Total Monthly** | | **~$3,175/month** |

**Cost Per Query:** $0.011/query

**Annual Cost:** ~$38,000/year

**Cost Attribution by Service:**
```python
{
    "llm_api": 31%,        # $1,000/month
    "infrastructure": 28%,  # $900/month
    "vector_db": 19%,      # $600/month
    "embedding": 13%,      # $400/month
    "monitoring": 6%,      # $200/month
    "other": 3%            # $75/month
}
```

---

### Cost Comparison by Provider

**LLM Costs for 1M tokens (Input + Output):**

| Provider | Model | Input ($/1M) | Output ($/1M) | Use Case |
|----------|-------|--------------|---------------|----------|
| **Groq** | llama-3.1-8b-instant | $0.05 | $0.08 | Fast, cheap |
| **Groq** | llama-3.1-70b-versatile | $0.59 | $0.79 | Better quality |
| **OpenAI** | gpt-3.5-turbo | $0.50 | $1.50 | Baseline |
| **OpenAI** | gpt-4-turbo | $10 | $30 | Highest quality |
| **Anthropic** | claude-3-haiku | $0.25 | $1.25 | Fast Claude |
| **Anthropic** | claude-3-sonnet | $3 | $15 | Best reasoning |

**Example: 100k queries/month**
- Average: 2k input tokens, 500 output tokens
- Total: 200M input + 50M output

| Provider | Model | Monthly Cost |
|----------|-------|--------------|
| Groq | llama-3.1-8b | $10 + $4 = **$14** |
| Groq | llama-3.1-70b | $118 + $40 = **$158** |
| OpenAI | gpt-3.5-turbo | $100 + $75 = **$175** |
| OpenAI | gpt-4-turbo | $2,000 + $1,500 = **$3,500** |

---

### Embedding Model Costs

**Self-Hosted (One-time setup):**

| Model | Size | GPU Required | Setup Cost | Monthly (compute) |
|-------|------|--------------|------------|-------------------|
| all-MiniLM-L6-v2 | 90MB | No | $0 | ~$20 (CPU) |
| all-mpnet-base-v2 | 420MB | No | $0 | ~$30 (CPU) |
| bge-large-en-v1.5 | 1.3GB | Recommended | $0 | ~$100 (T4 GPU) |

**API-Based:**

| Provider | Model | Cost per 1M tokens | Monthly (1M docs) |
|----------|-------|-------------------|-------------------|
| OpenAI | text-embedding-ada-002 | $0.10 | ~$100 |
| OpenAI | text-embedding-3-large | $0.13 | ~$130 |
| Cohere | embed-english-v3.0 | $0.10 | ~$100 |
| Voyage AI | voyage-2 | $0.12 | ~$120 |

**Recommendation:** Self-host for >10k queries/day to save costs.

---

### Vector Database Costs

**Qdrant Cloud Pricing (as of 2024):**

| Vectors | RAM | Storage | Replicas | Monthly Cost |
|---------|-----|---------|----------|--------------|
| 1M | 2GB | 5GB | 1 | $25 |
| 5M | 8GB | 20GB | 1 | $95 |
| 10M | 16GB | 40GB | 2 (HA) | $380 |
| 50M | 64GB | 200GB | 3 (HA) | $1,900 |

**Self-Hosted (AWS EC2):**

| Vectors | Instance | Monthly Cost |
|---------|----------|--------------|
| 1M | t3.small (2GB) | ~$15 |
| 10M | t3.xlarge (16GB) | ~$120 |
| 50M | r6g.2xlarge (64GB) | ~$380 |

**Recommendation:** 
- <5M vectors: Use Qdrant Cloud (simpler)
- >10M vectors: Self-host (cost-effective)

---

### Total Cost Calculator

```python
class RAGCostCalculator:
    """Calculate monthly RAG costs by usage."""
    
    COSTS = {
        "llm": {
            "groq-8b": {"input": 0.05/1_000_000, "output": 0.08/1_000_000},
            "groq-70b": {"input": 0.59/1_000_000, "output": 0.79/1_000_000},
            "gpt-3.5": {"input": 0.50/1_000_000, "output": 1.50/1_000_000},
            "gpt-4": {"input": 10/1_000_000, "output": 30/1_000_000},
        },
        "embedding": {
            "self-hosted": 0,
            "openai-ada": 0.10/1_000_000,
        },
        "qdrant_cloud": {
            1_000_000: 25,
            5_000_000: 95,
            10_000_000: 380,
        }
    }
    
    def calculate(
        self,
        queries_per_day: int,
        llm_model: str = "groq-8b",
        cache_hit_rate: float = 0.5,
        avg_input_tokens: int = 2000,
        avg_output_tokens: int = 500,
        total_vectors: int = 1_000_000,
        use_managed_db: bool = True
    ) -> dict:
        """Calculate monthly costs."""
        
        # Queries per month
        monthly_queries = queries_per_day * 30
        
        # Effective LLM calls (after cache)
        effective_llm_calls = monthly_queries * (1 - cache_hit_rate)
        
        # LLM costs
        llm_costs = self.COSTS["llm"][llm_model]
        llm_cost = (
            effective_llm_calls * avg_input_tokens * llm_costs["input"] +
            effective_llm_calls * avg_output_tokens * llm_costs["output"]
        )
        
        # Vector DB cost
        if use_managed_db:
            # Find closest tier
            vector_tiers = sorted(self.COSTS["qdrant_cloud"].keys())
            tier = next(t for t in vector_tiers if t >= total_vectors)
            vector_db_cost = self.COSTS["qdrant_cloud"][tier]
        else:
            # Self-hosted estimate
            vector_db_cost = (total_vectors / 1_000_000) * 15  # ~$15 per 1M
        
        # Other infrastructure
        postgres_cost = 15 if monthly_queries < 10000 else 50
        redis_cost = 5 if monthly_queries < 10000 else 30
        monitoring_cost = 0 if monthly_queries < 1000 else 50
        
        total_cost = (
            llm_cost +
            vector_db_cost +
            postgres_cost +
            redis_cost +
            monitoring_cost
        )
        
        return {
            "monthly_queries": monthly_queries,
            "effective_llm_calls": int(effective_llm_calls),
            "llm_cost": round(llm_cost, 2),
            "vector_db_cost": vector_db_cost,
            "postgres_cost": postgres_cost,
            "redis_cost": redis_cost,
            "monitoring_cost": monitoring_cost,
            "total_monthly_cost": round(total_cost, 2),
            "cost_per_query": round(total_cost / monthly_queries, 4)
        }

# Example usage
calc = RAGCostCalculator()

# Small team
small = calc.calculate(
    queries_per_day=100,
    llm_model="groq-8b",
    cache_hit_rate=0.3,
    total_vectors=500_000
)
print(f"Small team: ${small['total_monthly_cost']}/month (${small['cost_per_query']} per query)")

# Startup
startup = calc.calculate(
    queries_per_day=1000,
    llm_model="groq-70b",
    cache_hit_rate=0.5,
    total_vectors=5_000_000
)
print(f"Startup: ${startup['total_monthly_cost']}/month")

# Enterprise
enterprise = calc.calculate(
    queries_per_day=10000,
    llm_model="gpt-3.5",
    cache_hit_rate=0.7,
    total_vectors=50_000_000,
    use_managed_db=True
)
print(f"Enterprise: ${enterprise['total_monthly_cost']}/month")
```

**Output:**
```
Small team: $31.45/month ($0.0105 per query)
Startup: $276.50/month
Enterprise: $2,847.00/month
```

---

### Cost Optimization Strategies

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Caching (70% hit rate)** | 50-60% | Redis + smart TTL |
| **Use Groq vs OpenAI** | 90%+ | Switch LLM provider |
| **Self-host embeddings** | 100% | Run on CPU/GPU |
| **Deduplication** | 10-20% | Hash-based detection |
| **Smaller models for simple queries** | 30-40% | Query classification |
| **Batch processing** | 10-15% | Process docs in bulk |
| **Query rewriting** | 5-10% | Simplify verbose queries |

**ROI Calculation:**

If caching reduces LLM calls by 50%, and LLM costs are 30% of total:
- Savings = 30% × 50% = 15% total cost reduction
- For $1,000/month system → $150/month saved
- Redis cost = $30/month
- **Net savings: $120/month ($1,440/year)**

---


---

## 14. Conclusion

This design document provides a comprehensive blueprint for building an enterprise-grade RAG system. Key design principles:

1. **Modularity** - Clean separation of concerns (ingest, search, generate, API)
2. **Scalability** - Horizontal scaling via K8s, clustered Qdrant, async workers
3. **Security** - Multi-tenant isolation, RBAC, encryption, compliance
4. **Quality** - Re-ranking, semantic chunking, evaluation metrics
5. **Observability** - Structured logging, Prometheus metrics, distributed tracing

### Next Steps

1. **Initialize project structure** - Create directories per Project Structure section
2. **Set up development environment** - Docker Compose for local services
3. **Implement Phase 1 (MVP)** - Core RAG pipeline with CLI
4. **Add API layer** - FastAPI REST endpoints
5. **Productionize** - Add enterprise features progressively

### Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Author**: Sujith
**Repository**: `/home/sujith/github/rag`

## Appendix A: Quick Command Reference

This appendix provides quick reference commands for common RAG system operations.

---

### Local Development

#### Setup Commands

```bash
# Clone and setup
git clone https://github.com/yourusername/rag.git
cd rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker-compose -f docker/docker-compose.yml up -d

# Check services
docker-compose ps
curl http://localhost:6333/health  # Qdrant
redis-cli PING                      # Redis
psql -h localhost -U rag            # PostgreSQL
```

#### Development Workflow

```bash
# Run tests
pytest                              # All tests
pytest tests/unit                   # Unit tests only
pytest tests/integration            # Integration tests
pytest -v --cov=rag                # With coverage

# Code quality
ruff check .                        # Lint
ruff check . --fix                  # Auto-fix
mypy rag/                          # Type check
black rag/                         # Format code

# Run application
# CLI mode
python -m rag.cli add ./documents/
python -m rag.cli ask "What is RAG?"
python -m rag.cli status

# API mode
uvicorn src.rag.api:app --reload --port 8000
# Open http://localhost:8000/docs for API documentation
```

---

### Production Operations

#### Kubernetes Commands

```bash
# Deploy
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl rollout status deployment/rag-api -n rag-system

# Scale
kubectl scale deployment/rag-api --replicas=5 -n rag-system
kubectl scale deployment/celery-worker --replicas=3 -n rag-system

# Update image
kubectl set image deployment/rag-api \
  rag-api=ghcr.io/yourrepo/rag-api:v1.2.3 \
  -n rag-system

# Rollback
kubectl rollout undo deployment/rag-api -n rag-system
kubectl rollout undo deployment/rag-api --to-revision=5 -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system
kubectl logs -f deployment/rag-api -n rag-system --tail=100
kubectl logs -f deployment/rag-api -n rag-system --since=1h

# Execute commands in pod
kubectl exec -it rag-api-xxx -n rag-system -- /bin/bash
kubectl exec -it rag-api-xxx -n rag-system -- python -m rag.cli status

# Port forwarding
kubectl port-forward svc/rag-api-service 8000:80 -n rag-system
kubectl port-forward svc/grafana 3000:3000 -n rag-system
kubectl port-forward svc/qdrant 6333:6333 -n rag-system

# Resource usage
kubectl top nodes
kubectl top pods -n rag-system
kubectl describe pod rag-api-xxx -n rag-system
```

#### Database Operations

```bash
# PostgreSQL
# Connect
kubectl exec -it postgres-0 -n rag-system -- psql -U rag

# Backup
kubectl exec -it postgres-0 -n rag-system -- \
  pg_dump -U rag > backup_$(date +%Y%m%d).sql

# Restore
kubectl exec -i postgres-0 -n rag-system -- \
  psql -U rag < backup_20240115.sql

# Useful queries
kubectl exec -it postgres-0 -n rag-system -- psql -U rag << 'EOF'
-- Active connections
SELECT count(*) FROM pg_stat_activity;

-- Recent queries
SELECT query_text, created_at, latency_ms 
FROM query_logs 
ORDER BY created_at DESC 
LIMIT 10;

-- Failed queries in last hour
SELECT query_text, COUNT(*) as fail_count
FROM query_logs 
WHERE created_at > NOW() - INTERVAL '1 hour'
  AND latency_ms IS NULL  -- Failed queries
GROUP BY query_text
ORDER BY fail_count DESC;

-- Cost summary by tenant
SELECT tenant_id, SUM(cost_usd) as total_cost
FROM query_logs
WHERE created_at > DATE_TRUNC('month', CURRENT_DATE)
GROUP BY tenant_id
ORDER BY total_cost DESC;
EOF

# Qdrant
# Health check
curl http://qdrant:6333/health

# List collections
curl http://qdrant:6333/collections

# Collection info
curl http://qdrant:6333/collections/knowledge_base

# Count vectors
curl http://qdrant:6333/collections/knowledge_base/points/count

# Search (debug)
curl -X POST http://qdrant:6333/collections/knowledge_base/points/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, ...],
    "limit": 5,
    "with_payload": true
  }'

# Backup Qdrant
kubectl exec -it qdrant-0 -n rag-system -- \
  tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage
kubectl cp rag-system/qdrant-0:/tmp/qdrant-backup.tar.gz \
  ./qdrant-backup-$(date +%Y%m%d).tar.gz

# Redis
# Connect
kubectl exec -it redis-0 -n rag-system -- redis-cli

# Useful commands
INFO stats                          # Stats
DBSIZE                             # Key count
KEYS rag:query:*                   # List query cache keys
FLUSHALL                           # Clear all (DANGEROUS!)
CONFIG GET maxmemory               # Memory limit
MEMORY USAGE rag:query:abc123     # Check key size
```

---

### Monitoring & Debugging

#### Metrics & Logs

```bash
# Prometheus queries
# Query rate
rate(rag_queries_total[5m])

# P95 latency
histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))

#Error rate
rate(rag_errors_total[5m]) / rate(rag_queries_total[5m])

# Cache hit rate
rate(rag_cache_hits_total[5m]) / rate(rag_cache_requests_total[5m])

# Grafana - Access dashboards
kubectl port-forward svc/grafana 3000:3000 -n observability
# Open http://localhost:3000

# Loki - Query logs
kubectl port-forward svc/loki 3100:3100 -n observability
# Query: {namespace="rag-system"} |= "ERROR"

# Jaeger - View traces
kubectl port-forward svc/jaeger 16686:16686 -n observability
# Open http://localhost:16686
```

#### Common Debugging Commands

```bash
# Check why pod is failing
kubectl describe pod rag-api-xxx -n rag-system
kubectl logs rag-api-xxx -n rag-system --previous  # Previous crash logs

# Debug network issues
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
nslookup qdrant-service
wget -O- http://qdrant-service:6333/health

# Check resource constraints
kubectl get resourcequota -n rag-system
kubectl get limitrange -n rag-system

# View events
kubectl get events -n rag-system --sort-by='.lastTimestamp'

# Test API endpoint
kubectl exec -it rag-api-xxx -n rag-system -- \
  curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
```

---

### Maintenance Scripts

#### Document Management

```bash
# Ingest documents
python -m rag.cli add ./documents/ --recursive

# Re-index specific document
python scripts/reindex.py --file="path/to/document.pdf"

# Bulk reindex all documents
python scripts/bulk_reindex.py \
  --collection knowledge_base \
  --batch-size 100 \
  --parallel 4

# Delete document
python scripts/delete_document.py --doc-id="uuid-here"

# Check document status
python -m rag.cli status --detailed

# Export document metadata
python scripts/export_metadata.py --output metadata.json
```

#### Evaluation & Testing

```bash
# Run evaluation
python scripts/evaluate.py \
  --test-set data/eval/golden_set.json \
  --output results/eval_$(date +%Y%m%d).json

# Generate evaluation report
python scripts/evaluation_report.py \
  --results results/eval_20240115.json \
  --format html

# Benchmark performance
python scripts/benchmark.py \
  --queries 1000 \
  --concurrent 10

# Test specific query
python scripts/test_query.py \
  --question "What is RAG?" \
  --verbose \
  --show-chunks
```

#### Cost Analysis

```bash
# Monthly cost report
python scripts/cost_analysis.py \
  --month 2024-01 \
  --breakdown

# Per-tenant costs
python scripts/cost_per_tenant.py \
  --output costs/tenants_2024_01.csv

# Forecast costs
python scripts/cost_forecast.py \
  --based-on-last-days 30 \
  --predict-months 3
```

---

### CI/CD

#### GitHub Actions

```bash
# Trigger workflow manually
gh workflow run ci-cd.yaml \
  --ref main \
  -f environment=staging

# View workflow runs
gh run list --workflow=ci-cd.yaml

# View logs
gh run view <run-id> --log

# Cancel workflow
gh run cancel <run-id>
```

#### Docker

```bash
# Build locally
docker build -t rag-api:latest -f docker/Dockerfile .
docker build -t rag-worker:latest -f docker/Dockerfile.worker .

# Run locally
docker run -p 8000:8000 \
  -e QDRANT_HOST=host.docker.internal \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  rag-api:latest

# Push to registry
docker tag rag-api:latest ghcr.io/yourrepo/rag-api:v1.2.3
docker push ghcr.io/yourrepo/rag-api:v1.2.3

# Clean up
docker system prune -a  # Remove unused images
```

---

### Backup & Recovery

```bash
# Full backup
./scripts/backup_all.sh

# This runs:
# 1. PostgreSQL backup
kubectl exec -it postgres-0 -- pg_dump -U rag | gzip > \
  backups/postgres_$(date +%Y%m%d_%H%M%S).sql.gz

# 2. Qdrant snapshot
kubectl exec -it qdrant-0 -- \
  curl -X POST http://localhost:6333/collections/knowledge_base/snapshots

# 3. Upload to S3
aws s3 sync ./backups/ s3://rag-backups/$(date +%Y-%m-%d)/

# Restore from backup
./scripts/restore_from_backup.sh backups/postgres_20240115.sql.gz

# Test restore (dry-run)
./scripts/restore_from_backup.sh --dry-run backups/postgres_20240115.sql.gz
```

---

### Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Kubernetes
alias k='kubectl'
alias kns='kubectl config set-context --current --namespace'
alias kgp='kubectl get pods'
alias kdp='kubectl describe pod'
alias kl='kubectl logs -f'
alias kx='kubectl exec -it'

# RAG specific
alias rag-logs='kubectl logs -f deployment/rag-api -n rag-system'
alias rag-api='kubectl exec -it deployment/rag-api -n rag-system -- /bin/bash'
alias rag-db='kubectl exec -it postgres-0 -n rag-system -- psql -U rag'
alias rag-redis='kubectl exec -it redis-0 -n rag-system -- redis-cli'
alias rag-status='kubectl get pods -n rag-system'

# Quick restart
alias rag-restart='kubectl rollout restart deployment/rag-api -n rag-system'

# Port forwards
alias grafana-port='kubectl port-forward svc/grafana 3000:3000 -n observability'
alias qdrant-port='kubectl port-forward svc/qdrant 6333:6333 -n rag-system'
```

---


---

## Appendix B: Phase Completion Criteria

Clear "Definition of Done" for each development phase.

---

### Phase 1: MVP - Python Fundamentals

**Goal:** Working CLI that can ingest documents and answer questions.

#### Completion Criteria

- [x] **Project Setup**
  - [ ] `pyproject.toml` with all dependencies
  - [ ] `.env.example` with required variables
  - [ ] `README.md` with setup instructions
  - [ ] Virtual environment activates successfully
  - [ ] All dependencies install without errors

- [x] **Document Ingestion**
  - [ ] Can extract text from PDF files
  - [ ] Can extract text from TXT files
  - [ ] Can extract text from Markdown files
  - [ ] Sentence-aware chunking implemented
  - [ ] Chunk overlap working (50-100 chars)
  - [ ] File hash tracking prevents duplicates
  - [ ] Progress indicator during ingestion

- [x] **Embedding & Storage**
  - [ ] Sentence transformer model loads successfully
  - [ ] Embeddings generate for all chunks
  - [ ] Qdrant collection creates automatically
  - [ ] Vectors store with metadata (source, page, index)
  - [ ] Can handle 100+ documents without errors

- [x] **Search & Retrieval**
  - [ ] Query embedding works
  - [ ] Vector search returns results
  - [ ] Results ranked by relevance score
  - [ ] Can retrieve top 5 relevant chunks
  - [ ] Search completes in <2 seconds

- [x] **LLM Integration**
  - [ ] Groq API key configured
  - [ ] Context building from retrieved chunks
  - [ ] System prompt enforces grounding
  - [ ] Answer generation works
  - [ ] Answers cite sources

- [x] **CLI Interface**
  - [ ] `rag add <path>` command works
  - [ ] `rag ask <question>` command works
  - [ ] `rag status` shows document count
  - [ ] Interactive mode (optional)
  - [ ] Error messages are clear

#### Acceptance Tests

```bash
# Test 1: Ingest sample document
echo "RAG is Retrieval Augmented Generation." > test.txt
python -m rag.cli add test.txt
# Expected: Success message, 1 chunk created

# Test 2: Query system
python -m rag.cli ask "What is RAG?"
# Expected: Answer mentions "Retrieval Augmented Generation", cites source

# Test 3: Check status
python -m rag.cli status
# Expected: Shows 1 document, 1 chunk
```

#### Knowledge Check

Can you answer these questions?
- [ ] What's the difference between a list and a generator?
- [ ] Why use type hints?
- [ ] How do context managers work (`with` statement)?
- [ ] What does `__init__.py` do?
- [ ] How does `@dataclass` reduce boilerplate?

#### Deliverables

- [ ] Working Python package in `src/rag/`
- [ ] At least 3 unit tests passing
- [ ] Can process 10 documents in <1 minute
- [ ] README with "Quick Start" section
- [ ] Demo video (optional but recommended)

**Estimated Time:** 2-3 weeks (learning Python fundamentals)

---

### Phase 2: API & Quality - Intermediate Python

**Goal:** REST API with tests and documentation.

#### Completion Criteria

- [x] **FastAPI Implementation**
  - [ ] `/api/v1/documents/ingest` endpoint
  - [ ] `/api/v1/query` endpoint
  - [ ] `/api/v1/health` endpoint
  - [ ] OpenAPI docs auto-generated (`/docs`)
  - [ ] Pydantic models for request/response
  - [ ] Input validation with helpful errors

- [x] **Improved RAG Quality**
  - [ ] Cross-encoder reranking implemented
  - [ ] Retrieval limit configurable (default: 10)
  - [ ] Rerank top-k configurable (default: 5)
  - [ ] Citations in response format
  - [ ] Structured context building

- [x] **Error Handling**
  - [ ] Custom exception classes
  - [ ] Retry decorator for LLM calls (3 attempts)
  - [ ] Graceful degradation (skip reranking if fails)
  - [ ] HTTP status codes correct
  - [ ] Error responses include details

- [x] **Testing**
  - [ ] Unit tests for chunking logic
  - [ ] Unit tests for embedding logic
  - [ ] Mock tests for LLM calls
  - [ ] Integration test for full pipeline
  - [ ] Pytest fixtures for test data
  - [ ] Code coverage >70%

- [x] **Logging**
  - [ ] Structured logging (JSON format)
  - [ ] Log levels configured (INFO in prod, DEBUG in dev)
  - [ ] Query latency logged
  - [ ] Retrieval scores logged
  - [ ] Errors logged with stack traces

- [x] **Docker Setup**
  - [ ] `Dockerfile` builds successfully
  - [ ] `docker-compose.yml` for local dev
  - [ ] Environment variables configurable
  - [ ] Health checks in docker-compose
  - [ ] Can run entire stack with one command

#### Acceptance Tests

```bash
# Test 1: Start API
docker-compose up -d
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Test 2: Ingest document via API
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@test.pdf"
# Expected: {"status": "processing", "job_id": "..."}

# Test 3: Query via API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
# Expected: {"answer": "...", "sources": [...]}

# Test 4: Run tests
pytest -v --cov=rag
# Expected: All tests pass, coverage >70%
```

#### Knowledge Check

- [ ] How does dependency injection improve testability?
- [ ] What's the difference between `@staticmethod` and `@classmethod`?
- [ ] How do you mock external services in tests?
- [ ] What's the purpose of Pydantic validators?
- [ ] How does FastAPI generate OpenAPI docs?

#### Deliverables

- [ ] FastAPI application with 3+ endpoints
- [ ] 15+ unit tests passing
- [ ] Docker Compose stack works
- [ ] API documentation at `/docs`
- [ ] Postman/Insomnia collection (optional)

**Estimated Time:** 3-4 weeks (learning testing & APIs)

---

### Phase 3: Enterprise Features - Advanced Python

**Goal:** Production-ready multi-tenant API.

#### Completion Criteria

- [x] **Database Layer**
  - [ ] PostgreSQL schema created
  - [ ] SQLAlchemy models for tenants, users, documents
  - [ ] Alembic migrations setup
  - [ ] Repository pattern implemented
  - [ ] Database connection pooling configured

- [x] **Authentication & Authorization**
  - [ ] API key generation and validation
  - [ ] JWT token support (optional)
  - [ ] User roles (admin, user, viewer)
  - [ ] Middleware for auth
  - [ ] Per-tenant data isolation

- [x] **Multi-Tenancy**
  - [ ] Tenant creation endpoint
  - [ ] Collection-per-tenant OR tenant_id filtering
  - [ ] Query filtering by tenant
  - [ ] Per-tenant quotas (optional)
  - [ ] Tenant deletion with cleanup

- [x] **Async Processing**
  - [ ] Celery worker setup
  - [ ] Async document ingestion task
  - [ ] Job status tracking
  - [ ] Redis as message broker
  - [ ] Error handling in tasks

- [x] **Caching**
  - [ ] Redis caching for query results
  - [ ] TTL configuration (default: 1 hour)
  - [ ] Cache invalidation on document update
  - [ ] Cache hit rate monitoring

- [x] **Observability**
  - [ ] Prometheus metrics exported
  - [ ] Grafana dashboard configured
  - [ ] Request latency histogram
  - [ ] Error counter
  - [ ] Custom metrics (retrieval score, cache hits)

- [x] **Kubernetes Deployment**
  - [ ] K8s manifests created
  - [ ] Deployment, Service, Ingress configured
  - [ ] ConfigMap for configuration
  - [ ] Secrets for sensitive data
  - [ ] HPA (Horizontal Pod Autoscaler) for API

- [x] **CI/CD Pipeline**
  - [ ] GitHub Actions workflow
  - [ ] Automated tests on PR
  - [ ] Docker image build
  - [ ] Deploy to staging on merge
  - [ ] Manual approval for production

#### Acceptance Tests

```bash
# Test 1: Create tenant
curl -X POST http://api.rag.dev/api/v1/tenants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "Acme Corp"}'
# Expected: {"id": "uuid", "name": "Acme Corp", "created_at": "..."}

# Test 2: Multi-tenant isolation
# Query as Tenant A - should only see Tenant A's data
curl -X POST http://api.rag.dev/api/v1/query \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  -d '{"question": "test"}'

# Test 3: Cache effectiveness
# Run same query twice, second should be faster
time curl -X POST http://api.rag.dev/api/v1/query -d '{"question": "test"}'
time curl -X POST http://api.rag.dev/api/v1/query -d '{"question": "test"}'
# Expected: Second call <100ms (cached)

# Test 4: Kubernetes deployment
kubectl get pods -n rag-system
kubectl logs -f deployment/rag-api
# Expected: Pods running, logs showing requests

# Test 5: Metrics
curl http://api.rag.dev/metrics
# Expected: Prometheus metrics format
```

#### Knowledge Check

- [ ] When should you use async vs threading vs multiprocessing?
- [ ] How does SQLAlchemy manage database sessions?
- [ ] What's the difference between Factory and Strategy patterns?
- [ ] How does Redis caching improve performance?
- [ ] What's the purpose of database migrations?

#### Deliverables

- [ ] Multi-tenant API with authentication
- [ ] PostgreSQL schema + migrations
- [ ] Celery workers for async tasks
- [ ] Kubernetes manifests
- [ ] CI/CD pipeline functional
- [ ] Grafana dashboard with 5+ metrics

**Estimated Time:** 6-8 weeks (learning async, ORM, K8s)

---

### Phase 4: Advanced - Expert Python

**Goal:** Extensible, high-performance RAG system.

#### Completion Criteria

- [x] **Hybrid Search**
  - [ ] Sparse vector generation (BM25/TF-IDF)
  - [ ] Qdrant collection with dense + sparse vectors
  - [ ] Hybrid search scoring
  - [ ] Benchmarked against dense-only

- [x] **Document-Level ACL**
  - [ ] Permission model (user, document, permission)
  - [ ] Middleware to filter results by permissions
  - [ ] Permission inheritance (groups)
  - [ ] Audit logging for access

- [x] **Advanced Features**
  - [ ] Conversation history (multi-turn)
  - [ ] Query expansion / HyDE
  - [ ] PII detection and masking
  - [ ] A/B testing framework

- [x] **Data Connectors**
  - [ ] S3 connector
  - [ ] Confluence connector (nice to have)
  - [ ] Google Drive connector (nice to have)
  - [ ] Base connector interface (ABC)

- [x] **Performance Optimization**
  - [ ] Profiling with cProfile
  - [ ] Memory optimization
  - [ ] Batch processing
  - [ ] Query parallelization (if applicable)
  - [ ] Benchmarks documented

- [x] **Production Hardening**
  - [ ] Rate limiting per tenant
  - [ ] Circuit breakers for external services
  - [ ] Health checks (liveness, readiness)
  - [ ] Graceful shutdown
  - [ ] Disaster recovery plan documented

#### Acceptance Tests

```bash
# Test 1: Hybrid search
# Compare precision@5 for hybrid vs dense-only
python scripts/benchmark_hybrid_search.py
# Expected: Hybrid improves precision by 10-20%

# Test 2: Document permissions
# User A should NOT see User B's private documents
curl -H "Authorization: Bearer $USER_A_TOKEN" \
  http://api.rag.dev/api/v1/query?question="User B's secret"
# Expected: No results

# Test 3: Performance
# Handle 100 concurrent queries
ab -n 1000 -c 100 -p query.json -T 'application/json' \
  http://api.rag.dev/api/v1/query
# Expected: P95 latency <3s, error rate <1%

# Test 4: PII detection
echo "My SSN is 123-45-6789" > pii_test.txt
python -m rag.cli add pii_test.txt
# Expected: Warning logged, SSN masked in storage
```

#### Knowledge Check

- [ ] When would you use metaclasses?
- [ ] How do you profile Python code for bottlenecks?
- [ ] What's structural typing (Protocols)?
- [ ] How do you design a plugin system?
- [ ] What's the difference between eager and lazy evaluation?

#### Deliverables

- [ ] Hybrid search working
- [ ] 3+ data connectors
- [ ] Performance benchmarks documented
- [ ] Security audit completed
- [ ] Architecture documentation
- [ ] Production runbook

**Estimated Time:** 8-12 weeks (expert-level features)

---

### Overall Project Timeline

| Phase | Duration | Cumulative | Skill Level |
|-------|----------|------------|-------------|
| Phase 1 | 2-3 weeks | 3 weeks | Beginner → Intermediate |
| Phase 2 | 3-4 weeks | 7 weeks | Intermediate |
| Phase 3 | 6-8 weeks | 15 weeks | Intermediate → Advanced |
| Phase 4 | 8-12 weeks | 27 weeks | Advanced → Expert |

**Total: ~6 months (part-time) to 9 months (learning mode)**

---

### Success Metrics by Phase

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|
| **Lines of Code** | 500 | 2,000 | 5,000 | 8,000+ |
| **Test Coverage** | 30% | 70% | 80% | 85%+ |
| **Latency (P95)** | <5s | <3s | <2s | <1.5s |
| **Documents** | 100 | 1,000 | 10,000 | 100,000+ |
| **Concurrent Users** | 1 | 10 | 100 | 1,000+ |
| **Uptime** | - | - | 99% | 99.9% |

---

