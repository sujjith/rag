
## Table of Contents

1. [Project Overview](#project-overview)
2. [Python Learning Path](#python-learning-path) *(NEW)*
3. [Goals & Requirements](#goals--requirements)
4. [System Architecture](#system-architecture)
5. [Project Structure](#project-structure)
6. [Core Components](#core-components)
7. [Infrastructure](#infrastructure)
8. [Security & Compliance](#security--compliance)
9. [Deployment](#deployment)
10. [Development Roadmap](#development-roadmap)
11. **[Evaluation & Monitoring](#evaluation--monitoring)** *(ENHANCED)*
    - 9.1 Retrieval Metrics
    - 9.2 Logging
    - 9.3 **RAG-Specific Testing Strategy** *(NEW)*
12. **[Advanced Features](#advanced-features)** *(ENHANCED)*
    - 10.1 Conversation History
    - 10.2 Multi-Document Synthesis
    - 10.3 Document Update Detection
    - 10.4 **Common Pitfalls & Troubleshooting** *(NEW)*
    - 10.5 **Advanced RAG Techniques** *(NEW)*
      - Agentic RAG (ReAct, Function Calling)
      - Self-RAG (Adaptive Retrieval)
      - Corrective RAG (CRAG)
      - Graph RAG (Knowledge Graph Integration)
      - Multi-Modal RAG (Images, Tables, Charts)
13. **[Enterprise Infrastructure](#enterprise-infrastructure-details)** *(ENHANCED)*
    - 13.15 **Production Runbook** *(NEW)*
14. **[Cost & Conclusion](#cost-estimation-examples)** *(ENHANCED)*
    - 14.1 **Cost Estimation Examples** *(NEW)*
15. **[Appendices](#appendix-a-quick-command-reference)** *(NEW)*
    - Appendix A: Quick Command Reference
    - Appendix B: Phase Completion Criteria

---

## Project Overview

### What is RAG?

**RAG (Retrieval Augmented Generation)** is a technique that enhances Large Language Models by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the user's question with retrieved context
3. **Generating** accurate, grounded answers using an LLM

### Why Build This?

- **Knowledge Grounding**: Answers based on YOUR documents, not just LLM training data
- **Reduced Hallucination**: LLM responses are grounded in retrieved context
- **Up-to-date Information**: Add new documents anytime, no model retraining
- **Private Data**: Keep sensitive documents within your infrastructure
- **Enterprise Scale**: Multi-tenant, secure, observable, production-ready

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Enterprise RAG System                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Documents                                                              │
│       ↓                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐     │
│   │ Extract │ →  │  Chunk  │ →  │  Embed  │ →  │  Qdrant Vector  │     │
│   │  Text   │    │  Text   │    │ (BGE)   │    │    Database     │     │
│   └─────────┘    └─────────┘    └─────────┘    └────────┬────────┘     │
│                                                          │              │
│   User Question                                          │              │
│       ↓                                                  ↓              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐     │
│   │  Embed  │ →  │ Search  │ →  │ Rerank  │ →  │   LLM (Groq/    │     │
│   │  Query  │    │ Qdrant  │    │ Results │    │  OpenAI/Claude) │     │
│   └─────────┘    └─────────┘    └─────────┘    └────────┬────────┘     │
│                                                          │              │
│                                                          ↓              │
│                                                    ┌───────────┐        │
│                                                    │  Answer   │        │
│                                                    │ + Sources │        │
│                                                    └───────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Python Learning Path

This project is designed as a **hands-on Python learning journey**. Each phase introduces new concepts progressively.

### Learning Objectives

By completing this project, you will learn:

| Category | Concepts |
|----------|----------|
| **Python Fundamentals** | Type hints, dataclasses, enums, f-strings, comprehensions |
| **OOP** | Classes, inheritance, abstract base classes, protocols |
| **Functional** | Decorators, generators, context managers, closures |
| **Async** | asyncio, async/await, concurrent.futures |
| **Testing** | pytest, fixtures, mocking, parameterized tests |
| **APIs** | FastAPI, Pydantic, dependency injection |
| **Databases** | SQLAlchemy, async DB, migrations |
| **DevOps** | Docker, environment variables, logging |

### Python Concepts by Phase

```
Phase 1 (MVP)           Phase 2 (API)           Phase 3 (Enterprise)      Phase 4 (Advanced)
─────────────────       ─────────────────       ─────────────────────     ─────────────────
• Type hints            • FastAPI/Pydantic      • SQLAlchemy ORM          • Metaclasses
• Dataclasses           • Dependency injection  • Alembic migrations      • Descriptors
• File I/O              • Request validation    • Background tasks        • Custom decorators
• JSON handling         • Error handling        • Celery workers          • Design patterns
• f-strings             • HTTP clients          • Redis caching           • Performance tuning
• List comprehensions   • Async/await basics    • Connection pooling      • Memory profiling
• Context managers      • Logging               • Multi-tenancy           • Concurrency
• Basic OOP             • Unit testing          • Authentication          • Plugin architecture
```

### Skill Level Progression

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PYTHON SKILL PROGRESSION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Beginner          Intermediate         Advanced           Expert           │
│     │                   │                   │                  │            │
│     ▼                   ▼                   ▼                  ▼            │
│  ┌──────┐          ┌──────────┐       ┌──────────┐      ┌──────────┐       │
│  │Phase │    →     │  Phase   │   →   │  Phase   │  →   │  Phase   │       │
│  │  1   │          │    2     │       │    3     │      │    4     │       │
│  └──────┘          └──────────┘       └──────────┘      └──────────┘       │
│                                                                              │
│  • Variables        • Classes           • Async/await      • Metaclasses    │
│  • Functions        • Decorators        • SQLAlchemy       • C extensions   │
│  • File I/O         • Type hints        • Design patterns  • Profiling      │
│  • Data structures  • Testing           • Concurrency      • Optimization   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Before You Start - Python Setup for Beginners

If you're new to Python, start here before diving into RAG implementation.

#### Development Environment Setup

**1. Install Python 3.11+**

```bash
# Check if Python is installed
python --version  # Should show 3.11 or higher

# If not installed:
# - Windows: Download from python.org
# - Mac: brew install python@3.11
# - Linux: sudo apt install python3.11
```

**2. Choose Your IDE** (Pick ONE - VS Code recommended for beginners)

| IDE | Best For | Setup Difficulty |
|-----|----------|------------------|
| **VS Code** ✅ | Beginners, lightweight | Easy |
| **PyCharm Community** | Feature-rich development | Medium |
| **Cursor** | AI-powered coding | Easy |

**3. Essential VS Code Extensions**

Install these for the best Python experience:

```markdown
Required:
- Python (by Microsoft)
- Pylance (fast type checking)

Highly Recommended:
- Python Indent
- autoDocstring (auto-generates docstrings)
- Error Lens (shows errors inline)
- Black Formatter (code formatting)

Optional but Useful:
- GitLens (git history)
- Thunder Client (API testing)
- Todo Tree (track TODOs)
```

**4. First Project Setup**

```bash
# Create project directory
mkdir rag-learning
cd rag-learning

# Create virtual environment (ALWAYS do this!)
python -m venv .venv

# Activate it
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Verify you're in venv (should show .venv path)
which python
```

**5. Create Your First Python File**

```bash
# Create main.py
touch main.py

# Open in VS Code
code .
```

In `main.py`:
```python
# Your first Python program
print("Hello, RAG!")

# Check Python version
import sys
print(f"Python version: {sys.version}")
```

Run it:
```bash
python main.py
```

---

#### Learning Path: Zero to RAG

**If you're NEW to Python**, follow this order:

| Timeframe | Focus | What You'll Learn |
|-----------|-------|-------------------|
| **Week 1-2** | Python Basics | Variables, types, functions, loops, if/else |
| **Week 3-4** | Phase 1 RAG | File I/O, libraries, simple CLI |
| **Week 5-8** | Phase 2 RAG | APIs, testing, error handling |
| **Month 3-4** | Phase 3 RAG | Async, databases, ORM |
| **Month 5-6** | Phase 4 RAG | Advanced patterns, optimization |

> **⚠️ Important**: Don't skip ahead! Each phase builds on the previous. If you skip Phase 1, you'll struggle with Phase 2.

---

#### Quick Python Refresher (15 minutes)

If you know some Python but need a refresher:

```python
# 1. Variables and Types
name: str = "RAG System"
count: int = 42
price: float = 99.99
is_ready: bool = True

# 2. Lists and Dictionaries
documents = ["doc1.pdf", "doc2.txt"]  # List
metadata = {"author": "John", "pages": 10}  # Dict

# 3. Functions
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 4. List Comprehension
squares = [x**2 for x in range(10)]

# 5. File Reading
with open("data.txt") as f:
    content = f.read()

# 6. Loops
for doc in documents:
    print(f"Processing {doc}")

# 7. Conditionals
if count > 10:
    print("Many documents")
elif count > 0:
    print("Some documents")
else:
    print("No documents")
```

Got all of these? You're ready for Phase 1! 🚀

---

#### Pre-Phase 1 Checklist

Before starting Phase 1 implementation, ensure:

- [ ] Python 3.11+ installed and working
- [ ] VS Code (or your IDE) installed and configured
- [ ] Can create and activate virtual environment
- [ ] Can install packages with pip
- [ ] Understand basic Python syntax (variables, functions, loops)
- [ ] Know how to read error messages
- [ ] Have a terminal/command line open

**Not ready yet?** Complete this [30-minute Python tutorial](https://learnxinyminutes.com/docs/python/) first.

---

### Phase 1: Python Fundamentals

#### 1.1 Type Hints (PEP 484)

**Why?** Makes code self-documenting and enables IDE support.

```python
# ❌ Without type hints - unclear what types are expected
def process_document(file_path, chunk_size):
    pass

# ✅ With type hints - clear contract
def process_document(file_path: str, chunk_size: int = 500) -> list[str]:
    """Process a document and return chunks."""
    pass

# Advanced: Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Result(Generic[T]):
    def __init__(self, value: T, success: bool):
        self.value = value
        self.success = success

# Usage
result: Result[str] = Result("Hello", True)
```

**Learn More**: [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)

#### 1.2 Dataclasses (PEP 557)

**Why?** Reduce boilerplate for data containers.

```python
# ❌ Traditional class - lots of boilerplate
class DocumentOld:
    def __init__(self, id: str, title: str, content: str):
        self.id = id
        self.title = title
        self.content = content

    def __repr__(self):
        return f"Document(id={self.id}, title={self.title})"

    def __eq__(self, other):
        return self.id == other.id

# ✅ Dataclass - clean and automatic
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Document:
    id: str
    title: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def word_count(self) -> int:
        return len(self.content.split())

# Immutable version
@dataclass(frozen=True)
class ChunkMetadata:
    source: str
    page: int
    index: int
```

**Exercise**: Create a `QueryResult` dataclass with fields for answer, sources, and latency.

#### 1.3 Context Managers

**Why?** Ensure resources are properly cleaned up.

```python
# ❌ Manual resource management - error prone
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()

# ✅ Context manager - automatic cleanup
with open("data.txt", "r") as file:
    content = file.read()

# Creating your own context manager
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Time a block of code."""
    start = time.time()
    try:
        yield  # Code inside 'with' block runs here
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.2f} seconds")

# Usage
with timer("Document processing"):
    process_documents()

# Class-based context manager
class QdrantConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None

    def __enter__(self):
        self.client = QdrantClient(host=self.host, port=self.port)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
        return False  # Don't suppress exceptions

# Usage
with QdrantConnection("localhost", 6333) as client:
    client.search(...)
```

#### 1.4 Generators and Iterators

**Why?** Memory-efficient processing of large datasets.

```python
# ❌ Loading all documents into memory
def get_all_chunks_bad(documents: list[str]) -> list[str]:
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc)
        all_chunks.extend(chunks)
    return all_chunks  # Could be gigabytes!

# ✅ Generator - yields one at a time
def get_all_chunks(documents: list[str]):
    """Yield chunks one at a time - memory efficient."""
    for doc in documents:
        for chunk in split_into_chunks(doc):
            yield chunk

# Usage - processes one chunk at a time
for chunk in get_all_chunks(large_document_list):
    embed_and_store(chunk)

# Generator expression (like list comprehension but lazy)
chunks = (process(doc) for doc in documents)  # Nothing happens yet
first_chunk = next(chunks)  # Now it processes

# Useful built-in: itertools
from itertools import islice, chain, batched

# Process in batches of 100
for batch in batched(get_all_chunks(docs), 100):
    bulk_insert(batch)
```

**Exercise**: Write a generator that reads a large PDF page by page.

#### 1.5 Decorators

**Why?** Add behavior to functions without modifying them.

```python
import functools
import time
from typing import Callable, TypeVar

T = TypeVar('T')

# Simple decorator
def log_calls(func: Callable[..., T]) -> Callable[..., T]:
    """Log function calls."""
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs) -> T:
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a: int, b: int) -> int:
    return a + b

# Decorator with arguments
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry a function on failure."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def call_llm_api(prompt: str) -> str:
    """Call LLM API with automatic retry."""
    return groq_client.generate(prompt)

# Decorator for timing
def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper
```

**Exercise**: Create a `@cache_result` decorator that caches function results.

---

### Phase 2: Intermediate Python

#### 2.1 Pydantic Models

**Why?** Data validation and serialization for APIs.

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"

class DocumentCreate(BaseModel):
    """Request model for creating a document."""
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10)
    doc_type: DocumentType
    tags: list[str] = Field(default_factory=list)

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        return [tag.lower().strip() for tag in v]

class DocumentResponse(BaseModel):
    """Response model for a document."""
    id: str
    title: str
    doc_type: DocumentType
    chunk_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)  # Allow ORM objects

# Usage
doc = DocumentCreate(
    title="  My Document  ",  # Will be stripped
    content="This is the content...",
    doc_type=DocumentType.PDF,
    tags=["RAG", "Python"]
)
print(doc.model_dump_json())  # Serialize to JSON
```

#### 2.2 Abstract Base Classes and Protocols

**Why?** Define interfaces for extensibility.

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# Abstract Base Class approach
class BaseConnector(ABC):
    """Abstract base class for data source connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    def list_documents(self) -> list[str]:
        """List available documents."""
        pass

    @abstractmethod
    def fetch_document(self, doc_id: str) -> str:
        """Fetch document content."""
        pass

class S3Connector(BaseConnector):
    def __init__(self, bucket: str):
        self.bucket = bucket

    def connect(self) -> None:
        self.client = boto3.client('s3')

    def list_documents(self) -> list[str]:
        response = self.client.list_objects_v2(Bucket=self.bucket)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def fetch_document(self, doc_id: str) -> str:
        response = self.client.get_object(Bucket=self.bucket, Key=doc_id)
        return response['Body'].read().decode('utf-8')

# Protocol approach (structural typing - "duck typing" with types)
@runtime_checkable
class Embeddable(Protocol):
    """Any object that can be embedded."""
    def to_text(self) -> str: ...
    def metadata(self) -> dict: ...

class TextChunk:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source

    def to_text(self) -> str:
        return self.text

    def metadata(self) -> dict:
        return {"source": self.source}

def embed_item(item: Embeddable) -> list[float]:
    """Works with any Embeddable object."""
    text = item.to_text()
    return model.encode(text).tolist()

# Both approaches work
chunk = TextChunk("Hello world", "doc1.pdf")
assert isinstance(chunk, Embeddable)  # True! (runtime_checkable)
```

#### 2.3 Dependency Injection with FastAPI

**Why?** Testable, modular code.

```python
from fastapi import FastAPI, Depends, HTTPException
from functools import lru_cache

# Configuration
class Settings(BaseModel):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    groq_api_key: str

    model_config = ConfigDict(env_file=".env")

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Dependencies
def get_qdrant_client(settings: Settings = Depends(get_settings)):
    """Dependency that provides Qdrant client."""
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    try:
        yield client
    finally:
        client.close()

def get_embedding_model():
    """Dependency that provides embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# Service layer
class RAGService:
    def __init__(
        self,
        qdrant: QdrantClient,
        model: SentenceTransformer,
        settings: Settings
    ):
        self.qdrant = qdrant
        self.model = model
        self.settings = settings

    def search(self, query: str, limit: int = 5) -> list[dict]:
        vector = self.model.encode(query).tolist()
        return self.qdrant.search(
            collection_name="documents",
            query_vector=vector,
            limit=limit
        )

def get_rag_service(
    qdrant: QdrantClient = Depends(get_qdrant_client),
    model: SentenceTransformer = Depends(get_embedding_model),
    settings: Settings = Depends(get_settings)
) -> RAGService:
    return RAGService(qdrant, model, settings)

# Route using dependency injection
app = FastAPI()

@app.post("/query")
def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    results = rag_service.search(request.question)
    return {"results": results}

# Easy to test!
def test_query():
    mock_qdrant = MockQdrantClient()
    mock_model = MockEmbeddingModel()
    settings = Settings(groq_api_key="test")

    service = RAGService(mock_qdrant, mock_model, settings)
    results = service.search("test query")
    assert len(results) > 0
```

#### 2.4 Testing with pytest

**Why?** Ensure code quality and catch regressions.

```python
# tests/test_chunking.py
import pytest
from rag.core.chunking import split_into_chunks, ChunkingStrategy

# Basic test
def test_split_into_chunks_basic():
    text = "This is a test. This is another sentence."
    chunks = split_into_chunks(text, max_chars=20)

    assert len(chunks) > 0
    assert all(len(chunk) <= 20 for chunk in chunks)

# Parameterized test - test multiple inputs
@pytest.mark.parametrize("text,expected_count", [
    ("Short", 1),
    ("A" * 100, 1),
    ("A" * 600, 2),  # Should split at ~500 chars
])
def test_chunk_count(text: str, expected_count: int):
    chunks = split_into_chunks(text, max_chars=500)
    assert len(chunks) == expected_count

# Fixtures - reusable test setup
@pytest.fixture
def sample_document() -> str:
    return """
    Introduction to RAG.

    RAG stands for Retrieval Augmented Generation.
    It combines retrieval with generation.
    """

@pytest.fixture
def qdrant_client():
    """Create a test Qdrant client."""
    client = QdrantClient(":memory:")  # In-memory for tests
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    yield client
    client.close()

def test_search_returns_results(qdrant_client, sample_document):
    # Test using fixtures
    add_document(qdrant_client, sample_document)
    results = search(qdrant_client, "What is RAG?")

    assert len(results) > 0
    assert results[0].score > 0.5

# Mocking external services
from unittest.mock import Mock, patch

def test_llm_generation():
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test answer"))]

    with patch('rag.core.llm.groq_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response

        answer = generate_answer("What is RAG?", "RAG is...")

        assert answer == "Test answer"
        mock_client.chat.completions.create.assert_called_once()

# Testing exceptions
def test_empty_document_raises():
    with pytest.raises(ValueError, match="Document cannot be empty"):
        split_into_chunks("")

# Async test
@pytest.mark.asyncio
async def test_async_search():
    results = await async_search("test query")
    assert results is not None
```

---

### Phase 3: Advanced Python

#### 3.1 Async/Await

**Why?** Handle many concurrent I/O operations efficiently.

```python
import asyncio
import aiohttp
from typing import AsyncIterator

# Async function
async def fetch_url(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Async context manager
class AsyncQdrantClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._client = None

    async def __aenter__(self):
        self._client = await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close()

    async def _connect(self):
        # Async connection logic
        pass

# Async generator
async def fetch_documents(doc_ids: list[str]) -> AsyncIterator[str]:
    """Fetch documents concurrently."""
    for doc_id in doc_ids:
        content = await fetch_document(doc_id)
        yield content

# Concurrent execution
async def process_documents_concurrently(doc_ids: list[str]):
    """Process multiple documents at once."""
    tasks = [process_document(doc_id) for doc_id in doc_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for doc_id, result in zip(doc_ids, results):
        if isinstance(result, Exception):
            print(f"Error processing {doc_id}: {result}")
        else:
            print(f"Processed {doc_id}: {result}")

# Semaphore for rate limiting
async def fetch_with_rate_limit(urls: list[str], max_concurrent: int = 10):
    """Fetch URLs with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> str:
        async with semaphore:
            return await fetch_url(url)

    return await asyncio.gather(*[fetch_one(url) for url in urls])

# Running async code
async def main():
    async with AsyncQdrantClient("localhost", 6333) as client:
        results = await client.search("test query")
        print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 3.2 SQLAlchemy ORM

**Why?** Database abstraction and relationships.

```python
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime

Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="tenant")
    users = relationship("User", back_populates="tenant")

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    filename = Column(String(500), nullable=False)
    chunk_count = Column(Integer, default=0)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="documents")

# Database session management
class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

# Repository pattern
class DocumentRepository:
    def __init__(self, session):
        self.session = session

    def create(self, tenant_id: uuid.UUID, filename: str) -> Document:
        doc = Document(tenant_id=tenant_id, filename=filename)
        self.session.add(doc)
        self.session.commit()
        self.session.refresh(doc)
        return doc

    def get_by_tenant(self, tenant_id: uuid.UUID) -> list[Document]:
        return self.session.query(Document).filter(
            Document.tenant_id == tenant_id
        ).all()

    def update_chunk_count(self, doc_id: uuid.UUID, count: int):
        self.session.query(Document).filter(
            Document.id == doc_id
        ).update({"chunk_count": count})
        self.session.commit()
```

#### 3.3 Design Patterns

**Why?** Solve common problems with proven solutions.

```python
# 1. Factory Pattern - Create objects without specifying exact class
class ConnectorFactory:
    _connectors = {}

    @classmethod
    def register(cls, name: str, connector_class):
        cls._connectors[name] = connector_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseConnector:
        if name not in cls._connectors:
            raise ValueError(f"Unknown connector: {name}")
        return cls._connectors[name](**kwargs)

# Register connectors
ConnectorFactory.register("s3", S3Connector)
ConnectorFactory.register("local", LocalConnector)
ConnectorFactory.register("confluence", ConfluenceConnector)

# Usage
connector = ConnectorFactory.create("s3", bucket="my-docs")

# 2. Strategy Pattern - Interchangeable algorithms
class ChunkingStrategy(Protocol):
    def chunk(self, text: str) -> list[str]: ...

class FixedSizeChunking:
    def __init__(self, size: int = 500):
        self.size = size

    def chunk(self, text: str) -> list[str]:
        return [text[i:i+self.size] for i in range(0, len(text), self.size)]

class SentenceChunking:
    def chunk(self, text: str) -> list[str]:
        return sent_tokenize(text)

class DocumentProcessor:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy

    def process(self, text: str) -> list[str]:
        return self.strategy.chunk(text)

# Usage - easily swap strategies
processor = DocumentProcessor(SentenceChunking())
chunks = processor.process(document)

# 3. Singleton Pattern - Single instance
class EmbeddingModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._instance

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

# Both are the same instance
model1 = EmbeddingModel()
model2 = EmbeddingModel()
assert model1 is model2

# 4. Observer Pattern - Event handling
class EventEmitter:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable):
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def emit(self, event: str, *args, **kwargs):
        for handler in self._handlers.get(event, []):
            handler(*args, **kwargs)

# Usage
events = EventEmitter()

@events.on("document_processed")
def log_processing(doc_id: str):
    print(f"Document {doc_id} processed")

@events.on("document_processed")
def update_metrics(doc_id: str):
    metrics.increment("documents_processed")

# Triggers both handlers
events.emit("document_processed", "doc123")
```

---

### Learning Resources