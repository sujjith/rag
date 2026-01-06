# Enterprise RAG System - Design Document

> **Project**: Enterprise-grade Retrieval Augmented Generation (RAG) System
> **Version**: 1.0
> **Status**: Design Phase
> **Repository**: `/home/sujith/github/rag`
> **Purpose**: Python Learning Project + Production RAG System

---

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

#### 1.2 Semantic Chunking (Advanced)
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

#### 1.3 Document Structure-Aware Chunking
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

#### 2.2 Instruction-Tuned Embeddings
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
        │  │ (Metrics)│  │(Dashbord)│  │ (Traces) │  │  (Logs)  │    │
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
