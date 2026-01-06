# Phase 3: Enterprise Features

> **Goal**: Transform the Phase 2 API into a production-ready, multi-tenant system with PostgreSQL, Redis caching, async processing, and observability.

---

## Overview

This phase adds enterprise capabilities to make the RAG system production-ready. By the end, you'll have:

1. PostgreSQL for metadata storage and multi-tenancy
2. API key authentication
3. Redis caching for embeddings and queries
4. Celery for async document processing
5. Prometheus metrics and Grafana dashboards
6. Kubernetes deployment manifests
7. CI/CD pipeline with GitHub Actions

---

## Technology Stack (Phase 3)

| Component | Choice | Why |
|-----------|--------|-----|
| **Metadata DB** | PostgreSQL | ACID, JSON support, mature ecosystem |
| **ORM** | SQLAlchemy 2.0 | Async support, type hints, migrations |
| **Migrations** | Alembic | SQLAlchemy integration, version control |
| **Cache** | Redis | Fast, TTL support, pub/sub |
| **Task Queue** | Celery + Redis | Distributed processing, retries |
| **Metrics** | Prometheus | Industry standard, K8s native |
| **Dashboards** | Grafana | Visualization, alerting |
| **Container Orchestration** | Kubernetes | Scalability, self-healing |
| **CI/CD** | GitHub Actions | Native GitHub integration |

---

## Architecture (Phase 3)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Phase 3: Enterprise Architecture                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Clients (with API Key)                                                     │
│       │                                                                       │
│       ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                    FastAPI Application                               │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │    │
│   │  │  Auth       │  │  Rate       │  │  Prometheus                 │  │    │
│   │  │  Middleware │  │  Limiter    │  │  Metrics                    │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                       │
│       ├──────────────────┬──────────────────┬──────────────────┐             │
│       ▼                  ▼                  ▼                  ▼             │
│   ┌─────────┐      ┌─────────┐      ┌─────────────┐      ┌─────────┐        │
│   │ Sync    │      │ Async   │      │  Cache      │      │ Tenant  │        │
│   │ Queries │      │ Ingest  │      │  Layer      │      │ Filter  │        │
│   └────┬────┘      └────┬────┘      └──────┬──────┘      └────┬────┘        │
│        │                │                   │                  │             │
│        │                ▼                   ▼                  │             │
│        │         ┌─────────────┐     ┌─────────────┐          │             │
│        │         │   Celery    │     │    Redis    │          │             │
│        │         │   Workers   │     │    Cache    │          │             │
│        │         └──────┬──────┘     └─────────────┘          │             │
│        │                │                                      │             │
│        ▼                ▼                                      ▼             │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                         Data Layer                                   │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │    │
│   │  │ PostgreSQL  │  │   Qdrant    │  │  Groq API                   │  │    │
│   │  │ (Metadata)  │  │  (Vectors)  │  │  (LLM)                      │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Python Concepts You'll Learn

| Concept | Where Used |
|---------|------------|
| Async/await and asyncio | Database operations, API handlers |
| SQLAlchemy 2.0 ORM | Models, relationships, queries |
| Alembic migrations | Schema versioning |
| Celery tasks | Async document processing |
| Redis integration | Caching, task broker |
| Design patterns (Factory, Strategy) | Multi-tenant filtering |
| Connection pooling | Database connections |
| Decorators (advanced) | Caching, metrics, auth |
| Context managers (async) | Database sessions |
| Multi-threading vs multi-processing | Celery workers |

---

## Project Structure

```
rag_v1/
├── pyproject.toml
├── .env
├── Dockerfile
├── docker-compose.yml
├── alembic.ini                    # NEW: Alembic config
│
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py              # Updated: Add DB, Redis, Celery config
│       ├── models.py              # (Phase 1)
│       ├── exceptions.py          # (Phase 2)
│       ├── logging_config.py      # (Phase 2)
│       │
│       ├── db/                    # NEW: Database layer
│       │   ├── __init__.py
│       │   ├── base.py            # SQLAlchemy base, engine
│       │   ├── models.py          # ORM models
│       │   ├── repositories.py    # Data access layer
│       │   └── session.py         # Session management
│       │
│       ├── auth/                  # NEW: Authentication
│       │   ├── __init__.py
│       │   ├── api_keys.py        # API key validation
│       │   ├── middleware.py      # Auth middleware
│       │   └── dependencies.py    # Auth dependencies
│       │
│       ├── cache/                 # NEW: Caching layer
│       │   ├── __init__.py
│       │   ├── redis_client.py    # Redis connection
│       │   ├── decorators.py      # Cache decorators
│       │   └── keys.py            # Cache key builders
│       │
│       ├── tasks/                 # NEW: Celery tasks
│       │   ├── __init__.py
│       │   ├── celery_app.py      # Celery configuration
│       │   ├── ingestion.py       # Async ingestion tasks
│       │   └── maintenance.py     # Cleanup tasks
│       │
│       ├── metrics/               # NEW: Observability
│       │   ├── __init__.py
│       │   ├── prometheus.py      # Metrics definitions
│       │   └── middleware.py      # Metrics middleware
│       │
│       ├── tenants/               # NEW: Multi-tenancy
│       │   ├── __init__.py
│       │   ├── context.py         # Tenant context
│       │   └── filters.py         # Tenant filtering
│       │
│       ├── api/                   # (Phase 2, updated)
│       │   ├── main.py
│       │   ├── routes/
│       │   │   ├── documents.py
│       │   │   ├── query.py
│       │   │   ├── health.py
│       │   │   ├── tenants.py     # NEW: Tenant management
│       │   │   └── admin.py       # NEW: Admin endpoints
│       │   ├── schemas/
│       │   └── dependencies.py
│       │
│       ├── services/              # (Phase 2)
│       ├── ingestion/             # (Phase 1)
│       ├── embeddings/            # (Phase 1)
│       ├── vectorstore/           # (Phase 1)
│       ├── retrieval/             # (Phase 1)
│       └── llm/                   # (Phase 1)
│
├── migrations/                    # NEW: Alembic migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
│
├── k8s/                           # NEW: Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── monitoring/
│       ├── prometheus.yaml
│       └── grafana.yaml
│
├── .github/                       # NEW: CI/CD
│   └── workflows/
│       ├── ci.yaml
│       └── cd.yaml
│
└── tests/
    ├── conftest.py
    ├── unit/
    ├── integration/
    └── e2e/                       # NEW: End-to-end tests
```

---

## Implementation Tasks

| # | Task | Priority | Python Concepts | Files |
|---|------|----------|-----------------|-------|
| 1 | Add new dependencies | High | `pyproject.toml` | `pyproject.toml` |
| 2 | Update configuration | High | Pydantic settings | `config.py` |
| 3 | PostgreSQL + SQLAlchemy setup | High | Async SQLAlchemy | `db/` |
| 4 | Database models | High | ORM, relationships | `db/models.py` |
| 5 | Alembic migrations | High | Schema versioning | `migrations/` |
| 6 | Repository pattern | High | Data access layer | `db/repositories.py` |
| 7 | API key authentication | High | Hashing, middleware | `auth/` |
| 8 | Multi-tenancy | High | Context, filtering | `tenants/` |
| 9 | Redis caching | Medium | Decorators, TTL | `cache/` |
| 10 | Celery async tasks | Medium | Task queues | `tasks/` |
| 11 | Prometheus metrics | Medium | Counters, histograms | `metrics/` |
| 12 | Update API endpoints | High | Tenant-aware routes | `api/routes/` |
| 13 | Kubernetes manifests | Medium | YAML, configs | `k8s/` |
| 14 | CI/CD pipeline | Medium | GitHub Actions | `.github/workflows/` |

---

## Step-by-Step Implementation

### Step 1: Add New Dependencies

```bash
# Database
uv add sqlalchemy[asyncio] asyncpg alembic

# Caching
uv add redis

# Task queue
uv add celery[redis]

# Metrics
uv add prometheus-client

# Security
uv add passlib[bcrypt] python-jose[cryptography]

# Rate limiting
uv add slowapi
```

**Update pyproject.toml:**
```toml
[project]
name = "rag-system"
version = "0.3.0"
description = "Enterprise RAG system with multi-tenancy"
requires-python = ">=3.11"

dependencies = [
    # Phase 1 & 2 dependencies
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
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",
    "structlog>=24.1.0",
    "tenacity>=8.2.0",
    # Phase 3 dependencies
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "redis>=5.0.0",
    "celery[redis]>=5.3.0",
    "prometheus-client>=0.19.0",
    "passlib[bcrypt]>=1.7.4",
    "python-jose[cryptography]>=3.3.0",
    "slowapi>=0.1.9",
]
```

---

### Step 2: Update Configuration

```python
# src/rag/config.py
"""Configuration management with environment variables."""
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from functools import lru_cache


class Settings(BaseSettings):
    # === Groq API ===
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = "llama-3.1-8b-instant"

    # === Embedding ===
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768

    # === Qdrant ===
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # === Chunking ===
    chunk_size: int = 512
    chunk_overlap: int = 50

    # === Retrieval ===
    top_k: int = 5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # === Logging ===
    log_level: str = "INFO"
    log_json: bool = True

    # === PostgreSQL (NEW) ===
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    postgres_db: str = "rag"
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync URL for Alembic migrations."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # === Redis (NEW) ===
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 hour default

    @computed_field
    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # === Celery (NEW) ===
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    @computed_field
    @property
    def celery_broker(self) -> str:
        return self.celery_broker_url or self.redis_url

    @computed_field
    @property
    def celery_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    # === Auth (NEW) ===
    api_key_header: str = "X-API-Key"
    api_key_hash_rounds: int = 12
    admin_api_key: str = Field(default="", env="ADMIN_API_KEY")

    # === Rate Limiting (NEW) ===
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

---

### Step 3: PostgreSQL + SQLAlchemy Setup

```python
# src/rag/db/base.py
"""SQLAlchemy base configuration."""
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from rag.config import settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Create async engine
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    echo=False,  # Set True for SQL logging
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """Dependency for getting database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

```python
# src/rag/db/session.py
"""Database session management."""
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from rag.db.base import async_session_factory


@asynccontextmanager
async def get_db_session() -> AsyncSession:
    """Context manager for database sessions."""
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

---

### Step 4: Database Models

```python
# src/rag/db/models.py
"""SQLAlchemy ORM models."""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Text, Integer, Boolean, DateTime, ForeignKey,
    Index, UniqueConstraint, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from rag.db.base import Base
import uuid


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Tenant(Base):
    """Tenant for multi-tenancy."""
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey", back_populates="tenant", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="tenant", cascade="all, delete-orphan"
    )


class APIKey(Base):
    """API keys for authentication."""
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    scopes: Mapped[list] = mapped_column(JSON, default=list)
    rate_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index("ix_api_keys_key_prefix", "key_prefix"),
        Index("ix_api_keys_tenant_id", "tenant_id"),
    )


class Document(Base):
    """Document metadata stored in PostgreSQL."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="documents")

    # Indexes
    __table_args__ = (
        Index("ix_documents_tenant_id", "tenant_id"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_created_at", "created_at"),
    )


class QueryLog(Base):
    """Query audit log."""
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    api_key_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sources_count: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_ms: Mapped[float] = mapped_column(Integer, default=0)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("ix_query_logs_tenant_id", "tenant_id"),
        Index("ix_query_logs_created_at", "created_at"),
    )
```

---

### Step 5: Alembic Migrations

```bash
# Initialize Alembic
alembic init migrations
```

```python
# migrations/env.py
"""Alembic environment configuration."""
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from rag.config import settings
from rag.db.base import Base
from rag.db import models  # Import models to register them

config = context.config

# Set database URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Run migrations
alembic upgrade head
```

---

### Step 6: Repository Pattern

```python
# src/rag/db/repositories.py
"""Data access layer using repository pattern."""
from typing import Optional, Sequence
from datetime import datetime
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.models import Tenant, APIKey, Document, QueryLog


class TenantRepository:
    """Repository for Tenant operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, slug: str, settings: dict = None) -> Tenant:
        tenant = Tenant(name=name, slug=slug, settings=settings or {})
        self.session.add(tenant)
        await self.session.flush()
        return tenant

    async def get_by_id(self, tenant_id: str) -> Optional[Tenant]:
        result = await self.session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        result = await self.session.execute(
            select(Tenant).where(Tenant.slug == slug)
        )
        return result.scalar_one_or_none()

    async def list_active(self, limit: int = 100, offset: int = 0) -> Sequence[Tenant]:
        result = await self.session.execute(
            select(Tenant)
            .where(Tenant.is_active == True)
            .order_by(Tenant.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def update(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        await self.session.execute(
            update(Tenant).where(Tenant.id == tenant_id).values(**kwargs)
        )
        return await self.get_by_id(tenant_id)

    async def delete(self, tenant_id: str) -> bool:
        result = await self.session.execute(
            delete(Tenant).where(Tenant.id == tenant_id)
        )
        return result.rowcount > 0


class APIKeyRepository:
    """Repository for API key operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: str,
        name: str,
        key_hash: str,
        key_prefix: str,
        scopes: list = None,
        rate_limit: int = None,
        expires_at: datetime = None,
    ) -> APIKey:
        api_key = APIKey(
            tenant_id=tenant_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=scopes or ["read", "write"],
            rate_limit=rate_limit,
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.flush()
        return api_key

    async def get_by_prefix(self, prefix: str) -> Sequence[APIKey]:
        """Get API keys by prefix for validation."""
        result = await self.session.execute(
            select(APIKey)
            .where(APIKey.key_prefix == prefix)
            .where(APIKey.is_active == True)
        )
        return result.scalars().all()

    async def get_by_tenant(self, tenant_id: str) -> Sequence[APIKey]:
        result = await self.session.execute(
            select(APIKey).where(APIKey.tenant_id == tenant_id)
        )
        return result.scalars().all()

    async def update_last_used(self, api_key_id: str) -> None:
        await self.session.execute(
            update(APIKey)
            .where(APIKey.id == api_key_id)
            .values(last_used_at=func.now())
        )

    async def deactivate(self, api_key_id: str) -> bool:
        result = await self.session.execute(
            update(APIKey)
            .where(APIKey.id == api_key_id)
            .values(is_active=False)
        )
        return result.rowcount > 0


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        qdrant_collection: str,
        metadata: dict = None,
    ) -> Document:
        doc = Document(
            tenant_id=tenant_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            qdrant_collection=qdrant_collection,
            metadata=metadata or {},
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def get_by_id(
        self, doc_id: str, tenant_id: str = None
    ) -> Optional[Document]:
        query = select(Document).where(Document.id == doc_id)
        if tenant_id:
            query = query.where(Document.tenant_id == tenant_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: str,
        status: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        query = (
            select(Document)
            .where(Document.tenant_id == tenant_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status:
            query = query.where(Document.status == status)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = None,
        error_message: str = None,
    ) -> None:
        values = {"status": status}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if error_message:
            values["error_message"] = error_message
        if status == "completed":
            values["processed_at"] = func.now()

        await self.session.execute(
            update(Document).where(Document.id == doc_id).values(**values)
        )

    async def delete(self, doc_id: str, tenant_id: str) -> bool:
        result = await self.session.execute(
            delete(Document)
            .where(Document.id == doc_id)
            .where(Document.tenant_id == tenant_id)
        )
        return result.rowcount > 0

    async def count_by_tenant(self, tenant_id: str) -> int:
        result = await self.session.execute(
            select(func.count(Document.id)).where(Document.tenant_id == tenant_id)
        )
        return result.scalar_one()


class QueryLogRepository:
    """Repository for query audit logs."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: str,
        query: str,
        answer: str = None,
        sources_count: int = 0,
        processing_time_ms: float = 0,
        api_key_id: str = None,
        tokens_used: int = None,
    ) -> QueryLog:
        log = QueryLog(
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            query=query,
            answer=answer,
            sources_count=sources_count,
            processing_time_ms=processing_time_ms,
            tokens_used=tokens_used,
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_stats(
        self, tenant_id: str, days: int = 30
    ) -> dict:
        """Get query statistics for a tenant."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)

        result = await self.session.execute(
            select(
                func.count(QueryLog.id).label("total_queries"),
                func.avg(QueryLog.processing_time_ms).label("avg_time_ms"),
                func.sum(QueryLog.tokens_used).label("total_tokens"),
            )
            .where(QueryLog.tenant_id == tenant_id)
            .where(QueryLog.created_at >= cutoff)
        )
        row = result.one()
        return {
            "total_queries": row.total_queries or 0,
            "avg_processing_time_ms": float(row.avg_time_ms or 0),
            "total_tokens_used": row.total_tokens or 0,
        }
```

---

### Step 7: API Key Authentication

```python
# src/rag/auth/api_keys.py
"""API key generation and validation."""
import secrets
import hashlib
from passlib.context import CryptContext
from datetime import datetime
from typing import Optional
from rag.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_prefix, key_hash)
    """
    # Generate random key: prefix_random
    prefix = secrets.token_hex(4)  # 8 chars
    random_part = secrets.token_urlsafe(32)  # 43 chars
    full_key = f"rag_{prefix}_{random_part}"

    # Hash the key for storage
    key_hash = pwd_context.hash(full_key)

    return full_key, prefix, key_hash


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return pwd_context.verify(plain_key, hashed_key)


def get_key_prefix(api_key: str) -> Optional[str]:
    """Extract prefix from API key for lookup."""
    if not api_key or not api_key.startswith("rag_"):
        return None
    parts = api_key.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None
```

```python
# src/rag/auth/middleware.py
"""Authentication middleware."""
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
from datetime import datetime

from rag.config import settings
from rag.auth.api_keys import get_key_prefix, verify_api_key
from rag.db.session import get_db_session
from rag.db.repositories import APIKeyRepository


api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/metrics",
    }

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(settings.api_key_header)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Validate API key
        auth_context = await self._validate_api_key(api_key)
        if not auth_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )

        # Add auth context to request state
        request.state.tenant_id = auth_context["tenant_id"]
        request.state.api_key_id = auth_context["api_key_id"]
        request.state.scopes = auth_context["scopes"]

        return await call_next(request)

    async def _validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return auth context."""
        prefix = get_key_prefix(api_key)
        if not prefix:
            return None

        async with get_db_session() as session:
            repo = APIKeyRepository(session)
            api_keys = await repo.get_by_prefix(prefix)

            for key_record in api_keys:
                # Check expiration
                if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                    continue

                # Verify key
                if verify_api_key(api_key, key_record.key_hash):
                    # Update last used
                    await repo.update_last_used(key_record.id)
                    await session.commit()

                    return {
                        "tenant_id": key_record.tenant_id,
                        "api_key_id": key_record.id,
                        "scopes": key_record.scopes,
                    }

        return None
```

```python
# src/rag/auth/dependencies.py
"""Authentication dependencies for FastAPI."""
from fastapi import Request, HTTPException, status, Depends
from typing import Annotated


class AuthContext:
    """Authentication context from validated request."""

    def __init__(self, tenant_id: str, api_key_id: str, scopes: list):
        self.tenant_id = tenant_id
        self.api_key_id = api_key_id
        self.scopes = scopes

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


def get_auth_context(request: Request) -> AuthContext:
    """Get authentication context from request."""
    if not hasattr(request.state, "tenant_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    return AuthContext(
        tenant_id=request.state.tenant_id,
        api_key_id=request.state.api_key_id,
        scopes=request.state.scopes,
    )


def require_scope(scope: str):
    """Dependency to require a specific scope."""

    def check_scope(auth: Annotated[AuthContext, Depends(get_auth_context)]):
        if not auth.has_scope(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {scope}",
            )
        return auth

    return check_scope


# Common dependencies
CurrentTenant = Annotated[AuthContext, Depends(get_auth_context)]
RequireRead = Annotated[AuthContext, Depends(require_scope("read"))]
RequireWrite = Annotated[AuthContext, Depends(require_scope("write"))]
RequireAdmin = Annotated[AuthContext, Depends(require_scope("admin"))]
```

---

### Step 8: Multi-Tenancy

```python
# src/rag/tenants/context.py
"""Tenant context management."""
from contextvars import ContextVar
from typing import Optional
from dataclasses import dataclass


@dataclass
class TenantContext:
    """Current tenant context."""
    tenant_id: str
    tenant_slug: str
    collection_name: str


# Context variable for current tenant
_current_tenant: ContextVar[Optional[TenantContext]] = ContextVar(
    "current_tenant", default=None
)


def get_current_tenant() -> Optional[TenantContext]:
    """Get current tenant context."""
    return _current_tenant.get()


def set_current_tenant(context: TenantContext) -> None:
    """Set current tenant context."""
    _current_tenant.set(context)


def clear_current_tenant() -> None:
    """Clear current tenant context."""
    _current_tenant.set(None)


def get_tenant_collection(tenant_id: str) -> str:
    """Get Qdrant collection name for a tenant."""
    return f"tenant_{tenant_id}_documents"
```

```python
# src/rag/tenants/filters.py
"""Tenant-aware filtering for queries."""
from typing import TypeVar, Generic
from sqlalchemy import Select
from rag.tenants.context import get_current_tenant

T = TypeVar("T")


class TenantFilter(Generic[T]):
    """Apply tenant filtering to queries."""

    @staticmethod
    def apply(query: Select, tenant_column) -> Select:
        """Apply tenant filter to a SQLAlchemy query."""
        tenant = get_current_tenant()
        if tenant:
            return query.where(tenant_column == tenant.tenant_id)
        return query


def tenant_collection_name() -> str:
    """Get collection name for current tenant."""
    tenant = get_current_tenant()
    if tenant:
        return tenant.collection_name
    raise ValueError("No tenant context set")
```

---

### Step 9: Redis Caching

```python
# src/rag/cache/redis_client.py
"""Redis client configuration."""
import redis.asyncio as redis
from rag.config import settings


class RedisClient:
    """Async Redis client wrapper."""

    _instance: redis.Redis = None

    @classmethod
    async def get_client(cls) -> redis.Redis:
        """Get Redis client instance."""
        if cls._instance is None:
            cls._instance = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        """Close Redis connection."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None


async def get_redis() -> redis.Redis:
    """Dependency for getting Redis client."""
    return await RedisClient.get_client()
```

```python
# src/rag/cache/keys.py
"""Cache key builders."""
from typing import Optional


class CacheKeys:
    """Cache key patterns."""

    # Prefixes
    EMBEDDING = "emb"
    QUERY = "query"
    DOCUMENT = "doc"

    @staticmethod
    def embedding(text_hash: str, model: str) -> str:
        """Key for cached embeddings."""
        return f"{CacheKeys.EMBEDDING}:{model}:{text_hash}"

    @staticmethod
    def query_result(tenant_id: str, query_hash: str) -> str:
        """Key for cached query results."""
        return f"{CacheKeys.QUERY}:{tenant_id}:{query_hash}"

    @staticmethod
    def document_metadata(tenant_id: str, doc_id: str) -> str:
        """Key for cached document metadata."""
        return f"{CacheKeys.DOCUMENT}:{tenant_id}:{doc_id}"

    @staticmethod
    def tenant_stats(tenant_id: str) -> str:
        """Key for cached tenant statistics."""
        return f"stats:{tenant_id}"
```

```python
# src/rag/cache/decorators.py
"""Caching decorators."""
import hashlib
import json
from functools import wraps
from typing import Callable, Optional
import redis.asyncio as redis

from rag.cache.redis_client import RedisClient
from rag.config import settings
from rag.logging_config import get_logger

logger = get_logger("cache")


def cache_result(
    key_builder: Callable[..., str],
    ttl: Optional[int] = None,
    prefix: str = "",
):
    """
    Decorator to cache function results in Redis.

    Args:
        key_builder: Function to build cache key from arguments
        ttl: Time to live in seconds (default: settings.cache_ttl)
        prefix: Optional key prefix
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = key_builder(*args, **kwargs)
            if prefix:
                cache_key = f"{prefix}:{cache_key}"

            # Try to get from cache
            try:
                client = await RedisClient.get_client()
                cached = await client.get(cache_key)

                if cached:
                    logger.debug("cache_hit", key=cache_key)
                    return json.loads(cached)
            except Exception as e:
                logger.warning("cache_get_error", key=cache_key, error=str(e))

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await client.setex(
                    cache_key,
                    ttl or settings.cache_ttl,
                    json.dumps(result),
                )
                logger.debug("cache_set", key=cache_key, ttl=ttl)
            except Exception as e:
                logger.warning("cache_set_error", key=cache_key, error=str(e))

            return result

        return wrapper

    return decorator


def invalidate_cache(key_pattern: str):
    """
    Decorator to invalidate cache after function execution.

    Args:
        key_pattern: Redis key pattern to delete (supports *)
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Invalidate cache
            try:
                client = await RedisClient.get_client()
                if "*" in key_pattern:
                    keys = await client.keys(key_pattern)
                    if keys:
                        await client.delete(*keys)
                        logger.debug("cache_invalidated", pattern=key_pattern, count=len(keys))
                else:
                    await client.delete(key_pattern)
                    logger.debug("cache_invalidated", key=key_pattern)
            except Exception as e:
                logger.warning("cache_invalidate_error", pattern=key_pattern, error=str(e))

            return result

        return wrapper

    return decorator


def hash_for_cache(*args) -> str:
    """Create a hash from arguments for cache key."""
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

### Step 10: Celery Async Tasks

```python
# src/rag/tasks/celery_app.py
"""Celery application configuration."""
from celery import Celery
from rag.config import settings

celery_app = Celery(
    "rag",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["rag.tasks.ingestion", "rag.tasks.maintenance"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes (grace period)
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=86400,  # 24 hours
)

# Task routes
celery_app.conf.task_routes = {
    "rag.tasks.ingestion.*": {"queue": "ingestion"},
    "rag.tasks.maintenance.*": {"queue": "maintenance"},
}
```

```python
# src/rag/tasks/ingestion.py
"""Async document ingestion tasks."""
import asyncio
from celery import shared_task
from pathlib import Path
import tempfile

from rag.tasks.celery_app import celery_app
from rag.logging_config import get_logger
from rag.db.session import get_db_session
from rag.db.repositories import DocumentRepository
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import SentenceChunker
from rag.embeddings.embedder import Embedder
from rag.vectorstore.qdrant import QdrantStore
from rag.tenants.context import get_tenant_collection

logger = get_logger("tasks.ingestion")


@celery_app.task(bind=True, max_retries=3)
def process_document_async(
    self,
    document_id: str,
    tenant_id: str,
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """
    Async task to process a document.

    Args:
        document_id: Database document ID
        tenant_id: Tenant ID
        file_path: Path to temporary file
        chunk_size: Chunk size for splitting
        chunk_overlap: Overlap between chunks
    """
    logger.info(
        "processing_document",
        document_id=document_id,
        tenant_id=tenant_id,
    )

    try:
        # Run async processing in event loop
        asyncio.get_event_loop().run_until_complete(
            _process_document(
                document_id=document_id,
                tenant_id=tenant_id,
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    except Exception as e:
        logger.error(
            "processing_failed",
            document_id=document_id,
            error=str(e),
        )
        # Update status to failed
        asyncio.get_event_loop().run_until_complete(
            _update_document_status(document_id, "failed", error_message=str(e))
        )
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
    finally:
        # Cleanup temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


async def _process_document(
    document_id: str,
    tenant_id: str,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Internal async document processing."""
    # Update status to processing
    await _update_document_status(document_id, "processing")

    # Initialize components
    loader = DocumentLoader()
    chunker = SentenceChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    embedder = Embedder()

    # Get tenant-specific collection
    collection_name = get_tenant_collection(tenant_id)
    store = QdrantStore(collection_name=collection_name)

    # Load document
    path = Path(file_path)
    document = loader.load(path)

    # Chunk document
    chunks = list(chunker.chunk(document))
    logger.info("document_chunked", document_id=document_id, chunks=len(chunks))

    if not chunks:
        await _update_document_status(
            document_id, "failed", error_message="No chunks created"
        )
        return

    # Generate embeddings
    texts = [c.content for c in chunks]
    embeddings = embedder.embed(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        # Add tenant metadata
        chunk.metadata["tenant_id"] = tenant_id
        chunk.metadata["document_db_id"] = document_id

    # Store in Qdrant
    store.upsert(chunks)

    # Update status to completed
    await _update_document_status(
        document_id, "completed", chunk_count=len(chunks)
    )

    logger.info(
        "processing_complete",
        document_id=document_id,
        chunks=len(chunks),
    )


async def _update_document_status(
    document_id: str,
    status: str,
    chunk_count: int = None,
    error_message: str = None,
):
    """Update document status in database."""
    async with get_db_session() as session:
        repo = DocumentRepository(session)
        await repo.update_status(
            doc_id=document_id,
            status=status,
            chunk_count=chunk_count,
            error_message=error_message,
        )
        await session.commit()
```

```python
# src/rag/tasks/maintenance.py
"""Maintenance tasks."""
from celery import shared_task
from datetime import datetime, timedelta

from rag.tasks.celery_app import celery_app
from rag.logging_config import get_logger
from rag.cache.redis_client import RedisClient

logger = get_logger("tasks.maintenance")


@celery_app.task
def cleanup_expired_cache():
    """Clean up expired cache entries."""
    logger.info("cleaning_expired_cache")
    # Redis handles TTL automatically, but we can clean up patterns
    # This is more for custom cleanup logic


@celery_app.task
def cleanup_failed_documents(days: int = 7):
    """Clean up documents stuck in failed state."""
    import asyncio
    from rag.db.session import get_db_session
    from sqlalchemy import delete, and_
    from rag.db.models import Document

    async def _cleanup():
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with get_db_session() as session:
            result = await session.execute(
                delete(Document).where(
                    and_(
                        Document.status == "failed",
                        Document.created_at < cutoff,
                    )
                )
            )
            await session.commit()
            logger.info("cleaned_failed_documents", count=result.rowcount)

    asyncio.get_event_loop().run_until_complete(_cleanup())


# Periodic tasks (configured in celery beat)
celery_app.conf.beat_schedule = {
    "cleanup-failed-docs-daily": {
        "task": "rag.tasks.maintenance.cleanup_failed_documents",
        "schedule": 86400,  # Daily
        "args": (7,),
    },
}
```

---

### Step 11: Prometheus Metrics

```python
# src/rag/metrics/prometheus.py
"""Prometheus metrics definitions."""
from prometheus_client import Counter, Histogram, Gauge, Info

# Application info
APP_INFO = Info("rag_app", "RAG application information")
APP_INFO.info({"version": "0.3.0", "python_version": "3.11"})

# Request metrics
REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# RAG-specific metrics
DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
    ["tenant_id", "status"],
)

CHUNKS_CREATED = Counter(
    "rag_chunks_created_total",
    "Total chunks created",
    ["tenant_id"],
)

QUERIES_PROCESSED = Counter(
    "rag_queries_processed_total",
    "Total queries processed",
    ["tenant_id"],
)

QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "Query processing latency",
    ["tenant_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

EMBEDDING_LATENCY = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding generation latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Cache metrics
CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

# Connection pool metrics
DB_CONNECTIONS_ACTIVE = Gauge(
    "rag_db_connections_active",
    "Active database connections",
)

QDRANT_CONNECTIONS_ACTIVE = Gauge(
    "rag_qdrant_connections_active",
    "Active Qdrant connections",
)

# Task queue metrics
CELERY_TASKS_PENDING = Gauge(
    "rag_celery_tasks_pending",
    "Pending Celery tasks",
    ["queue"],
)

CELERY_TASKS_ACTIVE = Gauge(
    "rag_celery_tasks_active",
    "Active Celery tasks",
    ["queue"],
)
```

```python
# src/rag/metrics/middleware.py
"""Metrics middleware for FastAPI."""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from rag.metrics.prometheus import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""

    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()

        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        endpoint = self._get_endpoint(request)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    def _get_endpoint(self, request: Request) -> str:
        """Get normalized endpoint path."""
        path = request.url.path
        # Normalize dynamic segments
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part and len(part) == 36 and "-" in part:
                # Looks like a UUID
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)


async def metrics_endpoint(request: Request) -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
```

---

### Step 12: Update API with Multi-tenancy

```python
# src/rag/api/routes/documents.py (Updated)
"""Document management endpoints with multi-tenancy."""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from typing import Annotated
import tempfile
from pathlib import Path

from rag.api.schemas.documents import IngestResponse, DocumentListResponse
from rag.auth.dependencies import CurrentTenant, RequireWrite
from rag.db.session import get_db_session
from rag.db.repositories import DocumentRepository
from rag.tasks.ingestion import process_document_async
from rag.metrics.prometheus import DOCUMENTS_INGESTED

router = APIRouter(prefix="/documents")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document to ingest")],
    auth: RequireWrite,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """
    Upload and queue a document for async ingestion.
    """
    # Validate file type
    allowed_types = {".pdf", ".txt", ".md", ".docx"}
    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Create document record
    async with get_db_session() as session:
        repo = DocumentRepository(session)
        doc = await repo.create(
            tenant_id=auth.tenant_id,
            filename=filename,
            file_type=suffix,
            file_size=file_size,
            qdrant_collection=f"tenant_{auth.tenant_id}_documents",
        )
        await session.commit()
        doc_id = doc.id

    # Queue async processing
    process_document_async.delay(
        document_id=doc_id,
        tenant_id=auth.tenant_id,
        file_path=tmp_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    DOCUMENTS_INGESTED.labels(tenant_id=auth.tenant_id, status="queued").inc()

    return IngestResponse(
        document_id=doc_id,
        filename=filename,
        chunks_created=0,  # Will be updated async
        message="Document queued for processing",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    auth: CurrentTenant,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """List documents for current tenant."""
    async with get_db_session() as session:
        repo = DocumentRepository(session)
        documents = await repo.list_by_tenant(
            tenant_id=auth.tenant_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        total = await repo.count_by_tenant(auth.tenant_id)

    return DocumentListResponse(
        documents=[
            {
                "id": doc.id,
                "filename": doc.filename,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at.isoformat(),
            }
            for doc in documents
        ],
        total=total,
    )


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    auth: CurrentTenant,
):
    """Get document details."""
    async with get_db_session() as session:
        repo = DocumentRepository(session)
        doc = await repo.get_by_id(document_id, tenant_id=auth.tenant_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat(),
        "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    auth: RequireWrite,
):
    """Delete a document and its vectors."""
    from rag.vectorstore.qdrant import QdrantStore
    from rag.tenants.context import get_tenant_collection

    async with get_db_session() as session:
        repo = DocumentRepository(session)
        doc = await repo.get_by_id(document_id, tenant_id=auth.tenant_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from Qdrant
        collection = get_tenant_collection(auth.tenant_id)
        store = QdrantStore(collection_name=collection)
        store.delete_by_document_id(document_id)

        # Delete from database
        await repo.delete(document_id, auth.tenant_id)
        await session.commit()

    return {"success": True, "message": f"Document {document_id} deleted"}
```

```python
# src/rag/api/routes/tenants.py
"""Tenant management endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
from pydantic import BaseModel

from rag.auth.dependencies import RequireAdmin
from rag.auth.api_keys import generate_api_key
from rag.db.session import get_db_session
from rag.db.repositories import TenantRepository, APIKeyRepository

router = APIRouter(prefix="/tenants")


class CreateTenantRequest(BaseModel):
    name: str
    slug: str
    settings: dict = {}


class CreateAPIKeyRequest(BaseModel):
    name: str
    scopes: list[str] = ["read", "write"]


@router.post("")
async def create_tenant(
    request: CreateTenantRequest,
    auth: RequireAdmin,
):
    """Create a new tenant (admin only)."""
    async with get_db_session() as session:
        repo = TenantRepository(session)

        # Check if slug exists
        existing = await repo.get_by_slug(request.slug)
        if existing:
            raise HTTPException(status_code=400, detail="Slug already exists")

        tenant = await repo.create(
            name=request.name,
            slug=request.slug,
            settings=request.settings,
        )
        await session.commit()

        return {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
            "created_at": tenant.created_at.isoformat(),
        }


@router.get("")
async def list_tenants(auth: RequireAdmin):
    """List all tenants (admin only)."""
    async with get_db_session() as session:
        repo = TenantRepository(session)
        tenants = await repo.list_active()

        return {
            "tenants": [
                {
                    "id": t.id,
                    "name": t.name,
                    "slug": t.slug,
                    "is_active": t.is_active,
                    "created_at": t.created_at.isoformat(),
                }
                for t in tenants
            ]
        }


@router.post("/{tenant_id}/api-keys")
async def create_api_key(
    tenant_id: str,
    request: CreateAPIKeyRequest,
    auth: RequireAdmin,
):
    """Create an API key for a tenant (admin only)."""
    async with get_db_session() as session:
        tenant_repo = TenantRepository(session)
        key_repo = APIKeyRepository(session)

        # Verify tenant exists
        tenant = await tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Generate API key
        full_key, prefix, key_hash = generate_api_key()

        api_key = await key_repo.create(
            tenant_id=tenant_id,
            name=request.name,
            key_hash=key_hash,
            key_prefix=prefix,
            scopes=request.scopes,
        )
        await session.commit()

        return {
            "id": api_key.id,
            "name": api_key.name,
            "key": full_key,  # Only shown once!
            "key_prefix": prefix,
            "scopes": api_key.scopes,
            "message": "Save this key - it won't be shown again!",
        }
```

---

### Step 13: Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    app.kubernetes.io/name: rag-system
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  QDRANT_HOST: "qdrant.rag-system.svc.cluster.local"
  QDRANT_PORT: "6333"
  POSTGRES_HOST: "postgres.rag-system.svc.cluster.local"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "rag"
  REDIS_HOST: "redis.rag-system.svc.cluster.local"
  REDIS_PORT: "6379"
  LOG_LEVEL: "INFO"
  LOG_JSON: "true"
```

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
stringData:
  GROQ_API_KEY: "your-groq-api-key"
  POSTGRES_PASSWORD: "your-db-password"
  REDIS_PASSWORD: ""
  ADMIN_API_KEY: "your-admin-api-key"
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: rag-api
          image: rag-system:0.3.0
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /api/v1/health/live
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /api/v1/health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-worker
  namespace: rag-system
  labels:
    app: rag-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-worker
  template:
    metadata:
      labels:
        app: rag-worker
    spec:
      containers:
        - name: rag-worker
          image: rag-system:0.3.0
          command: ["celery", "-A", "rag.tasks.celery_app", "worker", "-l", "info"]
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
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

```yaml
# k8s/hpa.yaml
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
```

---

### Step 14: CI/CD Pipeline

```yaml
# .github/workflows/ci.yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Run ruff
        run: ruff check .

      - name: Run mypy
        run: mypy src/rag

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: rag
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: rag_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

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
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Run tests
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PASSWORD: testpass
          REDIS_HOST: localhost
          QDRANT_HOST: localhost
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          pytest tests/ -v --cov=rag --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: rag-system:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

```yaml
# .github/workflows/cd.yaml
name: CD

on:
  push:
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

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
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/rag-api \
            rag-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }} \
            -n rag-system

          kubectl set image deployment/rag-worker \
            rag-worker=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }} \
            -n rag-system

          kubectl rollout status deployment/rag-api -n rag-system
          kubectl rollout status deployment/rag-worker -n rag-system
```

---

## Running Phase 3

### Local Development

```bash
# Start infrastructure
docker-compose up -d postgres redis qdrant

# Run migrations
alembic upgrade head

# Start Celery worker
celery -A rag.tasks.celery_app worker -l info -Q ingestion,maintenance

# Start API
uvicorn rag.api.main:app --reload --port 8000
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml (Updated for Phase 3)
version: "3.8"

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - REDIS_HOST=redis
      - QDRANT_HOST=qdrant
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy

  rag-worker:
    build: .
    command: celery -A rag.tasks.celery_app worker -l info
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - REDIS_HOST=redis
      - QDRANT_HOST=qdrant
    depends_on:
      - postgres
      - redis
      - qdrant

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=rag
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=rag
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./k8s/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

### API Usage Examples

```bash
# Create tenant (admin)
curl -X POST "http://localhost:8000/api/v1/tenants" \
  -H "X-API-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme"}'

# Create API key for tenant
curl -X POST "http://localhost:8000/api/v1/tenants/{tenant_id}/api-keys" \
  -H "X-API-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "Production Key", "scopes": ["read", "write"]}'

# Ingest document (async)
curl -X POST "http://localhost:8000/api/v1/documents/ingest" \
  -H "X-API-Key: your-tenant-key" \
  -F "file=@document.pdf"

# Check document status
curl "http://localhost:8000/api/v1/documents/{doc_id}" \
  -H "X-API-Key: your-tenant-key"

# Query (tenant-isolated)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your-tenant-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'

# View metrics
curl http://localhost:8000/metrics
```

---

## Milestone Checklist

- [ ] PostgreSQL with SQLAlchemy async configured
- [ ] Alembic migrations working
- [ ] Repository pattern for data access
- [ ] API key authentication with hashing
- [ ] Multi-tenant data isolation
- [ ] Redis caching with decorators
- [ ] Celery async document processing
- [ ] Prometheus metrics exposed
- [ ] All endpoints tenant-aware
- [ ] Kubernetes manifests ready
- [ ] CI pipeline with lint, test, build
- [ ] CD pipeline with auto-deploy
- [ ] docker-compose full stack working
- [ ] Documentation updated

---

## Next Steps

After completing Phase 3:

1. **Phase 4**: Add advanced RAG techniques (Agentic RAG, Graph RAG, Multi-modal)
2. **Production Hardening**: Add rate limiting, circuit breakers, advanced monitoring

---

**Ready to start?** Begin with Step 1: Add New Dependencies!
