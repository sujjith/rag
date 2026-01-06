# Enterprise Infrastructure

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

