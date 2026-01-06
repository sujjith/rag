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

