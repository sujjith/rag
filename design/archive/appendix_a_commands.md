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

