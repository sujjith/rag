# Production Runbook

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

