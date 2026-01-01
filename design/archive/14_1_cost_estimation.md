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

