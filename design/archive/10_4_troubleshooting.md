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

