# Testing & Debugging Guide

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
