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

