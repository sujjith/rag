# RAG System - Quick Start

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
