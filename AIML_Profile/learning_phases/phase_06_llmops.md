# Phase 06: LLMOps & RAG

**Duration**: 4 weeks | **Prerequisites**: Phase 05 completed

---

## Week 1: Local LLMs

### Day 1-2: Ollama Setup
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

```python
import ollama
response = ollama.chat(model='llama3.2:3b', messages=[
    {'role': 'user', 'content': 'Explain MLOps'}
])
```

### Day 3-7: Embeddings
```bash
uv add sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["MLOps is...", "Machine learning..."])
```

---

## Week 2: Vector Databases

### Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
uv add qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(host="localhost", port=6333)
client.create_collection("docs", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
```

---

## Week 3: RAG Pipeline

### LangChain RAG
```bash
uv add langchain langchain-community langchain-ollama
```

```python
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA

llm = OllamaLLM(model="llama3.2:3b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

---

## Week 4: Evaluation

### Ragas
```bash
uv add ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

---

## Milestone Checklist
- [ ] Ollama running locally
- [ ] Embeddings with Sentence Transformers
- [ ] Qdrant vector search working
- [ ] RAG pipeline with LangChain
- [ ] Evaluation with Ragas

**Next**: [Phase 07](./phase_07_infrastructure.md)
