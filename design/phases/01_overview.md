# RAG System - Overview

> **100% Open Source, Self-Hosted on Ubuntu**
> All components can run on your Ubuntu machine without external dependencies.

---

## What is RAG?

**RAG (Retrieval Augmented Generation)** is a technique that enhances Large Language Models by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the user's question with retrieved context
3. **Generating** accurate, grounded answers using an LLM

### Why Build This?

- **Knowledge Grounding**: Answers based on YOUR documents, not just LLM training data
- **Reduced Hallucination**: LLM responses are grounded in retrieved context
- **Up-to-date Information**: Add new documents anytime, no model retraining
- **Private Data**: Keep sensitive documents within your infrastructure
- **100% Open Source**: No vendor lock-in, complete control

---

## Technology Stack - Open Source Components

### Core RAG Components

| Component | Primary Choice | Open Source Alternatives |
|-----------|---------------|-------------------------|
| **Vector Database** | Qdrant | Milvus, Weaviate, ChromaDB, pgvector (PostgreSQL extension) |
| **Embedding Model** | sentence-transformers | all-MiniLM-L6-v2, BGE, E5, Instructor |
| **LLM** | Groq API (cloud) | Ollama (local), llama.cpp, vLLM, OpenAI API |
| **Document Processing** | PyMuPDF, python-docx | Apache Tika, Pandoc, unstructured.io |
| **Chunking** | nltk, spacy | LangChain TextSplitter, semantic-chunking |

### API & Backend

| Component | Primary Choice | Open Source Alternatives |
|-----------|---------------|-------------------------|
| **Web Framework** | FastAPI | Flask, Django, Litestar, Starlette |
| **Validation** | Pydantic | marshmallow, attrs, dataclasses |
| **Task Queue** | Celery | Huey, RQ, Dramatiq, TaskTiger |
| **Message Broker** | Redis | RabbitMQ, Apache Kafka, NATS |
| **Database** | PostgreSQL | MySQL, MariaDB, SQLite |
| **ORM** | SQLAlchemy | Django ORM, Peewee, SQLModel |

### Infrastructure & Observability

| Component | Primary Choice | Open Source Alternatives |
|-----------|---------------|-------------------------|
| **Container** | Docker | Podman, LXC/LXD |
| **Orchestration** | Kubernetes (K3s) | Docker Swarm, Nomad, OpenShift |
| **Metrics** | Prometheus | VictoriaMetrics, Thanos, Cortex |
| **Visualization** | Grafana | Kibana, Chronograf, Netdata |
| **Logging** | Loki + Promtail | ELK Stack (Elasticsearch + Logstash + Kibana), Fluentd |
| **Tracing** | Jaeger | Zipkin, SigNoz, Tempo |
| **Load Balancer** | Traefik | Nginx, HAProxy, Caddy, Envoy |

### Development & Testing

| Component | Primary Choice | Open Source Alternatives |
|-----------|---------------|-------------------------|
| **Testing** | pytest | unittest, nose2, Robot Framework |
| **Code Quality** | pylint, black | flake8, ruff, isort, mypy |
| **CI/CD** | GitLab CI | Jenkins, Drone CI, Woodpecker CI, Gitea Actions |
| **Git Hosting** | GitLab (self-hosted) | Gitea, Forgejo, Gogs |

---

## High-Level Architecture (Open Source Stack)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Enterprise RAG System (Open Source + Groq API)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Documents (PDF, TXT, MD, DOCX)                                        │
│       ↓                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────┐│
│   │  PyMuPDF /  │ →  │    nltk /   │ →  │  sentence-   │ →  │ Qdrant  ││
│   │  Pandoc     │    │    spaCy    │    │ transformers │    │ Vector  ││
│   │  (Extract)  │    │  (Chunk)    │    │   (Embed)    │    │   DB    ││
│   └─────────────┘    └─────────────┘    └──────────────┘    └────┬────┘│
│                                                                    │     │
│   User Question                                                    │     │
│       ↓                                                            ↓     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐│
│   │   Embed     │ →  │   Search    │ →  │   Rerank    │ →  │ Groq API ││
│   │   Query     │    │   Qdrant    │    │  (optional) │    │ (Cloud)  ││
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────┬────┘│
│                                                                    │     │
│                                                                    ↓     │
│                                              ┌──────────────────────────┐│
│                                              │   Answer + Citations    ││
│                                              └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘

Note: Groq API is cloud-based, everything else runs on your Ubuntu server
```

---

## LLM Options

### Option 1: Groq API (Recommended) ⭐

**Why Groq:**
- ⚡ **Blazingly fast** - 500+ tokens/second
- 💰 **Free tier** - Generous limits for development
- 🔧 **Easy setup** - Just an API key
- 🔌 **OpenAI-compatible** - Drop-in replacement

```bash
# Get API key from https://console.groq.com
export GROQ_API_KEY="your-key-here"

# Install Python client
uv pip install groq

# Use in code
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",  # or mixtral-8x7b-32768
    messages=[{"role": "user", "content": "What is RAG?"}]
)
print(response.choices[0].message.content)
```

**Available Models:**
- `llama-3.1-70b-versatile` - Best quality, reasoning
- `llama-3.1-8b-instant` - Fast, good for simple tasks
- `mixtral-8x7b-32768` - Large context window (32K tokens)
- `gemma2-9b-it` - Efficient, good quality

**Pricing:**
- Free tier: 30 requests/minute, 6000 requests/day
- Paid: $0.27-0.79 per 1M tokens (very cheap!)

**Pros:** Fast, cheap, no local GPU needed
**Cons:** Requires internet, data leaves your server

---

### Option 2: Ollama (Optional - Fully Local)

Use this if you need **100% offline** or have sensitive data.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:3b    # Small, fast (2GB)
ollama pull mistral:7b     # Good quality (4GB)
ollama pull llama3.1:70b   # Best quality (40GB - needs powerful PC)

# Use in Python
from ollama import Client
client = Client()
response = client.chat(model='llama3.2:3b', messages=[...])
```

**Hardware Requirements:**
| Model | RAM Needed | Speed | Quality |
|-------|-----------|-------|---------|
| llama3.2:3b | 8GB | Fast | Good |
| mistral:7b | 16GB | Medium | Very Good |
| llama3.1:70b | 64GB+ | Slow | Excellent |

**Pros:** 100% private, no API costs, offline
**Cons:** Requires powerful hardware, slower than Groq

---

### Option 3: Other APIs (Alternatives)

If Groq doesn't work for you:

```python
# OpenAI (most capable, expensive)
from openai import OpenAI
client = OpenAI(api_key="...")
# Cost: $0.60-$60 per 1M tokens

# Together AI (many open models)
# https://api.together.xyz
# Cost: $0.20-$1.20 per 1M tokens

# Replicate (pay-per-use)
# https://replicate.com
# Cost: $0.05 per 1M tokens
```

---

### Recommended Setup: Groq + Ollama Fallback

**Best of both worlds:**

```python
import os
from groq import Groq
from ollama import Client as OllamaClient

class LLMClient:
    def __init__(self):
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.ollama = OllamaClient()  # Fallback
    
    def generate(self, prompt: str) -> str:
        try:
            # Try Groq first (fast!)
            response = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to local Ollama
            print(f"Groq failed, using Ollama: {e}")
            response = self.ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
```

**For this project, we'll use Groq API as primary.**

---

### Option 1: Ollama (Recommended for Beginners)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2  # 3B parameters, fast
ollama pull mistral   # 7B parameters, good quality
ollama pull llama3.1  # 70B parameters, best quality (needs 40GB+ RAM)

# Use in Python
from ollama import Client
client = Client()
response = client.chat(model='llama3.2', messages=[...])
```

**Pros:** Easy to use, automatic model management
**Cons:** Slightly less control than llama.cpp

### Option 2: llama.cpp (Maximum Control)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a model (GGUF format)
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Run server
./server -m llama-2-7b.Q4_K_M.gguf -c 2048

# Use OpenAI-compatible API
import openai
openai.api_base = "http://localhost:8080/v1"
```

**Pros:** Maximum performance, GPU acceleration
**Cons:** More manual setup

### Option 3: vLLM (Production-Grade)

```bash
# Install vLLM
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000

# OpenAI-compatible API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "..."}'
```

**Pros:** Best for production, high throughput
**Cons:** Higher memory usage

### Model Recommendations by Hardware

| Hardware | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| **8GB RAM** | Llama 3.2 3B | 2GB | Fast | Good |
| **16GB RAM** | Mistral 7B | 4GB | Medium | Very Good |
| **32GB RAM** | Llama 3.1 13B | 8GB | Medium | Excellent |
| **64GB+ RAM** | Llama 3.1 70B | 40GB | Slow | Best |

---

## Goals & Requirements

### Functional Requirements

| ID | Requirement | Open Source Solution |
|----|-------------|---------------------|
| FR-1 | Ingest documents (PDF, TXT, MD, DOCX) | PyMuPDF, python-docx, Pandoc |
| FR-2 | Semantic search across documents | Qdrant, sentence-transformers |
| FR-3 | Question answering with source citations | Ollama, llama.cpp, vLLM |
| FR-4 | REST API for all operations | FastAPI, Pydantic |
| FR-5 | Multi-tenant data isolation | PostgreSQL + SQLAlchemy |
| FR-6 | Document-level access control | PostgreSQL RBAC |
| FR-7 | Conversation history / multi-turn | PostgreSQL, Redis |
| FR-8 | 50+ data source connectors | Apache Airflow, Custom connectors |

### Non-Functional Requirements

| ID | Requirement | Target | Open Source Solution |
|----|-------------|--------|---------------------|
| NFR-1 | Query latency (P95) | < 2 seconds | Qdrant + Ollama (local) |
| NFR-2 | Availability | 99.9% uptime | Kubernetes (K3s) + HAProxy |
| NFR-3 | Concurrent users | 1000+ | Gunicorn + Nginx |
| NFR-4 | Document scale | 10M+ chunks | Qdrant (scales to billions) |
| NFR-5 | Security | Encryption, RBAC | PostgreSQL + TLS |

---

## System Architecture (Detailed - Open Source)

```
                            ┌─────────────────┐
                            │   Caddy / Nginx │
                            │  (TLS Termination)
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Traefik /     │
                            │   HAProxy       │
                            │ (Load Balancer) │
                            └────────┬────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
  ┌────────▼────────┐     ┌─────────▼────────┐     ┌─────────▼────────┐
  │   FastAPI #1    │     │   FastAPI #2     │     │   FastAPI #3     │
  │   (Gunicorn)    │     │   (Gunicorn)     │     │   (Gunicorn)     │
  └────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           │                         │                         │
  ┌────────▼────────┐     ┌─────────▼────────┐     ┌─────────▼────────┐
  │   PostgreSQL    │     │     Qdrant       │     │   Ollama / vLLM  │
  │   (Metadata)    │     │  (Vector Store)  │     │   (LLM Server)   │
  └─────────────────┘     └──────────────────┘     └──────────────────┘
           │                         
  ┌────────▼────────┐     ┌──────────────────┐     ┌──────────────────┐
  │     Redis       │     │  Celery Workers  │     │   Prometheus +   │
  │ (Cache/Sessions)│     │  (Background)    │     │   Grafana        │
  └─────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Deployment Options

### Option 1: Single Server (Development)

**Hardware:** 16GB RAM, 4 CPU cores, 100GB SSD
**Stack:**
- Docker Compose
- All services on one machine
- Perfect for learning and testing

```bash
# docker-compose.yml has everything
docker-compose up -d
```

### Option 2: Multi-Server (Small Production)

**Hardware:** 3 servers (8GB RAM each)
**Stack:**
- Server 1: FastAPI + Nginx
- Server 2: PostgreSQL + Qdrant
- Server 3: Ollama + Redis

### Option 3: Kubernetes (Enterprise)

**Hardware:** 5+ nodes
**Stack:**
- K3s (lightweight Kubernetes)
- Helm charts for deployment
- Auto-scaling, high availability

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -

# Deploy RAG system
kubectl apply -f k8s/
```

---

## Cost Comparison: Open Source + Groq vs Full Cloud

| Component | Open Source (Ubuntu + Groq) | Full Cloud (AWS/Azure) |
|-----------|---------------------------|-------------------|
| **Compute** | $50-200/month (VPS) | $500-2000/month |
| **Storage** | Included in VPS | $100-500/month (S3, EBS) |
| **LLM** | $10-50/month (Groq API) | $1000-5000/month (OpenAI API) |
| **Vector DB** | Free (Qdrant self-hosted) | $200-1000/month (Pinecone) |
| **Monitoring** | Free (Prometheus/Grafana) | $100-300/month (DataDog) |
| **Total** | **$60-250/month** | **$1900-8800/month** |

**Savings:** 85-95% cost reduction! 🎉

**Groq Free Tier:**
- 30 requests/minute
- 6000 requests/day
- Perfect for development and small projects
- Upgrade to paid later if needed

---

## Next Steps

1. **Review this architecture** - Make sure it fits your needs
2. **Choose your LLM** - Ollama (easy) or llama.cpp (control)
3. **Proceed to Quick Start** - [02_quick_start.md](02_quick_start.md)
4. **Start Phase 1** - [03_phase_01_mvp.md](03_phase_01_mvp.md)

---

## Ubuntu-Specific Setup Notes

### System Requirements

**Minimum (Development):**
- Ubuntu 22.04 LTS or 24.04 LTS
- 16GB RAM
- 4 CPU cores
- 50GB free disk space

**Recommended (Production):**
- Ubuntu 22.04 LTS (better LTS support)
- 32GB+ RAM
- 8+ CPU cores
- 200GB+ SSD

### Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3-pip

# Install system dependencies
sudo apt install build-essential git curl wget

# Reboot
sudo reboot
```

---

**Ready to build?** → [02_quick_start.md](02_quick_start.md)
