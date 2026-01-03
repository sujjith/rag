# RAG System Commands - Phase 01 MVP

Quick reference guide for testing and using your RAG system.

## 🚀 Setup (One-time)

```bash
cd /home/sujith/github/rag/rag_v1

# Start Qdrant
docker compose up -d

# Install RAG package
uv sync
```

## 📊 Check Status

```bash
uv run rag status
```

Shows:
- Qdrant connection status
- Number of vectors stored
- Embedding model configuration
- LLM model configuration
- Reranker status

## 📥 Ingest Documents

### Single file:
```bash
uv run rag ingest data/documents/macbeth.pdf
uv run rag ingest /path/to/your/document.txt
```

### Entire directory:
```bash
uv run rag ingest data/documents/
uv run rag ingest /path/to/your/documents/
```

**Supported formats:** `.pdf`, `.txt`, `.md`, `.docx`

## 💬 Ask Questions

### Basic query (5 results):
```bash
uv run rag ask "What is the main theme of Macbeth?"
```

### With reranking (better accuracy):
```bash
uv run rag ask "Who killed Duncan?" --rerank
```

### Specify number of results:
```bash
uv run rag ask "What happens in Act 1?" --top-k 3
```

### Reranking + custom results:
```bash
uv run rag ask "Describe Lady Macbeth" --rerank --top-k 5
```

## 🛑 Stop Services

```bash
docker compose down
```

## 🧪 Run Tests

```bash
# All tests
uv run pytest -v

# Specific test file
uv run pytest tests/test_cli.py -v
uv run pytest tests/test_qdrant.py -v
uv run pytest tests/test_loader.py -v
uv run pytest tests/test_reranker.py -v
uv run pytest tests/test_groq.py -v
```

## 📁 Add Your Own Documents

```bash
# Create your documents folder
mkdir -p data/my_docs

# Copy your PDFs/text files there
cp ~/Downloads/*.pdf data/my_docs/

# Ingest them
uv run rag ingest data/my_docs/
```

## 🔄 Quick Test Workflow

```bash
# 1. Check everything is running
uv run rag status

# 2. Add a document
uv run rag ingest data/documents/macbeth.pdf

# 3. Ask a question
uv run rag ask "Who is the main character?"

# 4. Ask with reranking for better results
uv run rag ask "What are the major themes?" --rerank --top-k 3
```

## 📝 Configuration

Configuration is managed via `.env` file in `/home/sujith/github/rag/rag_v1/.env`:

```bash
# Groq API (for LLM)
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6335
COLLECTION_NAME=documents

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval
TOP_K=5
USE_RERANKER=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 🐛 Troubleshooting

### Qdrant not connecting:
```bash
# Check if Qdrant is running
docker ps

# Check Qdrant logs
docker compose logs qdrant

# Restart Qdrant
docker compose restart qdrant
```

### API key error:
```bash
# Check if .env file exists
ls -la .env

# Verify API key is set
grep GROQ_API_KEY .env
```

### Import errors:
```bash
# Reinstall package
uv sync
```
