# Phase 09: Specialization & Projects

**Duration**: 4 weeks | **Prerequisites**: Phases 01-08 completed

---

## Overview

Build 4 end-to-end projects to consolidate all skills learned.

---

## Project 1: Complete ML Pipeline (Week 1)

### Goal
Build production ML pipeline with DVC, MLflow, and Prefect.

### Components
- [ ] Data versioning with DVC
- [ ] Experiment tracking with MLflow
- [ ] Pipeline orchestration with Prefect
- [ ] Model registry and deployment
- [ ] Monitoring with Evidently

### Architecture
```
Data (DVC) → Prefect Pipeline → MLflow Tracking → BentoML → Prometheus/Grafana
```

---

## Project 2: RAG Application (Week 2)

### Goal
Build production RAG system with evaluation.

### Components
- [ ] Document ingestion pipeline
- [ ] Embedding generation
- [ ] Qdrant vector store
- [ ] LangChain/LlamaIndex RAG
- [ ] Ragas evaluation
- [ ] FastAPI interface

### Architecture
```
Documents → Embeddings → Qdrant → RAG Chain → Evaluation → API
```

---

## Project 3: Multi-Agent System (Week 3)

### Goal
Build collaborative AI agents.

### Components
- [ ] LangGraph state machine
- [ ] Tool integration
- [ ] Memory management
- [ ] Guardrails
- [ ] Agent orchestration

### Architecture
```
User → Router Agent → [Research Agent, Code Agent, Writer Agent] → Output
```

---

## Project 4: MLOps Platform (Week 4)

### Goal
Deploy complete MLOps infrastructure.

### Components
- [ ] Kubernetes cluster
- [ ] GitHub Actions CI/CD
- [ ] MLflow server
- [ ] Prometheus + Grafana
- [ ] Documentation (MkDocs)

### Architecture
```
Git Push → GitHub Actions → Docker Build → K8s Deploy → Monitoring
```

---

## Documentation

### MkDocs Setup
```bash
uv add mkdocs mkdocs-material
mkdocs new docs
mkdocs serve
```

### Model Cards
```bash
uv add model-card-toolkit
```

```python
from model_card_toolkit import ModelCardToolkit
toolkit = ModelCardToolkit()
model_card = toolkit.scaffold_assets()
```

---

## Final Checklist

- [ ] Project 1: ML Pipeline completed
- [ ] Project 2: RAG Application completed
- [ ] Project 3: Multi-Agent System completed
- [ ] Project 4: MLOps Platform completed
- [ ] All projects documented
- [ ] Portfolio ready

---

## Congratulations! 🎉

You've completed the 9-phase MLOps & AIOps learning journey covering:

| Phase | Skills |
|-------|--------|
| 01 | Python, uv, Jupyter |
| 02 | MLflow, DVC, Prefect |
| 03 | FastAPI, BentoML, Ray Serve |
| 04 | Polars, Great Expectations, Feast |
| 05 | Evidently, Prometheus, Grafana |
| 06 | Ollama, LangChain, RAG |
| 07 | Docker, Kubernetes, CI/CD |
| 08 | Agents, ONNX, SHAP |
| 09 | End-to-end Projects |

**Total Duration**: ~26 weeks
