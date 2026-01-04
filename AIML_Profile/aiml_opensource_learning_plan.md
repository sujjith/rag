# AIML/MLOps/AIOps Learning Plan

> A 9-phase structured approach to mastering open-source MLOps & AIOps tools

---

## Overview

| Phase | Focus Area | Duration | Key Outcomes |
|-------|------------|----------|--------------|
| 01 | Foundations & Dev Environment | 2 weeks | Python, Jupyter, Git basics |
| 02 | ML Experiment Lifecycle | 3 weeks | Tracking, versioning, pipelines |
| 03 | Model Serving & APIs | 2 weeks | Deploy models as APIs |
| 04 | Data Engineering | 3 weeks | ETL, feature stores, validation |
| 05 | Monitoring & Observability | 2 weeks | Drift detection, dashboards |
| 06 | LLMOps & RAG | 4 weeks | LLMs, embeddings, vector DBs |
| 07 | Production Infrastructure | 3 weeks | Docker, K8s, CI/CD, security |
| 08 | Advanced Topics | 3 weeks | Agents, optimization, privacy |
| 09 | Specialization & Projects | 4 weeks | Build end-to-end projects |

**Total Duration**: ~26 weeks (6 months)

---

## Phase 01: Foundations & Dev Environment
**Duration**: 2 weeks

### Goals
- Set up local development environment
- Master Python project management with `uv`
- Get comfortable with notebooks

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | ⭐⭐⭐ |
| JupyterLab | `uv add jupyterlab` | ⭐⭐⭐ |
| Marimo | `uv add marimo` | ⭐⭐ |
| VS Code + Extensions | Download | ⭐⭐⭐ |

### Hands-on Tasks
- [ ] Install uv and create first project
- [ ] Set up JupyterLab
- [ ] Try Marimo reactive notebooks
- [ ] Configure VS Code for Python/ML

### Milestone
✅ Can create Python projects, run notebooks, manage dependencies

---

## Phase 02: ML Experiment Lifecycle
**Duration**: 3 weeks

### Goals
- Track experiments systematically
- Version data and models
- Build reproducible pipelines

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| MLflow | `uv add mlflow` | ⭐⭐⭐ |
| DVC | `uv add dvc` | ⭐⭐⭐ |
| Prefect | `uv add prefect` | ⭐⭐⭐ |
| Aim | `uv add aim` | ⭐⭐ |
| Dagster | `uv add dagster dagit` | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | Experiment Tracking | MLflow tracking, Aim UI |
| 2 | Data Versioning | DVC setup, remote storage |
| 3 | Pipeline Orchestration | Prefect flows, scheduling |

### Hands-on Tasks
- [ ] Track ML experiment with MLflow
- [ ] Version dataset with DVC
- [ ] Compare runs in Aim UI
- [ ] Build 3-step pipeline in Prefect
- [ ] Register model in MLflow Model Registry

### Milestone
✅ Can track experiments, version data, build pipelines

---

## Phase 03: Model Serving & APIs
**Duration**: 2 weeks

### Goals
- Package ML models for deployment
- Build production-ready APIs
- Understand inference patterns

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| FastAPI | `uv add fastapi uvicorn` | ⭐⭐⭐ |
| BentoML | `uv add bentoml` | ⭐⭐⭐ |
| Ray Serve | `uv add "ray[serve]"` | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | FastAPI + BentoML | Build inference APIs |
| 2 | Scaling | Ray Serve, async inference |

### Hands-on Tasks
- [ ] Build ML API with FastAPI
- [ ] Package model with BentoML
- [ ] Deploy BentoML service locally
- [ ] Scale with Ray Serve
- [ ] Add request validation & error handling

### Milestone
✅ Can deploy models as production APIs

---

## Phase 04: Data Engineering
**Duration**: 3 weeks

### Goals
- Build data pipelines
- Implement feature stores
- Ensure data quality

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| Great Expectations | `uv add great_expectations` | ⭐⭐⭐ |
| Feast | `uv add feast` | ⭐⭐⭐ |
| Polars | `uv add polars` | ⭐⭐⭐ |
| PySpark | `uv add pyspark` | ⭐⭐ |
| dbt | `uv add dbt-core` | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | Data Processing | Polars, PySpark basics |
| 2 | Data Quality | Great Expectations suites |
| 3 | Feature Store | Feast setup & serving |

### Hands-on Tasks
- [ ] Process data with Polars
- [ ] Create data validation suite
- [ ] Set up Feast feature store
- [ ] Serve features for inference
- [ ] Build dbt transformation

### Milestone
✅ Can build data pipelines with quality checks and feature serving

---

## Phase 05: Monitoring & Observability
**Duration**: 2 weeks

### Goals
- Detect data and model drift
- Build monitoring dashboards
- Set up alerting

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| Evidently | `uv add evidently` | ⭐⭐⭐ |
| Prometheus | Docker | ⭐⭐⭐ |
| Grafana | Docker | ⭐⭐⭐ |
| Whylogs | `uv add whylogs` | ⭐⭐ |
| NannyML | `uv add nannyml` | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | ML Monitoring | Evidently reports, drift detection |
| 2 | Infrastructure | Prometheus + Grafana stack |

### Hands-on Tasks
- [ ] Generate Evidently drift report
- [ ] Profile data with Whylogs
- [ ] Set up Prometheus + Grafana
- [ ] Create ML metrics dashboard
- [ ] Configure alerts for drift

### Milestone
✅ Can monitor models in production and detect issues

---

## Phase 06: LLMOps & RAG
**Duration**: 4 weeks

### Goals
- Run LLMs locally
- Build RAG applications
- Evaluate and guard LLM outputs

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| Ollama | Binary install | ⭐⭐⭐ |
| LangChain | `uv add langchain` | ⭐⭐⭐ |
| LlamaIndex | `uv add llama-index` | ⭐⭐⭐ |
| Qdrant | `docker run qdrant/qdrant` | ⭐⭐⭐ |
| Chroma | `uv add chromadb` | ⭐⭐ |
| Ragas | `uv add ragas` | ⭐⭐⭐ |
| Sentence Transformers | `uv add sentence-transformers` | ⭐⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | Local LLMs | Ollama, vLLM, LiteLLM |
| 2 | Embeddings & Vector DBs | Sentence Transformers, Qdrant |
| 3 | RAG Pipeline | LangChain/LlamaIndex RAG |
| 4 | Evaluation & Guardrails | Ragas, Guardrails AI |

### Hands-on Tasks
- [ ] Run local LLM with Ollama
- [ ] Generate embeddings with Sentence Transformers
- [ ] Set up Qdrant vector database
- [ ] Build RAG pipeline with LangChain
- [ ] Evaluate RAG with Ragas
- [ ] Add guardrails to LLM outputs

### Milestone
✅ Can build and evaluate production RAG applications

---

## Phase 07: Production Infrastructure
**Duration**: 3 weeks

### Goals
- Containerize ML applications
- Deploy on Kubernetes
- Implement CI/CD for ML

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| Docker | System install | ⭐⭐⭐ |
| Minikube/Kind | Binary install | ⭐⭐⭐ |
| Helm | Binary install | ⭐⭐ |
| GitHub Actions | GitHub | ⭐⭐⭐ |
| CML | GitHub/GitLab | ⭐⭐ |
| Vault | Docker | ⭐⭐ |
| Trivy | Binary install | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | Containerization | Docker, multi-stage builds |
| 2 | Kubernetes | Minikube, deployments, services |
| 3 | CI/CD & Security | GitHub Actions, secrets, scanning |

### Hands-on Tasks
- [ ] Containerize ML model with Docker
- [ ] Deploy on Minikube
- [ ] Create Helm chart
- [ ] Set up GitHub Actions ML pipeline
- [ ] Use CML for model reports
- [ ] Scan containers with Trivy

### Milestone
✅ Can deploy ML systems on Kubernetes with CI/CD

---

## Phase 08: Advanced Topics
**Duration**: 3 weeks

### Goals
- Build AI agents
- Optimize models for production
- Understand privacy-preserving ML

### Tools to Learn
| Tool | Install | Priority |
|------|---------|----------|
| LangGraph | `uv add langgraph` | ⭐⭐⭐ |
| AutoGen | `uv add pyautogen` | ⭐⭐⭐ |
| CrewAI | `uv add crewai` | ⭐⭐ |
| ONNX Runtime | `uv add onnxruntime` | ⭐⭐⭐ |
| bitsandbytes | `uv add bitsandbytes` | ⭐⭐ |
| Flower | `uv add flwr` | ⭐⭐ |
| SHAP | `uv add shap` | ⭐⭐ |

### Week-by-Week
| Week | Focus | Tasks |
|------|-------|-------|
| 1 | AI Agents | LangGraph, AutoGen, CrewAI |
| 2 | Optimization | ONNX, quantization, TensorRT |
| 3 | Privacy & Explainability | Federated learning, SHAP |

### Hands-on Tasks
- [ ] Build stateful agent with LangGraph
- [ ] Create multi-agent system with AutoGen
- [ ] Convert model to ONNX
- [ ] Quantize LLM with bitsandbytes
- [ ] Simulate federated learning with Flower
- [ ] Generate SHAP explanations

### Milestone
✅ Can build agents, optimize models, ensure fairness

---

## Phase 09: Specialization & Projects
**Duration**: 4 weeks

### Goals
- Build end-to-end projects
- Document your work
- Specialize in your interest area

### Project Ideas

#### Project 1: ML Pipeline (Week 1)
Build complete ML pipeline with:
- Data versioning (DVC)
- Experiment tracking (MLflow)
- Pipeline orchestration (Prefect)
- Model registry & deployment

#### Project 2: RAG Application (Week 2)
Build production RAG system with:
- Document ingestion
- Vector database (Qdrant)
- LLM integration (Ollama)
- Evaluation (Ragas)
- Monitoring (Evidently)

#### Project 3: AI Agent System (Week 3)
Build multi-agent application with:
- LangGraph orchestration
- Tool integration
- Memory management
- Guardrails

#### Project 4: MLOps Platform (Week 4)
Deploy complete MLOps platform:
- Kubernetes deployment
- CI/CD pipeline
- Monitoring stack
- Documentation (MkDocs)

### Documentation Tasks
- [ ] Document projects with MkDocs
- [ ] Create model cards
- [ ] Write README files
- [ ] Build portfolio

### Milestone
✅ Have 4 complete projects demonstrating MLOps skills

---

## Quick Reference: All Tools by Phase

```
Phase 01: uv, JupyterLab, Marimo, VS Code
Phase 02: MLflow, DVC, Prefect, Aim, Dagster
Phase 03: FastAPI, BentoML, Ray Serve
Phase 04: Great Expectations, Feast, Polars, PySpark, dbt
Phase 05: Evidently, Prometheus, Grafana, Whylogs
Phase 06: Ollama, LangChain, LlamaIndex, Qdrant, Ragas
Phase 07: Docker, Kubernetes, Helm, GitHub Actions, Vault
Phase 08: LangGraph, AutoGen, ONNX, Flower, SHAP
Phase 09: Projects + MkDocs + Model Cards
```

---

## Progress Tracker

| Phase | Status | Start Date | End Date |
|-------|--------|------------|----------|
| 01 | [ ] Not Started | | |
| 02 | [ ] Not Started | | |
| 03 | [ ] Not Started | | |
| 04 | [ ] Not Started | | |
| 05 | [ ] Not Started | | |
| 06 | [ ] Not Started | | |
| 07 | [ ] Not Started | | |
| 08 | [ ] Not Started | | |
| 09 | [ ] Not Started | | |

---

*Created: January 2026*
