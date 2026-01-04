# Open-Source MLOps & AIOps Learning List

> Tools you can install, try, and learn on your local Ubuntu environment

---

## 1. ML Experiment Tracking & Model Registry

| Tool | Description | Try It |
|------|-------------|--------|
| **MLflow** | End-to-end ML lifecycle management | `uv add mlflow` |
| **DVC** | Data & model version control | `uv add dvc` |
| **Weights & Biases** | Experiment tracking (free tier) | `uv add wandb` |
| **ClearML** | MLOps suite (open-source version) | `uv add clearml` |
| **Neptune** | Metadata store (free tier) | `uv add neptune` |
| **Aim** | Experiment comparison UI | `uv add aim` |

### Hands-on Tasks
- [ ] Track experiments with MLflow
- [ ] Version datasets with DVC
- [ ] Compare model runs in Aim UI
- [ ] Create model registry in MLflow

---

## 2. ML Pipeline Orchestration

| Tool | Description | Try It |
|------|-------------|--------|
| **Apache Airflow** | Workflow orchestration | `uv add apache-airflow` |
| **Kubeflow Pipelines** | K8s-native ML workflows | Kind/Minikube + Kubeflow |
| **Prefect** | Modern workflow orchestration | `uv add prefect` |
| **Dagster** | Data-aware orchestration | `uv add dagster dagit` |
| **Argo Workflows** | K8s-native workflows | K8s + Argo install |
| **Metaflow** | Netflix's ML framework | `uv add metaflow` |
| **ZenML** | MLOps framework | `uv add zenml` |

### Hands-on Tasks
- [ ] Build DAG pipeline in Airflow
- [ ] Create ML pipeline with Prefect
- [ ] Deploy Kubeflow on Minikube
- [ ] Build reproducible pipeline with ZenML

---

## 3. Model Serving & Deployment

| Tool | Description | Try It |
|------|-------------|--------|
| **KServe** | K8s model inference | K8s + KServe install |
| **Seldon Core** | ML deployment on K8s | K8s + Seldon install |
| **BentoML** | Model packaging & serving | `uv add bentoml` |
| **Ray Serve** | Scalable model serving | `uv add "ray[serve]"` |
| **TorchServe** | PyTorch model serving | `uv add torchserve` |
| **TensorFlow Serving** | TF model serving | Docker image |
| **Triton Inference Server** | NVIDIA's model server | Docker/NGC |
| **FastAPI** | Build ML APIs | `uv add fastapi uvicorn` |

### Hands-on Tasks
- [ ] Serve model with BentoML
- [ ] Deploy with FastAPI
- [ ] Scale inference with Ray Serve
- [ ] Try TorchServe for PyTorch models

---

## 4. Feature Stores

| Tool | Description | Try It |
|------|-------------|--------|
| **Feast** | Open-source feature store | `uv add feast` |
| **Hopsworks** | Feature store platform | Community edition |
| **Featureform** | Virtual feature store | `uv add featureform` |

### Hands-on Tasks
- [ ] Set up Feast with local registry
- [ ] Create feature views
- [ ] Serve features for inference

---

## 5. Data Versioning & Lineage

| Tool | Description | Try It |
|------|-------------|--------|
| **DVC** | Git for data & models | `uv add dvc` |
| **LakeFS** | Data lake version control | Docker compose |
| **Pachyderm** | Data versioning & pipelines | Docker/K8s |
| **Great Expectations** | Data validation | `uv add great_expectations` |
| **Pandera** | DataFrame validation | `uv add pandera` |

### Hands-on Tasks
- [ ] Version datasets with DVC
- [ ] Create data validation suite with Great Expectations
- [ ] Set up LakeFS locally

---

## 6. Model Monitoring & Observability

| Tool | Description | Try It |
|------|-------------|--------|
| **Evidently AI** | ML monitoring & drift | `uv add evidently` |
| **Whylogs** | Data logging & profiling | `uv add whylogs` |
| **NannyML** | Post-deployment monitoring | `uv add nannyml` |
| **Alibi Detect** | Drift & outlier detection | `uv add alibi-detect` |
| **Prometheus** | Metrics collection | Docker/binary |
| **Grafana** | Visualization & dashboards | Docker/binary |

### Hands-on Tasks
- [ ] Detect data drift with Evidently
- [ ] Set up Prometheus + Grafana stack
- [ ] Create ML monitoring dashboard
- [ ] Profile data with Whylogs

---

## 7. LLM & RAG Tools (Open Source)

| Tool | Description | Try It |
|------|-------------|--------|
| **LangChain** | LLM application framework | `uv add langchain` |
| **LlamaIndex** | Data framework for LLMs | `uv add llama-index` |
| **Haystack** | NLP/LLM framework | `uv add farm-haystack` |
| **vLLM** | Fast LLM inference | `uv add vllm` |
| **Ollama** | Run LLMs locally | Binary install |
| **LiteLLM** | Unified LLM API | `uv add litellm` |
| **Text Generation Inference** | HF's LLM serving | Docker |

### Hands-on Tasks
- [ ] Run local LLM with Ollama
- [ ] Build RAG with LangChain
- [ ] Create chatbot with LlamaIndex
- [ ] Serve LLM with vLLM

---

## 8. Vector Databases (Open Source)

| Tool | Description | Try It |
|------|-------------|--------|
| **Qdrant** | Vector similarity search | `docker run qdrant/qdrant` |
| **Milvus** | Scalable vector DB | Docker compose |
| **Weaviate** | Vector search engine | Docker |
| **Chroma** | Embedding database | `uv add chromadb` |
| **FAISS** | Facebook's similarity search | `uv add faiss-cpu` |
| **pgvector** | Postgres vector extension | PostgreSQL extension |

### Hands-on Tasks
- [ ] Set up Qdrant locally
- [ ] Build vector search with FAISS
- [ ] Create embeddings with Chroma
- [ ] Add pgvector to PostgreSQL

---

## 9. AutoML & Hyperparameter Tuning

| Tool | Description | Try It |
|------|-------------|--------|
| **Optuna** | Hyperparameter optimization | `uv add optuna` |
| **Ray Tune** | Scalable tuning | `uv add "ray[tune]"` |
| **Hyperopt** | Bayesian optimization | `uv add hyperopt` |
| **Auto-sklearn** | AutoML for sklearn | `uv add auto-sklearn` |
| **FLAML** | Fast AutoML | `uv add flaml` |
| **PyCaret** | Low-code ML | `uv add pycaret` |

### Hands-on Tasks
- [ ] Tune hyperparameters with Optuna
- [ ] Run distributed tuning with Ray Tune
- [ ] Try AutoML with PyCaret

---

## 10. Distributed Training

| Tool | Description | Try It |
|------|-------------|--------|
| **Ray** | Distributed computing | `uv add ray` |
| **Dask** | Parallel computing | `uv add dask distributed` |
| **Horovod** | Distributed deep learning | `uv add horovod` |
| **DeepSpeed** | Deep learning optimization | `uv add deepspeed` |
| **PyTorch DDP** | Built into PyTorch | PyTorch |
| **Accelerate** | HF distributed training | `uv add accelerate` |

### Hands-on Tasks
- [ ] Distribute training with Ray
- [ ] Scale pandas with Dask
- [ ] Train LLM with DeepSpeed

---

## 11. CI/CD for ML

| Tool | Description | Try It |
|------|-------------|--------|
| **GitHub Actions** | CI/CD workflows | GitHub repo |
| **GitLab CI** | GitLab pipelines | GitLab repo |
| **Jenkins** | Automation server | Docker |
| **CML** | CI/CD for ML (DVC) | GitHub/GitLab |
| **ArgoCD** | GitOps for K8s | K8s |

### Hands-on Tasks
- [ ] Set up ML pipeline with GitHub Actions
- [ ] Use CML for model reports
- [ ] Deploy models with ArgoCD

---

## 12. Containerization & Orchestration

| Tool | Description | Try It |
|------|-------------|--------|
| **Docker** | Containerization | Install Docker |
| **Kubernetes** | Container orchestration | Minikube/Kind |
| **Kind** | K8s in Docker | `go install kind` |
| **Minikube** | Local K8s cluster | Binary install |
| **K3s** | Lightweight K8s | Binary install |
| **Helm** | K8s package manager | Binary install |

### Hands-on Tasks
- [ ] Containerize ML model with Docker
- [ ] Deploy on Minikube
- [ ] Create Helm chart for ML service

---

## 13. Explainable AI & Fairness

| Tool | Description | Try It |
|------|-------------|--------|
| **SHAP** | Model explanations | `uv add shap` |
| **LIME** | Local interpretability | `uv add lime` |
| **Alibi** | ML explanations | `uv add alibi` |
| **Fairlearn** | Fairness assessment | `uv add fairlearn` |
| **AI Fairness 360** | IBM's fairness toolkit | `uv add aif360` |
| **TrustyAI** | Trustworthy AI toolkit | pip/Maven |

### Hands-on Tasks
- [ ] Explain predictions with SHAP
- [ ] Detect bias with Fairlearn
- [ ] Create LIME explanations

---

## 14. Data Processing & ETL

| Tool | Description | Try It |
|------|-------------|--------|
| **Apache Spark** | Big data processing | `uv add pyspark` |
| **Polars** | Fast DataFrames | `uv add polars` |
| **Apache Beam** | Unified batch/stream | `uv add apache-beam` |
| **dbt** | Data transformation | `uv add dbt-core` |
| **Apache Kafka** | Event streaming | Docker |
| **Apache Flink** | Stream processing | Docker |

### Hands-on Tasks
- [ ] Process data with PySpark
- [ ] Build transforms with dbt
- [ ] Stream data with Kafka

---

## 15. AIOps Specific Tools

| Tool | Description | Try It |
|------|-------------|--------|
| **Prometheus** | Metrics & alerting | Docker |
| **Grafana** | Visualization | Docker |
| **OpenTelemetry** | Observability framework | pip/SDK |
| **Jaeger** | Distributed tracing | Docker |
| **Elastic Stack** | Log analysis | Docker |
| **Datadog** | Monitoring (free tier) | Agent install |

### Hands-on Tasks
- [ ] Set up Prometheus + Grafana
- [ ] Implement OpenTelemetry tracing
- [ ] Analyze logs with ELK stack

---

## 16. LLM Evaluation & Guardrails

| Tool | Description | Try It |
|------|-------------|--------|
| **Ragas** | RAG evaluation framework | `uv add ragas` |
| **DeepEval** | LLM evaluation framework | `uv add deepeval` |
| **Guardrails AI** | LLM output validation | `uv add guardrails-ai` |
| **NeMo Guardrails** | NVIDIA's LLM guardrails | `uv add nemoguardrails` |
| **Phoenix (Arize)** | LLM observability | `uv add arize-phoenix` |
| **LangSmith** | LangChain evaluation | LangChain Cloud |
| **PromptFoo** | Prompt testing CLI | `npm install -g promptfoo` |

### Hands-on Tasks
- [ ] Evaluate RAG pipeline with Ragas
- [ ] Add guardrails to LLM outputs
- [ ] Test prompts with PromptFoo
- [ ] Monitor LLM with Phoenix

---

## 17. Model Optimization & Quantization

| Tool | Description | Try It |
|------|-------------|--------|
| **ONNX Runtime** | Cross-platform inference | `uv add onnxruntime` |
| **TensorRT** | NVIDIA optimization | NVIDIA toolkit |
| **OpenVINO** | Intel optimization | `uv add openvino` |
| **llama.cpp** | CPU LLM inference | Build from source |
| **GPTQ** | LLM quantization | `uv add auto-gptq` |
| **bitsandbytes** | 8-bit optimizers | `uv add bitsandbytes` |
| **ctransformers** | C bindings for transformers | `uv add ctransformers` |

### Hands-on Tasks
- [ ] Convert model to ONNX
- [ ] Quantize LLM with GPTQ
- [ ] Run inference with llama.cpp
- [ ] Optimize with bitsandbytes

---

## 18. Embedding Models (Open Source)

| Tool | Description | Try It |
|------|-------------|--------|
| **Sentence Transformers** | Text embeddings | `uv add sentence-transformers` |
| **FastEmbed** | Fast CPU embeddings | `uv add fastembed` |
| **Instructor** | Customizable embeddings | `uv add instructor-embedding` |
| **E5 Models** | Microsoft embeddings | Hugging Face |
| **BGE Models** | BAAI embeddings | Hugging Face |
| **Nomic Embed** | Long-context embeddings | `uv add nomic` |

### Hands-on Tasks
- [ ] Generate embeddings with Sentence Transformers
- [ ] Use FastEmbed for CPU inference
- [ ] Compare embedding models

---

## 19. Data Labeling & Annotation

| Tool | Description | Try It |
|------|-------------|--------|
| **Label Studio** | Multi-modal labeling | Docker / `uv add label-studio` |
| **Argilla** | Data curation for LLMs | `uv add argilla` |
| **Prodigy** | Active learning labeling | Commercial (trial) |
| **CVAT** | Computer vision annotation | Docker |
| **Doccano** | Text annotation | Docker |

### Hands-on Tasks
- [ ] Set up Label Studio locally
- [ ] Curate LLM data with Argilla
- [ ] Label images with CVAT

---

## 20. Secret Management & Security

| Tool | Description | Try It |
|------|-------------|--------|
| **HashiCorp Vault** | Secrets management | Docker |
| **SOPS** | Encrypted secrets in Git | Binary install |
| **git-crypt** | Git encryption | `apt install git-crypt` |
| **python-dotenv** | Environment variables | `uv add python-dotenv` |
| **Trivy** | Container security scanning | Binary install |

### Hands-on Tasks
- [ ] Manage secrets with Vault
- [ ] Encrypt configs with SOPS
- [ ] Scan containers with Trivy

---

## 21. Testing for ML

| Tool | Description | Try It |
|------|-------------|--------|
| **pytest** | Python testing | `uv add pytest` |
| **Deepchecks** | ML validation suite | `uv add deepchecks` |
| **Checklist** | NLP behavioral testing | `uv add checklist` |
| **MLtest** | ML system testing | `uv add mltest` |
| **Locust** | Load testing | `uv add locust` |
| **pytest-benchmark** | Performance testing | `uv add pytest-benchmark` |

### Hands-on Tasks
- [ ] Write ML pipeline tests with pytest
- [ ] Validate model with Deepchecks
- [ ] Load test API with Locust

---

## Quick Start: Recommended Learning Path

### Week 1-2: Foundations
```bash
# Install core tools
uv add mlflow dvc prefect
uv add fastapi uvicorn
docker pull qdrant/qdrant
```
- [ ] Version data with DVC
- [ ] Track experiments with MLflow
- [ ] Build simple pipeline with Prefect

### Week 3-4: Model Serving
```bash
uv add bentoml "ray[serve]"
```
- [ ] Package model with BentoML
- [ ] Serve with FastAPI
- [ ] Scale with Ray Serve

### Week 5-6: Monitoring
```bash
uv add evidently whylogs
docker-compose up prometheus grafana
```
- [ ] Detect drift with Evidently
- [ ] Create monitoring dashboards

### Week 7-8: LLMOps
```bash
uv add langchain llama-index chromadb
# Install Ollama for local LLMs
```
- [ ] Build RAG pipeline
- [ ] Run local LLM with Ollama

---

## Docker Compose Quick Starters

### MLflow + MinIO
```yaml
# mlflow-docker-compose.yaml
version: '3'
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
```

### Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Prometheus + Grafana
```bash
docker run -p 9090:9090 prom/prometheus
docker run -p 3000:3000 grafana/grafana
```

---

*Last Updated: January 2026*
