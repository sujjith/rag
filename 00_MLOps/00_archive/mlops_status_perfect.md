# MLOps Platform Deployment Status

**Last Updated:** 2026-01-13

## Architecture Decision

| Decision | Choice | Reason |
|----------|--------|--------|
| **Data Pipeline Orchestration** | Prefect Cloud + Local Worker | Lightweight, no k3s overhead for POC |
| **Package Manager** | uv | Fast, modern Python package manager |
| **ML Pipeline Orchestration** | Kubeflow Pipelines | Standard for ML workflows (kept in k3s) |
| **Airflow** | Installed but not used | Available for production later |

## Kubernetes Deployments

| Tool | Namespace | Status | Pods | Notes |
|------|-----------|--------|------|-------|
| **MinIO** | minio | ✅ Running | 1/1 | Object storage |
| **PostgreSQL** | postgresql | ✅ Running | 1/1 | Shared database |
| **Redis** | redis | ✅ Running | 1/1 | Cache/Feature store |
| **NGINX Ingress** | ingress-nginx | ✅ Running | 1/1 | Ingress controller |
| **Apache Airflow** | airflow | ✅ Running | 5 pods | Installed (not used for POC) |
| **Marquez** | marquez | ✅ Running | 2/2 | Data lineage (API + Web) |
| **Feast** | feast | ✅ Running | 1/1 | Feature store |
| **Argo Workflows** | argo-workflows | ✅ Running | 2/2 | CI pipelines (optional for POC) |
| **MLflow** | mlflow | ✅ Running | 1/1 | Experiment tracking |
| **Kubeflow Pipelines** | kubeflow | ✅ Running | 14 pods | ML pipelines |
| **Argo CD** | argocd | ✅ Running | 7 pods | GitOps CD |
| **Cert-Manager** | cert-manager | ✅ Running | 3/3 | Certificate management |
| **KServe** | kserve | ✅ Running | 2/2 | Model serving |
| **Iter8** | iter8-system | ✅ Running | 1/1 | A/B testing |

## Local Components (Prefect Approach)

| Component | Status | Notes |
|-----------|--------|-------|
| **Prefect Cloud** | 🔲 Setup required | Free tier - https://app.prefect.cloud |
| **Prefect Worker** | 🔲 Setup required | Runs locally, connects to k3s |
| **uv** | 🔲 Setup required | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

## Python Libraries (via uv)

```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv sync  # Installs all dependencies from pyproject.toml
```

| Tool | Purpose |
|------|---------|
| prefect | Pipeline orchestration |
| dvc[s3] | Data versioning |
| great-expectations | Data validation |
| feast | Feature store client |
| kfp | Kubeflow Pipelines SDK |
| mlflow | Experiment tracking |
| evidently | Drift detection |
| witwidget | Fairness analysis |
| model-card-toolkit | Model documentation |

## Pipeline Structure

```
/home/sujith/github/rag/00_MLOps/04_usecases/
├── pyproject.toml              # uv project config
├── config.yaml                 # Shared configuration
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
└── uc_1_churn_prediction/      # Use Case 1
    ├── flows/
    │   └── churn_pipeline.py   # Prefect flow
    └── tasks/
        ├── data_ingestion.py
        ├── data_versioning.py
        ├── data_validation.py
        ├── feature_engineering.py
        ├── model_training.py
        └── model_serving.py
```

## Design Documents

| Document | Description | Status |
|----------|-------------|--------|
| `mlops_flow.md` | Original architecture with Airflow | ✅ Complete |
| `mlops_flow_prefect.md` | Updated architecture with Prefect | ✅ Complete |
| `mlops_usecases.md` | 6 use cases detailed | ✅ Complete |
| `prefect_pipeline_approach.md` | Prefect implementation guide | ✅ Complete |
| `manual_pipeline_approach.md` | Plain Python alternative | ✅ Complete |

## Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:30901 | minioadmin / minioadmin123 |
| Airflow | http://localhost:30800 | admin / admin |
| Marquez | http://localhost:30501 | - |
| Feast | http://localhost:30656 | - |
| Argo Workflows | http://localhost:30746 | - |
| MLflow | http://localhost:30050 | - |
| Kubeflow Pipelines | http://localhost:30880 | - |
| Argo CD | https://localhost:30444 | admin / (see command below) |
| **Prefect Cloud** | https://app.prefect.cloud | Your account |

```bash
# Get Argo CD password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

## Quick Start (Prefect)

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup project
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv sync

# 3. Login to Prefect Cloud
uv run prefect cloud login

# 4. Create work pool
uv run prefect work-pool create local-pool --type process

# 5. Start worker (Terminal 1)
uv run prefect worker start --pool local-pool

# 6. Run pipeline (Terminal 2)
uv run python -m uc_1_churn_prediction.flows.churn_pipeline
```

## Implementation Progress

| Use Case | Status | Description |
|----------|--------|-------------|
| UC1: Churn Prediction | 🔲 In Progress | End-to-end ML pipeline |
| UC2: A/B Testing | 🔲 Pending | KServe + Iter8 |
| UC3: Drift Retraining | 🔲 Pending | Evidently + Kubeflow |
| UC4: Data Lineage | 🔲 Pending | OpenLineage + Marquez |
| UC5: Fairness & Docs | 🔲 Pending | What-If Tool + Model Card |
| UC6: GitOps Deploy | 🔲 Pending | Git + Argo CD + KServe |

## Verification Commands

```bash
# Check all k3s pods
kubectl get pods -A | grep -E "minio|postgresql|redis|airflow|marquez|feast|argo|mlflow|kubeflow|kserve|iter8|cert-manager|ingress"

# Check services
kubectl get svc -A --field-selector spec.type=NodePort

# Check Prefect worker status
uv run prefect worker ls

# Check Prefect flow runs
uv run prefect flow-run ls
```
