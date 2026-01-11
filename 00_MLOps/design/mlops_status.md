# MLOps Platform Deployment Status

**Last Updated:** 2026-01-11

## Kubernetes Deployments

| Tool | Namespace | Status | Pods | Notes |
|------|-----------|--------|------|-------|
| **MinIO** | minio | ✅ Running | 1/1 | Object storage |
| **PostgreSQL** | postgresql | ✅ Running | 1/1 | Shared database |
| **Redis** | redis | ✅ Running | 1/1 | Cache/Feature store |
| **NGINX Ingress** | ingress-nginx | ✅ Running | 1/1 | Ingress controller |
| **Apache Airflow** | airflow | ✅ Running | 5 pods | Data orchestration |
| **Marquez** | marquez | ✅ Running | 2/2 | Data lineage (API + Web) |
| **Feast** | feast | ✅ Running | 1/1 | Feature store |
| **Argo Workflows** | argo-workflows | ✅ Running | 2/2 | CI pipelines |
| **MLflow** | mlflow | ✅ Running | 1/1 | Experiment tracking |
| **Kubeflow Pipelines** | kubeflow | ✅ Running | 14 pods | ML pipelines |
| **Argo CD** | argocd | ✅ Running | 7 pods | GitOps CD |
| **Cert-Manager** | cert-manager | ✅ Running | 3/3 | Certificate management |
| **KServe** | kserve | ✅ Running | 2/2 | Model serving |
| **Iter8** | iter8-system | ✅ Running | 1/1 | A/B testing |

## Python Libraries (Local Installation)

| Tool | Status | Install Command |
|------|--------|-----------------|
| **DVC** | 📦 Install locally | `pip install dvc[s3]` |
| **Great Expectations** | 📦 Install locally | `pip install great-expectations` |
| **What-If Tool** | 📦 Install locally | `pip install witwidget` |
| **Model Card Toolkit** | 📦 Install locally | `pip install model-card-toolkit` |
| **Evidently AI** | 📦 Install locally | `pip install evidently` |

### Quick Install All Python Tools
```bash
pip install dvc[s3] great-expectations witwidget model-card-toolkit evidently
```

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
| Argo CD | https://localhost:30444 | admin / `kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" \| base64 -d` |
| Evidently (local) | http://localhost:8000 | - |

## Verification Commands

```bash
# Check all pods
kubectl get pods -A | grep -E "minio|postgresql|redis|airflow|marquez|feast|argo|mlflow|kubeflow|kserve|iter8|cert-manager|ingress"

# Check services
kubectl get svc -A --field-selector spec.type=NodePort
```
