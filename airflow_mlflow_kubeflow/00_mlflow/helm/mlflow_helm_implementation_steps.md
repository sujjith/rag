# MLflow Helm Implementation Steps - Minikube Deployment

## Overview

Deploy MLflow using the community Helm chart on Minikube with PostgreSQL backend.

### Why Helm?
- **Production-ready**: Parameterized, reusable configuration
- **Easy upgrades**: Simple version management with `helm upgrade`
- **Community-maintained**: Well-tested, regularly updated
- **Values-based**: Change settings via `values.yaml` without editing templates

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Minikube Cluster                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              mlflow Namespace (Helm Release)            │ │
│  │                                                         │ │
│  │  ┌─────────────────────┐   ┌─────────────────────┐    │ │
│  │  │  MLflow Server      │   │  PostgreSQL         │    │ │
│  │  │  (Deployment)       │──▶│  (StatefulSet)      │    │ │
│  │  │  Port: 5000         │   │  Port: 5432         │    │ │
│  │  └─────────────────────┘   └─────────────────────┘    │ │
│  │           │                          │                 │ │
│  │           ▼                          ▼                 │ │
│  │  ┌─────────────────────┐   ┌─────────────────────┐    │ │
│  │  │  Service (NodePort) │   │  PVC (5Gi)          │    │ │
│  │  │  Port: 31343        │   │                     │    │ │
│  │  └─────────────────────┘   └─────────────────────┘    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Verify Tools

```bash
# Check Helm
helm version

# Check Minikube
minikube status

# Check kubectl
kubectl version --client
```

### 2. Start Minikube (if not running)

```bash
minikube start --cpus=4 --memory=8192 --driver=docker
```

---

## Step 1: Add Helm Repository

```bash
# Add community-charts repository
helm repo add community-charts https://community-charts.github.io/helm-charts

# Update repositories
helm repo update

# Search for MLflow chart
helm search repo mlflow
```

**Expected output:**
```
NAME                            CHART VERSION   APP VERSION     DESCRIPTION
community-charts/mlflow         1.8.1           3.7.0           A Helm chart for Mlflow...
```

---

## Step 2: Inspect Chart

```bash
# View default values
helm show values community-charts/mlflow > helm/helm-values-default.yaml

# View chart details
helm show chart community-charts/mlflow

# View README
helm show readme community-charts/mlflow
```

---

## Step 3: Create Custom Values File

Create `helm/helm-values-custom.yaml`:

```yaml
# Custom values for MLflow Helm chart
# Using bundled PostgreSQL subchart

# Enable PostgreSQL subchart
postgresql:
  enabled: true
  auth:
    username: "mlflow"
    password: "mlflow123"
    database: "mlflow"
  primary:
    persistence:
      enabled: true
      size: 5Gi

# Service configuration - NodePort for external access
service:
  enabled: true
  type: NodePort
  port: 80
  containerPort: 5000

# Resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

---

## Step 4: Install MLflow

```bash
# Install with custom values
helm install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace \
  --values helm/helm-values-custom.yaml

# Watch deployment
kubectl get pods -n mlflow -w
```

**Expected output:**
```
NAME: mlflow
LAST DEPLOYED: Thu Jan  8 22:07:44 2026
NAMESPACE: mlflow
STATUS: deployed
REVISION: 1
```

---

## Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n mlflow

# Expected:
# NAME                     READY   STATUS    RESTARTS   AGE
# mlflow-xxxxx             1/1     Running   0          2m
# mlflow-postgresql-0      1/1     Running   0          2m

# Check services
kubectl get svc -n mlflow

# Check Helm release
helm list -n mlflow

# Check logs
kubectl logs -l app.kubernetes.io/name=mlflow -n mlflow
```

---

## Step 6: Access MLflow UI

### Option 1: NodePort (Direct Access)

```bash
# Get access URL
export NODE_PORT=$(kubectl get --namespace mlflow -o jsonpath="{.spec.ports[0].nodePort}" services mlflow)
export NODE_IP=$(minikube ip)
echo "MLflow UI: http://$NODE_IP:$NODE_PORT"
```

Open the URL in your browser (e.g., `http://192.168.49.2:31343`)

### Option 2: Port Forward (Local Access)

```bash
# Forward to localhost
kubectl port-forward svc/mlflow 5000:80 -n mlflow

# Access at: http://localhost:5000
```

### Option 3: Port Forward (Network Access)

```bash
# Forward and bind to all interfaces
kubectl port-forward svc/mlflow 5000:80 -n mlflow --address 0.0.0.0

# Access from other devices: http://<your-ubuntu-ip>:5000
```

---

## Helm Management Commands

### View Configuration

```bash
# Get current values
helm get values mlflow -n mlflow

# Get all values (including defaults)
helm get values mlflow -n mlflow --all

# Get manifest
helm get manifest mlflow -n mlflow
```

### Upgrade Release

```bash
# Modify helm/helm-values-custom.yaml, then:
helm upgrade mlflow community-charts/mlflow \
  -n mlflow \
  --values helm/helm-values-custom.yaml

# Upgrade to new chart version
helm upgrade mlflow community-charts/mlflow \
  -n mlflow \
  --version 1.9.0 \
  --values helm/helm-values-custom.yaml
```

### Rollback

```bash
# View release history
helm history mlflow -n mlflow

# Rollback to previous version
helm rollback mlflow -n mlflow

# Rollback to specific revision
helm rollback mlflow 1 -n mlflow
```

### Uninstall

```bash
# Uninstall release (keeps namespace)
helm uninstall mlflow -n mlflow

# Delete namespace
kubectl delete namespace mlflow
```

---

## Troubleshooting

### Check Pod Status

```bash
# Describe pod
kubectl describe pod -l app.kubernetes.io/name=mlflow -n mlflow

# View logs
kubectl logs -l app.kubernetes.io/name=mlflow -n mlflow --tail=50

# Check PostgreSQL
kubectl logs mlflow-postgresql-0 -n mlflow
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Pod stuck in Pending | Check PVC: `kubectl get pvc -n mlflow` |
| Connection refused | Wait for pod to be ready: `kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mlflow -n mlflow` |
| Helm install fails | Check values schema: `helm lint helm/helm-values-custom.yaml` |
| Can't access UI | Verify service: `kubectl get svc mlflow -n mlflow` |

### Debug Commands

```bash
# Get all resources
kubectl get all -n mlflow

# Check events
kubectl get events -n mlflow --sort-by='.lastTimestamp'

# Exec into pod
kubectl exec -it deployment/mlflow -n mlflow -- /bin/bash

# Test database connection
kubectl exec -it mlflow-postgresql-0 -n mlflow -- psql -U mlflow -d mlflow
```

---

## Python Examples

### Setup Environment

```bash
cd /home/sujith/github/rag/airflow_mlflow_kubeflow/00_mlflow/examples

# Create virtual environment with uv
uv venv
source .venv/bin/activate
uv sync
```

### Run Examples

```bash
# Basic tracking
uv run python 01_basic_tracking.py

# Model registry
uv run python 02_model_registry.py

# Model serving (if configured)
uv run python 03_test_serving.py
```

**Note:** Update `MLFLOW_TRACKING_URI` in scripts to match your access method:
- NodePort: `http://192.168.49.2:31343`
- Port-forward: `http://localhost:5000`

---

## Comparison: Helm vs Raw Manifests

| Aspect | Helm | Raw Manifests |
|--------|------|---------------|
| **Setup** | Single command | Multiple files |
| **Upgrades** | `helm upgrade` | Manual kubectl apply |
| **Rollback** | Built-in | Manual |
| **Customization** | Values file | Edit YAMLs |
| **Learning Curve** | Moderate | Low |
| **Production** | ✅ Recommended | Manual maintenance |
| **Debugging** | Abstracted | Direct visibility |

---

## Advanced Configuration

### Custom Artifact Storage (S3/MinIO)

```yaml
# In helm-values-custom.yaml
artifactRoot:
  s3:
    enabled: true
    bucket: "mlflow-artifacts"
    awsAccessKeyId: "your-key"
    awsSecretAccessKey: "your-secret"
```

### Enable Ingress

```yaml
# In helm-values-custom.yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: mlflow.local
      paths:
        - path: /
          pathType: Prefix
```

### External PostgreSQL

```yaml
# In helm-values-custom.yaml
postgresql:
  enabled: false

backendStore:
  postgres:
    enabled: true
    host: "external-postgres.example.com"
    port: 5432
    database: "mlflow"
    user: "mlflow"
    password: "secure-password"
```

---

## Quick Reference

```bash
# Install
helm install mlflow community-charts/mlflow -n mlflow --create-namespace --values helm/helm-values-custom.yaml

# Access
minikube_ip=$(minikube ip)
node_port=$(kubectl get svc mlflow -n mlflow -o jsonpath='{.spec.ports[0].nodePort}')
echo "http://$minikube_ip:$node_port"

# Upgrade
helm upgrade mlflow community-charts/mlflow -n mlflow --values helm/helm-values-custom.yaml

# Uninstall
helm uninstall mlflow -n mlflow
kubectl delete namespace mlflow
```

---

## Next Steps

1. **Explore MLflow UI** - Create experiments, log runs
2. **Run Python examples** - Test tracking and model registry
3. **Learn Helm** - Understand charts, templates, and values
4. **Production Setup** - Add ingress, external storage, monitoring
