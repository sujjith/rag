# MLOps Platform Setup Guide

Complete step-by-step guide to deploy the enterprise MLOps stack on Ubuntu Server with K3s.

## Target Architecture

```
Ubuntu Server 24.04
└── K3s (Kubernetes)
    ├── Infrastructure Layer
    │   ├── MinIO (Object Storage)
    │   ├── PostgreSQL (Shared Database)
    │   └── Redis (Cache/Feature Store)
    │
    ├── Data Loop
    │   ├── Apache Airflow
    │   ├── Great Expectations
    │   ├── OpenLineage
    │   └── Marquez
    │
    ├── Code Loop
    │   └── Argo Workflows
    │
    ├── Model Loop
    │   ├── Kubeflow Pipelines
    │   ├── MLflow
    │   ├── What-If Tool
    │   └── Model Card Toolkit
    │
    ├── Deployment Loop
    │   ├── Argo CD
    │   ├── KServe
    │   └── Iter8
    │
    └── Monitoring Loop
        └── Evidently AI
```

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Ubuntu | 22.04 / 24.04 LTS | 24.04 LTS |
| CPU | 6 cores | 8+ cores |
| RAM | 24 GB | 32 GB |
| Disk | 100 GB SSD | 200 GB SSD |
| Swap | Disabled | Disabled |

### Pre-flight Checks

```bash
# Check resources
free -h
df -h
nproc

# Disable swap (required for Kubernetes)
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab

# Verify swap is off
swapon --show  # Should return nothing
```

---

## Phase 0: K3s Installation

### 0.1 Install K3s

```bash
# Install K3s (single node, with Traefik disabled - we'll use our own ingress)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -

# Wait for K3s to be ready
sudo systemctl status k3s

# Setup kubectl access for current user
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
chmod 600 ~/.kube/config

# Verify cluster is running
kubectl get nodes
kubectl get pods -A
```

### 0.2 Install Helm

```bash
# Install Helm package manager
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

### 0.3 Create Namespaces

```bash
# Create namespaces for each layer
kubectl create namespace mlops-infra      # Infrastructure (MinIO, PostgreSQL, Redis)
kubectl create namespace mlops-data       # Data tools (Airflow, Feast, Marquez)
kubectl create namespace mlops-ci         # CI tools (Argo Workflows)
kubectl create namespace mlops-ml         # ML tools (Kubeflow, MLflow)
kubectl create namespace mlops-cd         # CD tools (Argo CD)
kubectl create namespace mlops-serving    # Serving (KServe, Iter8)
kubectl create namespace mlops-monitor    # Monitoring (Evidently)

# Verify namespaces
kubectl get namespaces | grep mlops
```

---

## Phase 1: Infrastructure Layer

### 1.1 Install NGINX Ingress Controller

```bash
# Add ingress-nginx repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install NGINX Ingress
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=NodePort \
  --set controller.service.nodePorts.http=30080 \
  --set controller.service.nodePorts.https=30443

# Verify
kubectl get pods -n ingress-nginx
```

### 1.2 Install MinIO (Object Storage)

```bash
# Add MinIO repo
helm repo add minio https://charts.min.io/
helm repo update

# Create MinIO values file
cat <<EOF > /tmp/minio-values.yaml
mode: standalone
replicas: 1
persistence:
  enabled: true
  size: 50Gi
resources:
  requests:
    memory: 512Mi
    cpu: 250m
  limits:
    memory: 1Gi
    cpu: 500m
rootUser: minioadmin
rootPassword: minioadmin123
consoleIngress:
  enabled: false
buckets:
  - name: mlflow-artifacts
    policy: none
  - name: airflow-logs
    policy: none
  - name: dvc-storage
    policy: none
  - name: feast-offline
    policy: none
  - name: kubeflow-pipelines
    policy: none
EOF

# Install MinIO
helm install minio minio/minio \
  --namespace mlops-infra \
  --values /tmp/minio-values.yaml

# Verify
kubectl get pods -n mlops-infra -l app=minio

# Get MinIO service URL (internal)
echo "MinIO API: http://minio.mlops-infra.svc.cluster.local:9000"
echo "MinIO Console: http://minio.mlops-infra.svc.cluster.local:9001"
```

### 1.3 Install PostgreSQL (Shared Database)

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create PostgreSQL values file
cat <<EOF > /tmp/postgres-values.yaml
auth:
  postgresPassword: postgres123
  database: mlops
primary:
  persistence:
    enabled: true
    size: 10Gi
  resources:
    requests:
      memory: 256Mi
      cpu: 100m
    limits:
      memory: 1Gi
      cpu: 500m
  initdb:
    scripts:
      init-databases.sql: |
        CREATE DATABASE airflow;
        CREATE DATABASE mlflow;
        CREATE DATABASE marquez;
        CREATE DATABASE kubeflow;
        CREATE DATABASE feast;
EOF

# Install PostgreSQL
helm install postgresql bitnami/postgresql \
  --namespace mlops-infra \
  --values /tmp/postgres-values.yaml

# Verify
kubectl get pods -n mlops-infra -l app.kubernetes.io/name=postgresql

# Get PostgreSQL connection string (internal)
echo "PostgreSQL: postgresql://postgres:postgres123@postgresql.mlops-infra.svc.cluster.local:5432"
```

### 1.4 Install Redis (Cache)

```bash
# Create Redis values file
cat <<EOF > /tmp/redis-values.yaml
architecture: standalone
auth:
  enabled: false
master:
  persistence:
    enabled: true
    size: 5Gi
  resources:
    requests:
      memory: 128Mi
      cpu: 100m
    limits:
      memory: 512Mi
      cpu: 250m
EOF

# Install Redis
helm install redis bitnami/redis \
  --namespace mlops-infra \
  --values /tmp/redis-values.yaml

# Verify
kubectl get pods -n mlops-infra -l app.kubernetes.io/name=redis

# Get Redis connection string (internal)
echo "Redis: redis://redis-master.mlops-infra.svc.cluster.local:6379"
```

### 1.5 Verify Infrastructure

```bash
# Check all infrastructure pods are running
kubectl get pods -n mlops-infra

# Expected output:
# NAME                     READY   STATUS    RESTARTS   AGE
# minio-xxx                1/1     Running   0          5m
# postgresql-0             1/1     Running   0          4m
# redis-master-0           1/1     Running   0          3m
```

---

## Phase 2: Data Loop Tools

### 2.1 Install Apache Airflow

```bash
# Add Apache Airflow repo
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Create Airflow values file
cat <<EOF > /tmp/airflow-values.yaml
executor: KubernetesExecutor
webserverSecretKey: $(openssl rand -hex 16)

# Use existing PostgreSQL
postgresql:
  enabled: false
data:
  metadataConnection:
    user: postgres
    pass: postgres123
    protocol: postgresql
    host: postgresql.mlops-infra.svc.cluster.local
    port: 5432
    db: airflow

# Use MinIO for logs
logs:
  persistence:
    enabled: false
config:
  logging:
    remote_logging: 'True'
    remote_base_log_folder: 's3://airflow-logs'
    remote_log_conn_id: 'minio_default'

# Enable OpenLineage
extraEnv:
  - name: AIRFLOW__LINEAGE__BACKEND
    value: "openlineage.lineage_backend.OpenLineageBackend"
  - name: OPENLINEAGE_URL
    value: "http://marquez-api.mlops-data.svc.cluster.local:5000"
  - name: OPENLINEAGE_NAMESPACE
    value: "airflow"

# Resource limits for laptop
webserver:
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m
scheduler:
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m

# Web UI access
webserver:
  service:
    type: NodePort
    ports:
      - name: airflow-ui
        port: 8080
        nodePort: 30800
EOF

# Install Airflow
helm install airflow apache-airflow/airflow \
  --namespace mlops-data \
  --values /tmp/airflow-values.yaml \
  --timeout 10m

# Verify
kubectl get pods -n mlops-data -l release=airflow

# Get Airflow UI URL
echo "Airflow UI: http://localhost:30800"
echo "Default credentials: admin / admin"
```

### 2.2 Install Marquez (Data Lineage)

```bash
# Create Marquez deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marquez-api
  namespace: mlops-data
spec:
  replicas: 1
  selector:
    matchLabels:
      app: marquez-api
  template:
    metadata:
      labels:
        app: marquez-api
    spec:
      containers:
      - name: marquez
        image: marquezproject/marquez:latest
        ports:
        - containerPort: 5000
        - containerPort: 5001
        env:
        - name: MARQUEZ_CONFIG
          value: /etc/marquez/marquez.yml
        - name: POSTGRES_HOST
          value: postgresql.mlops-infra.svc.cluster.local
        - name: POSTGRES_PORT
          value: "5432"
        - name: POSTGRES_DB
          value: marquez
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          value: postgres123
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 250m
---
apiVersion: v1
kind: Service
metadata:
  name: marquez-api
  namespace: mlops-data
spec:
  selector:
    app: marquez-api
  ports:
  - name: api
    port: 5000
    targetPort: 5000
    nodePort: 30500
  - name: admin
    port: 5001
    targetPort: 5001
  type: NodePort
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marquez-web
  namespace: mlops-data
spec:
  replicas: 1
  selector:
    matchLabels:
      app: marquez-web
  template:
    metadata:
      labels:
        app: marquez-web
    spec:
      containers:
      - name: marquez-web
        image: marquezproject/marquez-web:latest
        ports:
        - containerPort: 3000
        env:
        - name: MARQUEZ_HOST
          value: marquez-api.mlops-data.svc.cluster.local
        - name: MARQUEZ_PORT
          value: "5000"
        resources:
          requests:
            memory: 128Mi
            cpu: 50m
          limits:
            memory: 256Mi
            cpu: 100m
---
apiVersion: v1
kind: Service
metadata:
  name: marquez-web
  namespace: mlops-data
spec:
  selector:
    app: marquez-web
  ports:
  - port: 3000
    targetPort: 3000
    nodePort: 30501
  type: NodePort
EOF

# Verify
kubectl get pods -n mlops-data -l app=marquez-api
kubectl get pods -n mlops-data -l app=marquez-web

# Get Marquez UI URL
echo "Marquez UI: http://localhost:30501"
echo "Marquez API: http://localhost:30500"
```

### 2.3 Install Feast (Feature Store)

```bash
# Create Feast deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feast-feature-server
  namespace: mlops-data
spec:
  replicas: 1
  selector:
    matchLabels:
      app: feast
  template:
    metadata:
      labels:
        app: feast
    spec:
      containers:
      - name: feast
        image: feastdev/feature-server:latest
        ports:
        - containerPort: 6566
        env:
        - name: FEAST_USAGE
          value: "false"
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 250m
---
apiVersion: v1
kind: Service
metadata:
  name: feast-feature-server
  namespace: mlops-data
spec:
  selector:
    app: feast
  ports:
  - port: 6566
    targetPort: 6566
    nodePort: 30656
  type: NodePort
EOF

# Verify
kubectl get pods -n mlops-data -l app=feast

echo "Feast Feature Server: http://localhost:30656"
```

### 2.4 Install DVC (Data Version Control)

DVC is a CLI tool installed on your local machine (or in pipeline containers):

```bash
# Install DVC on Ubuntu
pip install dvc[s3]

# Configure DVC to use MinIO
dvc remote add -d minio s3://dvc-storage
dvc remote modify minio endpointurl http://localhost:30900
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# Verify DVC
dvc version
```

### 2.5 Install Great Expectations

Great Expectations is a Python library used within Airflow DAGs:

```bash
# Install Great Expectations (in your Python environment or Docker image)
pip install great-expectations

# Great Expectations is typically used:
# 1. In Airflow DAGs for data validation
# 2. As part of Kubeflow pipeline steps
# 3. In CI/CD pipelines for data testing

# No separate Kubernetes deployment needed
echo "Great Expectations: Installed as Python library in Airflow/Kubeflow images"
```

---

## Phase 3: Code Loop Tools

### 3.1 Install Argo Workflows

```bash
# Create namespace
kubectl create namespace argo

# Install Argo Workflows
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.2/install.yaml

# Patch to use NodePort
kubectl patch svc argo-server -n argo -p '{"spec": {"type": "NodePort", "ports": [{"port": 2746, "nodePort": 30746}]}}'

# Set auth mode to server (for local development)
kubectl patch deployment argo-server -n argo --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--auth-mode=server"}]'

# Verify
kubectl get pods -n argo

echo "Argo Workflows UI: http://localhost:30746"
```

---

## Phase 4: Model Loop Tools

### 4.1 Install MLflow

```bash
# Create MLflow deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops-ml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.9.2
        command: ["mlflow", "server"]
        args:
          - "--backend-store-uri=postgresql://postgres:postgres123@postgresql.mlops-infra.svc.cluster.local:5432/mlflow"
          - "--default-artifact-root=s3://mlflow-artifacts"
          - "--host=0.0.0.0"
          - "--port=5000"
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio.mlops-infra.svc.cluster.local:9000"
        - name: AWS_ACCESS_KEY_ID
          value: "minioadmin"
        - name: AWS_SECRET_ACCESS_KEY
          value: "minioadmin123"
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 250m
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlops-ml
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
    nodePort: 30050
  type: NodePort
EOF

# Verify
kubectl get pods -n mlops-ml -l app=mlflow

echo "MLflow UI: http://localhost:30050"
```

### 4.2 Install Kubeflow Pipelines

```bash
# Install Kubeflow Pipelines (standalone)
export PIPELINE_VERSION=2.0.5

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s

# Patch to use NodePort
kubectl patch svc ml-pipeline-ui -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "nodePort": 30880}]}}'

# Verify
kubectl get pods -n kubeflow

echo "Kubeflow Pipelines UI: http://localhost:30880"
```

### 4.3 What-If Tool & Model Card Toolkit

These are Python libraries used within Kubeflow pipeline steps or Jupyter notebooks:

```bash
# Install in your Python environment or Docker images
pip install witwidget              # What-If Tool
pip install model-card-toolkit     # Model Card Toolkit

# These tools are used:
# 1. In Jupyter notebooks for interactive analysis
# 2. As Kubeflow pipeline components
# 3. Integrated into MLflow artifact logging

echo "What-If Tool: Installed as Python library (witwidget)"
echo "Model Card Toolkit: Installed as Python library (model-card-toolkit)"
```

---

## Phase 5: Deployment Loop Tools

### 5.1 Install Argo CD

```bash
# Create namespace
kubectl create namespace argocd

# Install Argo CD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Patch to use NodePort
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort", "ports": [{"port": 443, "nodePort": 30443, "name": "https"}, {"port": 80, "nodePort": 30080, "name": "http"}]}}'

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

# Verify
kubectl get pods -n argocd

echo "Argo CD UI: http://localhost:30080"
echo "Username: admin"
echo "Password: (run the command above to get password)"
```

### 5.2 Install KServe

```bash
# Install cert-manager (required by KServe)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=120s

# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml

# Install KServe built-in serving runtimes
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve-runtimes.yaml

# Verify
kubectl get pods -n kserve

echo "KServe: Installed and ready for InferenceServices"
```

### 5.3 Install Iter8

```bash
# Install Iter8
kubectl apply -f https://github.com/iter8-tools/iter8/releases/latest/download/install.yaml

# Verify
kubectl get pods -n iter8-system

echo "Iter8: Installed and ready for A/B testing"
```

---

## Phase 6: Monitoring Loop Tools

### 6.1 Install Evidently AI

```bash
# Create Evidently deployment (collector service)
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evidently-collector
  namespace: mlops-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: evidently
  template:
    metadata:
      labels:
        app: evidently
    spec:
      containers:
      - name: evidently
        image: evidentlyai/evidently-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 250m
---
apiVersion: v1
kind: Service
metadata:
  name: evidently-collector
  namespace: mlops-monitor
spec:
  selector:
    app: evidently
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30800
  type: NodePort
EOF

# Alternative: Evidently is commonly used as a Python library
pip install evidently

# Verify
kubectl get pods -n mlops-monitor -l app=evidently

echo "Evidently Collector: http://localhost:30800"
```

---

## Phase 7: Verification

### 7.1 Check All Pods

```bash
# Check all namespaces
echo "=== Infrastructure ==="
kubectl get pods -n mlops-infra

echo -e "\n=== Data Tools ==="
kubectl get pods -n mlops-data

echo -e "\n=== CI Tools ==="
kubectl get pods -n argo

echo -e "\n=== ML Tools ==="
kubectl get pods -n mlops-ml
kubectl get pods -n kubeflow

echo -e "\n=== CD Tools ==="
kubectl get pods -n argocd

echo -e "\n=== Serving Tools ==="
kubectl get pods -n kserve
kubectl get pods -n iter8-system

echo -e "\n=== Monitoring Tools ==="
kubectl get pods -n mlops-monitor
```

### 7.2 Service Endpoints Summary

| Service | URL | Credentials |
|---------|-----|-------------|
| **MinIO Console** | http://localhost:30901 | minioadmin / minioadmin123 |
| **Airflow** | http://localhost:30800 | admin / admin |
| **Marquez** | http://localhost:30501 | - |
| **MLflow** | http://localhost:30050 | - |
| **Kubeflow Pipelines** | http://localhost:30880 | - |
| **Argo Workflows** | http://localhost:30746 | - |
| **Argo CD** | http://localhost:30443 | admin / (see command) |

---

## Quick Reference: Tool Installation Commands

```bash
# Full installation in order
# Phase 0: K3s
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -

# Phase 1: Infrastructure
helm install minio minio/minio -n mlops-infra --values /tmp/minio-values.yaml
helm install postgresql bitnami/postgresql -n mlops-infra --values /tmp/postgres-values.yaml
helm install redis bitnami/redis -n mlops-infra --values /tmp/redis-values.yaml

# Phase 2: Data Loop
helm install airflow apache-airflow/airflow -n mlops-data --values /tmp/airflow-values.yaml
kubectl apply -f marquez.yaml
kubectl apply -f feast.yaml

# Phase 3: Code Loop
kubectl apply -n argo -f argo-workflows-install.yaml

# Phase 4: Model Loop
kubectl apply -f mlflow.yaml
kubectl apply -k kubeflow-pipelines

# Phase 5: Deployment Loop
kubectl apply -n argocd -f argo-cd-install.yaml
kubectl apply -f kserve.yaml
kubectl apply -f iter8.yaml

# Phase 6: Monitoring Loop
kubectl apply -f evidently.yaml
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Pod stuck in Pending | Check resources: `kubectl describe pod <pod>` |
| Pod CrashLoopBackOff | Check logs: `kubectl logs <pod> -n <namespace>` |
| Service not accessible | Check NodePort: `kubectl get svc -A` |
| Database connection failed | Verify PostgreSQL is running and credentials are correct |
| Storage issues | Check PVC status: `kubectl get pvc -A` |

### Useful Commands

```bash
# Check cluster resources
kubectl top nodes
kubectl top pods -A

# Check events
kubectl get events -A --sort-by='.lastTimestamp'

# Restart a deployment
kubectl rollout restart deployment <name> -n <namespace>

# Check logs
kubectl logs -f <pod-name> -n <namespace>

# Port forward for debugging
kubectl port-forward svc/<service> <local-port>:<service-port> -n <namespace>
```

---

## Next Steps

1. **Configure Airflow DAGs** - Create data pipelines with Great Expectations validation
2. **Setup DVC** - Initialize DVC in your ML project repository
3. **Configure Feast** - Define feature definitions and materialize features
4. **Create Kubeflow Pipelines** - Build ML training pipelines
5. **Setup GitOps** - Configure Argo CD to watch your deployment repository
6. **Deploy Models** - Create KServe InferenceServices
7. **Configure Monitoring** - Setup Evidently for drift detection

---

## Resource Usage Estimate

| Component | Memory | CPU |
|-----------|--------|-----|
| K3s system | 1 GB | 0.5 |
| MinIO | 1 GB | 0.5 |
| PostgreSQL | 1 GB | 0.5 |
| Redis | 0.5 GB | 0.25 |
| Airflow | 2 GB | 1.0 |
| Marquez | 0.5 GB | 0.25 |
| MLflow | 0.5 GB | 0.25 |
| Kubeflow Pipelines | 3 GB | 1.0 |
| Argo Workflows | 0.5 GB | 0.25 |
| Argo CD | 1 GB | 0.5 |
| KServe | 1 GB | 0.5 |
| Iter8 | 0.25 GB | 0.1 |
| Evidently | 0.5 GB | 0.25 |
| **Total** | **~13 GB** | **~5.5** |
| **Headroom** | 8 GB | 2.5 |
| **Grand Total** | **~21 GB** | **~8** |

Your system (31 GB RAM, 8 CPU) has sufficient resources with ~10 GB headroom for workloads.
