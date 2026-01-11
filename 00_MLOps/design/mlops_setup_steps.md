# MLOps Platform Setup Guide

Complete step-by-step guide to deploy the enterprise MLOps stack on Ubuntu Server with K3s.

## Target Architecture

```
Ubuntu Server 24.04
└── K3s (Kubernetes)
    ├── Infrastructure Layer
    │   ├── namespace: minio        → MinIO (Object Storage)
    │   ├── namespace: postgresql   → PostgreSQL (Shared Database)
    │   └── namespace: redis        → Redis (Cache/Feature Store)
    │
    ├── Data Loop
    │   ├── namespace: airflow      → Apache Airflow
    │   ├── namespace: marquez      → Marquez + OpenLineage
    │   └── namespace: feast        → Feast (Feature Store)
    │
    ├── Code Loop
    │   └── namespace: argo         → Argo Workflows
    │
    ├── Model Loop
    │   ├── namespace: mlflow       → MLflow
    │   └── namespace: kubeflow     → Kubeflow Pipelines
    │
    ├── Deployment Loop
    │   ├── namespace: argocd       → Argo CD
    │   ├── namespace: kserve       → KServe
    │   └── namespace: iter8-system → Iter8
    │
    └── Monitoring Loop
        └── namespace: evidently    → Evidently AI
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

### 0.3 Setup Directories

```bash
# Create directories for charts and value files
mkdir -p /home/sujith/github/rag/00_MLOps/helm_charts
mkdir -p /home/sujith/github/rag/00_MLOps/helm_value_files
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Add Repositories
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add minio https://charts.min.io/
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add apache-airflow https://airflow.apache.org
helm repo update
```

### 0.4 Create Namespaces

```bash
# Infrastructure Layer
kubectl create namespace minio
kubectl create namespace postgresql
kubectl create namespace redis
kubectl create namespace ingress-nginx

# Data Loop
kubectl create namespace airflow
kubectl create namespace marquez
kubectl create namespace feast

# Code Loop
kubectl create namespace argo

# Model Loop
kubectl create namespace mlflow
kubectl create namespace kubeflow

# Deployment Loop
kubectl create namespace argocd
# Note: kserve and iter8 create their own namespaces during installation

# Monitoring Loop
kubectl create namespace evidently

# Verify namespaces
kubectl get namespaces
```

---

## Phase 1: Infrastructure Layer

### 1.1 Install NGINX Ingress Controller

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Chart
helm pull ingress-nginx/ingress-nginx --untar

# Install from local folder
helm install ingress-nginx ./ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.type=NodePort \
  --set controller.service.nodePorts.http=30080 \
  --set controller.service.nodePorts.https=30443

# Verify
kubectl get pods -n ingress-nginx
```

### 1.2 Install MinIO (Object Storage)

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Chart
helm pull minio/minio --untar

# Create MinIO values file
cat <<EOF > /home/sujith/github/rag/00_MLOps/helm_value_files/minio-values.yaml
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

# Install from local folder
helm install minio ./minio \
  --namespace minio \
  --values /home/sujith/github/rag/00_MLOps/helm_value_files/minio-values.yaml

# Verify
kubectl get pods -n minio

# Get MinIO service URL (internal)
echo "MinIO API: http://minio.minio.svc.cluster.local:9000"
echo "MinIO Console: http://minio.minio.svc.cluster.local:9001"
```

### 1.3 Install PostgreSQL (Shared Database)

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Chart
helm pull bitnami/postgresql --untar

# Create PostgreSQL values file
cat <<EOF > /home/sujith/github/rag/00_MLOps/helm_value_files/postgres-values.yaml
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

# Install from local folder
helm install postgresql ./postgresql \
  --namespace postgresql \
  --values /home/sujith/github/rag/00_MLOps/helm_value_files/postgres-values.yaml

# Verify
kubectl get pods -n postgresql

# Get PostgreSQL connection string (internal)
echo "PostgreSQL: postgresql://postgres:postgres123@postgresql.postgresql.svc.cluster.local:5432"
```

### 1.4 Install Redis (Cache)

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Chart
helm pull bitnami/redis --untar

# Create Redis values file
cat <<EOF > /home/sujith/github/rag/00_MLOps/helm_value_files/redis-values.yaml
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

# Install from local folder
helm install redis ./redis \
  --namespace redis \
  --values /home/sujith/github/rag/00_MLOps/helm_value_files/redis-values.yaml

# Verify
kubectl get pods -n redis

# Get Redis connection string (internal)
echo "Redis: redis://redis-master.redis.svc.cluster.local:6379"
```

### 1.5 Verify Infrastructure

```bash
echo "=== MinIO ==="
kubectl get pods -n minio

echo "=== PostgreSQL ==="
kubectl get pods -n postgresql

echo "=== Redis ==="
kubectl get pods -n redis

echo "=== Ingress ==="
kubectl get pods -n ingress-nginx
```

---

## Phase 2: Data Loop Tools

### 2.1 Install Apache Airflow

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Chart
helm pull apache-airflow/airflow --untar

# Create Airflow values file (Note: connections point to other namespaces)
cat <<EOF > /home/sujith/github/rag/00_MLOps/helm_value_files/airflow-values.yaml
executor: KubernetesExecutor
webserverSecretKey: $(openssl rand -hex 16)

# Use external PostgreSQL (in 'postgresql' namespace)
postgresql:
  enabled: false
data:
  metadataConnection:
    user: postgres
    pass: postgres123
    protocol: postgresql
    host: postgresql.postgresql.svc.cluster.local
    port: 5432
    db: airflow

# Use MinIO for logs (in 'minio' namespace)
logs:
  persistence:
    enabled: false
config:
  logging:
    remote_logging: 'True'
    remote_base_log_folder: 's3://airflow-logs'
    remote_log_conn_id: 'minio_default'

# Enable OpenLineage (connects to Marquez in 'marquez' namespace)
extraEnv:
  - name: AIRFLOW__LINEAGE__BACKEND
    value: "openlineage.lineage_backend.OpenLineageBackend"
  - name: OPENLINEAGE_URL
    value: "http://marquez-api.marquez.svc.cluster.local:5000"
  - name: OPENLINEAGE_NAMESPACE
    value: "airflow"

# Resource limits
webserver:
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m
  service:
    type: NodePort
    ports:
      - name: airflow-ui
        port: 8080
        nodePort: 30800
scheduler:
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m
EOF

# Install from local folder
helm install airflow ./airflow \
  --namespace airflow \
  --values /home/sujith/github/rag/00_MLOps/helm_value_files/airflow-values.yaml \
  --timeout 10m

# Verify
kubectl get pods -n airflow

echo "Airflow UI: http://localhost:30800"
echo "Default credentials: admin / admin"
```

### 2.2 Install Marquez (Data Lineage)

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Save manifest locally
cat <<EOF > marquez.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marquez-api
  namespace: marquez
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
          value: postgresql.postgresql.svc.cluster.local
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
  namespace: marquez
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
  namespace: marquez
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
          value: marquez-api.marquez.svc.cluster.local
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
  namespace: marquez
spec:
  selector:
    app: marquez-web
  ports:
  - port: 3000
    targetPort: 3000
    nodePort: 30501
  type: NodePort
EOF

kubectl apply -f marquez.yaml

# Verify
kubectl get pods -n marquez

echo "Marquez UI: http://localhost:30501"
echo "Marquez API: http://localhost:30500"
```

### 2.3 Install Feast (Feature Store)

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Save manifest locally
cat <<EOF > feast.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feast-feature-server
  namespace: feast
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
  namespace: feast
spec:
  selector:
    app: feast
  ports:
  - port: 6566
    targetPort: 6566
    nodePort: 30656
  type: NodePort
EOF

kubectl apply -f feast.yaml

# Verify
kubectl get pods -n feast

echo "Feast Feature Server: http://localhost:30656"
```

### 2.4 Install DVC (Data Version Control)

DVC is a CLI tool installed on your local machine:

```bash
# Install DVC
pip install dvc[s3]

# Configure DVC to use MinIO (in 'minio' namespace)
dvc remote add -d minio s3://dvc-storage
dvc remote modify minio endpointurl http://localhost:30900
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# Verify DVC
dvc version
```

### 2.5 Install Great Expectations

Great Expectations is a Python library:

```bash
# Install Great Expectations
pip install great-expectations

echo "Great Expectations: Installed as Python library"
```

---

## Phase 3: Code Loop Tools

### 3.1 Install Argo Workflows

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Manifest
wget https://github.com/argoproj/argo-workflows/releases/download/v3.5.2/install.yaml -O argo-workflows-install.yaml

# Install
kubectl apply -n argo -f argo-workflows-install.yaml

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
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Create MLflow deployment
cat <<EOF > mlflow.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlflow
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
          - "--backend-store-uri=postgresql://postgres:postgres123@postgresql.postgresql.svc.cluster.local:5432/mlflow"
          - "--default-artifact-root=s3://mlflow-artifacts"
          - "--host=0.0.0.0"
          - "--port=5000"
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio.minio.svc.cluster.local:9000"
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
  namespace: mlflow
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
    nodePort: 30050
  type: NodePort
EOF

kubectl apply -f mlflow.yaml

# Verify
kubectl get pods -n mlflow

echo "MLflow UI: http://localhost:30050"
```

### 4.2 Install Kubeflow Pipelines

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# NOTE: Kubeflow is complex and pulls many resources. 
# We will download the kustomize manifests.
export PIPELINE_VERSION=2.0.5

# We can't easily "download" a kustomize build to a single file without running kustomize build.
# We will pull the resources and pipe to file.

# Cluster Scoped
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" > kubeflow-cluster-scoped.yaml
# Env
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION" > kubeflow-platform-agnostic.yaml

# Apply local files
kubectl apply -f kubeflow-cluster-scoped.yaml
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -f kubeflow-platform-agnostic.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s

# Patch to use NodePort
kubectl patch svc ml-pipeline-ui -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "nodePort": 30880}]}}'

# Verify
kubectl get pods -n kubeflow

echo "Kubeflow Pipelines UI: http://localhost:30880"
```

### 4.3 What-If Tool & Model Card Toolkit

These are Python libraries:

```bash
pip install witwidget              # What-If Tool
pip install model-card-toolkit     # Model Card Toolkit
```

---

## Phase 5: Deployment Loop Tools

### 5.1 Install Argo CD

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Manifest
wget https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml -O argo-cd-install.yaml

# Install
kubectl apply -n argocd -f argo-cd-install.yaml

# Patch to use NodePort
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort", "ports": [{"port": 443, "nodePort": 30443, "name": "https"}, {"port": 80, "nodePort": 30081, "name": "http"}]}}'

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""

# Verify
kubectl get pods -n argocd

echo "Argo CD UI: http://localhost:30081"
echo "Username: admin"
```

### 5.2 Install KServe

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Cert Manager
wget https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install Cert Manager
kubectl apply -f cert-manager.yaml
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=120s

# Download KServe
wget https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml
wget https://github.com/kserve/kserve/releases/download/v0.12.0/kserve-runtimes.yaml

# Install KServe
kubectl apply -f kserve.yaml
kubectl apply -f kserve-runtimes.yaml

# Verify
kubectl get pods -n kserve

echo "KServe: Installed and ready for InferenceServices"
```

### 5.3 Install Iter8

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Download Iter8
wget https://github.com/iter8-tools/iter8/releases/latest/download/install.yaml -O iter8-install.yaml

# Install
kubectl apply -f iter8-install.yaml

# Verify
kubectl get pods -n iter8-system

echo "Iter8: Installed and ready for A/B testing"
```

---

## Phase 6: Monitoring Loop Tools

### 6.1 Install Evidently AI

```bash
cd /home/sujith/github/rag/00_MLOps/helm_charts

# Create Evidently deployment
cat <<EOF > evidently.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evidently-collector
  namespace: evidently
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
  namespace: evidently
spec:
  selector:
    app: evidently
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30850
  type: NodePort
EOF

kubectl apply -f evidently.yaml

# Verify
kubectl get pods -n evidently

echo "Evidently Collector: http://localhost:30850"
```

---

## Phase 7: Verification

### 7.1 Check All Pods

```bash
echo "=== minio ==="
kubectl get pods -n minio

echo -e "\n=== postgresql ==="
kubectl get pods -n postgresql

echo -e "\n=== redis ==="
kubectl get pods -n redis

echo -e "\n=== airflow ==="
kubectl get pods -n airflow

echo -e "\n=== marquez ==="
kubectl get pods -n marquez

echo -e "\n=== feast ==="
kubectl get pods -n feast

echo -e "\n=== argo ==="
kubectl get pods -n argo

echo -e "\n=== mlflow ==="
kubectl get pods -n mlflow

echo -e "\n=== kubeflow ==="
kubectl get pods -n kubeflow

echo -e "\n=== argocd ==="
kubectl get pods -n argocd
```

### 7.2 Service Endpoints Summary

| Service | Namespace | URL | Credentials |
|---------|-----------|-----|-------------|
| **MinIO Console** | minio | http://localhost:30901 | minioadmin / minioadmin123 |
| **Airflow** | airflow | http://localhost:30800 | admin / admin |
| **Marquez** | marquez | http://localhost:30501 | - |
| **Feast** | feast | http://localhost:30656 | - |
| **Argo Workflows** | argo | http://localhost:30746 | - |
| **MLflow** | mlflow | http://localhost:30050 | - |
| **Kubeflow Pipelines** | kubeflow | http://localhost:30880 | - |
| **Argo CD** | argocd | http://localhost:30081 | admin / (see command) |
| **Evidently** | evidently | http://localhost:30850 | - |
