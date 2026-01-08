# MLflow Implementation Steps - Minikube Deployment

## Overview

Deploy MLflow v3.8.1 on Minikube with PostgreSQL backend, persistent artifact storage, and model serving capabilities.

### Features
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version control and stage management for models
- **Model Serving**: REST API for model inference on Kubernetes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Minikube Cluster                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        mlflow Namespace                             │ │
│  │                                                                     │ │
│  │  ┌─────────────────────┐     ┌─────────────────────┐               │ │
│  │  │  MLflow Tracking    │     │  MLflow Model       │               │ │
│  │  │  Server (Pod)       │     │  Server (Pod)       │               │ │
│  │  │  Port: 5000         │     │  Port: 5001         │               │ │
│  │  └─────────┬───────────┘     └──────────┬──────────┘               │ │
│  │            │                            │                           │ │
│  │            ▼                            │                           │ │
│  │  ┌─────────────────────┐               │                           │ │
│  │  │  PostgreSQL (Pod)   │               │                           │ │
│  │  │  Port: 5432         │               │                           │ │
│  │  │  + PVC (metadata)   │               │                           │ │
│  │  └─────────────────────┘               │                           │ │
│  │                                        │                           │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │                 PVC: mlflow-artifacts                        │  │ │
│  │  │                 (Shared artifact storage)                    │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                     │ │
│  │  ┌─────────────────────┐     ┌─────────────────────┐               │ │
│  │  │  Service: mlflow    │     │  Service: mlflow-   │               │ │
│  │  │  NodePort: 30500    │     │  model-server       │               │ │
│  │  │                     │     │  NodePort: 30501    │               │ │
│  │  └─────────────────────┘     └─────────────────────┘               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Port | NodePort | Purpose |
|-----------|------|----------|---------|
| MLflow Tracking Server | 5000 | 30500 | Experiment tracking, model registry, UI |
| PostgreSQL | 5432 | - | Metadata storage (internal only) |
| MLflow Model Server | 5001 | 30501 | Model serving / inference |

---

## Directory Structure

```
/home/sujith/github/rag/airflow_mlflow_kubeflow/00_mlflow/
├── mlflow_implementation_steps.md   # This document
├── k8s/
│   ├── namespace.yaml               # mlflow namespace
│   ├── configmap.yaml               # Environment configuration
│   ├── secrets.yaml                 # Database credentials
│   ├── postgres-pvc.yaml            # PostgreSQL persistent volume
│   ├── postgres-deployment.yaml     # PostgreSQL deployment + service
│   ├── artifacts-pvc.yaml           # Artifact storage persistent volume
│   ├── mlflow-deployment.yaml       # MLflow server deployment + service
│   └── model-server-deployment.yaml # Model serving deployment + service
├── scripts/
│   ├── deploy.sh                    # Deploy all resources
│   ├── destroy.sh                   # Remove all resources
│   └── port-forward.sh              # Port forwarding script
└── examples/
    ├── pyproject.toml               # uv project file
    ├── 01_basic_tracking.py         # Basic experiment tracking
    ├── 02_model_registry.py         # Model registration example
    └── 03_test_serving.py           # Model serving test
```

---

## Prerequisites

### 1. Install uv (Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 2. Install Minikube and kubectl

```bash
# Check if minikube is installed
minikube version

# Check if kubectl is installed
kubectl version --client
```

### 3. Start Minikube

```bash
# Start minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker

# Verify cluster is running
kubectl cluster-info
```

---

## Step 1: Create Kubernetes Manifests

### 1.1 Namespace

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlflow
  labels:
    app: mlflow
```

### 1.2 ConfigMap

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-config
  namespace: mlflow
data:
  MLFLOW_TRACKING_URI: "http://mlflow-server:5000"
  POSTGRES_HOST: "postgres"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "mlflow"
```

### 1.3 Secrets

Create `k8s/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-secrets
  namespace: mlflow
type: Opaque
stringData:
  POSTGRES_USER: "mlflow"
  POSTGRES_PASSWORD: "mlflow123"
```

### 1.4 PostgreSQL PVC

Create `k8s/postgres-pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: mlflow
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### 1.5 Artifacts PVC

Create `k8s/artifacts-pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-artifacts-pvc
  namespace: mlflow
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### 1.6 PostgreSQL Deployment

Create `k8s/postgres-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15-alpine
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: mlflow-secrets
                  key: POSTGRES_USER
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mlflow-secrets
                  key: POSTGRES_PASSWORD
            - name: POSTGRES_DB
              valueFrom:
                configMapKeyRef:
                  name: mlflow-config
                  key: POSTGRES_DB
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - mlflow
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: mlflow
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  type: ClusterIP
```

### 1.7 MLflow Server Deployment

Create `k8s/mlflow-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      initContainers:
        - name: wait-for-postgres
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              until nc -z postgres 5432; do
                echo "Waiting for PostgreSQL..."
                sleep 2
              done
              echo "PostgreSQL is ready!"
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v2.19.0
          ports:
            - containerPort: 5000
          command:
            - mlflow
            - server
            - --host=0.0.0.0
            - --port=5000
            - --backend-store-uri=postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@postgres:5432/mlflow
            - --default-artifact-root=/mlartifacts
            - --serve-artifacts
          env:
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: mlflow-secrets
                  key: POSTGRES_USER
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mlflow-secrets
                  key: POSTGRES_PASSWORD
          volumeMounts:
            - name: artifacts-storage
              mountPath: /mlartifacts
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 60
            periodSeconds: 30
      volumes:
        - name: artifacts-storage
          persistentVolumeClaim:
            claimName: mlflow-artifacts-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: mlflow
spec:
  selector:
    app: mlflow-server
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: 30500
  type: NodePort
```

### 1.8 Model Server Deployment

Create `k8s/model-server-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-model-server
  namespace: mlflow
  labels:
    app: mlflow-model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-model-server
  template:
    metadata:
      labels:
        app: mlflow-model-server
    spec:
      containers:
        - name: model-server
          image: ghcr.io/mlflow/mlflow:v2.19.0
          ports:
            - containerPort: 5001
          command:
            - sh
            - -c
            - |
              echo "Model server ready. Waiting for model URI via environment..."
              echo "To serve a model, update MODEL_URI env var and restart the pod."
              # Default: serve a placeholder that returns 503
              python -c "
              from http.server import HTTPServer, BaseHTTPRequestHandler
              import json
              class Handler(BaseHTTPRequestHandler):
                  def do_GET(self):
                      if self.path == '/health':
                          self.send_response(200)
                          self.send_header('Content-Type', 'application/json')
                          self.end_headers()
                          self.wfile.write(json.dumps({'status': 'ok', 'model': 'none'}).encode())
                      else:
                          self.send_response(404)
                          self.end_headers()
                  def do_POST(self):
                      self.send_response(503)
                      self.send_header('Content-Type', 'application/json')
                      self.end_headers()
                      self.wfile.write(json.dumps({'error': 'No model loaded. Set MODEL_URI env var.'}).encode())
              HTTPServer(('0.0.0.0', 5001), Handler).serve_forever()
              "
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-server:5000"
            - name: MODEL_URI
              value: ""  # Set this to models:/<name>/<stage> when ready
          volumeMounts:
            - name: artifacts-storage
              mountPath: /mlartifacts
              readOnly: true
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
      volumes:
        - name: artifacts-storage
          persistentVolumeClaim:
            claimName: mlflow-artifacts-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-model-server
  namespace: mlflow
spec:
  selector:
    app: mlflow-model-server
  ports:
    - port: 5001
      targetPort: 5001
      nodePort: 30501
  type: NodePort
```

---

## Step 2: Create Deployment Scripts

### 2.1 Deploy Script

Create `scripts/deploy.sh`:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

echo "=========================================="
echo "Deploying MLflow on Minikube"
echo "=========================================="

# Check minikube status
echo "[1/6] Checking Minikube status..."
if ! minikube status | grep -q "Running"; then
    echo "Starting Minikube..."
    minikube start --cpus=4 --memory=8192 --driver=docker
fi

# Apply namespace
echo "[2/6] Creating namespace..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

# Apply config and secrets
echo "[3/6] Applying configuration..."
kubectl apply -f "$K8S_DIR/configmap.yaml"
kubectl apply -f "$K8S_DIR/secrets.yaml"

# Apply PVCs
echo "[4/6] Creating persistent volumes..."
kubectl apply -f "$K8S_DIR/postgres-pvc.yaml"
kubectl apply -f "$K8S_DIR/artifacts-pvc.yaml"

# Deploy PostgreSQL
echo "[5/6] Deploying PostgreSQL..."
kubectl apply -f "$K8S_DIR/postgres-deployment.yaml"
echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n mlflow --timeout=120s

# Deploy MLflow
echo "[6/6] Deploying MLflow Server..."
kubectl apply -f "$K8S_DIR/mlflow-deployment.yaml"
kubectl apply -f "$K8S_DIR/model-server-deployment.yaml"
echo "Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow-server -n mlflow --timeout=180s

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Access MLflow UI:"
echo "  URL: http://$(minikube ip):30500"
echo ""
echo "Or use port-forward:"
echo "  kubectl port-forward svc/mlflow-server 5000:5000 -n mlflow"
echo "  Then open: http://localhost:5000"
echo ""
echo "Check status:"
echo "  kubectl get pods -n mlflow"
echo ""
```

### 2.2 Destroy Script

Create `scripts/destroy.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Removing MLflow deployment"
echo "=========================================="

# Delete all resources in the namespace
kubectl delete namespace mlflow --ignore-not-found

echo ""
echo "MLflow deployment removed!"
echo ""
```

### 2.3 Port Forward Script

Create `scripts/port-forward.sh`:

```bash
#!/bin/bash

echo "Starting port forwards..."
echo "MLflow UI: http://localhost:5000"
echo "Model Server: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run both port-forwards in background
kubectl port-forward svc/mlflow-server 5000:5000 -n mlflow &
kubectl port-forward svc/mlflow-model-server 5001:5001 -n mlflow &

# Wait for both
wait
```

---

## Step 3: Create Python Examples

### 3.1 Project Setup with uv

Create `examples/pyproject.toml`:

```toml
[project]
name = "mlflow-examples"
version = "0.1.0"
description = "MLflow example scripts"
requires-python = ">=3.12"
dependencies = [
    "mlflow>=2.19.0",
    "scikit-learn>=1.5.0",
    "pandas>=2.2.0",
    "matplotlib>=3.9.0",
    "requests>=2.32.0",
]

[tool.uv]
dev-dependencies = []
```

### 3.2 Basic Tracking Example

Create `examples/01_basic_tracking.py`:

```python
"""
Basic MLflow experiment tracking example.
Run after deploying MLflow on Minikube.
"""
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Configure MLflow (use port-forward or minikube IP)
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-classification")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter configurations to try
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 10},
]

print("=" * 60)
print("MLflow Basic Tracking Example")
print("=" * 60)
print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
print()

for config in configs:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Train model
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        print(f"Config: {config}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print()

print("=" * 60)
print(f"View results at: {MLFLOW_TRACKING_URI}")
print("=" * 60)
```

### 3.3 Model Registry Example

Create `examples/02_model_registry.py`:

```python
"""
Model Registry example - register, version, and promote models.
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "iris-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("model-registry-demo")
client = MlflowClient()

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Model Registry Demo")
print("=" * 60)

# Train and register model
with mlflow.start_run(run_name="production-candidate") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log and register model
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    result = mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:2],
        registered_model_name=MODEL_NAME
    )
    
    print(f"Model registered: {MODEL_NAME}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# Get latest version
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))
print(f"\nLatest version: {latest_version.version}")

# Transition to Staging
print("\nPromoting to Staging...")
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Staging"
)
print(f"Version {latest_version.version} is now in Staging")

# Transition to Production
print("\nPromoting to Production...")
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Version {latest_version.version} is now in Production")

# List all versions
print("\n" + "=" * 60)
print("Model Versions:")
print("=" * 60)
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"  Version {mv.version}: {mv.current_stage}")

print(f"\nView at: {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")
```

### 3.4 Test Serving Example

Create `examples/03_test_serving.py`:

```python
"""
Test model serving endpoint.
"""
import requests
import json
from sklearn.datasets import load_iris

MODEL_SERVER = "http://localhost:5001"
ENDPOINT = f"{MODEL_SERVER}/invocations"

iris = load_iris()

# Test samples (one of each class)
samples = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.2, 2.9, 4.3, 1.3],  # versicolor
    [7.7, 3.0, 6.1, 2.3],  # virginica
]

print("=" * 60)
print("Testing MLflow Model Server")
print("=" * 60)

# Health check
print("\n[Health Check]")
try:
    response = requests.get(f"{MODEL_SERVER}/health", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Server not reachable: {e}")
    print("\nMake sure to:")
    print("1. Register a model using 02_model_registry.py")
    print("2. Update the model-server deployment with MODEL_URI")
    print("3. Run: kubectl port-forward svc/mlflow-model-server 5001:5001 -n mlflow")
    exit(1)

# Test predictions
print("\n[Predictions]")
try:
    payload = {"inputs": samples}
    response = requests.post(
        ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            class_name = iris.target_names[pred] if isinstance(pred, int) else pred
            print(f"  Input: {sample}")
            print(f"  Prediction: {pred} ({class_name})")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
except Exception as e:
    print(f"Prediction failed: {e}")

print("=" * 60)
```

---

## Step 4: Deploy MLflow

### 4.1 Create Directory Structure

```bash
cd /home/sujith/github/rag/airflow_mlflow_kubeflow/00_mlflow

# Create directories
mkdir -p k8s scripts examples
```

### 4.2 Apply All Manifests

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy
./scripts/deploy.sh
```

### 4.3 Verify Deployment

```bash
# Check pods
kubectl get pods -n mlflow

# Expected output:
# NAME                                   READY   STATUS    RESTARTS   AGE
# postgres-xxxxx                         1/1     Running   0          2m
# mlflow-server-xxxxx                    1/1     Running   0          1m
# mlflow-model-server-xxxxx              1/1     Running   0          1m

# Check services
kubectl get svc -n mlflow

# Check PVCs
kubectl get pvc -n mlflow
```

---

## Step 5: Run Examples

### 5.1 Setup Python Environment

```bash
cd /home/sujith/github/rag/airflow_mlflow_kubeflow/00_mlflow/examples

# Create virtual environment with uv
uv venv

# Activate environment
source .venv/bin/activate

# Sync dependencies
uv sync
```

### 5.2 Start Port Forwarding

```bash
# In a separate terminal
cd /home/sujith/github/rag/airflow_mlflow_kubeflow/00_mlflow
./scripts/port-forward.sh
```

### 5.3 Run Examples

```bash
# Basic tracking
uv run python 01_basic_tracking.py

# Model registry
uv run python 02_model_registry.py

# Test serving (after updating model-server)
uv run python 03_test_serving.py
```

---

## Step 6: Serve a Model

After registering a model, update the model server to serve it:

### 6.1 Update Model Server Deployment

```bash
# Edit the deployment to set MODEL_URI
kubectl set env deployment/mlflow-model-server \
  MODEL_URI="models:/iris-classifier/Production" \
  -n mlflow

# Or patch the deployment
kubectl patch deployment mlflow-model-server -n mlflow --type='json' -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": [
    "mlflow", "models", "serve",
    "-m", "models:/iris-classifier/Production",
    "-h", "0.0.0.0",
    "-p", "5001",
    "--env-manager", "local"
  ]}
]'

# Restart the deployment
kubectl rollout restart deployment/mlflow-model-server -n mlflow
```

### 6.2 Verify Model Server

```bash
# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=mlflow-model-server -n mlflow --timeout=120s

# Test the endpoint
curl http://$(minikube ip):30501/health
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pods stuck in Pending | Check PVC status: `kubectl get pvc -n mlflow` |
| MLflow can't connect to PostgreSQL | Check postgres logs: `kubectl logs -l app=postgres -n mlflow` |
| Model server returns 503 | Verify MODEL_URI is set and model exists in registry |
| Port forward disconnects | Use `minikube service mlflow-server -n mlflow` instead |

### Debug Commands

```bash
# View pod logs
kubectl logs -l app=mlflow-server -n mlflow -f
kubectl logs -l app=postgres -n mlflow -f
kubectl logs -l app=mlflow-model-server -n mlflow -f

# Describe pod for events
kubectl describe pod -l app=mlflow-server -n mlflow

# Enter container shell
kubectl exec -it deployment/mlflow-server -n mlflow -- /bin/bash

# Check database
kubectl exec -it deployment/postgres -n mlflow -- psql -U mlflow -d mlflow -c "\dt"

# Reset everything
./scripts/destroy.sh
./scripts/deploy.sh
```

### Access Without Port Forwarding

```bash
# Get Minikube IP
minikube ip

# Access MLflow UI
# http://<minikube-ip>:30500

# Or use minikube service command
minikube service mlflow-server -n mlflow --url
```

---

## Quick Reference

```bash
# Start environment
minikube start
./scripts/deploy.sh
./scripts/port-forward.sh

# Python setup
cd examples
uv venv && source .venv/bin/activate && uv sync

# Run experiments
uv run python 01_basic_tracking.py
uv run python 02_model_registry.py

# Access UI
open http://localhost:5000

# Cleanup
./scripts/destroy.sh
minikube stop
```
