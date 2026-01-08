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
