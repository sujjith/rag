#!/bin/bash

echo "=========================================="
echo "  Kubeflow Port Forwarding"
echo "=========================================="

# Kill existing port-forwards
pkill -f "kubectl port-forward.*istio-ingressgateway" 2>/dev/null || true
pkill -f "kubectl port-forward.*minio" 2>/dev/null || true
pkill -f "kubectl port-forward.*ml-pipeline" 2>/dev/null || true

sleep 2

echo ""
echo "Starting port forwards..."

# Main Kubeflow Dashboard (Istio Gateway)
echo "[1/3] Kubeflow Dashboard (8080)..."
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &
sleep 2

# MinIO Console (optional)
echo "[2/3] MinIO Console (9001)..."
kubectl port-forward svc/minio-service -n kubeflow 9000:9000 9001:9001 &>/dev/null &
sleep 1

# Pipelines API (optional, for SDK access)
echo "[3/3] Pipelines API (8888)..."
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8888:80 &>/dev/null &
sleep 1

echo ""
echo "=========================================="
echo "  Port Forwarding Active"
echo "=========================================="
echo ""
echo "Access URLs:"
echo "  Kubeflow Dashboard: http://localhost:8080"
echo "  MinIO Console:      http://localhost:9001"
echo "  Pipelines UI:       http://localhost:8888"
echo ""
echo "Credentials:"
echo "  Kubeflow: user@example.com / 12341234"
echo "  MinIO:    minio / minio123"
echo ""
echo "Press Ctrl+C to stop port forwarding"
echo "=========================================="

# Keep script running
wait
