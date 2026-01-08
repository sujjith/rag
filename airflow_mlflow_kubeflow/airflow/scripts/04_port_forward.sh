#!/bin/bash

echo "=========================================="
echo "  Airflow Port Forwarding"
echo "=========================================="

NAMESPACE=${NAMESPACE:-airflow}

# Kill existing port-forwards
pkill -f "kubectl port-forward.*airflow" 2>/dev/null || true
sleep 2

echo ""
echo "Starting port forwards..."

# Airflow Webserver
echo "[1/2] Airflow UI (8080)..."
kubectl port-forward svc/airflow-webserver -n ${NAMESPACE} 8080:8080 &
sleep 2

# Flower (Celery monitoring)
echo "[2/2] Flower UI (5555)..."
kubectl port-forward svc/airflow-flower -n ${NAMESPACE} 5555:5555 &>/dev/null &
sleep 1

echo ""
echo "=========================================="
echo "  Port Forwarding Active"
echo "=========================================="
echo ""
echo "Access URLs:"
echo "  Airflow UI:  http://localhost:8080"
echo "  Flower UI:   http://localhost:5555"
echo ""
echo "Credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Press Ctrl+C to stop port forwarding"
echo "=========================================="

# Keep script running
wait
