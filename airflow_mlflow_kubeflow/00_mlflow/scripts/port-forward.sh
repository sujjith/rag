#!/bin/bash

echo "Starting port forwards..."
echo "MLflow UI: http://localhost:5000"
echo "Model Server: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Trap to kill background processes on exit
trap 'kill $(jobs -p)' EXIT

# Run both port-forwards in background
kubectl port-forward svc/mlflow-server 5000:5000 -n mlflow &
kubectl port-forward svc/mlflow-model-server 5001:5001 -n mlflow &

# Wait for both
wait
