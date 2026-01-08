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
