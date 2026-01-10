#!/bin/bash

echo "=========================================="
echo "  Kubeflow Cleanup"
echo "=========================================="

echo ""
read -p "This will delete the Minikube cluster. Continue? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "[1/3] Stopping port forwards..."
pkill -f "kubectl port-forward" 2>/dev/null || true

echo ""
echo "[2/3] Deleting Minikube cluster..."
minikube delete

echo ""
echo "[3/3] Cleaning up manifests..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests/kubeflow-manifests"

if [ -d "${MANIFESTS_DIR}" ]; then
    read -p "Delete cloned manifests? (y/N): " del_manifests
    if [[ "$del_manifests" =~ ^[Yy]$ ]]; then
        rm -rf "${MANIFESTS_DIR}"
        echo "Manifests deleted."
    fi
fi

echo ""
echo "=========================================="
echo "  Cleanup Complete"
echo "=========================================="
echo ""
echo "To reinstall:"
echo "  ./02_setup_minikube.sh"
echo "  ./03_install_kubeflow.sh"
echo "=========================================="
