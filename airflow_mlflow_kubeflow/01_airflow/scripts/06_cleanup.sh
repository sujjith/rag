#!/bin/bash

echo "=========================================="
echo "  Airflow Cleanup"
echo "=========================================="

NAMESPACE=${NAMESPACE:-airflow}
RELEASE_NAME=${RELEASE_NAME:-airflow}

echo ""
read -p "This will uninstall Airflow. Continue? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "[1/3] Stopping port forwards..."
pkill -f "kubectl port-forward.*airflow" 2>/dev/null || true

echo ""
echo "[2/3] Uninstalling Airflow Helm release..."
helm uninstall ${RELEASE_NAME} -n ${NAMESPACE} 2>/dev/null || true

echo ""
echo "[3/3] Cleaning up PVCs..."
read -p "Delete persistent volume claims? (y/N): " del_pvc
if [[ "$del_pvc" =~ ^[Yy]$ ]]; then
    kubectl delete pvc --all -n ${NAMESPACE} 2>/dev/null || true
    echo "PVCs deleted"
fi

read -p "Delete namespace ${NAMESPACE}? (y/N): " del_ns
if [[ "$del_ns" =~ ^[Yy]$ ]]; then
    kubectl delete namespace ${NAMESPACE} 2>/dev/null || true
    echo "Namespace deleted"
fi

echo ""
echo "=========================================="
echo "  Cleanup Complete"
echo "=========================================="
echo ""
echo "To reinstall:"
echo "  ./02_setup_namespace.sh"
echo "  ./03_install_airflow.sh"
echo "=========================================="
