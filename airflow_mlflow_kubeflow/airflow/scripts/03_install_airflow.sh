#!/bin/bash
set -e

echo "=========================================="
echo "  Apache Airflow Installation"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=${NAMESPACE:-airflow}
RELEASE_NAME=${RELEASE_NAME:-airflow}
VALUES_FILE="${SCRIPT_DIR}/../helm/values.yaml"

echo ""
echo "Configuration:"
echo "  Namespace: ${NAMESPACE}"
echo "  Release: ${RELEASE_NAME}"
echo "  Values: ${VALUES_FILE}"
echo ""

# Check prerequisites
if ! command -v helm &> /dev/null; then
    echo "ERROR: Helm is not installed"
    exit 1
fi

if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
    echo "ERROR: Namespace ${NAMESPACE} does not exist"
    echo "Run ./02_setup_namespace.sh first"
    exit 1
fi

echo "[1/4] Adding Airflow Helm repository..."
helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
helm repo update

echo ""
echo "[2/4] Installing Airflow..."
echo "This may take 5-10 minutes..."

helm upgrade --install ${RELEASE_NAME} apache-airflow/airflow \
    --namespace ${NAMESPACE} \
    -f ${VALUES_FILE} \
    --timeout 15m \
    --wait

echo ""
echo "[3/4] Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pods --all -n ${NAMESPACE} --timeout=600s 2>/dev/null || true

echo ""
echo "[4/4] Verifying installation..."
echo ""
echo "Pod Status:"
kubectl get pods -n ${NAMESPACE}

echo ""
echo "=========================================="
echo "  Airflow Installation Complete"
echo "=========================================="
echo ""
echo "Default credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Next step: ./04_port_forward.sh"
echo "=========================================="
