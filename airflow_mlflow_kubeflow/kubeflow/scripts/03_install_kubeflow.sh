#!/bin/bash
set -e

echo "=========================================="
echo "  Kubeflow Installation"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBEFLOW_VERSION=${KUBEFLOW_VERSION:-v1.8.0}
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests/kubeflow-manifests"

echo ""
echo "Configuration:"
echo "  Kubeflow Version: ${KUBEFLOW_VERSION}"
echo "  Manifests Dir: ${MANIFESTS_DIR}"
echo ""

# Check Minikube status
if ! minikube status &> /dev/null; then
    echo "ERROR: Minikube is not running!"
    echo "Run ./02_setup_minikube.sh first"
    exit 1
fi

echo "[1/5] Cloning Kubeflow manifests..."
if [ -d "${MANIFESTS_DIR}" ]; then
    echo "Manifests directory exists. Updating..."
    cd "${MANIFESTS_DIR}"
    git fetch --all
    git checkout ${KUBEFLOW_VERSION}
else
    git clone https://github.com/kubeflow/manifests.git "${MANIFESTS_DIR}"
    cd "${MANIFESTS_DIR}"
    git checkout ${KUBEFLOW_VERSION}
fi

echo ""
echo "[2/5] Installing Kubeflow components..."
echo "This will take 10-20 minutes. Please wait..."
echo ""

# Install with retries
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if kustomize build example | kubectl apply -f -; then
        echo "Kubeflow components applied successfully!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Retry ${RETRY_COUNT}/${MAX_RETRIES}..."
        sleep 30
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "WARNING: Some components may not have been applied. Continuing..."
fi

echo ""
echo "[3/5] Waiting for core namespaces..."
kubectl wait --for=condition=Ready namespace/kubeflow --timeout=60s || true
kubectl wait --for=condition=Ready namespace/istio-system --timeout=60s || true

echo ""
echo "[4/5] Waiting for pods to be ready..."
echo "This may take several minutes..."

# Wait for critical pods
namespaces=("istio-system" "kubeflow" "cert-manager")

for ns in "${namespaces[@]}"; do
    echo "Waiting for pods in ${ns}..."
    kubectl wait --for=condition=Ready pods --all -n ${ns} --timeout=600s 2>/dev/null || true
done

echo ""
echo "[5/5] Verifying installation..."

echo ""
echo "Pod Status by Namespace:"
echo "------------------------"

for ns in istio-system cert-manager kubeflow; do
    total=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | wc -l)
    running=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    echo "${ns}: ${running}/${total} running"
done

echo ""
echo "=========================================="
echo "  Kubeflow Installation Complete"
echo "=========================================="
echo ""
echo "Default credentials:"
echo "  Email: user@example.com"
echo "  Password: 12341234"
echo ""
echo "Next step: ./04_port_forward.sh"
echo "=========================================="
