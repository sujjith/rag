#!/bin/bash
set -e

echo "=========================================="
echo "  Airflow Namespace Setup"
echo "=========================================="

NAMESPACE=${NAMESPACE:-airflow}

# Check if Minikube is running
if ! minikube status &> /dev/null; then
    echo "Minikube is not running. Starting..."
    minikube start --cpus=4 --memory=8192 --disk-size=40g
fi

echo ""
echo "[1/3] Creating namespace: ${NAMESPACE}"
kubectl create namespace ${NAMESPACE} 2>/dev/null || echo "Namespace already exists"

echo ""
echo "[2/3] Setting up RBAC..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: airflow
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: airflow-role
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "pods/exec"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["secrets", "configmaps"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: airflow-rolebinding
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: airflow
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: airflow-role
  apiGroup: rbac.authorization.k8s.io
EOF

echo ""
echo "[3/3] Verifying namespace..."
kubectl get namespace ${NAMESPACE}

echo ""
echo "=========================================="
echo "  Namespace Setup Complete"
echo "=========================================="
echo ""
echo "Namespace: ${NAMESPACE}"
echo ""
echo "Next step: ./03_install_airflow.sh"
echo "=========================================="
