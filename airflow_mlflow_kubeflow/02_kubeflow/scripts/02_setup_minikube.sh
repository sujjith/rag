#!/bin/bash
set -e

echo "=========================================="
echo "  Minikube Setup for Kubeflow"
echo "=========================================="

# Configuration
CPUS=${CPUS:-4}
MEMORY=${MEMORY:-8192}
DISK_SIZE=${DISK_SIZE:-40g}
K8S_VERSION=${K8S_VERSION:-v1.28.0}
DRIVER=${DRIVER:-docker}

echo ""
echo "Configuration:"
echo "  CPUs: ${CPUS}"
echo "  Memory: ${MEMORY}MB"
echo "  Disk: ${DISK_SIZE}"
echo "  Kubernetes: ${K8S_VERSION}"
echo "  Driver: ${DRIVER}"
echo ""

# Check if Minikube is already running
if minikube status &> /dev/null; then
    echo "Minikube is already running."
    read -p "Delete and recreate? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "Deleting existing Minikube cluster..."
        minikube delete
    else
        echo "Using existing cluster."
        exit 0
    fi
fi

echo "[1/4] Starting Minikube..."
minikube start \
    --cpus=${CPUS} \
    --memory=${MEMORY} \
    --disk-size=${DISK_SIZE} \
    --driver=${DRIVER} \
    --kubernetes-version=${K8S_VERSION} \
    --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key \
    --extra-config=apiserver.service-account-issuer=kubernetes.default.svc

echo ""
echo "[2/4] Enabling addons..."
minikube addons enable default-storageclass
minikube addons enable storage-provisioner
minikube addons enable metrics-server

echo ""
echo "[3/4] Verifying cluster..."
kubectl cluster-info
echo ""
kubectl get nodes

echo ""
echo "[4/4] Setting up Docker environment..."
echo "Run this command to use Minikube's Docker daemon:"
echo ""
echo "  eval \$(minikube docker-env)"
echo ""

echo "=========================================="
echo "  Minikube Setup Complete"
echo "=========================================="
echo ""
echo "Cluster Status:"
minikube status
echo ""
echo "Next step: ./03_install_kubeflow.sh"
echo "=========================================="
