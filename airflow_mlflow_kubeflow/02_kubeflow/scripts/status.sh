#!/bin/bash

echo "=========================================="
echo "  Kubeflow Status Check"
echo "=========================================="

# Check Minikube
echo ""
echo "[Minikube Status]"
if minikube status &> /dev/null; then
    minikube status
else
    echo "Minikube is not running"
    exit 1
fi

# Check Kubernetes
echo ""
echo "[Kubernetes Nodes]"
kubectl get nodes

# Check namespaces
echo ""
echo "[Kubeflow Namespaces]"
kubectl get namespaces | grep -E "^(kubeflow|istio|cert-manager|knative)"

# Check pods by namespace
echo ""
echo "[Pod Status Summary]"
echo "-------------------"

namespaces=("istio-system" "cert-manager" "kubeflow" "kubeflow-user-example-com")

for ns in "${namespaces[@]}"; do
    if kubectl get namespace ${ns} &> /dev/null; then
        total=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | wc -l)
        running=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | grep -c "Running" || echo "0")
        pending=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | grep -c "Pending" || echo "0")
        failed=$(kubectl get pods -n ${ns} --no-headers 2>/dev/null | grep -c -E "(Error|CrashLoop)" || echo "0")
        echo "${ns}: ${running}/${total} running, ${pending} pending, ${failed} failed"
    fi
done

# Check key services
echo ""
echo "[Key Services]"
echo "--------------"

services=(
    "istio-system:istio-ingressgateway"
    "kubeflow:ml-pipeline"
    "kubeflow:ml-pipeline-ui"
    "kubeflow:jupyter-web-app-service"
    "kubeflow:katib-controller"
)

for svc in "${services[@]}"; do
    ns=$(echo $svc | cut -d: -f1)
    name=$(echo $svc | cut -d: -f2)
    if kubectl get svc ${name} -n ${ns} &> /dev/null; then
        echo "✓ ${name} (${ns})"
    else
        echo "✗ ${name} (${ns}) - NOT FOUND"
    fi
done

# Check training operators
echo ""
echo "[Training Operators]"
echo "-------------------"

operators=("tfjobs" "pytorchjobs" "mpijobs")
for op in "${operators[@]}"; do
    if kubectl api-resources | grep -q ${op}; then
        echo "✓ ${op}"
    else
        echo "✗ ${op} - NOT INSTALLED"
    fi
done

# Resource usage
echo ""
echo "[Resource Usage]"
echo "----------------"
kubectl top nodes 2>/dev/null || echo "Metrics not available"

echo ""
echo "=========================================="
