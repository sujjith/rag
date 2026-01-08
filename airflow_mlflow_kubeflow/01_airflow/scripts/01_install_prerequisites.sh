#!/bin/bash
set -e

echo "=========================================="
echo "  Airflow Prerequisites Installation"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

echo ""
echo "[1/4] Checking kubectl..."
if command -v kubectl &> /dev/null; then
    print_status "kubectl installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
else
    print_warning "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    print_status "kubectl installed"
fi

echo ""
echo "[2/4] Checking Helm..."
if command -v helm &> /dev/null; then
    print_status "Helm installed: $(helm version --short)"
else
    print_warning "Installing Helm..."
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    print_status "Helm installed"
fi

echo ""
echo "[3/4] Checking Minikube..."
if command -v minikube &> /dev/null; then
    print_status "Minikube installed: $(minikube version --short)"
else
    print_warning "Installing Minikube..."
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    chmod +x minikube-linux-amd64
    sudo mv minikube-linux-amd64 /usr/local/bin/minikube
    print_status "Minikube installed"
fi

echo ""
echo "[4/4] Adding Helm repositories..."
helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
helm repo update
print_status "Helm repos updated"

echo ""
echo "=========================================="
echo "  Prerequisites Installation Complete"
echo "=========================================="
echo ""
echo "Next step: ./02_setup_namespace.sh"
echo "=========================================="
