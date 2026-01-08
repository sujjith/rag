#!/bin/bash
set -e

echo "=========================================="
echo "  Kubeflow Prerequisites Installation"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run as root. Run as regular user with sudo access."
    exit 1
fi

echo ""
echo "[1/6] Checking Docker..."
if command -v docker &> /dev/null; then
    print_status "Docker already installed: $(docker --version)"
else
    print_warning "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    print_status "Docker installed. Please log out and back in for group changes."
fi

echo ""
echo "[2/6] Checking kubectl..."
if command -v kubectl &> /dev/null; then
    print_status "kubectl already installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
else
    print_warning "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    print_status "kubectl installed"
fi

echo ""
echo "[3/6] Checking Minikube..."
if command -v minikube &> /dev/null; then
    print_status "Minikube already installed: $(minikube version --short)"
else
    print_warning "Installing Minikube..."
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    chmod +x minikube-linux-amd64
    sudo mv minikube-linux-amd64 /usr/local/bin/minikube
    print_status "Minikube installed"
fi

echo ""
echo "[4/6] Checking kustomize..."
if command -v kustomize &> /dev/null; then
    print_status "kustomize already installed: $(kustomize version --short 2>/dev/null || kustomize version)"
else
    print_warning "Installing kustomize..."
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
    print_status "kustomize installed"
fi

echo ""
echo "[5/6] Checking Python and pip..."
if command -v python3 &> /dev/null; then
    print_status "Python already installed: $(python3 --version)"
else
    print_warning "Please install Python 3.9+ manually"
fi

if command -v pip3 &> /dev/null; then
    print_status "pip already installed"
else
    print_warning "Installing pip..."
    sudo apt-get update && sudo apt-get install -y python3-pip
fi

echo ""
echo "[6/6] Installing KFP SDK..."
pip3 install --user kfp==2.4.0 --quiet
print_status "KFP SDK installed"

echo ""
echo "=========================================="
echo "  Prerequisites Installation Complete"
echo "=========================================="
echo ""
echo "Installed versions:"
echo "  - Docker: $(docker --version 2>/dev/null || echo 'not available')"
echo "  - kubectl: $(kubectl version --client --short 2>/dev/null || echo 'not available')"
echo "  - Minikube: $(minikube version --short 2>/dev/null || echo 'not available')"
echo "  - kustomize: $(kustomize version --short 2>/dev/null || echo 'not available')"
echo ""
echo "Next step: ./02_setup_minikube.sh"
echo "=========================================="
