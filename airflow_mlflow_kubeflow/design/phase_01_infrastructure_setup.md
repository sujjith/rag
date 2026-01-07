# Phase 01: Infrastructure Setup

## Overview

This phase covers the complete infrastructure setup required for an enterprise-grade MLOps platform using Airflow, MLflow, and Kubeflow.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUBERNETES CLUSTER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INGRESS CONTROLLER                            │   │
│  │                    (NGINX / Traefik / Istio)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Airflow    │  │    MLflow    │  │   Kubeflow   │  │   KServe     │   │
│  │  Namespace   │  │  Namespace   │  │  Namespace   │  │  Namespace   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     STORAGE LAYER                                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ PostgreSQL │  │   MinIO    │  │   Redis    │  │  Prometheus │    │   │
│  │  │ (Metadata) │  │ (Artifacts)│  │  (Cache)   │  │ (Metrics)   │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Requirements

### Hardware Requirements

| Environment | CPU | RAM | Disk | GPU |
|-------------|-----|-----|------|-----|
| Development | 4+ cores | 16GB | 100GB SSD | Optional |
| Staging | 8+ cores | 32GB | 250GB SSD | 1x NVIDIA T4 |
| Production | 16+ cores | 64GB+ | 500GB+ SSD | 2x+ NVIDIA V100/A100 |

### Software Requirements

```bash
# Operating System
- Ubuntu 20.04/22.04 LTS (recommended)
- macOS 12+ (development only)
- Windows 11 with WSL2 (development only)

# Required Software Versions
- Docker: 24.0+
- Docker Compose: 2.20+
- Kubernetes: 1.27+
- Helm: 3.12+
- Python: 3.9 - 3.11
- kubectl: 1.27+
- kustomize: 5.0+
```

---

## Step 1: Install Docker

### Ubuntu/Debian

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Post-installation (run Docker without sudo)
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Configure Docker Daemon

Create `/etc/docker/daemon.json`:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "insecure-registries": [],
  "registry-mirrors": [],
  "exec-opts": ["native.cgroupdriver=systemd"],
  "features": {
    "buildkit": true
  }
}
```

```bash
# Restart Docker
sudo systemctl restart docker
sudo systemctl enable docker
```

---

## Step 2: Install Kubernetes (Minikube)

### Option A: Minikube (Development/Learning)

```bash
# Download Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# Start Minikube with production-like settings
minikube start \
    --cpus=6 \
    --memory=16384 \
    --disk-size=100g \
    --driver=docker \
    --kubernetes-version=v1.28.0 \
    --container-runtime=containerd \
    --extra-config=apiserver.enable-admission-plugins=PodSecurityPolicy \
    --addons=ingress,metrics-server,storage-provisioner,dashboard

# Enable GPU support (if available)
minikube addons enable nvidia-gpu-device-plugin

# Verify cluster
minikube status
kubectl cluster-info
kubectl get nodes -o wide
```

### Option B: Kind (Kubernetes in Docker)

```bash
# Install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create cluster config
cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
- role: worker
EOF

# Create cluster
kind create cluster --name mlops --config kind-config.yaml

# Verify
kubectl cluster-info --context kind-mlops
```

### Option C: K3s (Lightweight Production)

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -s - \
    --write-kubeconfig-mode 644 \
    --disable traefik \
    --docker

# Copy kubeconfig
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

# Verify
kubectl get nodes
```

---

## Step 3: Install kubectl and Helm

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Add common Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add apache-airflow https://airflow.apache.org
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Verify
kubectl version --client
helm version
```

---

## Step 4: Install Kustomize

```bash
# Install Kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify
kustomize version
```

---

## Step 5: Setup Namespaces

```bash
# Create namespaces for all components
kubectl create namespace airflow
kubectl create namespace mlflow
kubectl create namespace kubeflow
kubectl create namespace feast
kubectl create namespace monitoring
kubectl create namespace logging
kubectl create namespace istio-system
kubectl create namespace cert-manager
kubectl create namespace kserve

# Label namespaces for Istio injection (if using)
kubectl label namespace airflow istio-injection=enabled
kubectl label namespace mlflow istio-injection=enabled
kubectl label namespace kubeflow istio-injection=enabled

# Verify namespaces
kubectl get namespaces
```

---

## Step 6: Deploy Storage Infrastructure

### PostgreSQL (Metadata Store)

Create `infrastructure/postgres/values.yaml`:

```yaml
# PostgreSQL Helm values for MLOps
global:
  postgresql:
    auth:
      postgresPassword: "postgres123"
      database: "mlops"

primary:
  persistence:
    enabled: true
    size: 50Gi
    storageClass: "standard"
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"

  initdb:
    scripts:
      init.sql: |
        CREATE DATABASE airflow;
        CREATE DATABASE mlflow;
        CREATE DATABASE feast;
        CREATE USER airflow WITH PASSWORD 'airflow123';
        CREATE USER mlflow WITH PASSWORD 'mlflow123';
        CREATE USER feast WITH PASSWORD 'feast123';
        GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
        GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
        GRANT ALL PRIVILEGES ON DATABASE feast TO feast;

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring

backup:
  enabled: true
  cronjob:
    schedule: "0 2 * * *"
    storage:
      size: 20Gi
```

```bash
# Deploy PostgreSQL
helm install postgres bitnami/postgresql \
    --namespace mlflow \
    --values infrastructure/postgres/values.yaml \
    --wait

# Verify
kubectl get pods -n mlflow -l app.kubernetes.io/name=postgresql
```

### MinIO (Object Storage)

Create `infrastructure/minio/values.yaml`:

```yaml
# MinIO Helm values
mode: distributed

replicas: 4

persistence:
  enabled: true
  size: 100Gi
  storageClass: "standard"

resources:
  requests:
    memory: 2Gi
    cpu: 500m
  limits:
    memory: 4Gi
    cpu: 2

rootUser: minio
rootPassword: minio123

buckets:
  - name: mlflow
    policy: none
    purge: false
  - name: airflow-logs
    policy: none
    purge: false
  - name: kubeflow-pipelines
    policy: none
    purge: false
  - name: feast
    policy: none
    purge: false
  - name: datasets
    policy: none
    purge: false
  - name: models
    policy: none
    purge: false

users:
  - accessKey: mlflow
    secretKey: mlflow123
    policy: readwrite
  - accessKey: airflow
    secretKey: airflow123
    policy: readwrite
  - accessKey: kubeflow
    secretKey: kubeflow123
    policy: readwrite

ingress:
  enabled: true
  ingressClassName: nginx
  hosts:
    - minio.local

consoleIngress:
  enabled: true
  ingressClassName: nginx
  hosts:
    - minio-console.local

metrics:
  serviceMonitor:
    enabled: true
    namespace: monitoring
```

```bash
# Add MinIO repo
helm repo add minio https://charts.min.io/
helm repo update

# Deploy MinIO
helm install minio minio/minio \
    --namespace mlflow \
    --values infrastructure/minio/values.yaml \
    --wait

# Verify
kubectl get pods -n mlflow -l app=minio
```

### Redis (Caching & Message Broker)

Create `infrastructure/redis/values.yaml`:

```yaml
# Redis Helm values
architecture: replication

auth:
  enabled: true
  password: "redis123"

master:
  persistence:
    enabled: true
    size: 10Gi
  resources:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1

replica:
  replicaCount: 2
  persistence:
    enabled: true
    size: 10Gi
  resources:
    requests:
      memory: 1Gi
      cpu: 500m

sentinel:
  enabled: true

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
```

```bash
# Deploy Redis
helm install redis bitnami/redis \
    --namespace airflow \
    --values infrastructure/redis/values.yaml \
    --wait

# Verify
kubectl get pods -n airflow -l app.kubernetes.io/name=redis
```

---

## Step 7: Install Cert-Manager (TLS Certificates)

```bash
# Install cert-manager
helm install cert-manager jetstack/cert-manager \
    --namespace cert-manager \
    --set installCRDs=true \
    --wait

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: selfsigned-issuer
spec:
  selfSigned: {}
EOF

# Verify
kubectl get clusterissuers
```

---

## Step 8: Install NGINX Ingress Controller

```bash
# Install NGINX Ingress
helm install ingress-nginx ingress-nginx \
    --repo https://kubernetes.github.io/ingress-nginx \
    --namespace ingress-nginx \
    --create-namespace \
    --set controller.metrics.enabled=true \
    --set controller.metrics.serviceMonitor.enabled=true \
    --wait

# For Minikube, enable ingress addon instead
minikube addons enable ingress

# Verify
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx
```

---

## Step 9: Setup Local DNS (Development)

```bash
# Get Ingress IP
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# For Minikube
INGRESS_IP=$(minikube ip)

# Add to /etc/hosts
sudo tee -a /etc/hosts << EOF
${INGRESS_IP} airflow.local
${INGRESS_IP} mlflow.local
${INGRESS_IP} kubeflow.local
${INGRESS_IP} grafana.local
${INGRESS_IP} prometheus.local
${INGRESS_IP} minio.local
${INGRESS_IP} minio-console.local
${INGRESS_IP} feast.local
EOF

# Verify
cat /etc/hosts | grep local
```

---

## Step 10: Install Python Environment

```bash
# Install pyenv for Python version management
curl https://pyenv.run | bash

# Add to ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python
pyenv install 3.10.12
pyenv global 3.10.12

# Create virtual environment
python -m venv ~/mlops-venv
source ~/mlops-venv/bin/activate

# Install base packages
pip install --upgrade pip setuptools wheel

# Install MLOps packages
pip install \
    apache-airflow[celery,kubernetes,postgres,redis]==2.8.0 \
    mlflow==2.9.2 \
    kfp==2.5.0 \
    feast[redis,postgres]==0.35.0 \
    great-expectations==0.18.0 \
    evidently==0.4.0 \
    shap==0.44.0 \
    lime==0.2.0.1 \
    boto3==1.34.0 \
    kubernetes==28.1.0 \
    prometheus-client==0.19.0

# Verify installations
python -c "import airflow; print(f'Airflow: {airflow.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import kfp; print(f'KFP: {kfp.__version__}')"
```

---

## Step 11: GPU Support (Optional)

### Install NVIDIA Drivers

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Install NVIDIA Device Plugin for Kubernetes

```bash
# Deploy NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml

# Verify GPU availability
kubectl get nodes -o=custom-columns='NAME:.metadata.name,GPUs:.status.capacity.nvidia\.com/gpu'
```

---

## Step 12: Create Storage Classes

```bash
# Create StorageClasses for different workloads
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: k8s.io/minikube-hostpath
parameters:
  type: pd-ssd
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: Immediate
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: k8s.io/minikube-hostpath
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: Immediate
EOF

# Verify
kubectl get storageclass
```

---

## Step 13: Create Secrets

```bash
# Create secrets for all services
kubectl create secret generic postgres-secret \
    --namespace mlflow \
    --from-literal=postgres-password=postgres123 \
    --from-literal=airflow-password=airflow123 \
    --from-literal=mlflow-password=mlflow123

kubectl create secret generic minio-secret \
    --namespace mlflow \
    --from-literal=root-user=minio \
    --from-literal=root-password=minio123

kubectl create secret generic redis-secret \
    --namespace airflow \
    --from-literal=redis-password=redis123

# Copy secrets to other namespaces
for ns in airflow kubeflow feast monitoring; do
    kubectl get secret minio-secret -n mlflow -o yaml | \
        sed "s/namespace: mlflow/namespace: $ns/" | \
        kubectl apply -f -
done

# Verify
kubectl get secrets -n mlflow
kubectl get secrets -n airflow
```

---

## Verification Checklist

```bash
#!/bin/bash
# verification_script.sh

echo "=== Infrastructure Verification ==="

echo -e "\n1. Docker Status:"
docker info | grep -E "Server Version|Storage Driver|Cgroup Driver"

echo -e "\n2. Kubernetes Cluster:"
kubectl cluster-info
kubectl get nodes -o wide

echo -e "\n3. Namespaces:"
kubectl get namespaces | grep -E "airflow|mlflow|kubeflow|monitoring|feast"

echo -e "\n4. Storage Classes:"
kubectl get storageclass

echo -e "\n5. PostgreSQL Status:"
kubectl get pods -n mlflow -l app.kubernetes.io/name=postgresql

echo -e "\n6. MinIO Status:"
kubectl get pods -n mlflow -l app=minio

echo -e "\n7. Redis Status:"
kubectl get pods -n airflow -l app.kubernetes.io/name=redis

echo -e "\n8. Ingress Controller:"
kubectl get pods -n ingress-nginx

echo -e "\n9. Cert-Manager:"
kubectl get pods -n cert-manager

echo -e "\n10. Python Environment:"
python --version
pip list | grep -E "airflow|mlflow|kfp|feast"

echo -e "\n=== All checks completed ==="
```

```bash
# Run verification
chmod +x verification_script.sh
./verification_script.sh
```

---

## Project Directory Structure

```bash
# Create project structure
mkdir -p mlops-platform/{
    infrastructure/{postgres,minio,redis,secrets},
    airflow/{dags,plugins,config},
    mlflow/{experiments,models,config},
    kubeflow/{pipelines,components,notebooks},
    feast/{features,config},
    monitoring/{prometheus,grafana,alertmanager},
    cicd/{github-actions,argocd}
}

# Create initial files
touch mlops-platform/.gitignore
touch mlops-platform/.env.example
```

---

## Next Steps

After completing infrastructure setup:

1. **Phase 02**: MLflow Enterprise Setup
2. **Phase 03**: Airflow Enterprise Setup
3. **Phase 04**: Kubeflow Complete Setup
4. **Phase 05**: Feature Store & Data Validation

---

## Troubleshooting

### Common Issues

**Minikube won't start:**
```bash
minikube delete
minikube start --driver=docker --force
```

**Pods stuck in Pending:**
```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

**Storage issues:**
```bash
kubectl get pv,pvc -A
kubectl describe pvc <pvc-name> -n <namespace>
```

**Network issues:**
```bash
kubectl run test --image=busybox --rm -it --restart=Never -- wget -O- http://service-name.namespace.svc.cluster.local
```

---

**Status**: Phase 01 Complete
**Next**: Phase 02 - MLflow Enterprise Setup
