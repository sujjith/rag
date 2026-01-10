# Phase 02: Kubeflow - ML Platform on Kubernetes

## Overview

Kubeflow v1.11.0 (Latest Stable) implementation on Minikube with full component stack for ML workflows.

### Version: 1.11.0 Features
- **Full KFP v2 Support**: Native Kubeflow Pipelines v2 with improved artifact handling
- **Enhanced Model Registry**: Better model versioning and deployment workflows
- **Security Patches**: Latest CVE fixes and security improvements
- **Improved Katib**: Enhanced hyperparameter tuning algorithms
- **Updated Training Operators**: Better distributed training support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUBEFLOW ON MINIKUBE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        KUBEFLOW CENTRAL DASHBOARD                   │     │
│  │                          (Port 8080)                                │     │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │     │
│  │   │Pipelines│  │Notebooks│  │  Katib  │  │ Volumes │              │     │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│          ┌─────────────────────────┼─────────────────────────┐              │
│          │                         │                         │               │
│          ▼                         ▼                         ▼               │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   KF PIPELINES   │    │   KF NOTEBOOKS   │    │      KATIB       │      │
│  │                  │    │                  │    │                  │      │
│  │ • Pipeline UI    │    │ • Jupyter Server │    │ • HP Tuning      │      │
│  │ • Argo Workflows │    │ • VSCode Server  │    │ • NAS            │      │
│  │ • Artifact Store │    │ • Custom Images  │    │ • Early Stopping │      │
│  │ • Metadata Store │    │                  │    │                  │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│          │                                                │                  │
│          │                                                │                  │
│          ▼                                                ▼                  │
│  ┌──────────────────┐                           ┌──────────────────┐        │
│  │ TRAINING OPERATORS│                           │   EXPERIMENTS    │        │
│  │                  │                           │                  │        │
│  │ • TFJob          │                           │ • Trials         │        │
│  │ • PyTorchJob     │                           │ • Suggestions    │        │
│  │ • MPIJob         │                           │ • Metrics        │        │
│  │ • XGBoostJob     │                           │                  │        │
│  └──────────────────┘                           └──────────────────┘        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        KUBERNETES CLUSTER                           │     │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │     │
│  │   │  Istio  │  │  Dex    │  │  MinIO  │  │  MySQL  │              │     │
│  │   │ Service │  │  OIDC   │  │ Storage │  │Metadata │              │     │
│  │   │  Mesh   │  │  Auth   │  │         │  │         │              │     │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Purpose | Port |
|-----------|---------|------|
| Central Dashboard | Unified UI for all Kubeflow components | 8080 |
| Pipelines | ML workflow orchestration | - |
| Notebooks | Jupyter/VSCode development environment | - |
| Katib | Hyperparameter tuning & NAS | - |
| Training Operators | Distributed training (TF, PyTorch) | - |
| Istio | Service mesh, traffic management | - |
| Dex | Authentication (OIDC) | - |
| MinIO | Object storage for artifacts | 9000 |
| MySQL | Metadata storage | 3306 |

---

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 40 GB | 100 GB |
| Minikube | v1.30+ | Latest |
| Kubernetes | v1.25+ | v1.28 |

---

## Directory Structure

```
kubeflow/
├── design/
│   ├── phase_02_kubeflow.md     # This document
│   └── learning_guide.md        # Phased learning guide
├── scripts/
│   ├── 01_install_prerequisites.sh
│   ├── 02_setup_minikube.sh
│   ├── 03_install_kubeflow.sh
│   ├── 04_port_forward.sh
│   ├── 05_cleanup.sh
│   └── utils.sh
├── manifests/
│   ├── namespace.yaml
│   ├── minio-pvc.yaml
│   └── notebook-server.yaml
└── examples/
    ├── phase1/                  # Setup & Basic Concepts
    ├── phase2/                  # Kubeflow Pipelines
    ├── phase3/                  # Notebooks
    ├── phase4/                  # Katib (HP Tuning)
    ├── phase5/                  # Training Operators
    └── phase6/                  # MLflow Integration
```

---

## Learning Phases

| Phase | Topic | Key Concepts |
|-------|-------|--------------|
| 1 | Setup & Basics | Minikube, Kubeflow install, Dashboard |
| 2 | Pipelines | Components, DSL, Compilation, Runs |
| 3 | Notebooks | Jupyter servers, Custom images |
| 4 | Katib | HP tuning, Experiments, Trials |
| 5 | Training | TFJob, PyTorchJob, Distributed |
| 6 | Integration | MLflow tracking, End-to-end |

---

## Quick Start

```bash
# 1. Install prerequisites
./scripts/01_install_prerequisites.sh

# 2. Setup Minikube
./scripts/02_setup_minikube.sh

# 3. Install Kubeflow
./scripts/03_install_kubeflow.sh

# 4. Access Dashboard
./scripts/04_port_forward.sh

# Open: http://localhost:8080
# Default credentials: user@example.com / 12341234
```

---

## Access URLs (after port-forward)

| Service | URL | Credentials |
|---------|-----|-------------|
| Kubeflow Dashboard | http://localhost:8080 | user@example.com / 12341234 |
| MinIO Console | http://localhost:9001 | minio / minio123 |
| Pipelines API | http://localhost:8888 | - |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pods stuck in Pending | Increase Minikube resources |
| Dashboard not loading | Check Istio gateway: `kubectl get gateway -A` |
| Pipeline fails | Check Argo logs: `kubectl logs -n kubeflow -l app=workflow-controller` |
| OOM errors | Increase Minikube memory: `minikube config set memory 16384` |
