# Phase 03: Apache Airflow - Workflow Orchestration

## Overview

Apache Airflow v3.1.5 (Latest Stable) implementation on Kubernetes using Helm with CeleryExecutor and PostgreSQL.

### Version: 3.1.5 Features
- **Task Execution API & Task SDK (AIP-72)**: New task execution model
- **DAG Versioning (AIP-66)**: Track DAG versions over time
- **React UI Rewrite**: Modern, faster web interface
- **Human-in-the-Loop (HITL)**: Workflows can pause for human decisions
- **Asset-Based Scheduling**: Improved data-aware scheduling
- **Edge Executor**: Run tasks at the edge

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AIRFLOW ON KUBERNETES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        AIRFLOW WEBSERVER                            │     │
│  │                          (Port 8080)                                │     │
│  │                                                                     │     │
│  │   • DAG Visualization    • Task Logs      • Trigger DAGs           │     │
│  │   • Run History          • Connections    • Variables              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        AIRFLOW SCHEDULER                            │     │
│  │                                                                     │     │
│  │   • DAG Parsing          • Task Scheduling  • Executor Interface   │     │
│  │   • Dependency Check     • SLA Monitoring   • DAG Runs Creation    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    │                               │                         │
│                    ▼                               ▼                         │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐      │
│  │         REDIS            │    │       CELERY WORKERS             │      │
│  │    (Message Broker)      │    │                                  │      │
│  │                          │    │  ┌────────┐  ┌────────┐         │      │
│  │   • Task Queue           │    │  │Worker 1│  │Worker 2│  ...    │      │
│  │   • Result Backend       │    │  └────────┘  └────────┘         │      │
│  │                          │    │                                  │      │
│  └──────────────────────────┘    │   • Execute Tasks               │      │
│                                   │   • Log Results                 │      │
│                                   │   • Return Status               │      │
│                                   └──────────────────────────────────┘      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          TRIGGERER                                  │     │
│  │                                                                     │     │
│  │   • Async Sensors        • Deferrable Operators                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐      │
│  │       POSTGRESQL         │    │           FLOWER                  │      │
│  │     (Metadata DB)        │    │    (Celery Monitoring)           │      │
│  │                          │    │                                  │      │
│  │   • DAG Runs             │    │   • Worker Status                │      │
│  │   • Task Instances       │    │   • Task Progress                │      │
│  │   • Connections          │    │   • Queue Length                 │      │
│  │   • Variables            │    │                                  │      │
│  └──────────────────────────┘    └──────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Replicas | Purpose | Port |
|-----------|----------|---------|------|
| Webserver | 1 | UI and REST API | 8080 |
| Scheduler | 1 | DAG parsing and scheduling | - |
| Worker | 2 | Task execution (Celery) | - |
| Triggerer | 1 | Async sensor handling | - |
| Flower | 1 | Celery monitoring | 5555 |
| PostgreSQL | 1 | Metadata database | 5432 |
| Redis | 1 | Message broker | 6379 |

---

## Directory Structure

```
airflow/
├── design/
│   ├── phase_03_airflow.md      # This document
│   └── learning_guide.md        # Phased learning guide
├── scripts/
│   ├── 01_install_prerequisites.sh
│   ├── 02_setup_namespace.sh
│   ├── 03_install_airflow.sh
│   ├── 04_port_forward.sh
│   ├── 05_sync_dags.sh
│   └── 06_cleanup.sh
├── helm/
│   ├── values.yaml              # Helm values
│   └── values-dev.yaml          # Development overrides
├── dags/
│   ├── phase1/                  # Basic DAGs
│   ├── phase2/                  # DAG Fundamentals
│   ├── phase3/                  # Operators & Sensors
│   ├── phase4/                  # XCom & Dependencies
│   ├── phase5/                  # Advanced Features
│   └── phase6/                  # Integration
└── examples/
    └── requirements.txt
```

---

## Learning Phases

| Phase | Topic | Key Concepts |
|-------|-------|--------------|
| 1 | Setup & Basics | Installation, UI, CLI |
| 2 | DAG Fundamentals | DAG structure, scheduling, catchup |
| 3 | Operators & Sensors | BashOperator, PythonOperator, Sensors |
| 4 | XCom & Dependencies | Data passing, branching, triggers |
| 5 | Advanced | Pools, Connections, Variables, Hooks |
| 6 | Integration | MLflow, Kubeflow, external systems |

---

## Quick Start

```bash
# 1. Install prerequisites
./scripts/01_install_prerequisites.sh

# 2. Create namespace
./scripts/02_setup_namespace.sh

# 3. Install Airflow via Helm
./scripts/03_install_airflow.sh

# 4. Access UI
./scripts/04_port_forward.sh

# Open: http://localhost:8080
# Credentials: admin / admin
```

---

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| Flower | http://localhost:5555 | - |

---

## Key Concepts

### DAG (Directed Acyclic Graph)
- Collection of tasks with dependencies
- Defined in Python files
- Scheduled execution

### Operators
- Define what a task does
- BashOperator, PythonOperator, etc.
- Custom operators possible

### Sensors
- Wait for external conditions
- FileSensor, HttpSensor, etc.
- Can be deferrable (async)

### XCom
- Cross-communication between tasks
- Push/pull data
- Limited size (use external storage for large data)

### Executor
- CeleryExecutor: Distributed, scalable
- LocalExecutor: Single machine
- KubernetesExecutor: Pod per task

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| DAG not appearing | Check syntax, wait for scheduler parse |
| Task stuck in queued | Check worker logs, Redis connection |
| Import errors | Check Python path, dependencies |
| Database errors | Check PostgreSQL connection |

### Debug Commands

```bash
# Check pods
kubectl get pods -n airflow

# Scheduler logs
kubectl logs -n airflow -l component=scheduler -f

# Worker logs
kubectl logs -n airflow -l component=worker -f

# List DAGs
kubectl exec -n airflow deployment/airflow-webserver -- airflow dags list

# Trigger DAG
kubectl exec -n airflow deployment/airflow-webserver -- airflow dags trigger <dag_id>
```
