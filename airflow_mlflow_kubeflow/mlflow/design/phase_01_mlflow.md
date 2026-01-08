# Phase 01: MLflow - Experiment Tracking & Model Management

## Overview

MLflow implementation using Docker Compose with PostgreSQL backend and local filesystem artifact storage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLflow Platform                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   MLflow Tracking Server                  │   │
│  │                      (Port 5000)                          │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │  Tracking   │  │   Model     │  │  Artifact   │      │   │
│  │  │    API      │  │  Registry   │  │   Browser   │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              │                               │                   │
│              ▼                               ▼                   │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │     PostgreSQL      │        │   Local Filesystem  │        │
│  │     (Port 5432)     │        │   ./mlartifacts     │        │
│  │                     │        │                     │        │
│  │  • Experiments      │        │  • Models           │        │
│  │  • Runs             │        │  • Datasets         │        │
│  │  • Metrics          │        │  • Plots            │        │
│  │  • Parameters       │        │  • Artifacts        │        │
│  │  • Tags             │        │                     │        │
│  │  • Model Registry   │        │                     │        │
│  └─────────────────────┘        └─────────────────────┘        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   MLflow Model Serving                    │   │
│  │                      (Port 5001)                          │   │
│  │                                                           │   │
│  │  • REST API for model inference                          │   │
│  │  • Loads models from registry                            │   │
│  │  • /invocations endpoint                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Port | Purpose |
|-----------|------|---------|
| MLflow Tracking Server | 5000 | Experiment tracking, model registry UI |
| PostgreSQL | 5432 | Metadata storage |
| MLflow Model Server | 5001 | Model serving (inference) |

---

## Directory Structure

```
mlflow/
├── design/
│   └── phase_01_mlflow.md      # This document
├── docker/
│   ├── docker-compose.yml      # Main compose file
│   ├── Dockerfile.mlflow       # MLflow server image
│   └── .env                    # Environment variables
├── scripts/
│   ├── start.sh                # Start the platform
│   ├── stop.sh                 # Stop the platform
│   ├── serve-model.sh          # Serve a registered model
│   └── health-check.sh         # Check service health
├── examples/
│   ├── 01_basic_tracking.py    # Basic experiment tracking
│   ├── 02_model_registry.py    # Model registration example
│   ├── 03_model_serving.py     # Model serving example
│   └── requirements.txt        # Python dependencies
└── mlartifacts/                # Artifact storage (created at runtime)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MLFLOW_TRACKING_URI | http://localhost:5000 | MLflow server URL |
| POSTGRES_USER | mlflow | Database username |
| POSTGRES_PASSWORD | mlflow123 | Database password |
| POSTGRES_DB | mlflow | Database name |
| MLFLOW_ARTIFACT_ROOT | /mlartifacts | Artifact storage path |

---

## Implementation Steps

### Step 1: Start the Platform

```bash
cd mlflow/docker
docker-compose up -d
```

### Step 2: Verify Services

```bash
# Check MLflow UI
curl http://localhost:5000/health

# Check PostgreSQL
docker exec mlflow-postgres pg_isready -U mlflow
```

### Step 3: Access MLflow UI

Open browser: http://localhost:5000

### Step 4: Run Example Experiments

```bash
cd ../examples
pip install -r requirements.txt
python 01_basic_tracking.py
```

### Step 5: Register and Serve a Model

```bash
# After running training example
./scripts/serve-model.sh <model-name> <version>

# Test inference
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[1, 2, 3, 4]]}'
```

---

## MLflow Concepts

### Experiments
- Container for runs
- Organize by project/task
- Default experiment ID: 0

### Runs
- Single execution of ML code
- Logs parameters, metrics, artifacts
- Has unique run_id

### Model Registry
- Central model store
- Version control for models
- Stage transitions: None → Staging → Production → Archived

### Artifacts
- Files produced by runs
- Models, plots, data samples
- Stored in artifact root

---

## API Reference

### Tracking API

```python
import mlflow

# Set tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create/set experiment
mlflow.set_experiment("my-experiment")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)

    # Log artifacts
    mlflow.log_artifact("model.pkl")

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry API

```python
# Register model
mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="my-model"
)

# Transition stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="my-model",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/my-model/Production"
)
```

### REST API

```bash
# List experiments
curl http://localhost:5000/api/2.0/mlflow/experiments/search

# Get run details
curl http://localhost:5000/api/2.0/mlflow/runs/get?run_id=<run_id>

# Search runs
curl -X POST http://localhost:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["0"]}'
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot connect to tracking server | Check if container is running: `docker ps` |
| Database connection error | Verify PostgreSQL is healthy: `docker logs mlflow-postgres` |
| Artifact upload fails | Check volume mount permissions |
| Model serving fails | Ensure model is registered and version exists |

### Debug Commands

```bash
# View logs
docker-compose logs -f mlflow-server
docker-compose logs -f postgres

# Enter container
docker exec -it mlflow-server /bin/bash

# Check database
docker exec -it mlflow-postgres psql -U mlflow -d mlflow -c "\dt"

# Reset everything
docker-compose down -v
docker-compose up -d
```

---

## Next Steps

After completing MLflow setup:
1. Proceed to Phase 02: Apache Airflow
2. Integrate Airflow DAGs with MLflow tracking
3. Automate model training pipelines
