# Prefect Pipeline Approach

## Overview

Use **Prefect Cloud (free tier)** for orchestration with a **local worker** that triggers jobs in k3s. This gives you a nice UI and scheduling without adding infrastructure to your cluster.

```
Prefect Cloud (SaaS)     Local Worker          K3s Cluster
┌─────────────────┐     ┌─────────────┐       ┌─────────────────┐
│  - UI Dashboard │     │  - Runs on  │       │  - Kubeflow     │
│  - Scheduling   │────▶│    your     │──────▶│  - MLflow       │
│  - Monitoring   │     │    machine  │       │  - Feast        │
│  - Logs         │     │  - Executes │       │  - KServe       │
└─────────────────┘     │    flows    │       │  - MinIO        │
                        └─────────────┘       └─────────────────┘
```

---

## Why Prefect Cloud + Local Worker?

| Benefit | Description |
|---------|-------------|
| Zero k3s overhead | No Prefect server in cluster |
| Free tier | 3 users, unlimited flows, 10k task runs/month |
| Nice UI | Dashboard, run history, logs, scheduling |
| Easy setup | `uv add prefect` + API key |
| Python-native | Just decorators, no DAG files |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR MACHINE (Local)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Prefect Worker                         │  │
│  │                                                           │  │
│  │  @flow churn_prediction_pipeline()                        │  │
│  │    ├── @task extract_data()      → PostgreSQL (k3s)       │  │
│  │    ├── @task version_data()      → DVC + MinIO (k3s)      │  │
│  │    ├── @task validate_data()     → Great Expectations     │  │
│  │    ├── @task engineer_features() → Feast (k3s)            │  │
│  │    ├── @task train_model()       → Kubeflow (k3s)         │  │
│  │    └── @task deploy_model()      → KServe (k3s)           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    kubeconfig (~/.kube/config)                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    K3s Cluster      │
                    │  ┌───────────────┐  │
                    │  │ PostgreSQL    │  │
                    │  │ MinIO         │  │
                    │  │ Kubeflow      │  │
                    │  │ MLflow        │  │
                    │  │ Feast         │  │
                    │  │ KServe        │  │
                    │  └───────────────┘  │
                    └─────────────────────┘
```

---

## Setup Steps

### 1. Create Prefect Cloud Account
```bash
# Visit https://app.prefect.cloud
# Sign up (free tier)
# Create a workspace
```

### 2. Install via uv
```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv sync
```

### 3. Authenticate
```bash
# Get API key from Prefect Cloud UI → Settings → API Keys
uv run prefect cloud login --key YOUR_API_KEY
```

### 4. Create Work Pool
```bash
uv run prefect work-pool create local-pool --type process
```

### 5. Start Worker
```bash
# Run in a terminal (or as background service)
uv run prefect worker start --pool local-pool
```

---

## Directory Structure (Flat - Per Use Case)

```
/home/sujith/github/rag/00_MLOps/04_usecases/
│
├── pyproject.toml                    # uv project config
├── uv.lock                           # uv lockfile
├── config.yaml                       # Shared configuration
│
├── common/                           # Shared utilities
│   ├── __init__.py
│   ├── config.py                     # Load config.yaml
│   └── logger.py                     # Logging setup
│
├── uc_1_churn_prediction/            # Use Case 1: Churn Prediction
│   ├── __init__.py
│   ├── flows/
│   │   ├── __init__.py
│   │   └── churn_pipeline.py         # Main Prefect flow
│   └── tasks/
│       ├── __init__.py
│       ├── data_ingestion.py         # PostgreSQL → MinIO
│       ├── data_versioning.py        # DVC operations
│       ├── data_validation.py        # Great Expectations
│       ├── feature_engineering.py    # Feast
│       ├── model_training.py         # Kubeflow trigger
│       └── model_serving.py          # KServe deployment
│
├── uc_2_ab_testing/                  # Use Case 2: A/B Testing
│   ├── __init__.py
│   ├── flows/
│   │   └── ab_testing_pipeline.py
│   └── tasks/
│       ├── deploy_canary.py
│       ├── create_experiment.py
│       └── monitor_experiment.py
│
├── uc_3_drift_retraining/            # Use Case 3: Drift Detection
│   ├── __init__.py
│   ├── flows/
│   │   └── drift_pipeline.py
│   └── tasks/
│       ├── drift_detection.py
│       └── trigger_retrain.py
│
├── uc_4_data_lineage/                # Use Case 4: Lineage Tracking
│   ├── __init__.py
│   ├── flows/
│   │   └── lineage_pipeline.py
│   └── tasks/
│       ├── emit_lineage.py
│       └── query_lineage.py
│
├── uc_5_fairness_docs/               # Use Case 5: Fairness & Docs
│   ├── __init__.py
│   ├── flows/
│   │   └── fairness_pipeline.py
│   └── tasks/
│       ├── fairness_analysis.py
│       └── model_card.py
│
└── uc_6_gitops_deployment/           # Use Case 6: GitOps
    ├── __init__.py
    ├── flows/
    │   └── gitops_pipeline.py
    └── tasks/
        ├── promote_model.py
        └── sync_argocd.py
```

---

## pyproject.toml

```toml
[project]
name = "mlops-pipelines"
version = "0.1.0"
description = "MLOps Prefect Pipelines"
requires-python = ">=3.11"
dependencies = [
    "prefect>=2.14.0",
    "pandas>=2.0.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "boto3>=1.28.0",
    "pyarrow>=14.0.0",
    "great-expectations>=0.18.0",
    "feast>=0.35.0",
    "kfp>=2.4.0",
    "mlflow>=2.9.0",
    "kubernetes>=28.0.0",
    "evidently>=0.4.0",
    "pyyaml>=6.0.0",
]
```

---

## config.yaml

```yaml
environment: development

postgres:
  host: postgresql.postgresql.svc.cluster.local
  port: 5432
  database: customers
  user: postgres
  password: postgres123

minio:
  endpoint: http://minio.minio.svc.cluster.local:9000
  access_key: minioadmin
  secret_key: minioadmin123
  bucket: dvc-storage

mlflow:
  tracking_uri: http://mlflow.mlflow.svc.cluster.local:5000

kubeflow:
  host: http://localhost:8080  # via port-forward

feast:
  repo_path: ./feature_repo

uc1:
  raw_data_path: raw/customers.parquet
  model_name: churn-predictor
  namespace: models
```

---

## Code Examples

### common/config.py
```python
import yaml
from pathlib import Path

_config = None

def get_config():
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    return _config
```

### uc_1_churn_prediction/tasks/data_ingestion.py
```python
from prefect import task
import pandas as pd
from sqlalchemy import create_engine
import boto3
from common.config import get_config

@task(name="Extract Customer Data", retries=2, retry_delay_seconds=60)
def extract_from_postgres() -> str:
    """Extract customer data from PostgreSQL"""
    cfg = get_config()

    db_url = f"postgresql://{cfg['postgres']['user']}:{cfg['postgres']['password']}@" \
             f"{cfg['postgres']['host']}:{cfg['postgres']['port']}/{cfg['postgres']['database']}"

    engine = create_engine(db_url)

    query = """
    SELECT customer_id, age, gender, tenure_months, total_purchases,
           avg_order_value, days_since_last_purchase, support_tickets_count, churn
    FROM customers
    """
    df = pd.read_sql(query, engine)

    output_path = "/tmp/customers.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Extracted {len(df)} customer records")
    return output_path

@task(name="Upload to MinIO", retries=2)
def upload_to_minio(local_path: str) -> str:
    """Upload file to MinIO"""
    cfg = get_config()

    s3 = boto3.client(
        's3',
        endpoint_url=cfg['minio']['endpoint'],
        aws_access_key_id=cfg['minio']['access_key'],
        aws_secret_access_key=cfg['minio']['secret_key']
    )

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"raw/customers_{timestamp}.parquet"

    s3.upload_file(local_path, cfg['minio']['bucket'], s3_key)
    print(f"Uploaded to s3://{cfg['minio']['bucket']}/{s3_key}")

    return s3_key
```

### uc_1_churn_prediction/flows/churn_pipeline.py
```python
from prefect import flow, get_run_logger
from uc_1_churn_prediction.tasks.data_ingestion import extract_from_postgres, upload_to_minio
from uc_1_churn_prediction.tasks.data_versioning import version_with_dvc
from uc_1_churn_prediction.tasks.data_validation import validate_with_great_expectations
from uc_1_churn_prediction.tasks.feature_engineering import materialize_feast_features
from uc_1_churn_prediction.tasks.model_training import trigger_kubeflow_training
from uc_1_churn_prediction.tasks.model_serving import deploy_to_kserve

@flow(name="UC1: Churn Prediction Pipeline", log_prints=True)
def churn_prediction_pipeline(skip_training: bool = False):
    """
    End-to-end churn prediction pipeline.

    Steps:
    1. Extract data from PostgreSQL
    2. Upload to MinIO
    3. Version with DVC
    4. Validate with Great Expectations
    5. Materialize features with Feast
    6. Train model with Kubeflow
    7. Deploy with KServe
    """
    logger = get_run_logger()

    # Step 1: Extract
    logger.info("Step 1: Extracting data from PostgreSQL")
    local_path = extract_from_postgres()

    # Step 2: Upload to MinIO
    logger.info("Step 2: Uploading to MinIO")
    s3_path = upload_to_minio(local_path)

    # Step 3: Version with DVC
    logger.info("Step 3: Versioning with DVC")
    version_with_dvc(local_path)

    # Step 4: Validate
    logger.info("Step 4: Validating with Great Expectations")
    validation_result = validate_with_great_expectations(local_path)

    if not validation_result:
        raise ValueError("Data validation failed! Pipeline aborted.")

    # Step 5: Feature engineering
    logger.info("Step 5: Materializing Feast features")
    materialize_feast_features()

    if not skip_training:
        # Step 6: Train
        logger.info("Step 6: Triggering Kubeflow training")
        run_id = trigger_kubeflow_training(s3_path)

        # Step 7: Deploy
        # logger.info("Step 7: Deploying to KServe")
        # endpoint = deploy_to_kserve(model_uri)

    logger.info("Pipeline completed successfully!")
    return {"s3_path": s3_path, "validation": validation_result}


if __name__ == "__main__":
    churn_prediction_pipeline()
```

---

## Running Flows

### Local Execution (Testing)
```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv run python -m uc_1_churn_prediction.flows.churn_pipeline
```

### Via Prefect Cloud UI
1. Go to https://app.prefect.cloud
2. Navigate to Deployments
3. Click "Run" on your flow
4. Monitor progress in real-time

### Deploy with Schedule
```bash
# In the flow file, add:
if __name__ == "__main__":
    churn_prediction_pipeline.serve(
        name="churn-prediction-daily",
        cron="0 6 * * *",  # 6 AM daily
        tags=["uc1", "churn"]
    )
```

### CLI Trigger
```bash
uv run prefect deployment run "UC1: Churn Prediction Pipeline/churn-prediction-daily"
```

---

## Prefect Cloud Free Tier Limits

| Resource | Limit |
|----------|-------|
| Users | 3 |
| Workspaces | 1 |
| Task runs | 10,000/month |
| Flow runs | Unlimited |
| Retention | 7 days |

---

## Quick Start Commands

```bash
# 1. Initialize project (if not done)
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv init --name mlops-pipelines

# 2. Install dependencies
uv sync

# 3. Login to Prefect Cloud
uv run prefect cloud login

# 4. Create work pool
uv run prefect work-pool create local-pool --type process

# 5. Start worker (keep running in one terminal)
uv run prefect worker start --pool local-pool

# 6. Run a flow (in another terminal)
uv run python -m uc_1_churn_prediction.flows.churn_pipeline
```

---

## Comparison: Manual Python vs Prefect

| Feature | Manual Python | Prefect |
|---------|---------------|---------|
| UI Dashboard | None | Yes (Cloud) |
| Scheduling | Cron/manual | Built-in |
| Retries | Manual code | `@task(retries=3)` |
| Logging | Print/logging | Structured + UI |
| Run history | None | Full history |
| Monitoring | Manual | Built-in |
| Setup effort | Zero | ~10 minutes |
| Package manager | pip | uv |

---

## Migration Path

```
Manual Python          →    Prefect Cloud + uv    →    Prefect in K3s / Airflow
(POC learning)              (POC with UI)              (Production)
```
