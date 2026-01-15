# UC1: Customer Churn Prediction - Implementation Guide

Step-by-step guide to implement an end-to-end ML pipeline using:
**Prefect → DVC → Great Expectations → Feast → Kubeflow → MLflow → KServe**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR LOCAL MACHINE                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Prefect Worker (orchestrates all steps)            │  │
│  │                                                                       │  │
│  │   Step 1: extract_from_postgres()    → PostgreSQL (k3s)               │  │
│  │   Step 2: upload_to_minio()          → MinIO (k3s)                    │  │
│  │   Step 3: version_with_dvc()         → DVC + Git                      │  │
│  │   Step 4: validate_with_ge()         → Great Expectations             │  │
│  │   Step 5: materialize_features()     → Feast (k3s)                    │  │
│  │   Step 6: trigger_kubeflow()         → Kubeflow (k3s) + MLflow        │  │
│  │   Step 7: deploy_to_kserve()         → KServe (k3s)                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ (via kubeconfig)
┌─────────────────────────────────────────────────────────────────────────────┐
│                              K3s CLUSTER                                    │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │ PostgreSQL  │  │    MinIO    │  │    Feast    │  │  Kubeflow   │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └──────┬──────┘       │
│                                                              │              │
│   ┌─────────────┐                   ┌─────────────┐  ┌──────▼──────┐       │
│   │   MLflow    │◄──────────────────│   KServe    │◄─│   Model     │       │
│   └─────────────┘                   └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Services in K3s
- PostgreSQL (source database)
- MinIO (object storage)
- Feast (feature store)
- Kubeflow Pipelines (ML training)
- MLflow (experiment tracking)
- KServe (model serving)

### Local Requirements
- Python 3.11+
- uv (Python package manager)
- kubectl configured for k3s
- Git

---

## Step 1: Project Setup

### 1.1 Create Project Directory

```bash
mkdir -p /home/sujith/github/rag/00_MLOps/04_usecases
cd /home/sujith/github/rag/00_MLOps/04_usecases
```

### 1.2 Initialize uv Project

```bash
uv init --name mlops-pipelines
```

### 1.3 Create pyproject.toml

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
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0.0",
]
```

### 1.4 Install Dependencies

```bash
uv sync
```

### 1.5 Create Directory Structure

```bash
mkdir -p common
mkdir -p uc_1_churn_prediction/{flows,tasks}
mkdir -p feature_repo
mkdir -p pipelines

# Create __init__.py files
touch common/__init__.py
touch uc_1_churn_prediction/__init__.py
touch uc_1_churn_prediction/flows/__init__.py
touch uc_1_churn_prediction/tasks/__init__.py
```

**Final Structure:**
```
04_usecases/
├── pyproject.toml
├── uv.lock
├── config.yaml
├── common/
│   ├── __init__.py
│   └── config.py
├── uc_1_churn_prediction/
│   ├── __init__.py
│   ├── flows/
│   │   ├── __init__.py
│   │   └── churn_pipeline.py
│   └── tasks/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_versioning.py
│       ├── data_validation.py
│       ├── feature_engineering.py
│       ├── model_training.py
│       └── model_serving.py
├── feature_repo/
│   ├── feature_store.yaml
│   └── features.py
└── pipelines/
    └── churn_training.yaml
```

---

## Step 2: Configuration

### 2.1 Create config.yaml

```yaml
# config.yaml
environment: development

postgres:
  host: localhost          # Use port-forward: kubectl port-forward svc/postgresql 5432:5432 -n postgresql
  port: 5432
  database: customers
  user: postgres
  password: postgres123

minio:
  endpoint: http://localhost:9000    # kubectl port-forward svc/minio 9000:9000 -n minio
  access_key: minioadmin
  secret_key: minioadmin123
  bucket: dvc-storage

mlflow:
  tracking_uri: http://localhost:5000    # kubectl port-forward svc/mlflow 5000:5000 -n mlflow

kubeflow:
  host: http://localhost:8080    # kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow

feast:
  repo_path: ./feature_repo

kserve:
  namespace: models

dvc:
  remote_name: minio
  remote_url: s3://dvc-storage
```

### 2.2 Create common/config.py

```python
# common/config.py
import yaml
from pathlib import Path

_config = None

def get_config() -> dict:
    """Load configuration from config.yaml"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    return _config
```

---

## Step 3: Setup Prefect Cloud

### 3.1 Create Prefect Cloud Account

1. Visit https://app.prefect.cloud
2. Sign up (free tier: 3 users, 10k task runs/month)
3. Create a workspace

### 3.2 Authenticate

```bash
# Get API key from Prefect Cloud UI → Settings → API Keys
uv run prefect cloud login --key YOUR_API_KEY
```

### 3.3 Create Work Pool

```bash
uv run prefect work-pool create local-pool --type process
```

### 3.4 Start Worker (Terminal 1)

```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv run prefect worker start --pool local-pool
```

Keep this running in a separate terminal.

---

## Step 4: Implement Data Ingestion (PostgreSQL → MinIO)

### 4.1 Create uc_1_churn_prediction/tasks/data_ingestion.py

```python
# uc_1_churn_prediction/tasks/data_ingestion.py
from prefect import task
import pandas as pd
from sqlalchemy import create_engine
import boto3
from datetime import datetime
from common.config import get_config

@task(name="Extract Customer Data", retries=2, retry_delay_seconds=60)
def extract_from_postgres() -> str:
    """
    Step 1: Extract customer data from PostgreSQL

    Returns:
        str: Local path to the extracted parquet file
    """
    cfg = get_config()

    db_url = (
        f"postgresql://{cfg['postgres']['user']}:{cfg['postgres']['password']}@"
        f"{cfg['postgres']['host']}:{cfg['postgres']['port']}/{cfg['postgres']['database']}"
    )
    engine = create_engine(db_url)

    query = """
    SELECT
        customer_id,
        age,
        gender,
        tenure_months,
        total_purchases,
        avg_order_value,
        days_since_last_purchase,
        support_tickets_count,
        churn
    FROM customers
    """

    df = pd.read_sql(query, engine)

    output_path = "/tmp/customers.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Extracted {len(df)} customer records to {output_path}")
    return output_path


@task(name="Upload to MinIO", retries=2)
def upload_to_minio(local_path: str) -> str:
    """
    Step 2: Upload extracted data to MinIO object storage

    Args:
        local_path: Path to local parquet file

    Returns:
        str: S3 key of uploaded file
    """
    cfg = get_config()

    s3 = boto3.client(
        's3',
        endpoint_url=cfg['minio']['endpoint'],
        aws_access_key_id=cfg['minio']['access_key'],
        aws_secret_access_key=cfg['minio']['secret_key']
    )

    # Create bucket if not exists
    try:
        s3.head_bucket(Bucket=cfg['minio']['bucket'])
    except:
        s3.create_bucket(Bucket=cfg['minio']['bucket'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"raw/customers_{timestamp}.parquet"

    s3.upload_file(local_path, cfg['minio']['bucket'], s3_key)

    s3_uri = f"s3://{cfg['minio']['bucket']}/{s3_key}"
    print(f"Uploaded to {s3_uri}")
    return s3_key
```

---

## Step 5: Implement Data Versioning (DVC)

### 5.1 Initialize DVC (One-time Setup)

```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases

# Initialize DVC
uv run dvc init

# Configure MinIO as remote
uv run dvc remote add -d minio s3://dvc-storage
uv run dvc remote modify minio endpointurl http://localhost:9000
uv run dvc remote modify minio access_key_id minioadmin
uv run dvc remote modify minio secret_access_key minioadmin123
```

### 5.2 Create uc_1_churn_prediction/tasks/data_versioning.py

```python
# uc_1_churn_prediction/tasks/data_versioning.py
from prefect import task
import subprocess
import shutil
from pathlib import Path

@task(name="Version with DVC")
def version_with_dvc(file_path: str) -> str:
    """
    Step 3: Version dataset with DVC and push to MinIO

    Args:
        file_path: Path to file to version

    Returns:
        str: Path to .dvc file
    """
    # Copy to data directory for versioning
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dest_path = data_dir / Path(file_path).name
    shutil.copy(file_path, dest_path)

    # Add file to DVC
    subprocess.run(["dvc", "add", str(dest_path)], check=True)

    # Push to remote (MinIO)
    subprocess.run(["dvc", "push"], check=True)

    # Git commit the .dvc file
    dvc_file = f"{dest_path}.dvc"
    subprocess.run(["git", "add", dvc_file, ".gitignore"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add dataset version: {Path(file_path).name}"],
        check=True
    )

    print(f"Versioned: {file_path} → {dvc_file}")
    return dvc_file
```

---

## Step 6: Implement Data Validation (Great Expectations)

### 6.1 Initialize Great Expectations (One-time Setup)

```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv run great_expectations init
```

### 6.2 Create uc_1_churn_prediction/tasks/data_validation.py

```python
# uc_1_churn_prediction/tasks/data_validation.py
from prefect import task
import great_expectations as gx
import pandas as pd

@task(name="Validate with Great Expectations")
def validate_with_great_expectations(file_path: str) -> bool:
    """
    Step 4: Validate data quality with Great Expectations

    Args:
        file_path: Path to parquet file to validate

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails
    """
    df = pd.read_parquet(file_path)

    # Get GX context
    context = gx.get_context()

    # Create validator from dataframe
    validator = context.sources.pandas_default.read_dataframe(df)

    # Define expectations
    validator.expect_column_to_exist("customer_id")
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_unique("customer_id")

    validator.expect_column_to_exist("age")
    validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)

    validator.expect_column_to_exist("churn")
    validator.expect_column_values_to_be_in_set("churn", [0, 1])

    validator.expect_column_values_to_be_between("tenure_months", min_value=0)
    validator.expect_column_values_to_be_between("total_purchases", min_value=0)
    validator.expect_column_values_to_be_between("avg_order_value", min_value=0)

    # Run validation
    results = validator.validate()

    if not results.success:
        failed = [r for r in results.results if not r.success]
        error_msg = f"Data validation failed! {len(failed)} expectations failed."
        print(error_msg)
        for f in failed:
            print(f"  - {f.expectation_config.expectation_type}")
        raise ValueError(error_msg)

    print(f"Data validation passed! {len(results.results)} expectations checked.")
    return True
```

---

## Step 7: Implement Feature Engineering (Feast)

### 7.1 Create feature_repo/feature_store.yaml

```yaml
# feature_repo/feature_store.yaml
project: churn_prediction
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
```

### 7.2 Create feature_repo/features.py

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, Field
from feast.types import Float32, Int32, String

# Define entity
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Customer identifier"
)

# Define data source
customer_source = FileSource(
    path="data/customers.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="tenure_months", dtype=Int32),
        Field(name="total_purchases", dtype=Int32),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int32),
        Field(name="support_tickets_count", dtype=Int32),
    ],
    source=customer_source,
)
```

### 7.3 Create uc_1_churn_prediction/tasks/feature_engineering.py

```python
# uc_1_churn_prediction/tasks/feature_engineering.py
from prefect import task
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

@task(name="Prepare Feast Data")
def prepare_feast_data(file_path: str) -> str:
    """
    Prepare data for Feast by adding event_timestamp

    Args:
        file_path: Path to raw parquet file

    Returns:
        str: Path to prepared parquet file
    """
    df = pd.read_parquet(file_path)

    # Add event_timestamp for Feast
    df['event_timestamp'] = pd.Timestamp.now()

    output_path = "data/customers.parquet"
    Path("data").mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Prepared Feast data: {output_path}")
    return output_path


@task(name="Apply Feast Features")
def apply_feast_features() -> None:
    """
    Step 5a: Apply Feast feature definitions
    """
    subprocess.run(
        ["feast", "apply"],
        cwd="feature_repo",
        check=True
    )
    print("Feast features applied")


@task(name="Materialize Feast Features")
def materialize_feast_features() -> str:
    """
    Step 5b: Materialize features to online store

    Returns:
        str: Timestamp of materialization
    """
    end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    subprocess.run(
        ["feast", "materialize-incremental", end_date],
        cwd="feature_repo",
        check=True
    )

    print(f"Features materialized up to {end_date}")
    return end_date
```

---

## Step 8: Implement Model Training (Kubeflow + MLflow)

### 8.1 Create pipelines/churn_training.yaml

This is compiled from the Python pipeline definition.

### 8.2 Create uc_1_churn_prediction/tasks/model_training.py

```python
# uc_1_churn_prediction/tasks/model_training.py
from prefect import task
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common.config import get_config

@task(name="Train Model Locally")
def train_model_local(data_path: str) -> str:
    """
    Step 6a: Train model locally with MLflow tracking

    Use this for quick iteration. For production, use Kubeflow.

    Args:
        data_path: Path to training data

    Returns:
        str: MLflow run ID
    """
    cfg = get_config()
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment("churn-prediction")

    # Load data
    df = pd.read_parquet(data_path)

    # Prepare features
    feature_cols = [
        'age', 'tenure_months', 'total_purchases',
        'avg_order_value', 'days_since_last_purchase', 'support_tickets_count'
    ]
    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features", feature_cols)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "churn_model",
            registered_model_name="ChurnModel"
        )

        print(f"Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")

        return run.info.run_id


@task(name="Trigger Kubeflow Training")
def trigger_kubeflow_training(data_path: str) -> str:
    """
    Step 6b: Submit training pipeline to Kubeflow

    Use this for production training with GPU.

    Args:
        data_path: S3 path to training data

    Returns:
        str: Kubeflow run ID
    """
    from kfp import Client

    cfg = get_config()
    client = Client(host=cfg['kubeflow']['host'])

    run = client.create_run_from_pipeline_package(
        pipeline_file="pipelines/churn_training.yaml",
        arguments={"data_path": data_path},
        run_name="churn-training-run"
    )

    print(f"Submitted Kubeflow run: {run.run_id}")
    return run.run_id


@task(name="Get Latest Model Version")
def get_latest_model_version(model_name: str = "ChurnModel") -> dict:
    """
    Get the latest registered model version from MLflow

    Args:
        model_name: Name of registered model

    Returns:
        dict: Model version info with run_id and version
    """
    cfg = get_config()
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])

    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
    if not versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest = versions[-1]

    model_info = {
        "name": model_name,
        "version": latest.version,
        "run_id": latest.run_id,
        "source": latest.source,
        "status": latest.status
    }

    print(f"Latest model: {model_name} v{latest.version}")
    return model_info
```

---

## Step 9: Implement Model Serving (KServe)

### 9.1 Create uc_1_churn_prediction/tasks/model_serving.py

```python
# uc_1_churn_prediction/tasks/model_serving.py
from prefect import task
from kubernetes import client, config
from common.config import get_config
import time

@task(name="Deploy to KServe")
def deploy_to_kserve(model_uri: str, model_name: str = "churn-predictor") -> str:
    """
    Step 7: Deploy model to KServe

    Args:
        model_uri: S3 URI to model artifact
        model_name: Name for the inference service

    Returns:
        str: Endpoint URL
    """
    cfg = get_config()
    namespace = cfg['kserve']['namespace']

    config.load_kube_config()
    api = client.CustomObjectsApi()

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "storageUri": model_uri
                }
            }
        }
    }

    # Check if exists
    try:
        api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name
        )
        # Update existing
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service
        )
        print(f"Updated InferenceService: {model_name}")
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # Create new
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
            print(f"Created InferenceService: {model_name}")
        else:
            raise

    endpoint = f"{model_name}.{namespace}.svc.cluster.local"
    print(f"Model deployed at: {endpoint}")
    return endpoint


@task(name="Wait for KServe Ready")
def wait_for_kserve_ready(
    model_name: str = "churn-predictor",
    timeout: int = 300
) -> bool:
    """
    Wait for KServe InferenceService to be ready

    Args:
        model_name: Name of inference service
        timeout: Timeout in seconds

    Returns:
        bool: True if ready
    """
    cfg = get_config()
    namespace = cfg['kserve']['namespace']

    config.load_kube_config()
    api = client.CustomObjectsApi()

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            isvc = api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name
            )

            status = isvc.get("status", {})
            conditions = status.get("conditions", [])

            for condition in conditions:
                if condition.get("type") == "Ready":
                    if condition.get("status") == "True":
                        print(f"InferenceService {model_name} is ready!")
                        return True

            print(f"Waiting for {model_name} to be ready...")
            time.sleep(10)

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(10)

    raise TimeoutError(f"InferenceService {model_name} not ready after {timeout}s")
```

---

## Step 10: Create Main Pipeline Flow

### 10.1 Create uc_1_churn_prediction/flows/churn_pipeline.py

```python
# uc_1_churn_prediction/flows/churn_pipeline.py
from prefect import flow, get_run_logger

from uc_1_churn_prediction.tasks.data_ingestion import (
    extract_from_postgres,
    upload_to_minio
)
from uc_1_churn_prediction.tasks.data_versioning import version_with_dvc
from uc_1_churn_prediction.tasks.data_validation import validate_with_great_expectations
from uc_1_churn_prediction.tasks.feature_engineering import (
    prepare_feast_data,
    apply_feast_features,
    materialize_feast_features
)
from uc_1_churn_prediction.tasks.model_training import (
    train_model_local,
    get_latest_model_version
)
from uc_1_churn_prediction.tasks.model_serving import (
    deploy_to_kserve,
    wait_for_kserve_ready
)


@flow(name="UC1: Churn Prediction Pipeline", log_prints=True)
def churn_prediction_pipeline(
    skip_versioning: bool = False,
    skip_training: bool = False,
    skip_deployment: bool = False
) -> dict:
    """
    End-to-end customer churn prediction pipeline.

    Pipeline Steps:
    1. Extract data from PostgreSQL
    2. Upload to MinIO
    3. Version with DVC
    4. Validate with Great Expectations
    5. Engineer features with Feast
    6. Train model (logs to MLflow)
    7. Deploy to KServe

    Args:
        skip_versioning: Skip DVC versioning step
        skip_training: Skip model training step
        skip_deployment: Skip KServe deployment step

    Returns:
        dict: Pipeline results including paths and endpoints
    """
    logger = get_run_logger()
    results = {}

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting data from PostgreSQL")
    logger.info("=" * 60)
    local_path = extract_from_postgres()
    results["local_path"] = local_path

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Upload to Object Storage
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Uploading to MinIO")
    logger.info("=" * 60)
    s3_key = upload_to_minio(local_path)
    results["s3_key"] = s3_key

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Data Versioning
    # ─────────────────────────────────────────────────────────────
    if not skip_versioning:
        logger.info("=" * 60)
        logger.info("STEP 3: Versioning with DVC")
        logger.info("=" * 60)
        dvc_file = version_with_dvc(local_path)
        results["dvc_file"] = dvc_file
    else:
        logger.info("STEP 3: Skipping DVC versioning")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Data Validation
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Validating with Great Expectations")
    logger.info("=" * 60)
    validation_passed = validate_with_great_expectations(local_path)
    results["validation_passed"] = validation_passed

    if not validation_passed:
        raise ValueError("Data validation failed! Pipeline aborted.")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Feature Engineering
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Engineering features with Feast")
    logger.info("=" * 60)

    # Prepare data for Feast
    feast_data_path = prepare_feast_data(local_path)

    # Apply feature definitions
    apply_feast_features()

    # Materialize to online store
    materialize_timestamp = materialize_feast_features()
    results["feast_materialized"] = materialize_timestamp

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Model Training
    # ─────────────────────────────────────────────────────────────
    if not skip_training:
        logger.info("=" * 60)
        logger.info("STEP 6: Training model (MLflow tracking)")
        logger.info("=" * 60)
        run_id = train_model_local(local_path)
        results["mlflow_run_id"] = run_id

        # Get model info
        model_info = get_latest_model_version()
        results["model_info"] = model_info
    else:
        logger.info("STEP 6: Skipping model training")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Model Deployment
    # ─────────────────────────────────────────────────────────────
    if not skip_deployment and not skip_training:
        logger.info("=" * 60)
        logger.info("STEP 7: Deploying to KServe")
        logger.info("=" * 60)

        model_uri = results["model_info"]["source"]
        endpoint = deploy_to_kserve(model_uri)

        # Wait for ready
        wait_for_kserve_ready()
        results["endpoint"] = endpoint
    else:
        logger.info("STEP 7: Skipping KServe deployment")

    # ─────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results: {results}")

    return results


# ─────────────────────────────────────────────────────────────────
# Sub-flows for partial execution
# ─────────────────────────────────────────────────────────────────

@flow(name="UC1: Data Pipeline Only", log_prints=True)
def data_pipeline_only() -> dict:
    """Run only data ingestion, versioning, and validation steps"""
    return churn_prediction_pipeline(
        skip_training=True,
        skip_deployment=True
    )


@flow(name="UC1: Training Pipeline Only", log_prints=True)
def training_pipeline_only(data_path: str) -> dict:
    """Run only training step with existing data"""
    logger = get_run_logger()

    logger.info("Training model with existing data...")
    run_id = train_model_local(data_path)
    model_info = get_latest_model_version()

    return {
        "mlflow_run_id": run_id,
        "model_info": model_info
    }


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run full pipeline
    churn_prediction_pipeline()
```

---

## Step 11: Port-Forward K3s Services

Before running the pipeline, set up port-forwards to k3s services:

```bash
# Terminal 2: PostgreSQL
kubectl port-forward svc/postgresql 5432:5432 -n postgresql

# Terminal 3: MinIO
kubectl port-forward svc/minio 9000:9000 -n minio

# Terminal 4: MLflow
kubectl port-forward svc/mlflow 5000:5000 -n mlflow

# Terminal 5: Kubeflow (if using)
kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow
```

---

## Step 12: Run the Pipeline

### 12.1 Local Execution (Testing)

```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases

# Run full pipeline
uv run python -m uc_1_churn_prediction.flows.churn_pipeline

# Run data pipeline only
uv run python -c "from uc_1_churn_prediction.flows.churn_pipeline import data_pipeline_only; data_pipeline_only()"
```

### 12.2 Via Prefect Cloud

```bash
# Deploy with schedule
uv run python -c "
from uc_1_churn_prediction.flows.churn_pipeline import churn_prediction_pipeline
churn_prediction_pipeline.serve(
    name='churn-prediction-daily',
    cron='0 6 * * *',
    tags=['uc1', 'churn', 'production']
)
"
```

### 12.3 Trigger Deployed Flow

```bash
uv run prefect deployment run "UC1: Churn Prediction Pipeline/churn-prediction-daily"
```

---

## Step 13: Monitor and Verify

### 13.1 Prefect Cloud UI

```
https://app.prefect.cloud
```

View:
- Flow runs and status
- Task logs
- Run history
- Scheduled deployments

### 13.2 MLflow UI

```bash
# Access via port-forward
kubectl port-forward svc/mlflow 5000:5000 -n mlflow

# Open http://localhost:5000
```

View:
- Experiments and runs
- Parameters and metrics
- Model artifacts
- Model registry

### 13.3 KServe Inference

```bash
# Test prediction
curl -X POST http://churn-predictor.models.svc.cluster.local/v1/models/churn-predictor:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [[35, 24, 150, 75.50, 15, 2]]
  }'
```

---

## Quick Reference Commands

```bash
# Project directory
cd /home/sujith/github/rag/00_MLOps/04_usecases

# Install/update dependencies
uv sync

# Login to Prefect Cloud
uv run prefect cloud login

# Start worker
uv run prefect worker start --pool local-pool

# Run pipeline
uv run python -m uc_1_churn_prediction.flows.churn_pipeline

# View deployments
uv run prefect deployment ls

# Trigger deployment
uv run prefect deployment run "UC1: Churn Prediction Pipeline/churn-prediction-daily"

# DVC commands
uv run dvc status
uv run dvc push
uv run dvc pull

# Feast commands
cd feature_repo && feast apply && feast materialize-incremental $(date -Iseconds)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PostgreSQL connection failed | Check port-forward: `kubectl port-forward svc/postgresql 5432:5432 -n postgresql` |
| MinIO upload failed | Verify credentials in config.yaml and port-forward |
| DVC push failed | Check MinIO remote config: `dvc remote list` |
| Great Expectations init error | Run `uv run great_expectations init` |
| Feast apply error | Verify feature_store.yaml path |
| MLflow connection refused | Check port-forward to MLflow service |
| KServe deployment failed | Verify namespace exists and RBAC permissions |
| Prefect worker not picking up runs | Ensure work pool name matches: `local-pool` |

---

## Summary

| Step | Tool | Purpose |
|------|------|---------|
| 1 | PostgreSQL | Source data extraction |
| 2 | MinIO | Object storage |
| 3 | DVC | Data versioning |
| 4 | Great Expectations | Data validation |
| 5 | Feast | Feature engineering |
| 6 | Kubeflow + MLflow | Model training & tracking |
| 7 | KServe | Model serving |
| Orchestration | Prefect | Pipeline orchestration |
