# MLOps Platform Use Cases

Practical scenarios demonstrating how all tools in the MLOps platform work together.

**Orchestration**: Prefect Cloud + Local Worker (lightweight alternative to Airflow)

---

## Use Case 1: Customer Churn Prediction Pipeline

**Scenario**: Build an end-to-end ML pipeline to predict customer churn, from raw data to deployed model.

### Tools Used
`Prefect` → `DVC` → `Great Expectations` → `Feast` → `Kubeflow` → `MLflow` → `KServe`

### Step-by-Step Flow

## Prefect Setup

```bash
# 1. Setup Prefect
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv sync
uv run prefect cloud login

# 2. Start worker
uv run prefect worker start --pool local-pool

# 3. Run data pipeline (new terminal)
uv run python -m uc_1_churn_prediction.flows.churn_pipeline

# 4. Monitor in Prefect Cloud UI
# https://app.prefect.cloud
```

#### 1. Data Ingestion (Prefect)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/data_ingestion.py`

```python
# tasks/data_ingestion.py
from prefect import task
import pandas as pd
import boto3
from sqlalchemy import create_engine
from datetime import datetime
from common.config import get_config

@task(name="Extract Customer Data", retries=2, retry_delay_seconds=60)
def extract_from_postgres() -> str:
    """Pull customer data from PostgreSQL data warehouse"""
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
    """Upload extracted data to MinIO object storage"""
    cfg = get_config()

    s3 = boto3.client(
        's3',
        endpoint_url=cfg['minio']['endpoint'],
        aws_access_key_id=cfg['minio']['access_key'],
        aws_secret_access_key=cfg['minio']['secret_key']
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"raw/customers_{timestamp}.parquet"
    s3.upload_file(local_path, cfg['minio']['bucket'], s3_key)
    print(f"Uploaded to s3://{cfg['minio']['bucket']}/{s3_key}")
    return s3_key
```

**Run Command**:
```bash
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv run python -m uc_1_churn_prediction.flows.churn_pipeline
```

#### 2. Data Versioning (DVC)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/data_versioning.py`

```python
# tasks/data_versioning.py
from prefect import task
import subprocess

@task(name="Version with DVC")
def version_with_dvc(file_path: str) -> str:
    """Version dataset with DVC and push to MinIO"""

    # Add file to DVC
    subprocess.run(["dvc", "add", file_path], check=True)

    # Push to remote (MinIO)
    subprocess.run(["dvc", "push"], check=True)

    # Git commit the .dvc file
    subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"Add dataset version"], check=True)

    print(f"Versioned: {file_path}")
    return f"{file_path}.dvc"
```

#### 3. Data Validation (Great Expectations)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/data_validation.py`

```python
# tasks/data_validation.py
from prefect import task
import great_expectations as gx
import pandas as pd

@task(name="Validate with Great Expectations")
def validate_with_great_expectations(file_path: str) -> bool:
    """Validate data quality with Great Expectations"""

    df = pd.read_parquet(file_path)

    context = gx.get_context()

    # Create validator
    validator = context.sources.pandas_default.read_dataframe(df)

    # Define expectations
    validator.expect_column_to_exist("customer_id")
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)
    validator.expect_column_values_to_be_in_set("churn", [0, 1])

    # Run validation
    results = validator.validate()

    if not results.success:
        raise ValueError("Data validation failed!")

    print("Data validation passed!")
    return True
```

#### 4. Feature Engineering (Feast)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/feature_engineering.py`

```python
# tasks/feature_engineering.py
from prefect import task
import subprocess
from datetime import datetime

@task(name="Materialize Feast Features")
def materialize_feast_features() -> str:
    """Apply feature definitions and materialize to online store"""

    # Apply feature definitions
    subprocess.run(["feast", "apply"], check=True)

    # Materialize features to online store
    end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    subprocess.run([
        "feast", "materialize-incremental", end_date
    ], check=True)

    print(f"Features materialized up to {end_date}")
    return end_date
```

**Feature Definition** (`feature_repo/features.py`):
```python
from feast import Entity, Feature, FeatureView, FileSource
from datetime import timedelta

customer = Entity(name="customer_id", join_keys=["customer_id"])

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_purchases", dtype=Float32),
        Feature(name="avg_order_value", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int32),
        Feature(name="support_tickets_count", dtype=Int32),
    ],
    source=FileSource(path="s3://feast-offline/customer_features.parquet"),
)
```

#### 5. Model Training (Kubeflow Pipelines + MLflow)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/model_training.py`

```python
# tasks/model_training.py
from prefect import task
from kfp import Client

@task(name="Trigger Kubeflow Training")
def trigger_kubeflow_training(data_path: str) -> str:
    """Submit training pipeline to Kubeflow"""

    # Connect to Kubeflow (via port-forward)
    client = Client(host="http://localhost:8080")

    # Submit pipeline run
    run = client.create_run_from_pipeline_package(
        pipeline_file="pipelines/churn_training.yaml",
        arguments={"data_path": data_path},
        run_name="churn-training-run"
    )

    print(f"Submitted Kubeflow run: {run.run_id}")
    return run.run_id
```

**Inside Kubeflow Pipeline** (logs to MLflow):
```python
# kubeflow_pipeline.py
from kfp import dsl
from kfp.components import func_to_container_op
import mlflow

@func_to_container_op
def train_model(data_path: str) -> str:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from feast import FeatureStore

    # Get features from Feast
    store = FeatureStore(repo_path="feature_repo/")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=["customer_features:total_purchases", "customer_features:avg_order_value"]
    ).to_df()

    # Train model
    X = training_df.drop("churn", axis=1)
    y = training_df["churn"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Log to MLflow
    mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", model.score(X, y))
        mlflow.sklearn.log_model(model, "churn_model")
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/churn_model",
            "ChurnModel"
        )

    return "Model trained and registered"

@dsl.pipeline(name="Churn Training Pipeline")
def churn_pipeline():
    train_task = train_model(data_path="s3://dvc-storage/raw/customers.parquet")
```

#### 6. Model Serving (KServe)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/tasks/model_serving.py`

```python
# tasks/model_serving.py
from prefect import task
from kubernetes import client, config

@task(name="Deploy to KServe")
def deploy_to_kserve(model_uri: str) -> str:
    """Deploy model to KServe"""

    config.load_kube_config()
    api = client.CustomObjectsApi()

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": "churn-predictor",
            "namespace": "models"
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

    api.create_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace="models",
        plural="inferenceservices",
        body=inference_service
    )

    print(f"Deployed model: {model_uri}")
    return "churn-predictor.models.svc.cluster.local"
```

#### 7. Main Flow (Orchestrator)

**Implementation Location**: `00_MLOps/04_usecases/uc_1_churn_prediction/flows/churn_pipeline.py`

```python
# flows/churn_pipeline.py
from prefect import flow, get_run_logger
from uc_1_churn_prediction.tasks.data_ingestion import extract_from_postgres, upload_to_minio
from uc_1_churn_prediction.tasks.data_versioning import version_with_dvc
from uc_1_churn_prediction.tasks.data_validation import validate_with_great_expectations
from uc_1_churn_prediction.tasks.feature_engineering import materialize_feast_features
from uc_1_churn_prediction.tasks.model_training import trigger_kubeflow_training
from uc_1_churn_prediction.tasks.model_serving import deploy_to_kserve

@flow(name="UC1: Churn Prediction Pipeline", log_prints=True)
def churn_prediction_pipeline(skip_training: bool = False):
    """End-to-end churn prediction pipeline"""
    logger = get_run_logger()

    # Step 1: Extract data
    logger.info("Step 1: Extracting data from PostgreSQL")
    local_path = extract_from_postgres()

    # Step 2: Upload to MinIO
    logger.info("Step 2: Uploading to MinIO")
    s3_path = upload_to_minio(local_path)

    # Step 3: Version with DVC
    logger.info("Step 3: Versioning with DVC")
    version_with_dvc(local_path)

    # Step 4: Validate data
    logger.info("Step 4: Validating with Great Expectations")
    validate_with_great_expectations(local_path)

    # Step 5: Feature engineering
    logger.info("Step 5: Materializing Feast features")
    materialize_feast_features()

    if not skip_training:
        # Step 6: Train model
        logger.info("Step 6: Triggering Kubeflow training")
        run_id = trigger_kubeflow_training(s3_path)

    logger.info("Pipeline completed successfully!")
    return {"s3_path": s3_path}

if __name__ == "__main__":
    churn_prediction_pipeline()
```

---

## Use Case 2: A/B Testing Model Versions (Iter8 + KServe)

**Scenario**: Safely roll out a new model version with traffic splitting and automatic validation.

### Tools Used
`MLflow` → `KServe` → `Iter8`

### Step-by-Step Flow

#### 1. Deploy Canary Version
```yaml
# kserve-canary.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-predictor
spec:
  predictor:
    canaryTrafficPercent: 10  # 10% to new model
    model:
      modelFormat:
        name: sklearn
      storageUri: s3://mlflow-artifacts/ChurnModel/v2
```

#### 2. Create Iter8 Experiment
```yaml
# iter8-experiment.yaml
apiVersion: iter8.tools/v2beta2
kind: Metric
metadata:
  name: error-rate
spec:
  type: Gauge
  provider: prometheus
  params:
    query: sum(rate(inference_request_errors_total[30s])) / sum(rate(inference_request_total[30s]))
---
apiVersion: iter8.tools/v2beta2
kind: Experiment
metadata:
  name: churn-model-experiment
spec:
  target: churn-predictor
  strategy:
    type: Canary
    weights:
      maxCandidateWeight: 100
  criteria:
    requestCount: iter8-system/request-count
    objectives:
    - metric: error-rate
      upperLimit: 0.01  # Max 1% error rate
    - metric: latency-p95
      upperLimit: 100   # Max 100ms latency
  duration:
    intervalSeconds: 10
    iterationsPerLoop: 10
```

#### 3. Monitor and Auto-Promote
```bash
# Watch experiment progress
kubectl get experiment churn-model-experiment -w

# Result: If SLOs pass, Iter8 promotes to 100% traffic
# If SLOs fail, automatic rollback to previous version
```

---

## Use Case 3: Automated Retraining on Data Drift (Evidently + Prefect)

**Scenario**: Detect when production data drifts from training data and trigger automatic retraining.

### Tools Used
`KServe` → `Evidently` → `Prefect` → `Kubeflow`

### Step-by-Step Flow

#### 1. Drift Detection Task (Evidently)

```python
# tasks/drift_detection.py
from prefect import task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

@task(name="Check Data Drift")
def check_drift(reference_path: str, production_path: str) -> tuple[bool, float]:
    """Compare production data against training data"""

    reference_data = pd.read_parquet(reference_path)
    production_data = pd.read_parquet(production_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=production_data)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_score = result["metrics"][0]["result"]["drift_share"]

    print(f"Drift detected: {drift_detected}, Score: {drift_score}")
    return drift_detected, drift_score
```

#### 2. Drift Monitoring Flow (Prefect)

```python
# flows/drift_pipeline.py
from prefect import flow, get_run_logger
from uc_3_drift_retraining.tasks.drift_detection import check_drift
from uc_1_churn_prediction.tasks.model_training import trigger_kubeflow_training

@flow(name="UC3: Drift Monitoring Pipeline", log_prints=True)
def drift_monitoring_pipeline(
    reference_path: str = "s3://dvc-storage/training/customers.parquet",
    production_path: str = "s3://dvc-storage/production/customers_latest.parquet",
    drift_threshold: float = 0.3
):
    """Monitor for drift and trigger retraining if needed"""
    logger = get_run_logger()

    # Check for drift
    logger.info("Checking for data drift...")
    drift_detected, drift_score = check_drift(reference_path, production_path)

    if drift_detected and drift_score > drift_threshold:
        logger.warning(f"Drift detected ({drift_score:.2%})! Triggering retraining...")
        trigger_kubeflow_training(production_path)
        return {"action": "retrain", "drift_score": drift_score}

    logger.info("No significant drift detected")
    return {"action": "none", "drift_score": drift_score}

if __name__ == "__main__":
    drift_monitoring_pipeline()
```

#### 3. Schedule Drift Monitoring

```python
# Deploy with schedule
if __name__ == "__main__":
    drift_monitoring_pipeline.serve(
        name="drift-monitoring-hourly",
        cron="0 * * * *",  # Every hour
        tags=["drift", "monitoring"]
    )
```

---

## Use Case 4: Full Data Lineage Tracking (OpenLineage + Marquez)

**Scenario**: Track the complete journey of data from source to model prediction for compliance.

### Tools Used
`Prefect` → `OpenLineage` → `Marquez` → `Kubeflow` → `MLflow`

### Step-by-Step Flow

#### 1. Emit OpenLineage Events (Prefect)

```python
# tasks/emit_lineage.py
from prefect import task
import requests
from datetime import datetime
import uuid

@task(name="Emit Lineage Event")
def emit_lineage_event(
    job_name: str,
    input_datasets: list[str],
    output_datasets: list[str]
) -> str:
    """Emit OpenLineage event to Marquez"""

    marquez_url = "http://localhost:30500/api/v1/lineage"

    event = {
        "eventType": "COMPLETE",
        "eventTime": datetime.utcnow().isoformat() + "Z",
        "run": {
            "runId": str(uuid.uuid4())
        },
        "job": {
            "namespace": "prefect",
            "name": job_name
        },
        "inputs": [
            {"namespace": "prefect", "name": ds} for ds in input_datasets
        ],
        "outputs": [
            {"namespace": "prefect", "name": ds} for ds in output_datasets
        ],
        "producer": "prefect-worker"
    }

    response = requests.post(marquez_url, json=event)
    print(f"Lineage event emitted: {job_name}")
    return response.status_code
```

#### 2. View Lineage in Marquez UI

```
http://localhost:30501

# See:
# - Data sources: PostgreSQL tables, S3 files
# - Transformations: Prefect tasks, Kubeflow steps
# - Outputs: Feature tables, Model artifacts
# - Impact analysis: "If this table changes, which models affected?"
```

#### 3. Query Lineage API

```bash
# Get all jobs in namespace
curl http://localhost:30500/api/v1/namespaces/prefect/jobs

# Get lineage for specific dataset
curl http://localhost:30500/api/v1/lineage?nodeId=dataset:prefect:customers
```

---

## Use Case 5: Model Fairness & Documentation (What-If Tool + Model Card)

**Scenario**: Analyze model bias and generate compliance documentation before deployment.

### Tools Used
`Kubeflow` → `What-If Tool` → `Model Card Toolkit` → `MLflow`

### Step-by-Step Flow

#### 1. Fairness Analysis (What-If Tool)

```python
# tasks/fairness_analysis.py
from prefect import task
from witwidget.notebook.visualization import WitConfigBuilder, WitWidget

@task(name="Analyze Model Fairness")
def analyze_fairness(model_path: str, test_data_path: str) -> dict:
    """Analyze model fairness with What-If Tool"""

    model = load_model(model_path)
    test_examples = load_test_data(test_data_path)

    # Configure What-If Tool
    config = WitConfigBuilder(test_examples).set_model(model)
    config.set_label_vocab(["Not Churned", "Churned"])

    # Analyze fairness across demographic slices
    config.set_compare_custom_prediction(
        lambda x: model.predict_proba(x)[:, 1],
        "Churn Probability"
    )

    # Generate fairness report
    widget = WitWidget(config, height=800)

    return {"status": "analyzed", "widget": widget}
```

#### 2. Generate Model Card

```python
# tasks/model_card.py
from prefect import task
from model_card_toolkit import ModelCardToolkit
import mlflow

@task(name="Generate Model Card")
def generate_model_card(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float
) -> str:
    """Generate standardized model documentation"""

    mct = ModelCardToolkit()

    model_card = mct.scaffold_assets()
    model_card.model_details.name = model_name
    model_card.model_details.version.name = "v2"
    model_card.model_details.overview = "Predicts customer churn probability"

    # Add performance metrics
    model_card.quantitative_analysis.performance_metrics = [
        {"type": "accuracy", "value": accuracy},
        {"type": "precision", "value": precision},
        {"type": "recall", "value": recall}
    ]

    # Add fairness considerations
    model_card.considerations.ethical_considerations = [
        {"name": "Age Bias", "mitigation_strategy": "Balanced training data across age groups"}
    ]

    # Export and log to MLflow
    output_path = "/tmp/model_card.html"
    mct.export_format(model_card, output_path)
    mlflow.log_artifact(output_path)

    print(f"Model card generated: {output_path}")
    return output_path
```

---

## Use Case 6: GitOps Model Deployment (Argo CD)

**Scenario**: Deploy models using GitOps - all changes tracked in Git.

### Tools Used
`MLflow` → `Git` → `Argo CD` → `KServe`

### Step-by-Step Flow

#### 1. Model Promotion Task (Prefect)

```python
# tasks/promote_model.py
from prefect import task
import subprocess

@task(name="Promote Model via GitOps")
def promote_model_gitops(
    model_name: str,
    model_version: str,
    model_uri: str
) -> str:
    """Update deployment manifest and push to Git"""

    # Update KServe manifest
    manifest_path = "deployments/churn-predictor.yaml"

    manifest = f"""
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-predictor
  namespace: models
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: {model_uri}
"""

    with open(manifest_path, 'w') as f:
        f.write(manifest)

    # Commit and push
    subprocess.run(['git', 'add', manifest_path], check=True)
    subprocess.run([
        'git', 'commit', '-m',
        f'Promote {model_name} v{model_version}'
    ], check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], check=True)

    print(f"Promoted {model_name} v{model_version} via GitOps")
    return "Deployment triggered"
```

#### 2. Argo CD Application

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-models
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/ml-deployments
    targetRevision: main
    path: deployments/
  destination:
    server: https://kubernetes.default.svc
    namespace: models
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

#### 3. Automatic Sync

```bash
# Argo CD detects Git change and syncs Kubernetes
# View in Argo CD UI: https://localhost:30444

argocd app get ml-models
# Status: Synced, Health: Healthy
```

---

## Quick Reference: Tool Connections

| From | To | Integration Method |
|------|----|--------------------|
| Prefect | DVC | subprocess running `dvc push` |
| Prefect | Feast | subprocess or Feast SDK |
| Prefect | OpenLineage | HTTP POST to Marquez API |
| OpenLineage | Marquez | HTTP API |
| Feast | Kubeflow | Python SDK in pipeline steps |
| Kubeflow | MLflow | `mlflow` Python package |
| MLflow | KServe | Model URI in InferenceService |
| KServe | Iter8 | Experiment CR targeting service |
| KServe | Evidently | Log predictions for analysis |
| Evidently | Prefect | Task triggers retraining flow |
| Git | Argo CD | Automatic sync on commits |
| Argo CD | KServe | Kubernetes manifest sync |

---

## MLOps Use Cases - Summary

| Use Case | Scenario | Tools |
|----------|----------|-------|
| **UC1** | Customer Churn Prediction Pipeline | Prefect → DVC → GE → Feast → Kubeflow → MLflow → KServe |
| **UC2** | A/B Testing Model Versions | MLflow → KServe → Iter8 |
| **UC3** | Automated Retraining on Data Drift | KServe → Evidently → Prefect → Kubeflow |
| **UC4** | Full Data Lineage Tracking | Prefect → OpenLineage → Marquez → Kubeflow → MLflow |
| **UC5** | Model Fairness & Documentation | Kubeflow → What-If Tool → Model Card Toolkit → MLflow |
| **UC6** | GitOps Model Deployment | MLflow → Git → Argo CD → KServe |

---

## Running the Pipelines

```bash
# Setup
cd /home/sujith/github/rag/00_MLOps/04_usecases
uv sync
uv run prefect cloud login

# Start worker (Terminal 1)
uv run prefect worker start --pool local-pool

# Run UC1: Churn Prediction (Terminal 2)
uv run python -m uc_1_churn_prediction.flows.churn_pipeline

# Run UC3: Drift Monitoring
uv run python -m uc_3_drift_retraining.flows.drift_pipeline

# Schedule drift monitoring (hourly)
uv run python -c "from uc_3_drift_retraining.flows.drift_pipeline import drift_monitoring_pipeline; drift_monitoring_pipeline.serve(name='drift-hourly', cron='0 * * * *')"
```
