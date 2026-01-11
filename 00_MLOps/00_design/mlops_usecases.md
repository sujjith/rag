# MLOps Platform Use Cases

Practical scenarios demonstrating how all tools in the MLOps platform work together.

---

## Use Case 1: Customer Churn Prediction Pipeline

**Scenario**: Build an end-to-end ML pipeline to predict customer churn, from raw data to deployed model.

### Tools Used
`Airflow` → `DVC` → `Great Expectations` → `Feast` → `Kubeflow` → `MLflow` → `KServe`

### Step-by-Step Flow

#### 1. Data Ingestion (Airflow)
```python
# Airflow DAG: data_ingestion_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_customer_data():
    """Pull customer data from data warehouse"""
    import pandas as pd
    # Extract from PostgreSQL
    df = pd.read_sql("SELECT * FROM customers", connection)
    df.to_parquet("/data/raw/customers.parquet")

def push_to_minio():
    """Upload to MinIO object storage"""
    import boto3
    s3 = boto3.client('s3', endpoint_url='http://minio.minio.svc.cluster.local:9000')
    s3.upload_file('/data/raw/customers.parquet', 'raw-data', 'customers.parquet')

with DAG('customer_churn_ingestion', schedule='@daily', start_date=datetime(2026, 1, 1)):
    extract = PythonOperator(task_id='extract', python_callable=extract_customer_data)
    upload = PythonOperator(task_id='upload', python_callable=push_to_minio)
    extract >> upload
```

#### 2. Data Versioning (DVC)
```bash
# Version the dataset
cd /project
dvc add data/raw/customers.parquet
dvc push  # Pushes to MinIO (s3://dvc-storage)
git add data/raw/customers.parquet.dvc
git commit -m "Add customer data v2026.01.11"
```

#### 3. Data Validation (Great Expectations)
```python
import great_expectations as gx

# Create expectation suite
context = gx.get_context()
suite = context.add_expectation_suite("customer_data_suite")

# Define expectations
validator = context.get_validator(batch_request=batch_request, expectation_suite=suite)
validator.expect_column_to_exist("customer_id")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)
validator.expect_column_values_to_be_in_set("churn", [0, 1])

# Validate and fail pipeline if not passing
results = context.run_checkpoint(checkpoint_name="customer_checkpoint")
if not results.success:
    raise Exception("Data validation failed!")
```

#### 4. Feature Engineering (Feast)
```python
# feature_repo/features.py
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

```bash
# Apply to Feast
feast apply
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

#### 5. Model Training (Kubeflow Pipelines + MLflow)
```python
# kubeflow_pipeline.py
from kfp import dsl
from kfp.components import func_to_container_op
import mlflow

@func_to_container_op
def train_model(feast_features: str) -> str:
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
        mlflow.register_model("runs:/{}/churn_model".format(mlflow.active_run().info.run_id), "ChurnModel")
    
    return "Model trained and registered"

@dsl.pipeline(name="Churn Training Pipeline")
def churn_pipeline():
    train_task = train_model(feast_features="customer_features")
```

#### 6. Model Serving (KServe)
```yaml
# kserve-churn-model.yaml
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
      storageUri: s3://mlflow-artifacts/1/abc123/artifacts/churn_model
```

```bash
kubectl apply -f kserve-churn-model.yaml
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

## Use Case 3: Automated Retraining on Data Drift (Evidently + Airflow)

**Scenario**: Detect when production data drifts from training data and trigger automatic retraining.

### Tools Used
`KServe` → `Evidently` → `Airflow` → `Kubeflow`

### Step-by-Step Flow

#### 1. Set Up Drift Monitoring (Evidently)
```python
# drift_monitor.py
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def check_drift(reference_data: pd.DataFrame, production_data: pd.DataFrame):
    """Compare production data against training data"""
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=production_data)
    
    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_score = result["metrics"][0]["result"]["drift_share"]
    
    return drift_detected, drift_score
```

#### 2. Airflow Monitoring DAG
```python
# drift_monitoring_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

def check_for_drift(**context):
    """Check drift and decide whether to retrain"""
    from drift_monitor import check_drift
    
    reference = load_training_data()
    production = load_production_logs(last_24_hours=True)
    
    drift_detected, drift_score = check_drift(reference, production)
    
    if drift_detected and drift_score > 0.3:  # >30% features drifted
        return 'trigger_retrain'
    return 'no_action'

with DAG('drift_monitoring', schedule='@hourly'):
    check = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_for_drift
    )
    
    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retrain',
        trigger_dag_id='kubeflow_training_pipeline'
    )
    
    no_action = PythonOperator(task_id='no_action', python_callable=lambda: None)
    
    check >> [trigger_retrain, no_action]
```

---

## Use Case 4: Full Data Lineage Tracking (OpenLineage + Marquez)

**Scenario**: Track the complete journey of data from source to model prediction for compliance.

### Tools Used
`Airflow` → `OpenLineage` → `Marquez` → `Kubeflow` → `MLflow`

### Step-by-Step Flow

#### 1. Enable OpenLineage in Airflow
```bash
# Set environment variables in Airflow
export OPENLINEAGE_URL=http://marquez-api.marquez.svc.cluster.local:5000
export OPENLINEAGE_NAMESPACE=airflow
```

#### 2. Airflow Automatically Emits Lineage
```python
# Any Airflow task automatically sends lineage events
# Input datasets → Job → Output datasets
```

#### 3. View Lineage in Marquez UI
```
http://localhost:30501

# See:
# - Data sources: PostgreSQL tables, S3 files
# - Transformations: Airflow tasks, Kubeflow steps
# - Outputs: Feature tables, Model artifacts
# - Impact analysis: "If this table changes, which models affected?"
```

#### 4. Query Lineage API
```bash
# Get all jobs in namespace
curl http://localhost:30500/api/v1/namespaces/airflow/jobs

# Get lineage for specific dataset
curl http://localhost:30500/api/v1/lineage?nodeId=dataset:airflow:customers
```

---

## Use Case 5: Model Fairness & Documentation (What-If Tool + Model Card)

**Scenario**: Analyze model bias and generate compliance documentation before deployment.

### Tools Used
`Kubeflow` → `What-If Tool` → `Model Card Toolkit` → `MLflow`

### Step-by-Step Flow

#### 1. Fairness Analysis (What-If Tool)
```python
# fairness_analysis.py
from witwidget.notebook.visualization import WitConfigBuilder, WitWidget

# Load model and test data
model = load_model("churn_model")
test_examples = load_test_data()

# Configure What-If Tool
config = WitConfigBuilder(test_examples).set_model(model)
config.set_label_vocab(["Not Churned", "Churned"])

# Analyze fairness across demographic slices
config.set_compare_custom_prediction(
    lambda x: model.predict_proba(x)[:, 1],
    "Churn Probability"
)

# Generate fairness report
WitWidget(config, height=800)
```

#### 2. Generate Model Card
```python
# model_card.py
from model_card_toolkit import ModelCardToolkit
import mlflow

# Initialize toolkit
mct = ModelCardToolkit()

# Create model card
model_card = mct.scaffold_assets()
model_card.model_details.name = "Customer Churn Predictor"
model_card.model_details.version.name = "v2"
model_card.model_details.overview = "Predicts customer churn probability"

# Add performance metrics
model_card.quantitative_analysis.performance_metrics = [
    {"type": "accuracy", "value": 0.92},
    {"type": "precision", "value": 0.88},
    {"type": "recall", "value": 0.85}
]

# Add fairness considerations
model_card.considerations.ethical_considerations = [
    {"name": "Age Bias", "mitigation_strategy": "Balanced training data across age groups"}
]

# Export and log to MLflow
mct.export_format(model_card, "model_card.html")
mlflow.log_artifact("model_card.html")
```

---

## Use Case 6: GitOps Model Deployment (Argo CD)

**Scenario**: Deploy models using GitOps - all changes tracked in Git.

### Tools Used
`MLflow` → `Git` → `Argo CD` → `KServe`

### Step-by-Step Flow

#### 1. Model Promotion Webhook (MLflow → Git)
```python
# mlflow_webhook.py
from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/promote', methods=['POST'])
def promote_model():
    data = request.json
    model_name = data['model_name']
    model_version = data['model_version']
    model_uri = data['model_uri']
    
    # Update deployment manifest in Git
    update_manifest(model_uri)
    
    # Commit and push
    subprocess.run(['git', 'add', 'deployments/'])
    subprocess.run(['git', 'commit', '-m', f'Promote {model_name} v{model_version}'])
    subprocess.run(['git', 'push', 'origin', 'main'])
    
    return {"status": "Deployment triggered via GitOps"}
```

#### 2. Argo CD Watches Git Repository
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
| Airflow | DVC | BashOperator running `dvc push` |
| Airflow | Feast | PythonOperator with Feast SDK |
| Airflow | OpenLineage | Built-in plugin (auto-enabled) |
| OpenLineage | Marquez | HTTP API (OPENLINEAGE_URL) |
| Feast | Kubeflow | Python SDK in pipeline steps |
| Kubeflow | MLflow | `mlflow` Python package |
| MLflow | KServe | Model URI in InferenceService |
| KServe | Iter8 | Experiment CR targeting service |
| KServe | Evidently | Log predictions for analysis |
| Evidently | Airflow | Webhook/API trigger |
| Git | Argo CD | Automatic sync on commits |
| Argo CD | KServe | Kubernetes manifest sync |

## MLOps Use Cases - Flow

######################################################################################
### Use Case 1: Customer Churn Prediction Pipeline
Scenario: Build an end-to-end ML pipeline to predict customer churn, from raw data to deployed model.

Tools Used
Airflow → DVC → Great Expectations → Feast → Kubeflow → MLflow → KServe

######################################################################################
### Use Case 2: A/B Testing Model Versions (Iter8 + KServe)
Scenario: Safely roll out a new model version with traffic splitting and automatic validation.

Tools Used
MLflow → KServe → Iter8

######################################################################################
### Use Case 3: Automated Retraining on Data Drift (Evidently + Airflow)
Scenario: Detect when production data drifts from training data and trigger automatic retraining.

Tools Used
KServe → Evidently → Airflow → Kubeflow

######################################################################################
### Use Case 4: Full Data Lineage Tracking (OpenLineage + Marquez)
Scenario: Track the complete journey of data from source to model prediction for compliance.

Tools Used
Airflow → OpenLineage → Marquez → Kubeflow → MLflow

######################################################################################
### Use Case 5: Model Fairness & Documentation (What-If Tool + Model Card)
Scenario: Analyze model bias and generate compliance documentation before deployment.

Tools Used
Kubeflow → What-If Tool → Model Card Toolkit → MLflow

######################################################################################
### Use Case 6: GitOps Model Deployment (Argo CD)
Scenario: Deploy models using GitOps - all changes tracked in Git.

Tools Used
MLflow → Git → Argo CD → KServe
