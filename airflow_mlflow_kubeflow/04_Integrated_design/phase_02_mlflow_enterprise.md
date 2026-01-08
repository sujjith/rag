# Phase 02: MLflow Enterprise Setup

## Overview

Complete MLflow setup with all enterprise features including experiment tracking, model registry, model serving, authentication, and multi-tenant support.

---

## MLflow Feature Matrix

| Feature | Description | Status |
|---------|-------------|--------|
| Experiment Tracking | Log parameters, metrics, artifacts | Included |
| Model Registry | Version control, staging, production | Included |
| Model Serving | REST API deployment | Included |
| Model Signatures | Input/output schema validation | Included |
| MLflow Projects | Reproducible ML code packaging | Included |
| Dataset Tracking | Log and version datasets | Included |
| System Metrics | Auto-log CPU, GPU, memory | Included |
| Nested Runs | Parent-child experiment hierarchy | Included |
| Authentication | Basic auth, OAuth, LDAP | Included |
| Multi-tenancy | Team/project isolation | Included |
| Custom Flavors | PyFunc custom models | Included |
| Model Evaluation | Built-in evaluation | Included |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLflow Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  MLflow Tracking │    │  MLflow Registry │                  │
│  │     Server       │◄──►│     Server       │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────────────────────────────┐                  │
│  │            PostgreSQL Backend             │                  │
│  │     (experiments, runs, model versions)   │                  │
│  └──────────────────────────────────────────┘                  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────┐                  │
│  │              MinIO / S3                   │                  │
│  │   (artifacts, models, datasets, logs)     │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Model Server │  │ Model Server │  │ Model Server │         │
│  │  (Staging)   │  │ (Production) │  │  (Canary)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Deploy MLflow on Kubernetes

### Create MLflow Namespace and Secrets

```bash
# Create namespace
kubectl create namespace mlflow

# Create secrets
kubectl create secret generic mlflow-postgres \
    --namespace mlflow \
    --from-literal=connection-string="postgresql://mlflow:mlflow123@postgres-postgresql.mlflow.svc.cluster.local:5432/mlflow"

kubectl create secret generic mlflow-minio \
    --namespace mlflow \
    --from-literal=AWS_ACCESS_KEY_ID=minio \
    --from-literal=AWS_SECRET_ACCESS_KEY=minio123

kubectl create secret generic mlflow-auth \
    --namespace mlflow \
    --from-literal=username=admin \
    --from-literal=password=admin123
```

### MLflow Deployment Manifest

Create `mlflow/kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: mlflow
  labels:
    app: mlflow
    component: tracking-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "5000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: mlflow
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.9.2
        ports:
        - containerPort: 5000
          name: http
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-postgres
              key: connection-string
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio.mlflow.svc.cluster.local:9000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-minio
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-minio
              key: AWS_SECRET_ACCESS_KEY
        - name: MLFLOW_TRACKING_USERNAME
          valueFrom:
            secretKeyRef:
              name: mlflow-auth
              key: username
        - name: MLFLOW_TRACKING_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-auth
              key: password
        command:
        - mlflow
        - server
        - --backend-store-uri
        - $(MLFLOW_BACKEND_STORE_URI)
        - --default-artifact-root
        - s3://mlflow/
        - --host
        - "0.0.0.0"
        - --port
        - "5000"
        - --serve-artifacts
        - --workers
        - "4"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: mlflow-config
          mountPath: /etc/mlflow
      volumes:
      - name: mlflow-config
        configMap:
          name: mlflow-config
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlflow
  labels:
    app: mlflow
spec:
  type: ClusterIP
  ports:
  - port: 5000
    targetPort: 5000
    protocol: TCP
    name: http
  selector:
    app: mlflow
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlflow
  namespace: mlflow
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "0"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - mlflow.local
    secretName: mlflow-tls
  rules:
  - host: mlflow.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlflow
            port:
              number: 5000
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow
  namespace: mlflow
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-config
  namespace: mlflow
data:
  mlflow.conf: |
    [mlflow]
    default_artifact_root = s3://mlflow/
    [tracking]
    username = admin
    password = admin123
```

```bash
# Deploy MLflow
kubectl apply -f mlflow/kubernetes/deployment.yaml

# Verify
kubectl get pods -n mlflow
kubectl get svc -n mlflow
kubectl get ingress -n mlflow
```

---

## Step 2: Experiment Tracking (Complete Features)

### Basic Experiment Tracking

```python
# mlflow/experiments/basic_tracking.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Configure MLflow
mlflow.set_tracking_uri("http://mlflow.local:5000")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "admin123"

# Create or get experiment
experiment_name = "iris-classification"
mlflow.set_experiment(experiment_name)

def log_model_with_full_tracking(model_class, params, X_train, X_test, y_train, y_test):
    """Complete MLflow tracking example"""

    with mlflow.start_run(run_name=f"{model_class.__name__}-experiment") as run:
        # Set tags for organization
        mlflow.set_tags({
            "model_type": model_class.__name__,
            "dataset": "iris",
            "team": "data-science",
            "environment": "development",
            "version": "1.0.0"
        })

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
            "roc_auc_ovr": roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.savefig("/tmp/confusion_matrix.png")
        mlflow.log_artifact("/tmp/confusion_matrix.png", "plots")
        plt.close()

        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("/tmp/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("/tmp/classification_report.json", "reports")

        # Log feature importances (if available)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv("/tmp/feature_importance.csv", index=False)
            mlflow.log_artifact("/tmp/feature_importance.csv", "reports")

        # Log model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_pred)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="iris-classifier"
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

        return run.info.run_id, metrics

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models_params = [
    (RandomForestClassifier, {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
    (RandomForestClassifier, {"n_estimators": 200, "max_depth": 15, "random_state": 42}),
    (GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}),
]

results = []
for model_class, params in models_params:
    run_id, metrics = log_model_with_full_tracking(
        model_class, params, X_train, X_test, y_train, y_test
    )
    results.append({"run_id": run_id, "model": model_class.__name__, **metrics})

print("\nAll Results:")
print(pd.DataFrame(results))
```

### Nested Runs (Hyperparameter Tuning)

```python
# mlflow/experiments/nested_runs.py
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow.local:5000")
mlflow.set_experiment("hyperparameter-tuning")

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Parent run for the entire tuning process
with mlflow.start_run(run_name="grid-search-parent") as parent_run:
    mlflow.set_tag("tuning_method", "grid_search")
    mlflow.log_param("param_grid", str(param_grid))

    best_score = 0
    best_params = None
    best_run_id = None

    # Iterate through parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))

        # Nested run for each parameter combination
        with mlflow.start_run(run_name=f"trial-{i}", nested=True) as child_run:
            mlflow.log_params(params)

            # Train model
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            mlflow.log_metrics({
                "train_accuracy": train_score,
                "test_accuracy": test_score
            })

            # Track best
            if test_score > best_score:
                best_score = test_score
                best_params = params
                best_run_id = child_run.info.run_id

    # Log best results to parent
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_test_accuracy", best_score)
    mlflow.set_tag("best_child_run_id", best_run_id)

    print(f"Best Score: {best_score}")
    print(f"Best Params: {best_params}")
```

### System Metrics Logging

```python
# mlflow/experiments/system_metrics.py
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# Enable system metrics logging
mlflow.set_tracking_uri("http://mlflow.local:5000")
mlflow.set_experiment("system-metrics-demo")

# Enable autolog with system metrics
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    silent=False
)

# Enable system metrics
mlflow.enable_system_metrics_logging()

with mlflow.start_run(run_name="system-metrics-run"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import time

    # Generate large dataset to see system metrics in action
    X, y = make_classification(n_samples=100000, n_features=100, random_state=42)

    # Train model (will automatically log system metrics)
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time

    mlflow.log_metric("training_time_seconds", training_time)

    # System metrics are automatically logged:
    # - system/cpu_utilization_percentage
    # - system/memory_usage_megabytes
    # - system/disk_usage_percentage
    # - system/network_receive_megabytes
    # - system/network_transmit_megabytes
    # - GPU metrics (if available)
```

### Dataset Tracking

```python
# mlflow/experiments/dataset_tracking.py
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://mlflow.local:5000")
mlflow.set_experiment("dataset-tracking-demo")

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create MLflow datasets
train_dataset: PandasDataset = mlflow.data.from_pandas(
    train_data,
    source="synthetic_data",
    name="training_dataset",
    targets="target"
)

test_dataset: PandasDataset = mlflow.data.from_pandas(
    test_data,
    source="synthetic_data",
    name="test_dataset",
    targets="target"
)

with mlflow.start_run(run_name="dataset-tracking-run"):
    # Log datasets
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    # Dataset metadata is automatically logged:
    # - Schema
    # - Row count
    # - Digest (hash)
    # - Source information

    # Train model
    from sklearn.ensemble import RandomForestClassifier

    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

## Step 3: Model Registry (Complete Features)

### Model Registration and Lifecycle

```python
# mlflow/registry/model_lifecycle.py
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import time

mlflow.set_tracking_uri("http://mlflow.local:5000")
client = MlflowClient()

# Register a new model
def register_model_from_run(run_id: str, model_name: str, artifact_path: str = "model"):
    """Register model from a run"""
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Register model
    result = mlflow.register_model(model_uri, model_name)

    print(f"Model registered: {result.name} version {result.version}")
    return result

# Update model version description
def update_model_version(model_name: str, version: int, description: str):
    """Update model version metadata"""
    client.update_model_version(
        name=model_name,
        version=version,
        description=description
    )
    print(f"Updated {model_name} v{version}")

# Transition model to different stages
def transition_model_stage(model_name: str, version: int, stage: str):
    """
    Transition model to stage: None, Staging, Production, Archived
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production")  # Archive old production models
    )
    print(f"Transitioned {model_name} v{version} to {stage}")

# Add model version tags
def tag_model_version(model_name: str, version: int, tags: dict):
    """Add tags to model version"""
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    print(f"Tagged {model_name} v{version}")

# Example workflow
model_name = "churn-classifier"

# Get latest version
versions = client.search_model_versions(f"name='{model_name}'")
if versions:
    latest = max(versions, key=lambda x: int(x.version))

    # Update description
    update_model_version(
        model_name,
        int(latest.version),
        "Random Forest model for customer churn prediction. Trained on Q4 2024 data."
    )

    # Add tags
    tag_model_version(model_name, int(latest.version), {
        "validated": "true",
        "validation_accuracy": "0.92",
        "data_version": "2024-Q4",
        "approved_by": "ml-team"
    })

    # Transition to staging
    transition_model_stage(model_name, int(latest.version), "Staging")
```

### Model Aliases (MLflow 2.3+)

```python
# mlflow/registry/model_aliases.py
from mlflow import MlflowClient

client = MlflowClient()
model_name = "churn-classifier"

# Set aliases (alternative to stages)
def set_model_alias(model_name: str, alias: str, version: int):
    """Set alias for model version"""
    client.set_registered_model_alias(model_name, alias, version)
    print(f"Set alias '{alias}' for {model_name} v{version}")

def get_model_by_alias(model_name: str, alias: str):
    """Get model version by alias"""
    return client.get_model_version_by_alias(model_name, alias)

def delete_model_alias(model_name: str, alias: str):
    """Delete model alias"""
    client.delete_registered_model_alias(model_name, alias)
    print(f"Deleted alias '{alias}' from {model_name}")

# Example: Set up champion/challenger pattern
set_model_alias(model_name, "champion", 1)  # Current production model
set_model_alias(model_name, "challenger", 2)  # Model being tested

# Load model using alias
champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
challenger_model = mlflow.pyfunc.load_model(f"models:/{model_name}@challenger")
```

### Model Comparison

```python
# mlflow/registry/model_comparison.py
import mlflow
from mlflow import MlflowClient
import pandas as pd

client = MlflowClient()

def compare_model_versions(model_name: str, versions: list):
    """Compare multiple model versions"""
    comparison = []

    for version in versions:
        mv = client.get_model_version(model_name, str(version))
        run = client.get_run(mv.run_id)

        comparison.append({
            "version": version,
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "creation_time": mv.creation_timestamp,
            **{f"param_{k}": v for k, v in run.data.params.items()},
            **{f"metric_{k}": v for k, v in run.data.metrics.items()}
        })

    df = pd.DataFrame(comparison)
    print("\nModel Version Comparison:")
    print(df.to_string())
    return df

# Compare versions 1, 2, 3
compare_model_versions("churn-classifier", [1, 2, 3])
```

---

## Step 4: Model Signatures and Input Examples

```python
# mlflow/models/signatures.py
import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://mlflow.local:5000")
mlflow.set_experiment("model-signatures")

# Method 1: Infer signature from data
def log_model_with_inferred_signature():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    model = RandomForestClassifier()
    model.fit(X, y)

    # Infer signature
    signature = infer_signature(X, model.predict(X))

    with mlflow.start_run(run_name="inferred-signature"):
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X.head(5)
        )

# Method 2: Manually define signature
def log_model_with_manual_signature():
    from sklearn.ensemble import RandomForestClassifier

    # Define input schema
    input_schema = Schema([
        ColSpec("double", "sepal_length"),
        ColSpec("double", "sepal_width"),
        ColSpec("double", "petal_length"),
        ColSpec("double", "petal_width"),
    ])

    # Define output schema
    output_schema = Schema([ColSpec("long", "prediction")])

    # Create signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Create input example
    input_example = pd.DataFrame({
        "sepal_length": [5.1, 4.9],
        "sepal_width": [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width": [0.2, 0.2]
    })

    model = RandomForestClassifier()
    X = input_example
    y = [0, 0]
    model.fit(X, y)

    with mlflow.start_run(run_name="manual-signature"):
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example
        )

log_model_with_inferred_signature()
log_model_with_manual_signature()
```

---

## Step 5: Custom PyFunc Models

```python
# mlflow/models/custom_pyfunc.py
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Dict, Any

class ChurnPredictorWithPreprocessing(mlflow.pyfunc.PythonModel):
    """Custom model with preprocessing and postprocessing"""

    def __init__(self, model, scaler, feature_names, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold

    def load_context(self, context):
        """Load artifacts when model is loaded"""
        import pickle

        # Load any additional artifacts
        with open(context.artifacts["config"], "rb") as f:
            self.config = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with preprocessing"""

        # Validate input
        missing_cols = set(self.feature_names) - set(model_input.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Preprocess
        X = model_input[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Predict
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        # Return structured output
        return pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities,
            "risk_level": pd.cut(
                probabilities,
                bins=[0, 0.3, 0.7, 1.0],
                labels=["low", "medium", "high"]
            )
        })

# Train and log custom model
def train_custom_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    import pickle

    mlflow.set_tracking_uri("http://mlflow.local:5000")
    mlflow.set_experiment("custom-pyfunc-models")

    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Train components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Create custom model
    custom_model = ChurnPredictorWithPreprocessing(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        threshold=0.5
    )

    # Save config artifact
    config = {"version": "1.0", "threshold": 0.5}
    with open("/tmp/config.pkl", "wb") as f:
        pickle.dump(config, f)

    # Define conda environment
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {"pip": ["mlflow", "scikit-learn", "pandas", "numpy"]}
        ]
    }

    with mlflow.start_run(run_name="custom-pyfunc"):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=custom_model,
            artifacts={"config": "/tmp/config.pkl"},
            conda_env=conda_env,
            signature=mlflow.models.signature.infer_signature(
                X_df,
                pd.DataFrame({
                    "prediction": [0],
                    "probability": [0.5],
                    "risk_level": ["medium"]
                })
            ),
            input_example=X_df.head(5),
            registered_model_name="churn-predictor-custom"
        )

train_custom_model()
```

---

## Step 6: MLflow Projects

### Create MLflow Project

Create `mlflow/projects/churn_prediction/MLproject`:

```yaml
name: churn_prediction

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: string, default: "data/churn.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
      test_size: {type: float, default: 0.2}
    command: "python train.py --data-path {data_path} --n-estimators {n_estimators} --max-depth {max_depth} --test-size {test_size}"

  validate:
    parameters:
      model_uri: {type: string}
      data_path: {type: string}
    command: "python validate.py --model-uri {model_uri} --data-path {data_path}"

  serve:
    parameters:
      model_name: {type: string}
      port: {type: int, default: 5001}
    command: "mlflow models serve -m models:/{model_name}/Production -p {port}"
```

Create `mlflow/projects/churn_prediction/conda.yaml`:

```yaml
name: churn-prediction
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - mlflow>=2.9.0
    - scikit-learn>=1.3.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - click>=8.0.0
```

Create `mlflow/projects/churn_prediction/train.py`:

```python
#!/usr/bin/env python
import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

@click.command()
@click.option("--data-path", type=str, required=True)
@click.option("--n-estimators", type=int, default=100)
@click.option("--max-depth", type=int, default=10)
@click.option("--test-size", type=float, default=0.2)
def train(data_path, n_estimators, max_depth, test_size):
    """Train churn prediction model"""

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)

        # Load data
        df = pd.read_csv(data_path)
        X = df.drop("churn", axis=1)
        y = df["churn"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn-classifier"
        )

        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train()
```

### Run MLflow Project

```bash
# Run locally
mlflow run mlflow/projects/churn_prediction \
    -P data_path=data/churn.csv \
    -P n_estimators=200 \
    -P max_depth=15

# Run from Git
mlflow run https://github.com/your-repo/churn-prediction \
    -P data_path=s3://bucket/data/churn.csv

# Run on Kubernetes
mlflow run mlflow/projects/churn_prediction \
    --backend kubernetes \
    --backend-config kubernetes_config.json
```

---

## Step 7: Model Evaluation

```python
# mlflow/evaluation/model_evaluation.py
import mlflow
from mlflow.models import MetricThreshold
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://mlflow.local:5000")
mlflow.set_experiment("model-evaluation")

# Prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with mlflow.start_run(run_name="model-evaluation"):
    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Create evaluation dataset
    eval_data = X_test.copy()
    eval_data["target"] = y_test

    # Evaluate model
    result = mlflow.evaluate(
        model=model,
        data=eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={
            "log_model_explainability": True,
            "explainability_algorithm": "shap",
            "pos_label": 1
        }
    )

    # Print evaluation results
    print("Metrics:")
    for metric_name, metric_value in result.metrics.items():
        print(f"  {metric_name}: {metric_value}")

    # Access artifacts
    print("\nArtifacts:")
    for artifact_name, artifact_path in result.artifacts.items():
        print(f"  {artifact_name}: {artifact_path}")

# Model validation with thresholds
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.8,
        min_absolute_change=0.01,
        min_relative_change=0.01,
        greater_is_better=True
    ),
    "f1_score_weighted": MetricThreshold(
        threshold=0.75,
        greater_is_better=True
    )
}

# Validate against baseline
baseline_model_uri = "models:/iris-classifier/1"
candidate_model_uri = "models:/iris-classifier/2"

mlflow.evaluate(
    model=candidate_model_uri,
    data=eval_data,
    targets="target",
    model_type="classifier",
    baseline_model=baseline_model_uri,
    validation_thresholds=thresholds
)
```

---

## Step 8: Model Serving

### Serve Model Locally

```bash
# Serve model from registry
mlflow models serve \
    -m "models:/churn-classifier/Production" \
    -p 5001 \
    --env-manager conda

# Serve with custom config
mlflow models serve \
    -m "models:/churn-classifier@champion" \
    -p 5001 \
    --workers 4 \
    --timeout 60
```

### Serve Model in Kubernetes

Create `mlflow/serving/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-model-server
  namespace: mlflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlflow-model-server
  template:
    metadata:
      labels:
        app: mlflow-model-server
    spec:
      containers:
      - name: model-server
        image: ghcr.io/mlflow/mlflow:v2.9.2
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.mlflow.svc.cluster.local:5000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-minio
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-minio
              key: AWS_SECRET_ACCESS_KEY
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio.mlflow.svc.cluster.local:9000"
        command:
        - mlflow
        - models
        - serve
        - -m
        - models:/churn-classifier/Production
        - -p
        - "8080"
        - --workers
        - "2"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-model-server
  namespace: mlflow
spec:
  selector:
    app: mlflow-model-server
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlflow-model-server
  namespace: mlflow
spec:
  ingressClassName: nginx
  rules:
  - host: model.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlflow-model-server
            port:
              number: 8080
```

### Prediction API Usage

```python
# mlflow/serving/predict_client.py
import requests
import pandas as pd
import json

MODEL_SERVER_URL = "http://model.local/invocations"

def predict(data: pd.DataFrame) -> dict:
    """Make prediction request to MLflow model server"""

    # Format for split orientation
    payload = {
        "dataframe_split": {
            "columns": data.columns.tolist(),
            "data": data.values.tolist()
        }
    }

    response = requests.post(
        MODEL_SERVER_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

# Example usage
sample_data = pd.DataFrame({
    "feature_1": [1.5, 2.3],
    "feature_2": [0.8, 1.2],
    "feature_3": [3.2, 2.8]
})

predictions = predict(sample_data)
print(f"Predictions: {predictions}")
```

---

## Step 9: Multi-Tenancy Setup

```python
# mlflow/multi_tenancy/tenant_manager.py
import mlflow
from mlflow import MlflowClient
import os

class MLflowTenantManager:
    """Manage multi-tenant MLflow setup"""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri)

    def create_tenant_experiment(self, tenant_id: str, project_name: str) -> str:
        """Create experiment for tenant"""
        experiment_name = f"tenant/{tenant_id}/{project_name}"

        # Check if exists
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id

        # Create new experiment
        experiment_id = self.client.create_experiment(
            name=experiment_name,
            tags={
                "tenant_id": tenant_id,
                "project": project_name,
                "managed_by": "tenant_manager"
            }
        )

        return experiment_id

    def get_tenant_experiments(self, tenant_id: str) -> list:
        """Get all experiments for a tenant"""
        experiments = self.client.search_experiments(
            filter_string=f"tags.tenant_id = '{tenant_id}'"
        )
        return experiments

    def get_tenant_models(self, tenant_id: str) -> list:
        """Get all registered models for a tenant"""
        models = []
        for rm in self.client.search_registered_models():
            # Check tags
            tags = {tag.key: tag.value for tag in rm.tags}
            if tags.get("tenant_id") == tenant_id:
                models.append(rm)
        return models

# Usage
manager = MLflowTenantManager("http://mlflow.local:5000")

# Create tenant experiments
manager.create_tenant_experiment("team-a", "churn-prediction")
manager.create_tenant_experiment("team-b", "recommendation-engine")

# Get tenant's experiments
team_a_experiments = manager.get_tenant_experiments("team-a")
```

---

## Verification Script

```bash
#!/bin/bash
# verify_mlflow.sh

echo "=== MLflow Verification ==="

echo -e "\n1. MLflow Server Status:"
kubectl get pods -n mlflow -l app=mlflow

echo -e "\n2. MLflow Service:"
kubectl get svc -n mlflow

echo -e "\n3. Test MLflow Connection:"
curl -s http://mlflow.local/health

echo -e "\n4. List Experiments:"
curl -s http://mlflow.local/api/2.0/mlflow/experiments/search | jq '.experiments[].name'

echo -e "\n5. List Registered Models:"
curl -s http://mlflow.local/api/2.0/mlflow/registered-models/search | jq '.registered_models[].name'

echo -e "\n=== MLflow Verification Complete ==="
```

---

## Next Steps

- **Phase 03**: Apache Airflow Enterprise Setup
- **Phase 04**: Kubeflow Complete Setup

---

**Status**: Phase 02 Complete
**Features Covered**: All MLflow enterprise features
