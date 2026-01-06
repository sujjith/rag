# Phase 02: ML Experiment Lifecycle

**Duration**: 3 weeks | **Prerequisites**: Phase 01 completed

---

## Learning Objectives

By the end of this phase, you will:
- [ ] Track experiments systematically with MLflow
- [ ] Version datasets and models with DVC
- [ ] Build reproducible ML pipelines with Prefect
- [ ] Compare experiment runs visually

---

## Week 1: Experiment Tracking with MLflow

### Day 1-2: MLflow Setup

```bash
# Install MLflow
uv add mlflow

# Start tracking server
uv run mlflow server --host 0.0.0.0 --port 5000

# Access UI at http://localhost:5000
```

### Day 3-4: Logging Experiments

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start run
with mlflow.start_run(run_name="random-forest-v1"):
    # Log parameters
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy:.4f}")
```

### Day 5-7: Model Registry

```python
# Register model
mlflow.register_model(
    f"runs:/{run_id}/model",
    "iris-classifier"
)

# Transition to production
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/iris-classifier/Production")
```

**Hands-on Exercise:**
1. Run 5 experiments with different parameters
2. Compare runs in MLflow UI
3. Register best model
4. Load and use production model

---

## Week 2: Data Versioning with DVC

### Day 8-9: DVC Setup

```bash
# Install DVC
uv add dvc dvc-s3  # or dvc-gdrive, dvc-azure

# Initialize DVC
dvc init

# Configure remote storage (local for learning)
mkdir -p /tmp/dvc-storage
dvc remote add -d myremote /tmp/dvc-storage
```

### Day 10-11: Tracking Data

```bash
# Add data file to DVC
dvc add data/raw/dataset.csv

# This creates:
# - data/raw/dataset.csv.dvc (pointer file)
# - .gitignore update

# Commit to git
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Add dataset v1"

# Push data to remote
dvc push
```

### Day 12-14: Data Pipelines with DVC

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw/dataset.csv
      - src/prepare.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false
```

```bash
# Run pipeline
dvc repro

# Check metrics
dvc metrics show

# Compare with previous
dvc metrics diff
```

**Hands-on Exercise:**
1. Version 3 different versions of dataset
2. Switch between versions with `dvc checkout`
3. Build 3-stage pipeline (prepare, train, evaluate)

---

## Week 3: Pipeline Orchestration with Prefect

### Day 15-16: Prefect Basics

```bash
# Install Prefect
uv add prefect

# Start Prefect server (optional, for UI)
uv run prefect server start
```

```python
from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split

@task
def load_data(path: str) -> pd.DataFrame:
    """Load dataset from path."""
    return pd.read_csv(path)

@task
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

@task
def split_data(df: pd.DataFrame):
    """Split into train/test."""
    train, test = train_test_split(df, test_size=0.2)
    return train, test

@flow(name="data-preparation")
def prepare_data_flow(input_path: str, output_dir: str):
    """Main data preparation flow."""
    # Load
    df = load_data(input_path)
    
    # Preprocess
    df_clean = preprocess(df)
    
    # Split
    train, test = split_data(df_clean)
    
    # Save
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    
    return {"train_size": len(train), "test_size": len(test)}

# Run flow
if __name__ == "__main__":
    result = prepare_data_flow("data/raw/data.csv", "data/processed")
    print(result)
```

### Day 17-18: Advanced Prefect Features

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

# Caching
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def expensive_computation(data):
    # This result will be cached
    return process(data)

# Retries
@task(retries=3, retry_delay_seconds=10)
def unreliable_api_call():
    return call_external_api()

# Concurrency
@flow
def parallel_flow():
    futures = [process_item.submit(item) for item in items]
    results = [f.result() for f in futures]
    return results
```

### Day 19-21: ML Pipeline with Prefect

```python
from prefect import flow, task
import mlflow

@task
def train_model(X_train, y_train, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model = train(X_train, y_train, **params)
        mlflow.sklearn.log_model(model, "model")
    return model

@task
def evaluate_model(model, X_test, y_test):
    metrics = evaluate(model, X_test, y_test)
    mlflow.log_metrics(metrics)
    return metrics

@flow(name="ml-training-pipeline")
def training_pipeline(data_path: str, params: dict):
    # Load and prepare
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    # Train
    model = train_model(X_train, y_train, params)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    return metrics
```

**Hands-on Exercise:**
1. Build complete ML pipeline
2. Add caching and retries
3. Schedule daily runs
4. View runs in Prefect UI

---

## Milestone Checklist

- [ ] MLflow server running
- [ ] Logged 5+ experiments
- [ ] Model registered in MLflow
- [ ] DVC initialized with remote
- [ ] Data versioned with DVC
- [ ] DVC pipeline working
- [ ] Prefect flow created
- [ ] Complete ML pipeline with all tools

---

## Integration Project

Build an end-to-end pipeline that:
1. Loads data (tracked by DVC)
2. Runs preprocessing (Prefect task)
3. Trains model (MLflow tracking)
4. Evaluates and registers (MLflow registry)
5. All orchestrated by Prefect

---

**Next Phase**: [Phase 03 - Model Serving & APIs](./phase_03_model_serving.md)
