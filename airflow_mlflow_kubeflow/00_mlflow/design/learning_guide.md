# MLflow Learning Guide - Phased Implementation

## Overview

This guide breaks down MLflow into 5 learning phases, each building on the previous one.

| Phase | Topic | Duration | Outcome |
|-------|-------|----------|---------|
| 1 | Setup & Basic Tracking | Beginner | Run experiments, log metrics |
| 2 | Artifacts & Models | Beginner | Save/load models, log files |
| 3 | Model Registry | Intermediate | Version control for models |
| 4 | Model Serving | Intermediate | Deploy models as REST APIs |
| 5 | Advanced Features | Advanced | Custom flavors, plugins, autolog |

---

# Phase 1: Setup & Basic Tracking

## Objectives
- Install and run MLflow locally
- Understand experiments and runs
- Log parameters and metrics
- Use the MLflow UI

## 1.1 Installation

```bash
# Create virtual environment
cd /home/sujith/github-sujith/rag/airflow_mlflow_kubeflow/mlflow
python -m venv venv
source venv/bin/activate

# Install MLflow
pip install mlflow==3.8.1 scikit-learn pandas matplotlib
```

## 1.2 Start MLflow Tracking Server (Simple Mode)

```bash
# Start with local file storage (simplest setup)
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Open browser: http://localhost:5000

## 1.3 Your First Experiment

Create `phase1/01_hello_mlflow.py`:

```python
"""
Phase 1.3: Your first MLflow experiment
"""
import mlflow

# Connect to tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create an experiment
mlflow.set_experiment("phase1-hello-mlflow")

# Start a run
with mlflow.start_run():
    # Log a parameter (input configuration)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)

    # Log metrics (output measurements)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.15)

    # Add tags (metadata)
    mlflow.set_tag("author", "sujith")
    mlflow.set_tag("version", "v1")

    print("Run logged successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

Run it:
```bash
python phase1/01_hello_mlflow.py
```

## 1.4 Logging Metrics Over Time

Create `phase1/02_training_simulation.py`:

```python
"""
Phase 1.4: Simulate training with metrics over epochs
"""
import mlflow
import random

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase1-training-simulation")

with mlflow.start_run(run_name="simulated-training"):
    # Log parameters
    mlflow.log_param("model_type", "neural_network")
    mlflow.log_param("hidden_layers", 3)
    mlflow.log_param("learning_rate", 0.001)

    # Simulate training loop
    epochs = 20
    accuracy = 0.5
    loss = 1.0

    for epoch in range(epochs):
        # Simulate improvement
        accuracy += random.uniform(0.01, 0.03)
        loss -= random.uniform(0.02, 0.05)

        # Clamp values
        accuracy = min(accuracy, 0.99)
        loss = max(loss, 0.01)

        # Log metrics with step
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch)

        print(f"Epoch {epoch+1}/{epochs}: accuracy={accuracy:.4f}, loss={loss:.4f}")

    # Log final metrics
    mlflow.log_metric("final_accuracy", accuracy)
    mlflow.log_metric("final_loss", loss)

print("\nCheck the MLflow UI to see the metric charts!")
```

## 1.5 Comparing Multiple Runs

Create `phase1/03_hyperparameter_search.py`:

```python
"""
Phase 1.5: Run multiple experiments with different hyperparameters
"""
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase1-hyperparameter-search")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = [
    {"n_estimators": 10, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

print("Running hyperparameter search...")
print("-" * 50)

for params in param_grid:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        print(f"n_estimators={params['n_estimators']:3d}, "
              f"max_depth={str(params['max_depth']):4s} | "
              f"train={train_acc:.4f}, test={test_acc:.4f}")

print("-" * 50)
print("\nCompare runs in MLflow UI: http://localhost:5000")
print("Use the 'Compare' button to see differences!")
```

## 1.6 Exercise

1. Run all three scripts
2. Open MLflow UI and explore:
   - View experiment list
   - Click on individual runs
   - Compare multiple runs
   - View metric charts
3. Try modifying parameters and running again

## Phase 1 Checklist

- [ ] MLflow server running locally
- [ ] Created first experiment
- [ ] Logged parameters and metrics
- [ ] Logged metrics with steps (for charts)
- [ ] Compared multiple runs in UI

---

# Phase 2: Artifacts & Model Logging

## Objectives
- Log files and artifacts
- Save and load ML models
- Understand model signatures
- Use different model flavors (sklearn, pytorch, etc.)

## 2.1 Logging Artifacts

Create `phase2/01_artifacts.py`:

```python
"""
Phase 2.1: Logging artifacts (files)
"""
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-artifacts")

with mlflow.start_run(run_name="artifact-demo"):
    # Create a temporary directory for artifacts
    os.makedirs("temp_artifacts", exist_ok=True)

    # 1. Log a plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("Sample Plot")
    plot_path = "temp_artifacts/plot.png"
    fig.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)
    print("Logged: plot.png")

    # 2. Log a CSV file
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0]
    })
    csv_path = "temp_artifacts/data_sample.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    print("Logged: data_sample.csv")

    # 3. Log a JSON config
    config = {
        "model": "RandomForest",
        "preprocessing": ["scale", "normalize"],
        "features": ["feature1", "feature2"]
    }
    json_path = "temp_artifacts/config.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact(json_path)
    print("Logged: config.json")

    # 4. Log a text file
    text_path = "temp_artifacts/notes.txt"
    with open(text_path, "w") as f:
        f.write("Training notes:\n")
        f.write("- Used default hyperparameters\n")
        f.write("- Data was clean, no preprocessing needed\n")
    mlflow.log_artifact(text_path)
    print("Logged: notes.txt")

    # 5. Log multiple files in a directory
    os.makedirs("temp_artifacts/reports", exist_ok=True)
    for i in range(3):
        with open(f"temp_artifacts/reports/report_{i}.txt", "w") as f:
            f.write(f"Report {i} content")
    mlflow.log_artifacts("temp_artifacts/reports", artifact_path="reports")
    print("Logged: reports/ directory")

    # Clean up
    import shutil
    shutil.rmtree("temp_artifacts")

    print("\nArtifacts logged! Check the MLflow UI.")
```

## 2.2 Logging ML Models

Create `phase2/02_model_logging.py`:

```python
"""
Phase 2.2: Logging and loading ML models
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-model-logging")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="model-logging-demo") as run:
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature (input/output schema)
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

    # Log model with input example
    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        signature=signature,
        input_example=X_train.iloc[:3]
    )

    print(f"Model logged!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# Load the model back
print("\n--- Loading model back ---")
model_uri = f"runs:/{run.info.run_id}/random_forest_model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Make predictions with loaded model
predictions = loaded_model.predict(X_test[:5])
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test[:5]}")
```

## 2.3 Model Signatures

Create `phase2/03_signatures.py`:

```python
"""
Phase 2.3: Understanding model signatures
"""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-signatures")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

# Method 1: Infer signature automatically
print("Method 1: Inferred Signature")
inferred_signature = infer_signature(X, predictions)
print(f"Inputs:  {inferred_signature.inputs}")
print(f"Outputs: {inferred_signature.outputs}")

with mlflow.start_run(run_name="inferred-signature"):
    mlflow.sklearn.log_model(model, "model", signature=inferred_signature)
    print("Model logged with inferred signature\n")

# Method 2: Define signature manually
print("Method 2: Manual Signature")
input_schema = Schema([
    ColSpec("double", "sepal length (cm)"),
    ColSpec("double", "sepal width (cm)"),
    ColSpec("double", "petal length (cm)"),
    ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long", "prediction")])
manual_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print(f"Inputs:  {manual_signature.inputs}")
print(f"Outputs: {manual_signature.outputs}")

with mlflow.start_run(run_name="manual-signature"):
    mlflow.sklearn.log_model(model, "model", signature=manual_signature)
    print("Model logged with manual signature\n")

print("Check both models in MLflow UI to see their signatures!")
```

## 2.4 Different Model Flavors

Create `phase2/04_model_flavors.py`:

```python
"""
Phase 2.4: Different model flavors (sklearn, xgboost, custom)
"""
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-model-flavors")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Dictionary of models to try
models = {
    "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
}

print("Logging different model types...")
print("-" * 50)

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X, y)

        # Log model using sklearn flavor
        signature = mlflow.models.infer_signature(X, model.predict(X))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Log model type as tag
        mlflow.set_tag("model_type", type(model).__name__)

        print(f"Logged: {name} ({type(model).__name__})")

print("-" * 50)

# Custom PyFunc model example
print("\nLogging custom PyFunc model...")

class CustomModel(mlflow.pyfunc.PythonModel):
    """Custom model that wraps any sklearn classifier with preprocessing."""

    def __init__(self, classifier):
        self.classifier = classifier

    def predict(self, context, model_input):
        # Add custom preprocessing here if needed
        predictions = self.classifier.predict(model_input)
        # Add custom postprocessing here if needed
        return predictions

# Create and log custom model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
custom_model = CustomModel(rf)

with mlflow.start_run(run_name="custom-pyfunc"):
    mlflow.pyfunc.log_model(
        "model",
        python_model=custom_model,
        signature=mlflow.models.infer_signature(X, rf.predict(X))
    )
    mlflow.set_tag("model_type", "CustomPyFunc")
    print("Logged: custom-pyfunc (CustomModel)")

print("\nAll models logged! Check MLflow UI.")
```

## Phase 2 Checklist

- [ ] Logged various artifact types (plots, CSV, JSON)
- [ ] Logged and loaded sklearn models
- [ ] Understand model signatures
- [ ] Logged different model flavors

---

# Phase 3: Model Registry

## Objectives
- Register models in the registry
- Manage model versions
- Transition models between stages
- Load models by name/stage

## 3.1 Registering Models

Create `phase3/01_register_model.py`:

```python
"""
Phase 3.1: Registering models in the Model Registry
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase3-model-registry")

MODEL_NAME = "iris-classifier"

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and log model
with mlflow.start_run(run_name="registry-demo") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# Method 1: Register using mlflow.register_model
print("\n--- Registering model ---")
model_uri = f"runs:/{run.info.run_id}/model"
result = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered model: {result.name}")
print(f"Version: {result.version}")

# Method 2: Register during logging (alternative)
# with mlflow.start_run():
#     mlflow.sklearn.log_model(
#         model, "model",
#         registered_model_name="iris-classifier"  # Auto-registers
#     )

# Add description to the model
client = MlflowClient()
client.update_registered_model(
    name=MODEL_NAME,
    description="Iris flower classifier using Random Forest"
)

client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description="Initial version with 100 estimators"
)

print(f"\nModel registered! View at: http://localhost:5000/#/models/{MODEL_NAME}")
```

## 3.2 Model Versions and Stages

Create `phase3/02_model_stages.py`:

```python
"""
Phase 3.2: Managing model versions and stages
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase3-model-stages")

MODEL_NAME = "iris-classifier-staged"
client = MlflowClient()

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create multiple model versions
configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15},
]

print("Creating multiple model versions...")
print("-" * 50)

versions = []
for config in configs:
    with mlflow.start_run():
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)

        # Register model (creates new version each time)
        result = mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=MODEL_NAME
        )

        versions.append({
            "version": len(versions) + 1,
            "accuracy": accuracy,
            "config": config
        })

        print(f"Version {len(versions)}: accuracy={accuracy:.4f}, config={config}")
        time.sleep(1)  # Wait for registry to update

print("-" * 50)

# Find best model
best = max(versions, key=lambda x: x["accuracy"])
print(f"\nBest model: Version {best['version']} (accuracy={best['accuracy']:.4f})")

# Transition stages
print("\n--- Managing Stages ---")

# Stage transitions: None -> Staging -> Production -> Archived
for v in versions:
    version_num = str(v["version"])

    if v["version"] == best["version"]:
        # Best model goes to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_num,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"Version {version_num} -> Production")
    else:
        # Others go to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_num,
            stage="Staging",
            archive_existing_versions=False
        )
        print(f"Version {version_num} -> Staging")

# List all versions and their stages
print("\n--- Current Model Versions ---")
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"Version {mv.version}: {mv.current_stage}")

print(f"\nView stages at: http://localhost:5000/#/models/{MODEL_NAME}")
```

## 3.3 Loading Models from Registry

Create `phase3/03_load_from_registry.py`:

```python
"""
Phase 3.3: Loading models from the registry
"""
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "iris-classifier-staged"
client = MlflowClient()

# Load test data
iris = load_iris()
X_test = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
y_test = iris.target[:5]

print("=" * 60)
print("Loading Models from Registry")
print("=" * 60)

# Method 1: Load by version number
print("\n[Method 1: Load by version]")
model_v1 = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
predictions = model_v1.predict(X_test)
print(f"Model URI: models:/{MODEL_NAME}/1")
print(f"Predictions: {predictions}")

# Method 2: Load by stage
print("\n[Method 2: Load by stage - Production]")
try:
    model_prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    predictions = model_prod.predict(X_test)
    print(f"Model URI: models:/{MODEL_NAME}/Production")
    print(f"Predictions: {predictions}")
except Exception as e:
    print(f"No Production model found: {e}")

print("\n[Method 3: Load by stage - Staging]")
try:
    model_staging = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
    predictions = model_staging.predict(X_test)
    print(f"Model URI: models:/{MODEL_NAME}/Staging")
    print(f"Predictions: {predictions}")
except Exception as e:
    print(f"No Staging model found: {e}")

# Method 4: Load latest version
print("\n[Method 4: Load latest version]")
latest_versions = client.get_latest_versions(MODEL_NAME)
if latest_versions:
    latest = latest_versions[0]
    model_latest = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    predictions = model_latest.predict(X_test)
    print(f"Latest version: {latest.version} ({latest.current_stage})")
    print(f"Predictions: {predictions}")

# Get model metadata
print("\n" + "=" * 60)
print("Model Metadata")
print("=" * 60)

for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"\nVersion {mv.version}:")
    print(f"  Stage: {mv.current_stage}")
    print(f"  Run ID: {mv.run_id}")
    print(f"  Created: {mv.creation_timestamp}")
    print(f"  Description: {mv.description or 'N/A'}")
```

## 3.4 Model Registry Workflow

Create `phase3/04_registry_workflow.py`:

```python
"""
Phase 3.4: Complete model registry workflow
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "iris-production-workflow"

def train_model(params):
    """Train and log a model."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        return run.info.run_id, accuracy

def register_model(run_id, model_name):
    """Register a model from a run."""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    return result.version

def promote_to_staging(model_name, version):
    """Promote a model version to Staging."""
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Staging"
    )

def promote_to_production(model_name, version):
    """Promote a model version to Production (archives existing)."""
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True
    )

def get_production_model(model_name):
    """Load the current production model."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

def archive_model(model_name, version):
    """Archive an old model version."""
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Archived"
    )

# Workflow demonstration
print("=" * 60)
print("Model Registry Workflow Demo")
print("=" * 60)

mlflow.set_experiment("phase3-workflow")

# Step 1: Train initial model
print("\n[Step 1] Training initial model...")
run_id, accuracy = train_model({"n_estimators": 50, "max_depth": 5})
print(f"Accuracy: {accuracy:.4f}")

# Step 2: Register model
print("\n[Step 2] Registering model...")
version = register_model(run_id, MODEL_NAME)
print(f"Registered as version {version}")

# Step 3: Promote to Staging
print("\n[Step 3] Promoting to Staging...")
promote_to_staging(MODEL_NAME, version)
print(f"Version {version} is now in Staging")

# Step 4: Test in Staging (simulated)
print("\n[Step 4] Testing in Staging...")
staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
print("Staging tests passed!")

# Step 5: Promote to Production
print("\n[Step 5] Promoting to Production...")
promote_to_production(MODEL_NAME, version)
print(f"Version {version} is now in Production")

# Step 6: Train improved model
print("\n[Step 6] Training improved model...")
run_id2, accuracy2 = train_model({"n_estimators": 100, "max_depth": 10})
print(f"Accuracy: {accuracy2:.4f}")

# Step 7: Register and promote new version
print("\n[Step 7] Registering and promoting new version...")
version2 = register_model(run_id2, MODEL_NAME)
promote_to_staging(MODEL_NAME, version2)
print(f"Version {version2} is in Staging")

# Step 8: Compare models
print("\n[Step 8] Comparing models...")
print(f"  Production (v{version}): accuracy = {accuracy:.4f}")
print(f"  Staging (v{version2}):   accuracy = {accuracy2:.4f}")

if accuracy2 > accuracy:
    print("\n[Step 9] New model is better, promoting to Production...")
    promote_to_production(MODEL_NAME, version2)
    print(f"Version {version2} is now in Production")
    print(f"Version {version} has been archived")
else:
    print("\n[Step 9] Keeping current Production model")

# Final state
print("\n" + "=" * 60)
print("Final Model Registry State")
print("=" * 60)
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"Version {mv.version}: {mv.current_stage}")
```

## Phase 3 Checklist

- [ ] Registered models in the registry
- [ ] Created multiple model versions
- [ ] Transitioned models between stages
- [ ] Loaded models by version and stage
- [ ] Understand the full registry workflow

---

# Phase 4: Model Serving

## Objectives
- Serve models as REST APIs
- Make predictions via HTTP
- Understand input formats
- Deploy with Docker

## 4.1 Start the Full Platform

Now we'll use the Docker Compose setup:

```bash
cd /home/sujith/github-sujith/rag/airflow_mlflow_kubeflow/mlflow
./scripts/start.sh
```

## 4.2 Prepare a Model for Serving

Create `phase4/01_prepare_model.py`:

```python
"""
Phase 4.2: Prepare a model for serving
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase4-serving")

MODEL_NAME = "iris-serving-model"
client = MlflowClient()

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model for serving...")

with mlflow.start_run(run_name="serving-model") as run:
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature and input example
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:2],
        registered_model_name=MODEL_NAME
    )

    print(f"Model logged and registered!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# Promote to Production
print("\nPromoting to Production...")
latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Production"
)

print(f"\nModel ready for serving!")
print(f"Model name: {MODEL_NAME}")
print(f"Version: {latest.version}")
print(f"Stage: Production")
print(f"\nTo serve this model:")
print(f"  ./scripts/serve-model.sh {MODEL_NAME} Production")
```

## 4.3 Serve the Model

```bash
# Start the model server
./scripts/serve-model.sh iris-serving-model Production
```

## 4.4 Test the Serving Endpoint

Create `phase4/02_test_serving.py`:

```python
"""
Phase 4.4: Test model serving endpoint
"""
import requests
import json
import pandas as pd
from sklearn.datasets import load_iris

MODEL_SERVER = "http://localhost:5001"
ENDPOINT = f"{MODEL_SERVER}/invocations"

# Load sample data
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Sample inputs
samples = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.2, 2.9, 4.3, 1.3],  # versicolor
    [7.7, 3.0, 6.1, 2.3],  # virginica
]

print("=" * 60)
print("Testing MLflow Model Serving")
print("=" * 60)

# Check server health
print("\n[Health Check]")
try:
    response = requests.get(f"{MODEL_SERVER}/health", timeout=5)
    print(f"Server status: {'OK' if response.ok else 'FAILED'}")
except Exception as e:
    print(f"Server not reachable: {e}")
    print("Start the server with: ./scripts/serve-model.sh iris-serving-model Production")
    exit(1)

# Test 1: Simple inputs format
print("\n[Test 1: inputs format]")
payload = {"inputs": samples}
response = requests.post(ENDPOINT, json=payload)
print(f"Request: {json.dumps(payload)}")
print(f"Response: {response.json()}")

# Test 2: DataFrame split format
print("\n[Test 2: dataframe_split format]")
df = pd.DataFrame(samples, columns=feature_names)
payload = {
    "dataframe_split": {
        "columns": df.columns.tolist(),
        "data": df.values.tolist()
    }
}
response = requests.post(ENDPOINT, json=payload)
print(f"Response: {response.json()}")

# Test 3: Single prediction
print("\n[Test 3: Single prediction]")
payload = {"inputs": [[5.1, 3.5, 1.4, 0.2]]}
response = requests.post(ENDPOINT, json=payload)
prediction = response.json()["predictions"][0]
print(f"Input: {payload['inputs'][0]}")
print(f"Prediction: {prediction} ({target_names[prediction]})")

# Test 4: Batch prediction
print("\n[Test 4: Batch prediction (100 samples)]")
import numpy as np
np.random.seed(42)
batch = np.random.uniform(
    low=[4.0, 2.0, 1.0, 0.1],
    high=[8.0, 4.5, 7.0, 2.5],
    size=(100, 4)
).tolist()

import time
start = time.time()
response = requests.post(ENDPOINT, json={"inputs": batch})
elapsed = time.time() - start

predictions = response.json()["predictions"]
print(f"Predictions: {len(predictions)}")
print(f"Time: {elapsed:.3f}s")
print(f"Throughput: {100/elapsed:.1f} predictions/sec")

# Distribution
from collections import Counter
dist = Counter(predictions)
print("Distribution:")
for class_id, count in sorted(dist.items()):
    print(f"  {target_names[class_id]}: {count}")

print("\n" + "=" * 60)
print("Serving tests completed!")
print("=" * 60)
```

## 4.5 cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'

# Multiple predictions
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}'

# DataFrame format
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
      "data": [[5.1, 3.5, 1.4, 0.2]]
    }
  }'
```

## Phase 4 Checklist

- [ ] Started MLflow with Docker Compose
- [ ] Prepared and registered a model
- [ ] Served the model via REST API
- [ ] Tested predictions via Python and cURL
- [ ] Understand different input formats

---

# Phase 5: Advanced Features

## Objectives
- Use autologging
- Work with model flavors
- Use MLflow Projects
- Understand plugins and extensions

## 5.1 Autologging

Create `phase5/01_autolog.py`:

```python
"""
Phase 5.1: Automatic logging with autolog
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase5-autolog")

# Enable autologging for sklearn
mlflow.sklearn.autolog()

print("=" * 60)
print("MLflow Autologging Demo")
print("=" * 60)

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Example 1: Simple model training (auto-logged)
print("\n[Example 1: Simple training]")
print("Training RandomForest... (automatically logged)")

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("Check MLflow UI - parameters, metrics, and model auto-logged!")

# Example 2: Grid Search (auto-logged)
print("\n[Example 2: GridSearchCV]")
print("Running grid search... (automatically logged)")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
print("All CV results logged to MLflow!")

# Example 3: Different model
print("\n[Example 3: Logistic Regression]")
wine = load_wine()
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_w, y_train_w)
print("Logistic Regression logged automatically!")

# Disable autologging
mlflow.sklearn.autolog(disable=True)

print("\n" + "=" * 60)
print("Autologging demo complete!")
print("Check experiments at: http://localhost:5000")
print("=" * 60)
```

## 5.2 Custom Metrics and Evaluations

Create `phase5/02_custom_metrics.py`:

```python
"""
Phase 5.2: Custom metrics and model evaluation
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase5-custom-metrics")

# Disable autolog to show manual logging
mlflow.sklearn.autolog(disable=True)

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print("Custom Metrics and Evaluation Demo")
print("=" * 60)

with mlflow.start_run(run_name="comprehensive-evaluation"):
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 1. Basic metrics
    print("\n[Basic Metrics]")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"  {name}: {value:.4f}")

    # 2. Per-class metrics
    print("\n[Per-class Metrics]")
    for i, class_name in enumerate(iris.target_names):
        precision = precision_score(y_test, y_pred, labels=[i], average="micro")
        recall = recall_score(y_test, y_pred, labels=[i], average="micro")
        f1 = f1_score(y_test, y_pred, labels=[i], average="micro")

        mlflow.log_metric(f"precision_{class_name}", precision)
        mlflow.log_metric(f"recall_{class_name}", recall)
        mlflow.log_metric(f"f1_{class_name}", f1)

        print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # 3. Cross-validation scores
    print("\n[Cross-validation]")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 4. Confusion matrix plot
    print("\n[Artifacts]")
    os.makedirs("temp", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.savefig("temp/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("temp/confusion_matrix.png")
    print("  Logged: confusion_matrix.png")

    # 5. Feature importance plot
    importance = pd.Series(model.feature_importances_, index=iris.feature_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.savefig("temp/feature_importance.png", bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("temp/feature_importance.png")
    print("  Logged: feature_importance.png")

    # 6. Classification report as JSON
    report = classification_report(y_test, y_pred,
                                   target_names=iris.target_names,
                                   output_dict=True)
    with open("temp/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("temp/classification_report.json")
    print("  Logged: classification_report.json")

    # 7. Log model
    mlflow.sklearn.log_model(model, "model")
    print("  Logged: model")

    # Clean up
    import shutil
    shutil.rmtree("temp")

print("\n" + "=" * 60)
print("Custom metrics demo complete!")
print("=" * 60)
```

## 5.3 MLflow Projects

Create `phase5/03_mlproject/MLproject`:

```yaml
name: iris-training

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth}"

  evaluate:
    parameters:
      model_uri: {type: string}
    command: "python evaluate.py --model-uri {model_uri}"
```

Create `phase5/03_mlproject/conda.yaml`:

```yaml
name: iris-training
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
    - mlflow==3.8.1
    - scikit-learn==1.3.2
    - pandas==2.1.4
```

Create `phase5/03_mlproject/train.py`:

```python
"""MLflow Project training script"""
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
```

Run the project:
```bash
# Run with default parameters
mlflow run phase5/03_mlproject

# Run with custom parameters
mlflow run phase5/03_mlproject -P n_estimators=200 -P max_depth=10
```

## Phase 5 Checklist

- [ ] Used autologging
- [ ] Logged custom metrics and artifacts
- [ ] Created confusion matrix and feature importance plots
- [ ] Understand MLflow Projects structure

---

# Summary

## Learning Path Complete!

| Phase | Topic | Key Skills |
|-------|-------|------------|
| 1 | Setup & Tracking | Experiments, runs, parameters, metrics |
| 2 | Artifacts & Models | File logging, model saving, signatures |
| 3 | Model Registry | Versioning, stages, promotion workflow |
| 4 | Model Serving | REST API, Docker, predictions |
| 5 | Advanced | Autolog, custom metrics, Projects |

## Next Steps

1. **Practice**: Run all examples multiple times
2. **Experiment**: Modify parameters, try different models
3. **Explore UI**: Spend time navigating MLflow's web interface
4. **Move to Kubeflow**: Apply MLflow knowledge in Kubeflow pipelines

## Quick Reference

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Start with Docker Compose
./scripts/start.sh

# Set tracking URI in Python
mlflow.set_tracking_uri("http://localhost:5000")

# Basic logging
with mlflow.start_run():
    mlflow.log_param("key", value)
    mlflow.log_metric("metric", value)
    mlflow.sklearn.log_model(model, "model")

# Register model
mlflow.register_model("runs:/<run_id>/model", "model-name")

# Load model
model = mlflow.pyfunc.load_model("models:/model-name/Production")

# Serve model
mlflow models serve -m "models:/model-name/Production" -p 5001
```
