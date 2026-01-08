"""
Phase 3.2: Managing model versions and stages

This script demonstrates:
- Creating multiple model versions
- Model stages: None, Staging, Production, Archived
- Transitioning between stages

Run: python 02_model_stages.py
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Model Stages Demo")
print("=" * 60)

# Delete existing model if it exists (for clean demo)
try:
    client.delete_registered_model(MODEL_NAME)
    print(f"\nDeleted existing model: {MODEL_NAME}")
except:
    pass

# Create multiple model versions with different configs
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 10},
]

print("\n[1] Creating multiple model versions...")
print("-" * 50)

versions_info = []

for i, config in enumerate(configs, 1):
    with mlflow.start_run(run_name=f"version-{i}"):
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)

        # Register model (creates new version each time)
        result = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        versions_info.append({
            "version": i,
            "config": config,
            "accuracy": accuracy
        })

        print(f"  Version {i}: n_estimators={config['n_estimators']:3d}, "
              f"max_depth={config['max_depth']:2d} | accuracy={accuracy:.4f}")

        time.sleep(1)  # Small delay to ensure registry updates

print("-" * 50)

# Find best model
best = max(versions_info, key=lambda x: x["accuracy"])
print(f"\nBest model: Version {best['version']} (accuracy: {best['accuracy']:.4f})")

# Stage transitions
print("\n" + "=" * 60)
print("[2] Managing Model Stages")
print("=" * 60)

print("\nAvailable stages: None -> Staging -> Production -> Archived")

# Transition models to appropriate stages
for info in versions_info:
    version = str(info["version"])

    if info["version"] == best["version"]:
        # Best model goes to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"\n  Version {version} -> Production (best accuracy)")
    elif info["accuracy"] > 0.9:
        # Good models go to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )
        print(f"  Version {version} -> Staging (good accuracy)")
    else:
        # Keep in None stage
        print(f"  Version {version} -> None (needs improvement)")

# List current state
print("\n" + "=" * 60)
print("[3] Current Model Registry State")
print("=" * 60)

for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"\n  Version {mv.version}:")
    print(f"    Stage: {mv.current_stage}")
    print(f"    Run ID: {mv.run_id[:8]}...")
    print(f"    Status: {mv.status}")

# Demonstrate archiving
print("\n" + "=" * 60)
print("[4] Archiving Old Versions")
print("=" * 60)

# Archive version 1 (oldest)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version="1",
    stage="Archived",
    archive_existing_versions=False
)
print("  Version 1 -> Archived")

# Show final state
print("\n" + "=" * 60)
print("Final Registry State")
print("=" * 60)

print(f"\nModel: {MODEL_NAME}")
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"  Version {mv.version}: {mv.current_stage}")

print("\n" + "=" * 60)
print(f"View at: http://localhost:5000/#/models/{MODEL_NAME}")
print("=" * 60)
