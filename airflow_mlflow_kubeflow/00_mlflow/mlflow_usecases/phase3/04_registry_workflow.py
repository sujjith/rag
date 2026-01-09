"""
Phase 3.4: Complete model registry workflow

This script demonstrates a real-world ML workflow:
1. Train model
2. Register in registry
3. Promote to Staging
4. Test in Staging
5. Promote to Production
6. Train improved model
7. Compare and promote

Run: python 04_registry_workflow.py
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
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase3-workflow")

MODEL_NAME = "iris-production-workflow"
client = MlflowClient()

# Helper functions
def train_model(params, run_name):
    """Train and log a model, return run_id and accuracy."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=run_name) as run:
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


def promote_to_stage(model_name, version, stage, archive_existing=False):
    """Promote a model version to a stage."""
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing
    )


def load_and_test(model_name, stage):
    """Load model from stage and run tests."""
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

    # Test data
    iris = load_iris()
    X_test = pd.DataFrame(iris.data[:10], columns=iris.feature_names)
    y_test = iris.target[:10]

    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()

    return accuracy


# Clean up existing model
try:
    client.delete_registered_model(MODEL_NAME)
except:
    pass

print("=" * 70)
print("ML Model Registry Workflow")
print("=" * 70)

# Step 1: Train initial model
print("\n[Step 1] Training initial model...")
print("-" * 50)

params_v1 = {"n_estimators": 50, "max_depth": 5}
run_id_v1, accuracy_v1 = train_model(params_v1, "initial-model")
print(f"  Parameters: {params_v1}")
print(f"  Accuracy: {accuracy_v1:.4f}")

# Step 2: Register model
print("\n[Step 2] Registering model in registry...")
print("-" * 50)

version_v1 = register_model(run_id_v1, MODEL_NAME)
print(f"  Registered as: {MODEL_NAME}")
print(f"  Version: {version_v1}")
time.sleep(1)

# Step 3: Promote to Staging
print("\n[Step 3] Promoting to Staging for testing...")
print("-" * 50)

promote_to_stage(MODEL_NAME, version_v1, "Staging")
print(f"  Version {version_v1} -> Staging")

# Step 4: Test in Staging
print("\n[Step 4] Running tests in Staging...")
print("-" * 50)

staging_accuracy = load_and_test(MODEL_NAME, "Staging")
print(f"  Staging test accuracy: {staging_accuracy:.4f}")

if staging_accuracy >= 0.8:
    print("  Tests PASSED!")
else:
    print("  Tests FAILED!")

# Step 5: Promote to Production
print("\n[Step 5] Promoting to Production...")
print("-" * 50)

promote_to_stage(MODEL_NAME, version_v1, "Production")
print(f"  Version {version_v1} -> Production")

# Step 6: Train improved model
print("\n[Step 6] Training improved model...")
print("-" * 50)

params_v2 = {"n_estimators": 100, "max_depth": 10}
run_id_v2, accuracy_v2 = train_model(params_v2, "improved-model")
print(f"  Parameters: {params_v2}")
print(f"  Accuracy: {accuracy_v2:.4f}")

# Step 7: Register new version
print("\n[Step 7] Registering improved model...")
print("-" * 50)

version_v2 = register_model(run_id_v2, MODEL_NAME)
print(f"  Version: {version_v2}")
time.sleep(1)

# Step 8: Compare models
print("\n[Step 8] Comparing models...")
print("-" * 50)
print(f"  Production (v{version_v1}): accuracy = {accuracy_v1:.4f}")
print(f"  Candidate (v{version_v2}):  accuracy = {accuracy_v2:.4f}")

improvement = accuracy_v2 - accuracy_v1
print(f"  Improvement: {improvement:+.4f}")

# Step 9: Promote new version if better
print("\n[Step 9] Making promotion decision...")
print("-" * 50)

if accuracy_v2 > accuracy_v1:
    print("  New model is better!")

    # First to Staging
    promote_to_stage(MODEL_NAME, version_v2, "Staging")
    print(f"  Version {version_v2} -> Staging")

    # Test in Staging
    staging_accuracy = load_and_test(MODEL_NAME, "Staging")
    print(f"  Staging tests passed (accuracy: {staging_accuracy:.4f})")

    # Promote to Production (archives old production)
    promote_to_stage(MODEL_NAME, version_v2, "Production", archive_existing=True)
    print(f"  Version {version_v2} -> Production")
    print(f"  Version {version_v1} -> Archived (automatically)")
else:
    print("  New model is NOT better. Keeping current Production model.")
    promote_to_stage(MODEL_NAME, version_v2, "Archived")
    print(f"  Version {version_v2} -> Archived")

# Final state
print("\n" + "=" * 70)
print("Final Registry State")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    run = client.get_run(mv.run_id)
    accuracy = run.data.metrics.get("accuracy", "N/A")
    print(f"  Version {mv.version}: {mv.current_stage:12s} (accuracy: {accuracy:.4f})")

# Load current production model
print("\n" + "-" * 50)
prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
print("Production model ready for inference!")

print("\n" + "=" * 70)
print(f"View at: {TRACKING_URI}/#/models/{MODEL_NAME}")
print("=" * 70)
