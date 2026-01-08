"""
Phase 4.1: Prepare a model for serving

This script demonstrates:
- Training a production-ready model
- Registering with proper signature
- Promoting to Production stage

Prerequisites: MLflow server running (./scripts/start.sh)

Run: python 01_prepare_model.py
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase4-serving")

MODEL_NAME = "iris-serving-model"
client = MlflowClient()

# Clean up existing model
try:
    client.delete_registered_model(MODEL_NAME)
    print(f"Cleaned up existing model: {MODEL_NAME}")
except:
    pass

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Preparing Model for Serving")
print("=" * 60)

print(f"\nDataset: Iris")
print(f"Features: {list(X.columns)}")
print(f"Classes: {list(iris.target_names)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Train model
print("\n[1] Training model...")
print("-" * 40)

with mlflow.start_run(run_name="serving-model") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    for line in report.split('\n'):
        print(f"  {line}")

    # Log metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)

    # Create signature
    print("\n[2] Creating model signature...")
    print("-" * 40)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    print(f"  Input schema: {signature.inputs}")
    print(f"  Output schema: {signature.outputs}")

    # Log model with all metadata
    print("\n[3] Logging model...")
    print("-" * 40)

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train.iloc[:2],
        registered_model_name=MODEL_NAME
    )

    print(f"  Model logged and registered!")
    print(f"  Run ID: {run.info.run_id}")

# Promote to Production
print("\n[4] Promoting to Production...")
print("-" * 40)

import time
time.sleep(2)  # Wait for registration

latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Production"
)

print(f"  Version {latest.version} -> Production")

# Verify model is ready
print("\n[5] Verifying model...")
print("-" * 40)

prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
test_predictions = prod_model.predict(X_test[:3])
print(f"  Test predictions: {list(test_predictions)}")
print(f"  Expected: {list(y_test[:3])}")
print(f"  Model is ready for serving!")

# Summary
print("\n" + "=" * 60)
print("Model Ready for Serving")
print("=" * 60)
print(f"\n  Model name: {MODEL_NAME}")
print(f"  Version: {latest.version}")
print(f"  Stage: Production")
print(f"  Accuracy: {accuracy:.4f}")
print(f"\n  To serve this model, run:")
print(f"    ./scripts/serve-model.sh {MODEL_NAME} Production")
print(f"\n  Or use mlflow CLI:")
print(f"    mlflow models serve -m 'models:/{MODEL_NAME}/Production' -p 5001 --no-conda")
print("=" * 60)
