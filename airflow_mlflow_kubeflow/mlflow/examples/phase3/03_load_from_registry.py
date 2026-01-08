"""
Phase 3.3: Loading models from the registry

This script demonstrates:
- Loading models by version number
- Loading models by stage name
- Getting model metadata

Prerequisites: Run 02_model_stages.py first

Run: python 03_load_from_registry.py
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

# Check if model exists
try:
    model_info = client.get_registered_model(MODEL_NAME)
    print(f"\nModel: {model_info.name}")
    print(f"Description: {model_info.description or 'No description'}")
except:
    print(f"\nERROR: Model '{MODEL_NAME}' not found!")
    print("Please run 02_model_stages.py first.")
    exit(1)

print("\nTest data (first 5 samples):")
print(X_test.to_string(index=False))
print(f"\nActual labels: {list(y_test)}")
print(f"Class names: {[iris.target_names[i] for i in y_test]}")

# Method 1: Load by version number
print("\n" + "=" * 60)
print("[Method 1: Load by Version Number]")
print("-" * 40)

for version in ["1", "2", "3"]:
    model_uri = f"models:/{MODEL_NAME}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(X_test)
        print(f"\nVersion {version}:")
        print(f"  URI: {model_uri}")
        print(f"  Predictions: {list(predictions)}")
    except Exception as e:
        print(f"\nVersion {version}: Failed to load - {e}")

# Method 2: Load by stage name
print("\n" + "=" * 60)
print("[Method 2: Load by Stage Name]")
print("-" * 40)

stages = ["None", "Staging", "Production", "Archived"]

for stage in stages:
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(X_test)
        print(f"\n{stage}:")
        print(f"  URI: {model_uri}")
        print(f"  Predictions: {list(predictions)}")
    except Exception as e:
        print(f"\n{stage}: No model in this stage")

# Method 3: Load latest version in each stage
print("\n" + "=" * 60)
print("[Method 3: Get Latest Versions by Stage]")
print("-" * 40)

for stage in ["Staging", "Production"]:
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if versions:
        latest = versions[0]
        print(f"\nLatest in {stage}:")
        print(f"  Version: {latest.version}")
        print(f"  Run ID: {latest.run_id[:8]}...")

        # Load and predict
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest.version}")
        predictions = model.predict(X_test)
        print(f"  Predictions: {list(predictions)}")
    else:
        print(f"\nNo versions in {stage}")

# Method 4: Get detailed metadata
print("\n" + "=" * 60)
print("[Method 4: Model Metadata]")
print("-" * 40)

for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"\nVersion {mv.version}:")
    print(f"  Stage: {mv.current_stage}")
    print(f"  Status: {mv.status}")
    print(f"  Run ID: {mv.run_id}")
    print(f"  Source: {mv.source[:50]}...")
    print(f"  Created: {mv.creation_timestamp}")

    # Get run metrics
    run = client.get_run(mv.run_id)
    accuracy = run.data.metrics.get("accuracy", "N/A")
    print(f"  Accuracy: {accuracy}")

# Best practice: Load production model for inference
print("\n" + "=" * 60)
print("[Best Practice: Production Model Loading]")
print("-" * 40)

try:
    # This is how you'd load in production code
    production_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

    print("\nProduction model loaded successfully!")
    print(f"Making predictions...")

    predictions = production_model.predict(X_test)
    class_names = [iris.target_names[p] for p in predictions]

    print("\nResults:")
    for i, (pred, actual) in enumerate(zip(class_names, y_test)):
        actual_name = iris.target_names[actual]
        match = "✓" if pred == actual_name else "✗"
        print(f"  Sample {i+1}: Predicted={pred:12s} Actual={actual_name:12s} {match}")

except Exception as e:
    print(f"Failed to load production model: {e}")

print("\n" + "=" * 60)
print("Model loading demo complete!")
print("=" * 60)
