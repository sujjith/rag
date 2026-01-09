"""
Phase 2.3: Understanding model signatures

This script demonstrates:
- Automatic signature inference
- Manual signature definition
- Schema types

Run: python 03_signatures.py
"""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase2-signatures")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print("=" * 60)
print("Model Signatures Demo")
print("=" * 60)

# Method 1: Infer signature automatically
print("\n[Method 1: Inferred Signature]")
print("-" * 40)

inferred_signature = infer_signature(X, predictions)
print(f"Input Schema:\n{inferred_signature.inputs.to_dict()}")
print(f"\nOutput Schema:\n{inferred_signature.outputs.to_dict()}")

with mlflow.start_run(run_name="inferred-signature"):
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=inferred_signature
    )
    print("\nModel logged with inferred signature!")

# Method 2: Define signature manually with ColSpec
print("\n" + "=" * 60)
print("[Method 2: Manual Signature with ColSpec]")
print("-" * 40)

input_schema = Schema([
    ColSpec("double", "sepal length (cm)"),
    ColSpec("double", "sepal width (cm)"),
    ColSpec("double", "petal length (cm)"),
    ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long", "prediction")])
manual_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print(f"Input Schema:\n{manual_signature.inputs.to_dict()}")
print(f"\nOutput Schema:\n{manual_signature.outputs.to_dict()}")

with mlflow.start_run(run_name="manual-colspec-signature"):
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=manual_signature
    )
    print("\nModel logged with manual ColSpec signature!")

# Method 3: Signature with TensorSpec (for deep learning models)
print("\n" + "=" * 60)
print("[Method 3: TensorSpec Signature (for tensors)]")
print("-" * 40)

tensor_input_schema = Schema([
    TensorSpec(np.dtype("float64"), (-1, 4), "input_features")
])
tensor_output_schema = Schema([
    TensorSpec(np.dtype("int64"), (-1,), "predictions")
])
tensor_signature = ModelSignature(
    inputs=tensor_input_schema,
    outputs=tensor_output_schema
)

print(f"Input Schema (tensor):\n{tensor_signature.inputs.to_dict()}")
print(f"\nOutput Schema (tensor):\n{tensor_signature.outputs.to_dict()}")

with mlflow.start_run(run_name="tensor-signature"):
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=tensor_signature
    )
    print("\nModel logged with TensorSpec signature!")

# Demonstrate signature validation
print("\n" + "=" * 60)
print("[Signature Validation Demo]")
print("-" * 40)

# Load model with signature
run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
if run_id is None:
    # Get the last run
    runs = mlflow.search_runs(experiment_names=["phase2-signatures"])
    run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print("\nValid input (correct format):")
valid_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
print(f"Input shape: {valid_input.shape}")
try:
    result = loaded_model.predict(valid_input)
    print(f"Prediction: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nNumpy array input:")
numpy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
print(f"Input shape: {numpy_input.shape}")
try:
    result = loaded_model.predict(numpy_input)
    print(f"Prediction: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Signatures demo complete!")
print("Check the models in MLflow UI to see their signatures")
print("=" * 60)
