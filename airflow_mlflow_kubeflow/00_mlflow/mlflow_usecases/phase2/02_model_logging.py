"""
Phase 2.2: Logging and loading ML models

This script demonstrates:
- Logging sklearn models with mlflow.sklearn
- Model signatures
- Input examples
- Loading models back

Run: python 02_model_logging.py
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase2-model-logging")

# Load data with feature names
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Model Logging Demo")
print("=" * 60)
print(f"\nDataset: Iris")
print(f"Features: {list(iris.feature_names)}")
print(f"Classes: {list(iris.target_names)}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

with mlflow.start_run(run_name="model-logging-demo") as run:
    print("\n[1] Training model...")

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"    Accuracy: {accuracy:.4f}")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)

    print("\n[2] Creating model signature...")

    # Infer signature from training data
    signature = mlflow.models.infer_signature(
        X_train,
        model.predict(X_train)
    )

    print(f"    Input schema: {signature.inputs}")
    print(f"    Output schema: {signature.outputs}")

    print("\n[3] Logging model with signature and input example...")

    # Log model with all metadata
    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        signature=signature,
        input_example=X_train.iloc[:3]  # First 3 rows as example
    )

    print(f"    Model logged!")
    print(f"    Run ID: {run.info.run_id}")

# Load the model back
print("\n" + "-" * 60)
print("Loading model back from MLflow")
print("-" * 60)

model_uri = f"runs:/{run.info.run_id}/random_forest_model"
print(f"\n[4] Loading model from: {model_uri}")

loaded_model = mlflow.sklearn.load_model(model_uri)
print(f"    Model type: {type(loaded_model).__name__}")

print("\n[5] Making predictions with loaded model...")

# Test predictions
sample_data = X_test.iloc[:5]
predictions = loaded_model.predict(sample_data)

print("\n    Sample Predictions:")
print("    " + "-" * 50)
for i in range(5):
    actual = iris.target_names[y_test.iloc[i]]
    predicted = iris.target_names[predictions[i]]
    match = "✓" if actual == predicted else "✗"
    print(f"    {i+1}. Predicted: {predicted:12s} | Actual: {actual:12s} {match}")

# Also test pyfunc interface
print("\n[6] Testing pyfunc interface...")
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
pyfunc_predictions = pyfunc_model.predict(sample_data)
print(f"    Pyfunc predictions: {pyfunc_predictions}")

print("\n" + "=" * 60)
print("Model logging and loading complete!")
print(f"View at: {TRACKING_URI}/#/experiments")
print("=" * 60)
