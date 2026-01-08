"""
Phase 3.1: Registering models in the Model Registry

This script demonstrates:
- Training and logging a model
- Registering the model in MLflow Model Registry
- Adding descriptions

Run: python 01_register_model.py
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
client = MlflowClient()

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Model Registration Demo")
print("=" * 60)

# Train and log model
print("\n[1] Training model...")
with mlflow.start_run(run_name="registry-demo") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"    Run ID: {run.info.run_id}")
    print(f"    Accuracy: {accuracy:.4f}")

# Register the model
print("\n[2] Registering model...")
model_uri = f"runs:/{run.info.run_id}/model"

# Method 1: Using mlflow.register_model
result = mlflow.register_model(model_uri, MODEL_NAME)
print(f"    Registered: {result.name}")
print(f"    Version: {result.version}")

# Add description to the registered model
print("\n[3] Adding descriptions...")
client.update_registered_model(
    name=MODEL_NAME,
    description="Iris flower species classifier using Random Forest algorithm. "
                "Predicts setosa, versicolor, or virginica based on sepal/petal measurements."
)

client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description=f"Initial version trained with 100 estimators. Accuracy: {accuracy:.4f}"
)
print("    Descriptions added!")

# Alternative: Register during logging
print("\n[4] Alternative: Register during logging...")
with mlflow.start_run(run_name="auto-register-demo"):
    model2 = RandomForestClassifier(n_estimators=150, random_state=42)
    model2.fit(X_train, y_train)

    accuracy2 = accuracy_score(y_test, model2.predict(X_test))
    mlflow.log_metric("accuracy", accuracy2)

    # This automatically registers the model
    mlflow.sklearn.log_model(
        model2,
        "model",
        registered_model_name=MODEL_NAME  # Auto-registers!
    )
    print(f"    Model logged and registered automatically!")
    print(f"    Accuracy: {accuracy2:.4f}")

# Show registered models
print("\n" + "=" * 60)
print("Registered Models")
print("=" * 60)

for rm in client.search_registered_models():
    print(f"\nModel: {rm.name}")
    print(f"Description: {rm.description[:80]}..." if rm.description else "No description")

    for version in rm.latest_versions:
        print(f"  Version {version.version}: {version.current_stage}")

print("\n" + "=" * 60)
print(f"View at: http://localhost:5000/#/models/{MODEL_NAME}")
print("=" * 60)
