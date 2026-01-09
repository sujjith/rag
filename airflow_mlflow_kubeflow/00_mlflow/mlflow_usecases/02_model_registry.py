"""
MLflow Model Registry Example

This script demonstrates:
- Registering models in the Model Registry
- Managing model versions
- Transitioning models between stages
- Loading models from the registry
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import time

# Configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-classifier"


def train_and_log_model(X_train, y_train, X_test, y_test, params):
    """Train a model and log it to MLflow."""
    with mlflow.start_run() as run:
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:1],
        )

        return run.info.run_id, accuracy


def register_model(run_id, model_name, description=None):
    """Register a model from a run."""
    model_uri = f"runs:/{run_id}/model"

    # Register model
    result = mlflow.register_model(model_uri, model_name)

    print(f"Registered model '{model_name}' version {result.version}")

    # Add description if provided
    if description:
        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=description,
        )

    return result.version


def transition_model_stage(model_name, version, stage):
    """Transition a model version to a new stage."""
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    print(f"Model '{model_name}' version {version} transitioned to {stage}")


def load_model_from_registry(model_name, stage="Production"):
    """Load a model from the registry by stage."""
    model_uri = f"models:/{model_name}/{stage}"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model '{model_name}' from {stage} stage")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def list_registered_models():
    """List all registered models."""
    client = MlflowClient()

    print("\n" + "=" * 60)
    print("Registered Models")
    print("=" * 60)

    for rm in client.search_registered_models():
        print(f"\nModel: {rm.name}")
        print(f"  Description: {rm.description or 'N/A'}")
        print(f"  Latest Versions:")

        for version in rm.latest_versions:
            print(f"    - Version {version.version}: {version.current_stage}")


def main():
    # Set tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow Tracking URI: {TRACKING_URI}")

    # Create or get experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and split data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Train multiple model versions
    model_configs = [
        {"n_estimators": 50, "max_depth": 5, "random_state": 42},
        {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        {"n_estimators": 150, "max_depth": 15, "random_state": 42},
    ]

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    best_run_id = None
    best_accuracy = 0
    best_version = None

    for i, params in enumerate(model_configs, 1):
        print(f"\nTraining model {i}/{len(model_configs)}...")
        run_id, accuracy = train_and_log_model(
            X_train, y_train, X_test, y_test, params
        )

        print(f"  Run ID: {run_id}")
        print(f"  Accuracy: {accuracy:.4f}")

        # Register model
        version = register_model(
            run_id,
            MODEL_NAME,
            description=f"RandomForest with n_estimators={params['n_estimators']}, max_depth={params['max_depth']}",
        )

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id
            best_version = version

        # Wait a bit for the registry to update
        time.sleep(1)

    # Transition stages
    print("\n" + "=" * 60)
    print("Managing Model Stages")
    print("=" * 60)

    client = MlflowClient()

    # Get all versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    for version in versions:
        if version.version == best_version:
            # Best model goes to Production
            transition_model_stage(MODEL_NAME, version.version, "Production")
        else:
            # Others go to Staging
            transition_model_stage(MODEL_NAME, version.version, "Staging")

    # List all registered models
    list_registered_models()

    # Load and test production model
    print("\n" + "=" * 60)
    print("Testing Production Model")
    print("=" * 60)

    production_model = load_model_from_registry(MODEL_NAME, "Production")

    if production_model:
        # Make predictions
        predictions = production_model.predict(X_test[:5])
        print(f"\nSample predictions: {predictions}")
        print(f"Actual values:      {y_test[:5]}")

    print("\n" + "=" * 60)
    print("Model Registry Demo Complete")
    print("=" * 60)
    print(f"\nView models at: {TRACKING_URI}/#/models")


if __name__ == "__main__":
    main()
