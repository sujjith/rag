# uc_1_churn_prediction/tasks/model_training.py
"""Model training tasks using Kubeflow and MLflow."""

from prefect import task
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common.config import get_config


@task(name="Train Model Locally")
def train_model_local(data_path: str) -> str:
    """
    Step 6a: Train model locally with MLflow tracking.

    Use this for quick iteration. For production, use Kubeflow.

    Args:
        data_path: Path to training data

    Returns:
        str: MLflow run ID
    """
    cfg = get_config()
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment("churn-prediction")

    # Load data
    df = pd.read_parquet(data_path)

    # Prepare features
    feature_cols = [
        'age', 'tenure_months', 'total_purchases',
        'avg_order_value', 'days_since_last_purchase', 'support_tickets_count'
    ]
    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features", feature_cols)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "churn_model",
            registered_model_name="ChurnModel"
        )

        print(f"Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")

        return run.info.run_id


@task(name="Trigger Kubeflow Training")
def trigger_kubeflow_training(data_path: str) -> str:
    """
    Step 6b: Submit training pipeline to Kubeflow.

    Use this for production training with GPU.

    Args:
        data_path: S3 path to training data

    Returns:
        str: Kubeflow run ID
    """
    from kfp import Client

    cfg = get_config()
    client = Client(host=cfg['kubeflow']['host'])

    run = client.create_run_from_pipeline_package(
        pipeline_file="pipelines/churn_training.yaml",
        arguments={"data_path": data_path},
        run_name="churn-training-run"
    )

    print(f"Submitted Kubeflow run: {run.run_id}")
    return run.run_id


@task(name="Get Latest Model Version")
def get_latest_model_version(model_name: str = "ChurnModel") -> dict:
    """
    Get the latest registered model version from MLflow.

    Args:
        model_name: Name of registered model

    Returns:
        dict: Model version info with run_id and version
    """
    cfg = get_config()
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])

    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
    if not versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest = versions[-1]

    model_info = {
        "name": model_name,
        "version": latest.version,
        "run_id": latest.run_id,
        "source": latest.source,
        "status": latest.status
    }

    print(f"Latest model: {model_name} v{latest.version}")
    return model_info
