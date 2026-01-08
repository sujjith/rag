"""
Phase 6.1: MLflow Integration DAG

Demonstrates:
- MLflow tracking from Airflow
- Model training with experiment logging
- Model registry integration

Requirements:
- MLflow server running (see mlflow setup)
- Set MLFLOW_TRACKING_URI environment variable or Airflow connection

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="phase6_01_mlflow_integration",
    description="MLflow integration demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase6", "mlflow", "ml"],
    doc_md="""
    ## MLflow Integration DAG

    Integrates Airflow with MLflow for ML experiment tracking.

    **Prerequisites:**
    1. MLflow server running at localhost:5000
    2. Set MLflow tracking URI:
       - Via environment: `MLFLOW_TRACKING_URI=http://localhost:5000`
       - Or Airflow Variable: `mlflow_tracking_uri`

    **Features:**
    - Experiment tracking
    - Parameter logging
    - Metric logging
    - Model registration
    """,
)
def mlflow_integration():
    """ML Pipeline with MLflow tracking."""

    @task
    def prepare_data():
        """Prepare training data."""
        import numpy as np

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000

        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1) > 0

        # In production, save to shared storage
        # For demo, we'll regenerate in training task
        return {
            "n_samples": n_samples,
            "n_features": 10,
            "prepared": True,
        }

    @task
    def train_model(data_info: dict):
        """Train model with MLflow tracking."""
        import os

        # Get MLflow tracking URI
        # Priority: env var > Airflow variable > default
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

        try:
            import mlflow
            from mlflow.models.signature import infer_signature
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score

            # Set tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI: {tracking_uri}")

            # Create/set experiment
            experiment_name = "airflow_ml_pipeline"
            mlflow.set_experiment(experiment_name)

            # Generate data (same as prepare_data for demo)
            np.random.seed(42)
            X = np.random.randn(data_info["n_samples"], data_info["n_features"])
            y = (X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(data_info["n_samples"]) * 0.1) > 0

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model parameters
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42,
            }

            # Start MLflow run
            with mlflow.start_run(run_name="airflow_training"):
                # Log parameters
                mlflow.log_params(params)
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))

                # Train model
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                # Predictions and metrics
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                })

                # Log model
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name="airflow_rf_model",
                )

                run_id = mlflow.active_run().info.run_id
                print(f"Training complete! Run ID: {run_id}")
                print(f"Accuracy: {accuracy:.4f}")

                return {
                    "run_id": run_id,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }

        except ImportError as e:
            print(f"MLflow not installed: {e}")
            print("Install with: pip install mlflow scikit-learn")
            return {"error": str(e), "status": "mlflow_not_installed"}

        except Exception as e:
            print(f"Error during training: {e}")
            return {"error": str(e), "status": "failed"}

    @task
    def evaluate_model(training_result: dict):
        """Evaluate trained model and decide on deployment."""
        if "error" in training_result:
            print(f"Training failed: {training_result['error']}")
            return {"deploy": False, "reason": "training_failed"}

        accuracy = training_result.get("accuracy", 0)
        threshold = 0.7

        if accuracy >= threshold:
            print(f"Model passed threshold ({accuracy:.4f} >= {threshold})")
            return {
                "deploy": True,
                "run_id": training_result["run_id"],
                "accuracy": accuracy,
            }
        else:
            print(f"Model below threshold ({accuracy:.4f} < {threshold})")
            return {
                "deploy": False,
                "reason": "below_threshold",
                "accuracy": accuracy,
            }

    @task
    def register_model(evaluation_result: dict):
        """Register model in MLflow Model Registry if approved."""
        if not evaluation_result.get("deploy"):
            print(f"Model not deployed: {evaluation_result.get('reason')}")
            return {"registered": False}

        try:
            import mlflow
            import os

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)

            run_id = evaluation_result["run_id"]
            model_uri = f"runs:/{run_id}/model"

            # Model is already registered during training
            # Here we could transition the stage
            client = mlflow.tracking.MlflowClient()

            # Get latest version
            model_name = "airflow_rf_model"
            latest_versions = client.get_latest_versions(model_name)

            if latest_versions:
                latest_version = latest_versions[0].version
                print(f"Latest model version: {latest_version}")

                # Transition to staging (in production, use proper approval workflow)
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Staging",
                )
                print(f"Model {model_name} v{latest_version} transitioned to Staging")

                return {
                    "registered": True,
                    "model_name": model_name,
                    "version": latest_version,
                    "stage": "Staging",
                }

        except Exception as e:
            print(f"Registration error: {e}")
            return {"registered": False, "error": str(e)}

        return {"registered": False}

    # Pipeline flow
    data = prepare_data()
    training = train_model(data)
    evaluation = evaluate_model(training)
    register_model(evaluation)


mlflow_integration()
