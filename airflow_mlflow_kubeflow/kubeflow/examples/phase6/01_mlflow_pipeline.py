"""
Phase 6.1: Kubeflow Pipeline with MLflow Tracking

This pipeline trains a model and logs everything to MLflow.

Prerequisites:
- MLflow server running (from mlflow/ folder)
- MLflow accessible from Kubeflow cluster

Run:
  python 01_mlflow_pipeline.py
  # Upload mlflow_tracking_pipeline.yaml to Kubeflow
"""
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "mlflow==2.9.2",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "boto3==1.34.0"
    ]
)
def train_and_log_to_mlflow(
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int,
    max_depth: int,
    metrics: Output[Metrics]
) -> str:
    """
    Train a model and log to MLflow tracking server.
    Returns the MLflow run_id.
    """
    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    import pandas as pd

    print("=" * 50)
    print("Training with MLflow Tracking")
    print("=" * 50)
    print(f"MLflow URI: {mlflow_tracking_uri}")
    print(f"Experiment: {experiment_name}")

    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("source", "kubeflow_pipeline")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train model
        print(f"\nTraining RandomForest (n={n_estimators}, depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log metrics to Kubeflow
        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("f1_score", f1)

        # Log model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:2]
        )

        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        print("=" * 50)

        return run.info.run_id


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.9.2", "boto3==1.34.0"]
)
def register_model(
    mlflow_tracking_uri: str,
    run_id: str,
    model_name: str
) -> str:
    """Register the model in MLflow Model Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    print(f"Registering model from run {run_id}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    print(f"Registered as: {model_name} v{result.version}")

    return str(result.version)


@dsl.pipeline(
    name="mlflow-tracking-pipeline",
    description="Train model and log to MLflow tracking server"
)
def mlflow_tracking_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-server.mlflow.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-experiments",
    model_name: str = "kubeflow-iris-model",
    n_estimators: int = 100,
    max_depth: int = 10
):
    """
    Pipeline that:
    1. Trains a model and logs to MLflow
    2. Registers the model in MLflow Registry
    """
    # Train and log
    train_task = train_and_log_to_mlflow(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # Register model
    register_task = register_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        run_id=train_task.output,
        model_name=model_name
    )


if __name__ == "__main__":
    output_file = "mlflow_tracking_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=mlflow_tracking_pipeline,
        package_path=output_file
    )

    print("=" * 50)
    print("Pipeline compiled!")
    print("=" * 50)
    print(f"\nOutput: {output_file}")
    print("\nParameters:")
    print("  - mlflow_tracking_uri: MLflow server URL")
    print("  - experiment_name: MLflow experiment name")
    print("  - model_name: Name for registered model")
    print("  - n_estimators: Number of trees")
    print("  - max_depth: Max tree depth")
    print("\nNote: Ensure MLflow is accessible from Kubeflow cluster")
    print("=" * 50)
