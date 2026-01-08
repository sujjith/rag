"""
MLflow Basic Tracking Example

This script demonstrates:
- Setting up MLflow tracking
- Creating experiments
- Logging parameters, metrics, and artifacts
- Training a simple model
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification"


def main():
    # Set tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow Tracking URI: {TRACKING_URI}")

    # Create or get experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}")

    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameters to try
    param_grid = [
        {"n_estimators": 10, "max_depth": 3},
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
    ]

    print("\nRunning experiments...")
    print("-" * 50)

    for params in param_grid:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_param("random_state", 42)

            # Train model
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=X_train.iloc[:1],
            )

            # Create and log feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            importance = pd.Series(
                model.feature_importances_, index=iris.feature_names
            ).sort_values(ascending=True)
            importance.plot(kind="barh", ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")

            # Save plot as artifact
            plot_path = "feature_importance.png"
            fig.savefig(plot_path, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            plt.close()
            os.remove(plot_path)

            # Log dataset info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features", len(iris.feature_names))

            # Add tags
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("dataset", "iris")

            run_id = mlflow.active_run().info.run_id
            print(
                f"Run {run_id[:8]}... | "
                f"n_estimators={params['n_estimators']:3d}, "
                f"max_depth={params['max_depth']:2d} | "
                f"accuracy={accuracy:.4f}"
            )

    print("-" * 50)
    print(f"\nExperiments logged successfully!")
    print(f"View results at: {TRACKING_URI}")


if __name__ == "__main__":
    main()
