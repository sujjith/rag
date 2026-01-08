"""
MLflow Project - Evaluation Script

Evaluates a model from a previous run.

Usage:
  mlflow run . -e evaluate -P run_id=<run_id>
"""
import argparse
import mlflow
import mlflow.pyfunc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    print("=" * 50)
    print("MLflow Project - Evaluation")
    print("=" * 50)
    print(f"\nEvaluating run: {args.run_id}")

    # Load model
    model_uri = f"runs:/{args.run_id}/model"
    print(f"Loading model from: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"ERROR: Could not load model - {e}")
        return

    # Load test data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n" + "-" * 50)
    print("Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("-" * 50)
    print("Confusion Matrix:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm,
        index=iris.target_names,
        columns=iris.target_names
    ))

    # Log evaluation as new run
    with mlflow.start_run():
        mlflow.log_param("evaluated_run_id", args.run_id)
        mlflow.log_metric("eval_accuracy", (y_pred == y_test).mean())

        # Log confusion matrix as artifact
        cm_df = pd.DataFrame(
            cm,
            index=iris.target_names,
            columns=iris.target_names
        )
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")

        print(f"\nEvaluation logged to run: {mlflow.active_run().info.run_id}")

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
