"""
MLflow Project - Hyperparameter Search

Runs multiple training experiments with different hyperparameters.

Usage:
  mlflow run . -e hyperparameter_search -P n_trials=10
"""
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("MLflow Project - Hyperparameter Search")
    print("=" * 60)
    print(f"\nRunning {args.n_trials} trials...")

    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter ranges
    n_estimators_range = [10, 50, 100, 150, 200]
    max_depth_range = [3, 5, 10, 15, None]
    min_samples_split_range = [2, 5, 10]

    # Track results
    results = []

    # Start parent run
    with mlflow.start_run(run_name="hyperparameter-search") as parent_run:
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("search_type", "random")

        print("\n" + "-" * 60)

        for trial in range(args.n_trials):
            # Random hyperparameters
            n_estimators = random.choice(n_estimators_range)
            max_depth = random.choice(max_depth_range)
            min_samples_split = random.choice(min_samples_split_range)

            # Child run for each trial
            with mlflow.start_run(run_name=f"trial-{trial+1}", nested=True):
                # Log parameters
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)

                # Train model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Evaluate
                accuracy = accuracy_score(y_test, model.predict(X_test))
                mlflow.log_metric("accuracy", accuracy)

                results.append({
                    "trial": trial + 1,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "accuracy": accuracy
                })

                print(f"Trial {trial+1:2d}: n_est={n_estimators:3d}, "
                      f"depth={str(max_depth):4s}, "
                      f"min_split={min_samples_split:2d} | "
                      f"accuracy={accuracy:.4f}")

        print("-" * 60)

        # Find best trial
        best = max(results, key=lambda x: x["accuracy"])
        print(f"\nBest Trial: #{best['trial']}")
        print(f"  n_estimators: {best['n_estimators']}")
        print(f"  max_depth: {best['max_depth']}")
        print(f"  min_samples_split: {best['min_samples_split']}")
        print(f"  accuracy: {best['accuracy']:.4f}")

        # Log best results to parent run
        mlflow.log_metric("best_accuracy", best["accuracy"])
        mlflow.log_param("best_n_estimators", best["n_estimators"])
        mlflow.log_param("best_max_depth", best["max_depth"])
        mlflow.log_param("best_min_samples_split", best["min_samples_split"])

        # Train final model with best params
        print("\nTraining final model with best parameters...")
        final_model = RandomForestClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            min_samples_split=best["min_samples_split"],
            random_state=42
        )
        final_model.fit(X_train, y_train)

        signature = mlflow.models.infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, "best_model", signature=signature)

        # Save results as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("search_results.csv", index=False)
        mlflow.log_artifact("search_results.csv")

        print(f"\nParent Run ID: {parent_run.info.run_id}")

    print("\n" + "=" * 60)
    print("Hyperparameter search complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
