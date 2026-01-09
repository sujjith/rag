"""
Phase 1.5: Run multiple experiments with different hyperparameters

This script demonstrates:
- Running multiple experiments
- Comparing runs in MLflow UI
- Using sklearn with MLflow

Run: python 03_hyperparameter_search.py
"""
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase1-hyperparameter-search")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter grid
param_grid = [
    {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
]

print("=" * 70)
print("Hyperparameter Search")
print("=" * 70)
print(f"\nDataset: Iris")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Configurations to test: {len(param_grid)}")
print("\n" + "-" * 70)

results = []

for i, params in enumerate(param_grid, 1):
    with mlflow.start_run(run_name=f"config-{i}"):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("random_state", 42)

        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average="weighted")

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        # Add tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("config_id", i)

        results.append({
            "config": i,
            "params": params,
            "test_accuracy": test_acc
        })

        print(f"Config {i}: n_estimators={params['n_estimators']:3d}, "
              f"max_depth={str(params['max_depth']):4s}, "
              f"min_samples={params['min_samples_split']:2d} | "
              f"train={train_acc:.4f}, test={test_acc:.4f}")

print("-" * 70)

# Find best configuration
best = max(results, key=lambda x: x["test_accuracy"])
print(f"\nBest Configuration: Config {best['config']}")
print(f"  Parameters: {best['params']}")
print(f"  Test Accuracy: {best['test_accuracy']:.4f}")

print("\n" + "=" * 70)
print(f"Compare runs in MLflow UI: {TRACKING_URI}")
print("Use the 'Compare' button to see differences!")
print("=" * 70)
