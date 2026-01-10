"""
Phase 5.1: Automatic logging with autolog

This script demonstrates:
- Enabling autologging for sklearn
- Automatic parameter/metric/model logging
- GridSearchCV autologging

Run: python 01_autolog.py
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import os

warnings.filterwarnings("ignore")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase5-autolog")

print("=" * 60)
print("MLflow Autologging Demo")
print("=" * 60)

# Enable autologging for sklearn
mlflow.sklearn.autolog()
print("\nAutologging enabled for sklearn!")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Example 1: Simple model training
print("\n" + "-" * 60)
print("[Example 1: Simple Model Training]")
print("-" * 60)
print("Training RandomForestClassifier...")
print("(Parameters, metrics, and model will be logged automatically)")

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\nAccuracy: {accuracy:.4f}")
print("Check MLflow UI - everything was logged automatically!")

# Example 2: GridSearchCV
print("\n" + "-" * 60)
print("[Example 2: GridSearchCV]")
print("-" * 60)
print("Running GridSearchCV...")
print("(All CV results will be logged automatically)")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    return_train_score=True
)
grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
print("All CV results logged to MLflow!")

# Example 3: Multiple models
print("\n" + "-" * 60)
print("[Example 3: Multiple Models]")
print("-" * 60)

wine = load_wine()
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
]

for name, model in models:
    print(f"Training {name}...")
    model.fit(X_train_w, y_train_w)
    accuracy = model.score(X_test_w, y_test_w)
    print(f"  Accuracy: {accuracy:.4f}")

print("\nAll models logged automatically with autolog!")

# Example 4: Controlling autolog behavior
print("\n" + "-" * 60)
print("[Example 4: Autolog Configuration]")
print("-" * 60)

# You can configure autolog behavior
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_post_training_metrics=True,
    silent=False,  # Show autolog messages
    max_tuning_runs=5  # Limit GridSearchCV child runs
)

print("Autolog configured with custom settings:")
print("  - log_input_examples: True")
print("  - log_model_signatures: True")
print("  - max_tuning_runs: 5")

# Train one more model with custom config
model = RandomForestClassifier(n_estimators=75, max_depth=7, random_state=42)
model.fit(X_train, y_train)
print(f"\nFinal model accuracy: {model.score(X_test, y_test):.4f}")

# Disable autolog
mlflow.sklearn.autolog(disable=True)
print("\nAutolog disabled.")

print("\n" + "=" * 60)
print("Autologging demo complete!")
print(f"View experiments at: {TRACKING_URI}")
print("=" * 60)
