"""
Phase 2.4: Different model flavors

This script demonstrates:
- sklearn flavor
- pyfunc flavor (custom models)
- Loading models with different interfaces

Run: python 04_model_flavors.py
"""
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-model-flavors")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("=" * 60)
print("Model Flavors Demo")
print("=" * 60)

# Part 1: Different sklearn models
print("\n[Part 1: Sklearn Models]")
print("-" * 40)

models = {
    "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X, y)

        # Log model using sklearn flavor
        signature = mlflow.models.infer_signature(X, model.predict(X))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Add metadata
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("flavor", "sklearn")

        accuracy = (model.predict(X) == y).mean()
        mlflow.log_metric("train_accuracy", accuracy)

        print(f"  Logged: {name} (accuracy: {accuracy:.4f})")

# Part 2: Sklearn Pipeline
print("\n[Part 2: Sklearn Pipeline]")
print("-" * 40)

with mlflow.start_run(run_name="sklearn-pipeline"):
    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    pipeline.fit(X, y)

    signature = mlflow.models.infer_signature(X, pipeline.predict(X))
    mlflow.sklearn.log_model(pipeline, "model", signature=signature)
    mlflow.set_tag("model_type", "Pipeline")

    accuracy = (pipeline.predict(X) == y).mean()
    mlflow.log_metric("train_accuracy", accuracy)
    print(f"  Logged: sklearn-pipeline (accuracy: {accuracy:.4f})")

# Part 3: Custom PyFunc Model
print("\n[Part 3: Custom PyFunc Model]")
print("-" * 40)


class PreprocessingModel(mlflow.pyfunc.PythonModel):
    """
    Custom model that includes preprocessing and postprocessing.
    """

    def __init__(self, classifier, class_names):
        self.classifier = classifier
        self.class_names = class_names
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit the scaler and classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, context, model_input):
        """
        Predict with preprocessing and return class names.
        """
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = np.array(model_input)

        # Preprocess
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.classifier.predict(X_scaled)

        # Return class names instead of indices
        return [self.class_names[p] for p in predictions]


# Create and train custom model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
custom_model = PreprocessingModel(rf, list(iris.target_names))
custom_model.fit(X, y)

with mlflow.start_run(run_name="custom-pyfunc"):
    # Log custom model
    mlflow.pyfunc.log_model(
        "model",
        python_model=custom_model,
        signature=mlflow.models.infer_signature(
            X,
            custom_model.predict(None, X)
        )
    )
    mlflow.set_tag("model_type", "CustomPyFunc")
    mlflow.set_tag("flavor", "pyfunc")
    print("  Logged: custom-pyfunc")

# Part 4: Test loading different flavors
print("\n" + "=" * 60)
print("[Loading and Testing Models]")
print("-" * 40)

# Get recent runs
runs = mlflow.search_runs(
    experiment_names=["phase2-model-flavors"],
    max_results=5
)

sample_input = X.iloc[:3]

for _, run in runs.iterrows():
    run_name = run["tags.mlflow.runName"]
    run_id = run["run_id"]

    print(f"\n  {run_name}:")

    # Load as pyfunc (works for all flavors)
    model_uri = f"runs:/{run_id}/model"
    loaded = mlflow.pyfunc.load_model(model_uri)
    preds = loaded.predict(sample_input)
    print(f"    Predictions: {list(preds[:3])}")

print("\n" + "=" * 60)
print("Model flavors demo complete!")
print("=" * 60)
