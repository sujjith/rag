"""
Phase 2.2: Data Processing Pipeline

A pipeline that demonstrates:
- Data loading and saving
- Passing datasets between components
- Training and evaluation

Run:
  python 02_data_pipeline.py
  # Then upload training_pipeline.yaml to Kubeflow UI
"""
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2"]
)
def load_iris_data(dataset: Output[Dataset]):
    """Load the Iris dataset and save as CSV."""
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Save to output path
    df.to_csv(dataset.path, index=False)

    print(f"Dataset saved to: {dataset.path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2"]
)
def split_data(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
):
    """Split dataset into train and test sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data
    df = pd.read_csv(input_dataset.path)
    print(f"Loaded {len(df)} samples")

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )

    # Save
    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "joblib==1.3.2"]
)
def train_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    n_estimators: int = 100,
    max_depth: int = 5
):
    """Train a RandomForest classifier."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load training data
    df = pd.read_csv(train_dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Training on {len(X)} samples")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_artifact.path)
    print(f"Model saved to: {model_artifact.path}")

    # Log training accuracy
    train_accuracy = model.score(X, y)
    print(f"Training accuracy: {train_accuracy:.4f}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.4", "scikit-learn==1.3.2", "joblib==1.3.2"]
)
def evaluate_model(
    model_artifact: Input[Model],
    test_dataset: Input[Dataset],
    metrics: Output[Metrics]
) -> float:
    """Evaluate the trained model."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib

    # Load model and test data
    model = joblib.load(model_artifact.path)
    df = pd.read_csv(test_dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Evaluating on {len(X)} samples")

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy


@dsl.pipeline(
    name="iris-training-pipeline",
    description="Train and evaluate a model on the Iris dataset"
)
def training_pipeline(
    n_estimators: int = 100,
    max_depth: int = 5,
    test_size: float = 0.2
):
    """
    ML training pipeline:
    1. Load data
    2. Split into train/test
    3. Train model
    4. Evaluate model
    """
    # Step 1: Load data
    load_task = load_iris_data()

    # Step 2: Split data
    split_task = split_data(
        input_dataset=load_task.outputs["dataset"],
        test_size=test_size
    )

    # Step 3: Train model
    train_task = train_model(
        train_dataset=split_task.outputs["train_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # Step 4: Evaluate model
    eval_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        test_dataset=split_task.outputs["test_dataset"]
    )


if __name__ == "__main__":
    output_file = "training_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=output_file
    )

    print("=" * 50)
    print("Pipeline compiled successfully!")
    print("=" * 50)
    print(f"\nOutput: {output_file}")
    print("\nPipeline parameters:")
    print("  - n_estimators: Number of trees (default: 100)")
    print("  - max_depth: Max tree depth (default: 5)")
    print("  - test_size: Test set fraction (default: 0.2)")
    print("=" * 50)
