"""
Phase 1.3: Your first MLflow experiment

This script demonstrates:
- Connecting to MLflow tracking server
- Creating experiments
- Logging parameters and metrics
- Adding tags

Run: python 01_hello_mlflow.py
"""
import mlflow

# Connect to tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create an experiment
mlflow.set_experiment("phase1-hello-mlflow")

# Start a run
with mlflow.start_run():
    # Log parameters (input configuration)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)

    # Log metrics (output measurements)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.15)
    mlflow.log_metric("val_accuracy", 0.82)

    # Add tags (metadata)
    mlflow.set_tag("author", "sujith")
    mlflow.set_tag("version", "v1")
    mlflow.set_tag("environment", "development")

    print("=" * 50)
    print("Run logged successfully!")
    print("=" * 50)
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Experiment: phase1-hello-mlflow")
    print("")
    print("Parameters logged:")
    print("  - learning_rate: 0.01")
    print("  - epochs: 10")
    print("  - batch_size: 32")
    print("")
    print("Metrics logged:")
    print("  - accuracy: 0.85")
    print("  - loss: 0.15")
    print("  - val_accuracy: 0.82")
    print("")
    print("View at: http://localhost:5000")
    print("=" * 50)
