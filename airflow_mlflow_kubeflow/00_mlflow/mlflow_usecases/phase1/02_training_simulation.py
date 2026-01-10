"""
Phase 1.4: Simulate training with metrics over epochs

This script demonstrates:
- Logging metrics at different steps
- Viewing metric charts in MLflow UI
- Simulating a training loop

Run: python 02_training_simulation.py
"""
import mlflow
import random
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase1-training-simulation")

print("=" * 60)
print("Training Simulation")
print("=" * 60)

with mlflow.start_run(run_name="simulated-training"):
    # Log parameters
    mlflow.log_param("model_type", "neural_network")
    mlflow.log_param("hidden_layers", 3)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("optimizer", "adam")

    # Simulate training loop
    epochs = 20
    accuracy = 0.5
    loss = 1.0

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)

    for epoch in range(epochs):
        # Simulate improvement with some noise
        accuracy += random.uniform(0.01, 0.03)
        loss -= random.uniform(0.02, 0.05)

        # Clamp values to realistic range
        accuracy = min(accuracy, 0.99)
        loss = max(loss, 0.01)

        # Log metrics with step number
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)
        mlflow.log_metric("train_loss", loss, step=epoch)

        # Simulate validation metrics (slightly worse than training)
        val_accuracy = accuracy - random.uniform(0.02, 0.05)
        val_loss = loss + random.uniform(0.01, 0.03)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"acc={accuracy:.4f}, loss={loss:.4f}, "
              f"val_acc={val_accuracy:.4f}, val_loss={val_loss:.4f}")

    print("-" * 40)

    # Log final metrics
    mlflow.log_metric("final_accuracy", accuracy)
    mlflow.log_metric("final_loss", loss)

    print(f"\nFinal Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Loss: {loss:.4f}")

print("\n" + "=" * 60)
print("Check the MLflow UI to see the metric charts!")
print(f"{TRACKING_URI}")
print("=" * 60)
