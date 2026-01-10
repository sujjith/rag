"""
Phase 6.7: PyTorch Deep Learning Integration (Enterprise Pattern)

This script demonstrates:
- Logging PyTorch models with MLflow
- Autologging for PyTorch/Lightning
- Custom training loops with metric logging
- Model signatures for tensor inputs
- GPU training tracking
- Checkpoint management

Prerequisites:
    pip install torch torchvision

Run: python 07_pytorch_integration.py
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
import numpy as np

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-pytorch-integration")

client = MlflowClient()

print("=" * 70)
print("PyTorch Integration with MLflow")
print("=" * 70)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("\n  WARNING: PyTorch not installed.")
    print("  Install with: pip install torch torchvision")
    print("\n  Showing code patterns for reference...")


if PYTORCH_AVAILABLE:
    # =========================================================================
    # Define a Simple Neural Network
    # =========================================================================

    class SimpleClassifier(nn.Module):
        """Simple feedforward neural network for classification."""

        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
            super(SimpleClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )

        def forward(self, x):
            return self.model(x)


    # =========================================================================
    # Create Sample Dataset
    # =========================================================================
    print("\n[1] Creating Sample Dataset...")
    print("-" * 50)

    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    n_features = 20
    n_classes = 3

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int64)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")


    # =========================================================================
    # [2] Manual Training Loop with MLflow Logging
    # =========================================================================
    print("\n[2] Training with Manual Logging...")
    print("-" * 50)

    with mlflow.start_run(run_name="pytorch-manual-logging"):
        # Hyperparameters
        config = {
            "input_dim": n_features,
            "hidden_dim": 64,
            "output_dim": n_classes,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 32,
            "optimizer": "Adam"
        }

        # Log hyperparameters
        mlflow.log_params(config)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", str(device))

        # Initialize model
        model = SimpleClassifier(
            config["input_dim"],
            config["hidden_dim"],
            config["output_dim"],
            config["dropout"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Training loop
        print(f"\n  Training on {device}...")
        best_accuracy = 0.0

        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(test_loader)

            # Log metrics with step
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, "/tmp/best_checkpoint.pt")

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{config['epochs']}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_acc={val_accuracy:.4f}")

        # Log final metrics
        mlflow.log_metric("best_val_accuracy", best_accuracy)

        # Log model with signature
        print("\n  Logging model to MLflow...")
        signature = mlflow.models.infer_signature(
            X_train[:5],
            model(torch.FloatTensor(X_train[:5]).to(device)).cpu().detach().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:2]
        )

        # Log checkpoint as artifact
        mlflow.log_artifact("/tmp/best_checkpoint.pt", "checkpoints")

        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_metric("total_parameters", total_params)
        mlflow.log_metric("trainable_parameters", trainable_params)

        print(f"\n  Training complete!")
        print(f"  Best validation accuracy: {best_accuracy:.4f}")
        print(f"  Total parameters: {total_params:,}")


    # =========================================================================
    # [3] PyTorch Autologging
    # =========================================================================
    print("\n[3] Using PyTorch Autologging...")
    print("-" * 50)

    # Enable autologging
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name="pytorch-autolog"):
        print("  Autologging enabled - model and metrics logged automatically")

        model2 = SimpleClassifier(n_features, 32, n_classes, 0.1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

        # Quick training
        model2.train()
        for epoch in range(5):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model2(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Log model (autolog will capture this)
        mlflow.pytorch.log_model(model2, "model")
        print("  Model logged with autolog!")

    # Disable autolog
    mlflow.pytorch.autolog(disable=True)


    # =========================================================================
    # [4] Loading and Using PyTorch Models
    # =========================================================================
    print("\n[4] Loading PyTorch Models from MLflow...")
    print("-" * 50)

    # Find the latest run
    runs = mlflow.search_runs(
        experiment_names=["phase6-pytorch-integration"],
        filter_string="tags.mlflow.runName = 'pytorch-manual-logging'",
        max_results=1
    )

    if not runs.empty:
        run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{run_id}/model"

        # Load as PyTorch model
        loaded_model = mlflow.pytorch.load_model(model_uri)
        loaded_model.eval()

        # Make predictions
        with torch.no_grad():
            test_input = torch.FloatTensor(X_test[:5])
            predictions = loaded_model(test_input)
            predicted_classes = torch.argmax(predictions, dim=1)

        print(f"  Loaded model from: {model_uri[:50]}...")
        print(f"  Sample predictions: {predicted_classes.numpy()}")
        print(f"  Actual labels: {y_test[:5]}")

        # Load as pyfunc (framework-agnostic)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        pyfunc_predictions = pyfunc_model.predict(X_test[:5])
        print(f"  PyFunc predictions shape: {pyfunc_predictions.shape}")

    # Clean up
    os.remove("/tmp/best_checkpoint.pt")


else:
    # =========================================================================
    # Show Code Patterns (when PyTorch not available)
    # =========================================================================
    print("""
    PyTorch + MLflow Code Patterns:

    1. Manual Logging:
    ```python
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({"lr": 0.001, "epochs": 100})

        for epoch in range(epochs):
            # Training loop
            train_loss = train_one_epoch(model, train_loader)
            val_acc = evaluate(model, val_loader)

            # Log metrics with step
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")
    ```

    2. Autologging:
    ```python
    mlflow.pytorch.autolog()
    # Your training code - metrics logged automatically
    ```

    3. Loading Models:
    ```python
    # As PyTorch model
    model = mlflow.pytorch.load_model("runs:/<run_id>/model")

    # As generic pyfunc
    model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
    ```

    4. GPU Tracking:
    ```python
    mlflow.log_param("device", str(torch.cuda.get_device_name(0)))
    mlflow.log_metric("gpu_memory_allocated", torch.cuda.memory_allocated())
    ```
    """)


# =========================================================================
# [5] PyTorch Lightning Integration (Reference)
# =========================================================================
print("\n[5] PyTorch Lightning Integration (Reference)...")
print("-" * 50)

print("""
  PyTorch Lightning + MLflow:

  ```python
  import pytorch_lightning as pl
  from pytorch_lightning.loggers import MLFlowLogger

  # Create MLflow logger
  mlflow_logger = MLFlowLogger(
      experiment_name="lightning-experiment",
      tracking_uri=TRACKING_URI,
      log_model=True
  )

  # Create trainer with MLflow logger
  trainer = pl.Trainer(
      max_epochs=100,
      logger=mlflow_logger,
      callbacks=[
          pl.callbacks.ModelCheckpoint(monitor='val_loss'),
          pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)
      ]
  )

  # Train - all metrics logged to MLflow automatically
  trainer.fit(model, train_loader, val_loader)
  ```

  Benefits:
  - Automatic metric logging
  - Checkpoint management
  - Hyperparameter tracking
  - Model artifact logging
""")


# =========================================================================
# [6] Best Practices
# =========================================================================
print("\n[6] Best Practices for Deep Learning + MLflow...")
print("-" * 50)

print("""
  1. Hyperparameter Logging:
     - Log ALL hyperparameters (lr, batch_size, architecture, etc.)
     - Use config dictionaries for cleaner code

  2. Metric Logging:
     - Log at each epoch with step parameter
     - Track both training and validation metrics
     - Log final/best metrics separately

  3. Model Artifacts:
     - Save checkpoints as artifacts
     - Log model with signature for serving
     - Include input_example for documentation

  4. GPU Tracking:
     - Log GPU type and memory usage
     - Track CUDA version
     - Monitor GPU utilization

  5. Reproducibility:
     - Log random seeds
     - Log PyTorch/CUDA versions
     - Save model architecture definition

  6. Large Models:
     - Use artifact storage (S3/GCS) for large checkpoints
     - Consider model compression before logging
     - Log model size metrics
""")


print("\n" + "=" * 70)
print("PyTorch Integration Complete!")
print("=" * 70)
print(f"""
  Key Features Demonstrated:

  - Manual training loop with metric logging
  - PyTorch autologging
  - Model signature inference
  - Checkpoint management
  - Loading models for inference
  - PyTorch Lightning integration patterns

  Next Steps:
  - Try with your own models
  - Integrate with model registry
  - Set up distributed training tracking
  - Implement early stopping with MLflow

  View at: {TRACKING_URI}
""")
print("=" * 70)
