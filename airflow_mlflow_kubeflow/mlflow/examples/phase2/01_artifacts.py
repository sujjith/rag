"""
Phase 2.1: Logging artifacts (files)

This script demonstrates:
- Logging plots
- Logging CSV files
- Logging JSON configs
- Logging directories

Run: python 01_artifacts.py
"""
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import shutil

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("phase2-artifacts")

print("=" * 60)
print("Artifact Logging Demo")
print("=" * 60)

# Create temp directory for artifacts
TEMP_DIR = "temp_artifacts"
os.makedirs(TEMP_DIR, exist_ok=True)

with mlflow.start_run(run_name="artifact-demo"):

    # 1. Log a matplotlib plot
    print("\n[1] Creating and logging plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Line plot
    x = np.linspace(0, 10, 100)
    axes[0].plot(x, np.sin(x), label="sin(x)")
    axes[0].plot(x, np.cos(x), label="cos(x)")
    axes[0].set_title("Trigonometric Functions")
    axes[0].legend()
    axes[0].grid(True)

    # Scatter plot
    np.random.seed(42)
    axes[1].scatter(
        np.random.randn(100),
        np.random.randn(100),
        c=np.random.randn(100),
        cmap="viridis"
    )
    axes[1].set_title("Random Scatter")

    plot_path = f"{TEMP_DIR}/analysis_plot.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(plot_path)
    print(f"    Logged: analysis_plot.png")

    # 2. Log a CSV file with sample data
    print("\n[2] Creating and logging CSV...")
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.randn(100),
        "target": np.random.choice([0, 1, 2], 100)
    })
    csv_path = f"{TEMP_DIR}/sample_data.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    print(f"    Logged: sample_data.csv ({len(df)} rows)")

    # 3. Log a JSON configuration
    print("\n[3] Creating and logging JSON config...")
    config = {
        "model": {
            "type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10
        },
        "preprocessing": {
            "normalize": True,
            "handle_missing": "mean",
            "features": ["feature_1", "feature_2", "feature_3"]
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5
        }
    }
    json_path = f"{TEMP_DIR}/config.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact(json_path)
    print(f"    Logged: config.json")

    # 4. Log a text file with notes
    print("\n[4] Creating and logging text notes...")
    notes_path = f"{TEMP_DIR}/experiment_notes.txt"
    with open(notes_path, "w") as f:
        f.write("Experiment Notes\n")
        f.write("=" * 40 + "\n\n")
        f.write("Date: 2024-01-15\n")
        f.write("Author: Data Science Team\n\n")
        f.write("Observations:\n")
        f.write("- Model converged after 50 epochs\n")
        f.write("- Feature 2 shows high importance\n")
        f.write("- No overfitting detected\n\n")
        f.write("Next Steps:\n")
        f.write("- Try different hyperparameters\n")
        f.write("- Add more features\n")
    mlflow.log_artifact(notes_path)
    print(f"    Logged: experiment_notes.txt")

    # 5. Log multiple files in a directory
    print("\n[5] Creating and logging a directory...")
    reports_dir = f"{TEMP_DIR}/reports"
    os.makedirs(reports_dir, exist_ok=True)

    for i in range(3):
        report_path = f"{reports_dir}/report_{i+1}.txt"
        with open(report_path, "w") as f:
            f.write(f"Report {i+1}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {0.85 + i*0.03:.2f}\n")
            f.write(f"Loss: {0.15 - i*0.02:.2f}\n")

    mlflow.log_artifacts(reports_dir, artifact_path="reports")
    print(f"    Logged: reports/ directory (3 files)")

    # 6. Log a binary file (pickle)
    print("\n[6] Creating and logging pickle file...")
    import pickle
    model_data = {
        "weights": np.random.randn(10, 5),
        "bias": np.random.randn(5)
    }
    pickle_path = f"{TEMP_DIR}/model_weights.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model_data, f)
    mlflow.log_artifact(pickle_path)
    print(f"    Logged: model_weights.pkl")

    # Log some metrics too
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_param("experiment_type", "artifact_demo")

# Clean up
shutil.rmtree(TEMP_DIR)

print("\n" + "=" * 60)
print("All artifacts logged!")
print("View them at: http://localhost:5000")
print("Click on the run and go to 'Artifacts' tab")
print("=" * 60)
