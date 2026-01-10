"""
Phase 5.2: Custom metrics and model evaluation

This script demonstrates:
- Logging comprehensive metrics
- Creating evaluation plots
- Saving detailed reports

Run: python 02_custom_metrics.py
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import shutil

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase5-custom-metrics")

# Disable autolog for manual control
mlflow.sklearn.autolog(disable=True)

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create temp directory
TEMP_DIR = "temp_eval"
os.makedirs(TEMP_DIR, exist_ok=True)

print("=" * 60)
print("Comprehensive Model Evaluation")
print("=" * 60)

with mlflow.start_run(run_name="comprehensive-evaluation"):

    # Train model
    print("\n[1] Training model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # 1. Basic metrics
    print("\n[2] Logging basic metrics...")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"    {name}: {value:.4f}")

    # 2. Per-class metrics
    print("\n[3] Logging per-class metrics...")
    for i, class_name in enumerate(iris.target_names):
        # Create binary labels for this class
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        mlflow.log_metric(f"precision_{class_name}", precision)
        mlflow.log_metric(f"recall_{class_name}", recall)
        mlflow.log_metric(f"f1_{class_name}", f1)

        print(f"    {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # 3. Cross-validation
    print("\n[4] Cross-validation scores...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    mlflow.log_metric("cv_min", cv_scores.min())
    mlflow.log_metric("cv_max", cv_scores.max())
    print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 4. AUC-ROC (multiclass)
    print("\n[5] AUC-ROC scores...")
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_weighted = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    mlflow.log_metric("auc_roc_weighted", auc_weighted)
    mlflow.log_metric("auc_roc_macro", auc_macro)
    print(f"    AUC-ROC (weighted): {auc_weighted:.4f}")
    print(f"    AUC-ROC (macro): {auc_macro:.4f}")

    # 5. Confusion Matrix Plot
    print("\n[6] Creating artifacts...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.savefig(f"{TEMP_DIR}/confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(f"{TEMP_DIR}/confusion_matrix.png")
    print("    Logged: confusion_matrix.png")

    # 6. Feature Importance Plot
    importance = pd.Series(
        model.feature_importances_,
        index=iris.feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(f"{TEMP_DIR}/feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(f"{TEMP_DIR}/feature_importance.png")
    print("    Logged: feature_importance.png")

    # 7. Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=iris.target_names,
        output_dict=True
    )
    with open(f"{TEMP_DIR}/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(f"{TEMP_DIR}/classification_report.json")
    print("    Logged: classification_report.json")

    # 8. Prediction distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual distribution
    pd.Series(y_test).value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue"
    )
    axes[0].set_xticklabels(iris.target_names, rotation=45)
    axes[0].set_title("Actual Distribution")
    axes[0].set_ylabel("Count")

    # Predicted distribution
    pd.Series(y_pred).value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color="coral"
    )
    axes[1].set_xticklabels(iris.target_names, rotation=45)
    axes[1].set_title("Predicted Distribution")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(f"{TEMP_DIR}/distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(f"{TEMP_DIR}/distribution.png")
    print("    Logged: distribution.png")

    # 9. Log model
    print("\n[7] Logging model...")
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    print("    Logged: model")

    # 10. Summary JSON
    summary = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(metrics["accuracy"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_importance": importance.to_dict(),
        "classes": list(iris.target_names),
    }
    with open(f"{TEMP_DIR}/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    mlflow.log_artifact(f"{TEMP_DIR}/model_summary.json")
    print("    Logged: model_summary.json")

# Clean up
shutil.rmtree(TEMP_DIR)

print("\n" + "=" * 60)
print("Comprehensive evaluation complete!")
print(f"View at: {TRACKING_URI}")
print("=" * 60)
