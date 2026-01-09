"""
Phase 6.1: Model Comparison and Selection (Enterprise Pattern)

This script demonstrates:
- Comparing multiple models programmatically
- Statistical significance testing between models
- Automated model selection based on multiple criteria
- Creating comparison reports and visualizations

Run: python 01_model_comparison.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import json

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-model-comparison")

client = MlflowClient()

# Load data
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("Enterprise Model Comparison Framework")
print("=" * 70)


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5):
    """Comprehensive model evaluation with cross-validation."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cross-validation scores for statistical comparison
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }


def statistical_comparison(scores_a, scores_b, alpha=0.05):
    """Perform paired t-test to compare two models."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = p_value < alpha

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'better_model': 'A' if np.mean(scores_a) > np.mean(scores_b) else 'B'
    }


# Define models to compare
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

print("\n[1] Training and Evaluating Models...")
print("-" * 50)

results = {}
run_ids = {}

# Train and log each model
for name, model in models.items():
    with mlflow.start_run(run_name=f"comparison-{name}") as run:
        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Log parameters
        mlflow.log_param("model_type", name)
        mlflow.log_params({k: v for k, v in model.get_params().items()
                         if isinstance(v, (int, float, str, bool))})

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("f1_weighted", metrics['f1_weighted'])
        mlflow.log_metric("precision_weighted", metrics['precision_weighted'])
        mlflow.log_metric("recall_weighted", metrics['recall_weighted'])
        mlflow.log_metric("cv_mean", metrics['cv_mean'])
        mlflow.log_metric("cv_std", metrics['cv_std'])

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Add comparison tag
        mlflow.set_tag("comparison_group", "wine-classifier-comparison")

        results[name] = metrics
        run_ids[name] = run.info.run_id

        print(f"  {name:20s}: accuracy={metrics['accuracy']:.4f}, "
              f"cv_mean={metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")

print("-" * 50)


# [2] Statistical Comparison
print("\n[2] Statistical Significance Testing...")
print("-" * 50)

model_names = list(results.keys())
comparison_results = []

# Compare each pair of models
for i, model_a in enumerate(model_names):
    for model_b in model_names[i+1:]:
        comparison = statistical_comparison(
            results[model_a]['cv_scores'],
            results[model_b]['cv_scores']
        )
        comparison_results.append({
            'model_a': model_a,
            'model_b': model_b,
            **comparison
        })

        sig_text = "SIGNIFICANT" if comparison['significant'] else "not significant"
        print(f"  {model_a} vs {model_b}: p={comparison['p_value']:.4f} ({sig_text})")

print("-" * 50)


# [3] Automated Model Selection
print("\n[3] Automated Model Selection...")
print("-" * 50)

# Selection criteria (customize for your use case)
SELECTION_CRITERIA = {
    'min_accuracy': 0.90,
    'max_cv_std': 0.05,  # Stability requirement
    'primary_metric': 'accuracy',
    'secondary_metric': 'cv_mean'
}

# Filter models meeting minimum requirements
qualified_models = {
    name: metrics for name, metrics in results.items()
    if metrics['accuracy'] >= SELECTION_CRITERIA['min_accuracy']
    and metrics['cv_std'] <= SELECTION_CRITERIA['max_cv_std']
}

if qualified_models:
    # Select best based on primary metric
    best_model = max(qualified_models.items(),
                    key=lambda x: x[1][SELECTION_CRITERIA['primary_metric']])
    print(f"  Qualified models: {list(qualified_models.keys())}")
    print(f"  Selected model: {best_model[0]}")
    print(f"  Selection criteria: accuracy >= {SELECTION_CRITERIA['min_accuracy']}, "
          f"cv_std <= {SELECTION_CRITERIA['max_cv_std']}")
else:
    # Fallback to best available
    best_model = max(results.items(),
                    key=lambda x: x[1][SELECTION_CRITERIA['primary_metric']])
    print(f"  WARNING: No model met all criteria!")
    print(f"  Fallback selection: {best_model[0]} (best {SELECTION_CRITERIA['primary_metric']})")

print("-" * 50)


# [4] Create Comparison Report
print("\n[4] Creating Comparison Report...")
print("-" * 50)

# Create comparison dataframe
comparison_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': m['accuracy'],
        'F1 Score': m['f1_weighted'],
        'Precision': m['precision_weighted'],
        'Recall': m['recall_weighted'],
        'CV Mean': m['cv_mean'],
        'CV Std': m['cv_std'],
        'Run ID': run_ids[name][:8]
    }
    for name, m in results.items()
]).sort_values('Accuracy', ascending=False)

print("\n  Model Performance Summary:")
print(comparison_df.to_string(index=False))


# [5] Log Comparison as Parent Run
print("\n[5] Logging Comparison Summary...")
print("-" * 50)

with mlflow.start_run(run_name="comparison-summary") as parent_run:
    # Log the winner
    mlflow.log_param("selected_model", best_model[0])
    mlflow.log_param("selection_criteria", json.dumps(SELECTION_CRITERIA))
    mlflow.log_param("num_models_compared", len(models))

    # Log winner's metrics
    mlflow.log_metric("best_accuracy", best_model[1]['accuracy'])
    mlflow.log_metric("best_cv_mean", best_model[1]['cv_mean'])

    # Log comparison table as artifact
    comparison_df.to_csv("/tmp/model_comparison.csv", index=False)
    mlflow.log_artifact("/tmp/model_comparison.csv")

    # Log statistical comparison results
    stat_df = pd.DataFrame(comparison_results)
    stat_df.to_csv("/tmp/statistical_comparison.csv", index=False)
    mlflow.log_artifact("/tmp/statistical_comparison.csv")

    # Create and log visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of metrics
    metrics_to_plot = ['Accuracy', 'F1 Score', 'CV Mean']
    comparison_df.set_index('Model')[metrics_to_plot].plot(
        kind='bar', ax=axes[0], rot=45
    )
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0.8, 1.0)

    # CV score distribution (box plot style)
    cv_data = [results[name]['cv_scores'] for name in model_names]
    axes[1].boxplot(cv_data, labels=model_names)
    axes[1].set_title('Cross-Validation Score Distribution')
    axes[1].set_ylabel('Accuracy')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig("/tmp/model_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("/tmp/model_comparison.png")

    # Tag for easy retrieval
    mlflow.set_tag("comparison_type", "multi-model")
    mlflow.set_tag("winner", best_model[0])
    mlflow.set_tag("winner_run_id", run_ids[best_model[0]])

    print(f"  Summary run logged: {parent_run.info.run_id[:8]}...")
    print("  Artifacts: model_comparison.csv, statistical_comparison.csv, model_comparison.png")


# [6] Programmatic Query of Comparison Results
print("\n[6] Querying Comparison Results...")
print("-" * 50)

# Find all runs in this comparison group
runs = mlflow.search_runs(
    experiment_names=["phase6-model-comparison"],
    filter_string="tags.comparison_group = 'wine-classifier-comparison'",
    order_by=["metrics.accuracy DESC"]
)

print(f"  Found {len(runs)} comparison runs")
print(f"  Top 3 by accuracy:")
for i, row in runs.head(3).iterrows():
    print(f"    {row['tags.mlflow.runName']}: {row['metrics.accuracy']:.4f}")

# Clean up
os.remove("/tmp/model_comparison.csv")
os.remove("/tmp/statistical_comparison.csv")
os.remove("/tmp/model_comparison.png")

print("\n" + "=" * 70)
print("Model Comparison Complete!")
print(f"Winner: {best_model[0]} (accuracy={best_model[1]['accuracy']:.4f})")
print(f"View at: {TRACKING_URI}")
print("=" * 70)
