"""
Phase 6.5: Advanced Search and Queries (Enterprise Pattern)

This script demonstrates:
- Complex run filtering with MLflow Search API
- Metric-based run selection
- Tag-based organization and queries
- Model registry searches
- Bulk operations on runs
- Building dashboards and reports

Run: python 05_advanced_search.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()
EXPERIMENT_NAME = "phase6-advanced-search"


# Create experiment
try:
    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        tags={"team": "data-science", "project": "wine-classification"}
    )
except:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)


print("=" * 70)
print("Advanced MLflow Search and Queries")
print("=" * 70)


# [1] Create Sample Runs with Various Tags and Metrics
print("\n[1] Creating Sample Runs...")
print("-" * 50)

# Load data
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create runs with different configurations
run_configs = [
    {"model": "RandomForest", "n_estimators": 50, "team": "alpha", "env": "dev"},
    {"model": "RandomForest", "n_estimators": 100, "team": "alpha", "env": "staging"},
    {"model": "RandomForest", "n_estimators": 150, "team": "beta", "env": "production"},
    {"model": "GradientBoosting", "n_estimators": 50, "team": "alpha", "env": "dev"},
    {"model": "GradientBoosting", "n_estimators": 100, "team": "beta", "env": "staging"},
    {"model": "LogisticRegression", "solver": "lbfgs", "team": "gamma", "env": "production"},
]

for i, config in enumerate(run_configs):
    with mlflow.start_run(run_name=f"run-{i+1}"):
        # Create model
        if config["model"] == "RandomForest":
            model = RandomForestClassifier(n_estimators=config["n_estimators"], random_state=42)
        elif config["model"] == "GradientBoosting":
            model = GradientBoostingClassifier(n_estimators=config["n_estimators"], random_state=42)
        else:
            model = LogisticRegression(solver=config.get("solver", "lbfgs"), max_iter=1000, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log params
        mlflow.log_param("model_type", config["model"])
        if "n_estimators" in config:
            mlflow.log_param("n_estimators", config["n_estimators"])

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", np.random.uniform(0.5, 5.0))

        # Log tags
        mlflow.set_tag("team", config["team"])
        mlflow.set_tag("environment", config["env"])
        mlflow.set_tag("model_family", config["model"])
        mlflow.set_tag("priority", "high" if config["env"] == "production" else "normal")

        print(f"  Created: run-{i+1} ({config['model']}, team={config['team']}, env={config['env']})")

print(f"\n  Total runs created: {len(run_configs)}")


# [2] Basic Search Queries
print("\n[2] Basic Search Queries...")
print("-" * 50)

# Search all runs in experiment
print("\n  All runs:")
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
print(f"    Found {len(runs)} runs")

# Filter by parameter
print("\n  Runs with n_estimators = 100:")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="params.n_estimators = '100'"
)
print(f"    Found {len(runs)} runs")
for _, run in runs.iterrows():
    print(f"      {run['tags.mlflow.runName']}: {run['params.model_type']}")


# [3] Metric-Based Queries
print("\n[3] Metric-Based Queries...")
print("-" * 50)

# Runs with accuracy > 0.95
print("\n  Runs with accuracy > 0.95:")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="metrics.accuracy > 0.95"
)
print(f"    Found {len(runs)} runs")
for _, run in runs.iterrows():
    print(f"      {run['tags.mlflow.runName']}: accuracy={run['metrics.accuracy']:.4f}")

# Top 3 runs by accuracy
print("\n  Top 3 runs by accuracy:")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["metrics.accuracy DESC"],
    max_results=3
)
for _, run in runs.iterrows():
    print(f"    {run['tags.mlflow.runName']}: accuracy={run['metrics.accuracy']:.4f}, "
          f"model={run['params.model_type']}")


# [4] Tag-Based Queries
print("\n[4] Tag-Based Queries...")
print("-" * 50)

# Runs by team
print("\n  Runs by team 'alpha':")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.team = 'alpha'"
)
print(f"    Found {len(runs)} runs")

# Runs by environment
print("\n  Production runs only:")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.environment = 'production'"
)
print(f"    Found {len(runs)} runs")
for _, run in runs.iterrows():
    print(f"      {run['tags.mlflow.runName']}: {run['params.model_type']}, team={run['tags.team']}")


# [5] Complex Compound Queries
print("\n[5] Complex Compound Queries...")
print("-" * 50)

# Multiple conditions with AND
print("\n  RandomForest models with accuracy > 0.90:")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="params.model_type = 'RandomForest' AND metrics.accuracy > 0.90"
)
print(f"    Found {len(runs)} runs")

# Using LIKE for partial matching
print("\n  Models from team starting with 'a' (LIKE):")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.team LIKE 'a%'"
)
print(f"    Found {len(runs)} runs")

# Using IN for multiple values
print("\n  Runs in dev or staging environment (IN):")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.environment IN ('dev', 'staging')"
)
print(f"    Found {len(runs)} runs")


# [6] Programmatic Analysis
print("\n[6] Programmatic Analysis...")
print("-" * 50)

# Get all runs as DataFrame for analysis
all_runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

# Group by model type
print("\n  Average accuracy by model type:")
if 'params.model_type' in all_runs.columns:
    grouped = all_runs.groupby('params.model_type')['metrics.accuracy'].agg(['mean', 'std', 'count'])
    print(grouped.to_string())

# Group by team
print("\n  Average accuracy by team:")
if 'tags.team' in all_runs.columns:
    grouped = all_runs.groupby('tags.team')['metrics.accuracy'].agg(['mean', 'count'])
    print(grouped.to_string())


# [7] Model Registry Searches
print("\n[7] Model Registry Searches...")
print("-" * 50)

# Search registered models
print("\n  All registered models:")
models = client.search_registered_models()
for model in models[:5]:  # Show first 5
    print(f"    {model.name}")

# Search model versions with filters
print("\n  Model versions in Production stage:")
try:
    versions = client.search_model_versions("name LIKE '%'")
    production_versions = [v for v in versions if v.current_stage == "Production"]
    print(f"    Found {len(production_versions)} production model versions")
    for v in production_versions[:3]:
        print(f"      {v.name} v{v.version}")
except Exception as e:
    print(f"    (No models in registry or error: {e})")


# [8] Advanced Search Patterns
print("\n[8] Advanced Search Patterns...")
print("-" * 50)

print("""
  Search Filter Syntax Reference:

  Comparison Operators:
    =, !=, <, <=, >, >=

  String Operators:
    LIKE 'pattern%'     # Starts with
    LIKE '%pattern'     # Ends with
    LIKE '%pattern%'    # Contains
    IN ('a', 'b', 'c')  # In list

  Logical Operators:
    AND, OR

  Attribute Prefixes:
    params.<name>       # Parameter values
    metrics.<name>      # Metric values
    tags.<name>         # Tag values
    attributes.status   # Run status
    attributes.run_id   # Run ID

  Examples:
    "params.learning_rate > 0.01"
    "metrics.accuracy >= 0.95 AND metrics.loss < 0.1"
    "tags.environment = 'production'"
    "params.model_type LIKE 'Random%'"
    "tags.team IN ('alpha', 'beta')"
    "attributes.status = 'FINISHED'"
""")


# [9] Bulk Operations
print("\n[9] Bulk Operations on Runs...")
print("-" * 50)

# Add tags to multiple runs
print("\n  Adding 'reviewed' tag to top performers:")
top_runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="metrics.accuracy > 0.95",
    max_results=10
)

for _, run in top_runs.iterrows():
    run_id = run['run_id']
    client.set_tag(run_id, "reviewed", "true")
    client.set_tag(run_id, "review_date", datetime.now().isoformat())
    print(f"    Tagged run {run_id[:8]}...")

print(f"  Tagged {len(top_runs)} runs as reviewed")


# [10] Building Reports
print("\n[10] Building Analysis Reports...")
print("-" * 50)

all_runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

report = {
    "experiment": EXPERIMENT_NAME,
    "total_runs": len(all_runs),
    "date_generated": datetime.now().isoformat(),
    "summary": {
        "best_accuracy": all_runs['metrics.accuracy'].max(),
        "avg_accuracy": all_runs['metrics.accuracy'].mean(),
        "best_model": all_runs.loc[all_runs['metrics.accuracy'].idxmax(), 'params.model_type'],
    },
    "by_model": all_runs.groupby('params.model_type')['metrics.accuracy'].mean().to_dict(),
    "by_environment": all_runs.groupby('tags.environment')['metrics.accuracy'].mean().to_dict(),
}

print("\n  Experiment Report:")
print(f"    Total runs: {report['total_runs']}")
print(f"    Best accuracy: {report['summary']['best_accuracy']:.4f}")
print(f"    Best model: {report['summary']['best_model']}")
print(f"    Average accuracy: {report['summary']['avg_accuracy']:.4f}")


# [11] Pagination for Large Datasets
print("\n[11] Pagination for Large Datasets...")
print("-" * 50)

print("""
  For large experiment datasets, use pagination:

  ```python
  page_token = None
  all_runs = []

  while True:
      runs = mlflow.search_runs(
          experiment_names=["my-experiment"],
          max_results=100,
          page_token=page_token
      )

      all_runs.extend(runs.to_dict('records'))

      # Check if there are more results
      # The search_runs function returns a DataFrame
      # Use the _page_token attribute if available
      if len(runs) < 100:
          break

      # For MlflowClient, use:
      # result = client.search_runs(...)
      # page_token = result.token
  ```
""")


print("\n" + "=" * 70)
print("Advanced Search Complete!")
print("=" * 70)
print(f"""
  Key Capabilities Demonstrated:

  1. Basic filtering by params, metrics, tags
  2. Complex compound queries (AND, OR, IN, LIKE)
  3. Metric-based ordering and selection
  4. Tag-based organization and retrieval
  5. Model registry searches
  6. Bulk operations on runs
  7. Programmatic analysis with pandas
  8. Report generation patterns

  Enterprise Best Practices:

  - Use consistent tagging schema across teams
  - Create periodic reports from search results
  - Use tags for workflow status (reviewed, approved, etc.)
  - Implement custom dashboards using search API
  - Set up alerts based on metric queries

  View at: {TRACKING_URI}
""")
print("=" * 70)
