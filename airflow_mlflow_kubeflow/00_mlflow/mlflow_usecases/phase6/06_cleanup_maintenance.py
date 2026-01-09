"""
Phase 6.6: Cleanup and Maintenance (Enterprise Pattern)

This script demonstrates:
- Identifying and deleting old/failed runs
- Archiving experiments
- Cleaning up model registry
- Storage optimization strategies
- Retention policies implementation
- Safe deletion patterns

CAUTION: This script deletes data. Use with care in production!

Run: python 06_cleanup_maintenance.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List, Dict

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


class MLflowCleanupManager:
    """
    Enterprise cleanup manager for MLflow artifacts and runs.

    IMPORTANT: Always backup before running cleanup operations!
    """

    def __init__(self, client: MlflowClient, dry_run: bool = True):
        """
        Args:
            client: MLflow client
            dry_run: If True, only report what would be deleted (default: True)
        """
        self.client = client
        self.dry_run = dry_run
        self.deletion_log = []

    def _log_deletion(self, item_type: str, item_id: str, reason: str):
        """Log deletion action."""
        action = "WOULD DELETE" if self.dry_run else "DELETED"
        self.deletion_log.append({
            "action": action,
            "type": item_type,
            "id": item_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        print(f"    [{action}] {item_type}: {item_id[:20]}... ({reason})")

    def find_failed_runs(self, experiment_name: str) -> List[str]:
        """Find runs that failed or were killed."""
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string="attributes.status != 'FINISHED'"
        )
        return runs['run_id'].tolist() if not runs.empty else []

    def find_old_runs(self, experiment_name: str, days_old: int = 90) -> List[str]:
        """Find runs older than specified days."""
        cutoff = datetime.now() - timedelta(days=days_old)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        runs = mlflow.search_runs(
            experiment_names=[experiment_name]
        )

        old_runs = []
        for _, run in runs.iterrows():
            run_time = run.get('start_time')
            if run_time and run_time < cutoff_ms:
                old_runs.append(run['run_id'])

        return old_runs

    def find_low_performance_runs(self, experiment_name: str,
                                   metric_name: str, threshold: float) -> List[str]:
        """Find runs below performance threshold."""
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"metrics.{metric_name} < {threshold}"
        )
        return runs['run_id'].tolist() if not runs.empty else []

    def find_untagged_runs(self, experiment_name: str, required_tag: str) -> List[str]:
        """Find runs missing required tags."""
        all_runs = mlflow.search_runs(experiment_names=[experiment_name])

        tag_col = f'tags.{required_tag}'
        if tag_col in all_runs.columns:
            untagged = all_runs[all_runs[tag_col].isna()]
            return untagged['run_id'].tolist()
        return all_runs['run_id'].tolist()

    def delete_runs(self, run_ids: List[str], reason: str):
        """Delete specified runs."""
        for run_id in run_ids:
            self._log_deletion("run", run_id, reason)
            if not self.dry_run:
                try:
                    self.client.delete_run(run_id)
                except Exception as e:
                    print(f"      Error: {e}")

    def archive_experiment(self, experiment_name: str):
        """Archive (soft delete) an experiment."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            self._log_deletion("experiment", experiment_name, "archived")
            if not self.dry_run:
                self.client.delete_experiment(experiment.experiment_id)

    def cleanup_model_versions(self, model_name: str, keep_versions: int = 5):
        """Keep only the N most recent versions, archive older ones."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

            if len(versions) > keep_versions:
                old_versions = versions[keep_versions:]
                for v in old_versions:
                    # Don't delete Production versions
                    if v.current_stage != "Production":
                        self._log_deletion(
                            "model_version",
                            f"{model_name}/v{v.version}",
                            f"older than {keep_versions} versions"
                        )
                        if not self.dry_run:
                            try:
                                self.client.delete_model_version(model_name, v.version)
                            except Exception as e:
                                print(f"      Error: {e}")
        except Exception as e:
            print(f"    Error accessing model {model_name}: {e}")

    def get_deletion_report(self) -> pd.DataFrame:
        """Get summary of all deletions."""
        return pd.DataFrame(self.deletion_log)


# Create cleanup experiment for demo
CLEANUP_EXPERIMENT = "phase6-cleanup-demo"
mlflow.set_experiment(CLEANUP_EXPERIMENT)

print("=" * 70)
print("MLflow Cleanup and Maintenance")
print("=" * 70)

print("""
  WARNING: This script demonstrates deletion operations.
  Always use dry_run=True first to see what would be deleted!

  Backup your MLflow data before running cleanup in production:
  - Database: pg_dump / mysqldump
  - Artifact store: S3 sync / gsutil cp
""")


# [1] Create Sample Data for Cleanup Demo
print("\n[1] Creating Sample Runs for Demo...")
print("-" * 50)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create some successful runs
for i in range(5):
    with mlflow.start_run(run_name=f"good-run-{i+1}"):
        model = RandomForestClassifier(n_estimators=50+i*10, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("reviewed", "true")
        mlflow.set_tag("team", "alpha")

# Create some low-performance runs
for i in range(3):
    with mlflow.start_run(run_name=f"weak-run-{i+1}"):
        model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) * 0.5  # Artificially low
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("team", "beta")
        # No "reviewed" tag

print(f"  Created 8 sample runs")


# [2] Initialize Cleanup Manager (DRY RUN)
print("\n[2] Initializing Cleanup Manager (DRY RUN MODE)...")
print("-" * 50)

cleanup_manager = MLflowCleanupManager(client, dry_run=True)
print("  Mode: DRY RUN (no actual deletions)")


# [3] Find Cleanup Candidates
print("\n[3] Finding Cleanup Candidates...")
print("-" * 50)

# Find failed runs
print("\n  Failed/Killed runs:")
failed_runs = cleanup_manager.find_failed_runs(CLEANUP_EXPERIMENT)
print(f"    Found {len(failed_runs)} failed runs")

# Find low performance runs
print("\n  Low performance runs (accuracy < 0.5):")
low_perf_runs = cleanup_manager.find_low_performance_runs(
    CLEANUP_EXPERIMENT, "accuracy", 0.5
)
print(f"    Found {len(low_perf_runs)} low performance runs")

# Find untagged runs (missing 'reviewed' tag)
print("\n  Unreviewed runs (missing 'reviewed' tag):")
untagged_runs = cleanup_manager.find_untagged_runs(CLEANUP_EXPERIMENT, "reviewed")
print(f"    Found {len(untagged_runs)} unreviewed runs")


# [4] Simulate Cleanup Operations
print("\n[4] Simulating Cleanup Operations...")
print("-" * 50)

if failed_runs:
    print("\n  Deleting failed runs:")
    cleanup_manager.delete_runs(failed_runs, "run failed/killed")

if low_perf_runs:
    print("\n  Deleting low performance runs:")
    cleanup_manager.delete_runs(low_perf_runs, "accuracy below threshold")


# [5] Retention Policy Examples
print("\n[5] Retention Policy Examples...")
print("-" * 50)

print("""
  Example Retention Policies:

  1. Time-based:
     - Delete dev runs older than 30 days
     - Delete staging runs older than 90 days
     - Keep production runs indefinitely

  2. Performance-based:
     - Delete runs with accuracy < baseline
     - Keep top N runs per experiment

  3. Status-based:
     - Delete failed/killed runs after 7 days
     - Archive unreviewed runs after 30 days

  4. Model Registry:
     - Keep last 5 versions per model
     - Never delete Production versions
     - Archive Staging versions after 60 days
""")


# [6] Implementation Patterns
print("\n[6] Cleanup Implementation Patterns...")
print("-" * 50)

print("""
  Safe Deletion Pattern:

  ```python
  # 1. Always start with dry run
  manager = MLflowCleanupManager(client, dry_run=True)

  # 2. Find candidates
  candidates = manager.find_old_runs("my-experiment", days_old=90)

  # 3. Review candidates
  print(f"Found {len(candidates)} runs to delete")
  for run_id in candidates[:5]:
      run = client.get_run(run_id)
      print(f"  {run.info.run_name}: {run.data.metrics}")

  # 4. Confirm deletion
  if input("Proceed with deletion? (yes/no): ") == "yes":
      manager.dry_run = False
      manager.delete_runs(candidates, "retention policy")

  # 5. Generate report
  report = manager.get_deletion_report()
  report.to_csv("deletion_log.csv")
  ```
""")


# [7] Scheduled Cleanup Script
print("\n[7] Scheduled Cleanup Script Template...")
print("-" * 50)

print("""
  Cron Job Script (cleanup_mlflow.py):

  ```python
  #!/usr/bin/env python
  import mlflow
  from datetime import datetime
  import logging

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  def daily_cleanup():
      '''Run daily cleanup tasks.'''
      client = mlflow.tracking.MlflowClient()

      # 1. Delete failed runs older than 7 days
      logger.info("Cleaning up failed runs...")
      # ... implementation

      # 2. Archive old dev experiments
      logger.info("Archiving old dev experiments...")
      # ... implementation

      # 3. Clean up orphaned artifacts
      logger.info("Checking for orphaned artifacts...")
      # ... implementation

      # 4. Generate and send report
      logger.info("Generating cleanup report...")
      # ... implementation

  if __name__ == "__main__":
      daily_cleanup()
  ```

  Cron entry (run daily at 2 AM):
  0 2 * * * /path/to/python /path/to/cleanup_mlflow.py
""")


# [8] Storage Optimization
print("\n[8] Storage Optimization Tips...")
print("-" * 50)

print("""
  Artifact Storage Optimization:

  1. Compress large artifacts before logging:
     ```python
     import gzip
     with gzip.open('model.pkl.gz', 'wb') as f:
         pickle.dump(model, f)
     mlflow.log_artifact('model.pkl.gz')
     ```

  2. Use appropriate artifact types:
     - Log model with mlflow.<flavor>.log_model (optimized)
     - Don't log redundant files (temp files, caches)

  3. Clean up local artifacts after logging:
     ```python
     mlflow.log_artifact(temp_file)
     os.remove(temp_file)  # Clean up local copy
     ```

  4. Set artifact retention in object storage:
     - S3: Lifecycle policies
     - GCS: Object lifecycle management
     - Azure: Blob storage lifecycle

  5. Use external storage for large datasets:
     - Log reference/path instead of actual data
     - Use data versioning tools (DVC, Delta Lake)
""")


# [9] Generate Deletion Report
print("\n[9] Deletion Report...")
print("-" * 50)

report = cleanup_manager.get_deletion_report()
if not report.empty:
    print("\n  Summary of planned deletions:")
    print(report.groupby(['type', 'reason']).size().to_string())
else:
    print("  No deletions planned")


# [10] Restore Deleted Items
print("\n[10] Restoring Deleted Items...")
print("-" * 50)

print("""
  Restore Patterns:

  1. Restore deleted run:
     ```python
     client.restore_run(run_id)
     ```

  2. Restore archived experiment:
     ```python
     client.restore_experiment(experiment_id)
     ```

  3. Model versions cannot be restored after deletion!
     - Use 'Archived' stage instead of deleting
     - Implement soft-delete pattern for models

  Best Practice: Use archiving before deletion:
  ```python
  # Archive first (recoverable)
  client.delete_experiment(exp_id)  # Soft delete

  # Permanent deletion (use with caution)
  # Requires direct database access or API call
  ```
""")


print("\n" + "=" * 70)
print("Cleanup and Maintenance Complete!")
print("=" * 70)
print(f"""
  Key Takeaways:

  1. Always use dry_run=True first
  2. Implement retention policies early
  3. Schedule regular cleanup jobs
  4. Keep deletion logs for audit
  5. Never delete Production models
  6. Use archiving before permanent deletion
  7. Backup before bulk operations

  Enterprise Checklist:
  [ ] Define retention policies per environment
  [ ] Set up scheduled cleanup jobs
  [ ] Implement deletion approval workflow
  [ ] Configure artifact storage lifecycle
  [ ] Create deletion audit trail
  [ ] Test restore procedures

  View at: {TRACKING_URI}
""")
print("=" * 70)
