"""
Phase 10.3: Data Profiling & Drift Detection

Demonstrates:
- Statistical profiling
- Data drift detection
- Anomaly detection
- Trend analysis
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
import json
import math


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


class DataProfiler:
    """Generate statistical profile of data."""

    def __init__(self, data: list):
        self.data = data

    def profile_numeric(self, column: str) -> dict:
        """Profile numeric column."""
        values = [row.get(column) for row in self.data if row.get(column) is not None]

        if not values:
            return {"column": column, "error": "No valid values"}

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(variance)

        sorted_vals = sorted(values)
        median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2

        return {
            "column": column,
            "type": "numeric",
            "count": n,
            "null_count": len(self.data) - n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": min(values),
            "max": max(values),
            "median": median,
            "p25": sorted_vals[int(n * 0.25)],
            "p75": sorted_vals[int(n * 0.75)],
        }

    def profile_categorical(self, column: str) -> dict:
        """Profile categorical column."""
        values = [row.get(column) for row in self.data if row.get(column) is not None]

        if not values:
            return {"column": column, "error": "No valid values"}

        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1

        return {
            "column": column,
            "type": "categorical",
            "count": len(values),
            "null_count": len(self.data) - len(values),
            "unique_count": len(value_counts),
            "top_values": dict(sorted(value_counts.items(), key=lambda x: -x[1])[:5]),
        }


class DriftDetector:
    """Detect data drift between datasets."""

    def __init__(self, baseline_profile: dict, current_profile: dict):
        self.baseline = baseline_profile
        self.current = current_profile

    def detect_numeric_drift(self, column: str, threshold: float = 0.2) -> dict:
        """Detect drift in numeric column using relative change."""
        baseline = self.baseline.get(column, {})
        current = self.current.get(column, {})

        if not baseline or not current:
            return {"column": column, "drift_detected": False, "error": "Missing profile"}

        metrics = ["mean", "std", "min", "max"]
        drift_results = {}

        for metric in metrics:
            baseline_val = baseline.get(metric, 0)
            current_val = current.get(metric, 0)

            if baseline_val != 0:
                change = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                change = abs(current_val) if current_val != 0 else 0

            drift_results[metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "change_pct": round(change * 100, 2),
                "drift": change > threshold,
            }

        overall_drift = any(r["drift"] for r in drift_results.values())

        return {
            "column": column,
            "drift_detected": overall_drift,
            "metrics": drift_results,
        }

    def detect_categorical_drift(self, column: str, threshold: float = 0.1) -> dict:
        """Detect drift in categorical distribution."""
        baseline = self.baseline.get(column, {})
        current = self.current.get(column, {})

        if not baseline or not current:
            return {"column": column, "drift_detected": False, "error": "Missing profile"}

        baseline_dist = baseline.get("top_values", {})
        current_dist = current.get("top_values", {})

        # Normalize to proportions
        baseline_total = sum(baseline_dist.values()) or 1
        current_total = sum(current_dist.values()) or 1

        baseline_pct = {k: v/baseline_total for k, v in baseline_dist.items()}
        current_pct = {k: v/current_total for k, v in current_dist.items()}

        # Check for new/missing categories
        new_categories = set(current_dist.keys()) - set(baseline_dist.keys())
        missing_categories = set(baseline_dist.keys()) - set(current_dist.keys())

        # Check distribution shift
        distribution_shift = {}
        for cat in set(baseline_pct.keys()) | set(current_pct.keys()):
            b = baseline_pct.get(cat, 0)
            c = current_pct.get(cat, 0)
            shift = abs(c - b)
            distribution_shift[cat] = {
                "baseline_pct": round(b * 100, 2),
                "current_pct": round(c * 100, 2),
                "shift": round(shift * 100, 2),
            }

        max_shift = max(abs(v["shift"]) for v in distribution_shift.values()) if distribution_shift else 0
        drift_detected = max_shift > threshold * 100 or new_categories or missing_categories

        return {
            "column": column,
            "drift_detected": drift_detected,
            "new_categories": list(new_categories),
            "missing_categories": list(missing_categories),
            "distribution_shift": distribution_shift,
        }


@dag(
    dag_id="phase10_03_data_profiling",
    description="Data profiling and drift detection",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase10", "enterprise", "data-quality", "profiling", "drift"],
    doc_md="""
    ## Data Profiling & Drift Detection

    **Patterns Covered:**
    - Statistical profiling (mean, std, percentiles)
    - Categorical distribution analysis
    - Data drift detection
    - Anomaly alerting

    **Use Cases:**
    - Monitor data quality over time
    - Detect schema/data changes
    - ML feature drift detection
    - Alert on data anomalies
    """,
)
def data_profiling():

    @task
    def get_baseline_profile():
        """Get baseline profile from historical data."""
        # In production, load from database/storage
        baseline = {
            "age": {
                "type": "numeric",
                "count": 1000,
                "mean": 35.5,
                "std": 12.3,
                "min": 18,
                "max": 80,
                "median": 34,
            },
            "income": {
                "type": "numeric",
                "count": 1000,
                "mean": 65000,
                "std": 25000,
                "min": 20000,
                "max": 200000,
                "median": 58000,
            },
            "category": {
                "type": "categorical",
                "count": 1000,
                "unique_count": 4,
                "top_values": {"A": 400, "B": 300, "C": 200, "D": 100},
            }
        }
        return baseline

    @task
    def get_current_data():
        """Get current batch of data."""
        import random
        random.seed(42)

        # Simulate data with some drift
        data = []
        for i in range(100):
            data.append({
                "id": i,
                "age": random.randint(20, 90),  # Slightly different range
                "income": random.randint(25000, 250000),  # Higher incomes
                "category": random.choice(["A", "B", "C", "E"]),  # D missing, E new
            })

        return {"data": data, "timestamp": datetime.now().isoformat()}

    @task
    def profile_current_data(current: dict):
        """Generate profile for current data."""
        data = current["data"]
        profiler = DataProfiler(data)

        profile = {
            "age": profiler.profile_numeric("age"),
            "income": profiler.profile_numeric("income"),
            "category": profiler.profile_categorical("category"),
        }

        print("Current Data Profile:")
        print(json.dumps(profile, indent=2))

        return profile

    @task
    def detect_drift(baseline: dict, current_profile: dict):
        """Detect drift between baseline and current."""
        detector = DriftDetector(baseline, current_profile)

        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "columns": {},
            "overall_drift": False,
        }

        # Numeric columns
        for col in ["age", "income"]:
            result = detector.detect_numeric_drift(col)
            drift_report["columns"][col] = result
            if result["drift_detected"]:
                drift_report["overall_drift"] = True

        # Categorical columns
        for col in ["category"]:
            result = detector.detect_categorical_drift(col)
            drift_report["columns"][col] = result
            if result["drift_detected"]:
                drift_report["overall_drift"] = True

        print("\nDrift Detection Report:")
        print("=" * 50)
        for col, result in drift_report["columns"].items():
            status = "🔴 DRIFT" if result["drift_detected"] else "🟢 OK"
            print(f"{status} {col}")
            if result["drift_detected"]:
                if "metrics" in result:
                    for metric, info in result.get("metrics", {}).items():
                        if info.get("drift"):
                            print(f"     {metric}: {info['baseline']} -> {info['current']} ({info['change_pct']}%)")
                if "new_categories" in result and result["new_categories"]:
                    print(f"     New categories: {result['new_categories']}")
                if "missing_categories" in result and result["missing_categories"]:
                    print(f"     Missing categories: {result['missing_categories']}")

        return drift_report

    @task
    def alert_on_drift(drift_report: dict):
        """Alert if significant drift detected."""
        if not drift_report["overall_drift"]:
            print("No drift detected - no alert needed")
            return {"alert_sent": False}

        drifted_columns = [
            col for col, result in drift_report["columns"].items()
            if result["drift_detected"]
        ]

        alert = {
            "severity": "warning",
            "title": "Data Drift Detected",
            "message": f"Drift detected in columns: {drifted_columns}",
            "timestamp": datetime.now().isoformat(),
            "details": drift_report,
        }

        print("\n🚨 ALERT: Data Drift Detected")
        print(f"Columns affected: {drifted_columns}")
        print("Sending notification...")

        # In production: send to Slack, email, PagerDuty
        return {"alert_sent": True, "alert": alert}

    @task
    def store_profile(current_profile: dict, drift_report: dict):
        """Store profile for future comparison."""
        record = {
            "profile_id": f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "profile": current_profile,
            "drift_detected": drift_report["overall_drift"],
        }

        print(f"Storing profile: {record['profile_id']}")
        # In production: store in database/data warehouse

        return {"stored": True, "profile_id": record["profile_id"]}

    # DAG flow
    baseline = get_baseline_profile()
    current = get_current_data()
    profile = profile_current_data(current)
    drift = detect_drift(baseline, profile)
    alert_on_drift(drift)
    store_profile(profile, drift)


data_profiling()
