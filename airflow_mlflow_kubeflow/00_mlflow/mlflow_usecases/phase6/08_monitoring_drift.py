"""
Phase 6.8: Production Monitoring and Drift Detection (Enterprise Pattern)

This script demonstrates:
- Tracking production predictions
- Data drift detection
- Model performance monitoring
- Feature distribution analysis
- Alerting patterns
- Retraining triggers

Run: python 08_monitoring_drift.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-monitoring-drift")

client = MlflowClient()


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    feature: str
    drift_score: float
    p_value: float
    severity: DriftSeverity
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float


class DataDriftDetector:
    """Detects drift between baseline and current data distributions."""

    def __init__(self, baseline_data: pd.DataFrame, thresholds: Dict = None):
        self.baseline = baseline_data
        self.thresholds = thresholds or {
            'low': 0.1,
            'medium': 0.2,
            'high': 0.3,
            'critical': 0.5
        }

        # Compute baseline statistics
        self.baseline_stats = {
            col: {
                'mean': baseline_data[col].mean(),
                'std': baseline_data[col].std(),
                'min': baseline_data[col].min(),
                'max': baseline_data[col].max(),
                'distribution': baseline_data[col].values
            }
            for col in baseline_data.columns
        }

    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect drift for all features."""
        results = []

        for col in current_data.columns:
            if col not in self.baseline_stats:
                continue

            baseline_dist = self.baseline_stats[col]['distribution']
            current_dist = current_data[col].values

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_dist, current_dist)

            # Population Stability Index (PSI)
            psi = self._calculate_psi(baseline_dist, current_dist)

            # Determine severity
            severity = self._get_severity(psi)

            results.append(DriftResult(
                feature=col,
                drift_score=psi,
                p_value=p_value,
                severity=severity,
                baseline_mean=self.baseline_stats[col]['mean'],
                current_mean=current_data[col].mean(),
                baseline_std=self.baseline_stats[col]['std'],
                current_std=current_data[col].std()
            ))

        return results

    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray,
                       buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create buckets based on baseline
        breakpoints = np.percentile(baseline, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]

        # Avoid division by zero
        baseline_props = (baseline_counts + 0.0001) / len(baseline)
        current_props = (current_counts + 0.0001) / len(current)

        # PSI formula
        psi = np.sum((current_props - baseline_props) *
                     np.log(current_props / baseline_props))

        return psi

    def _get_severity(self, psi: float) -> DriftSeverity:
        """Map PSI to severity level."""
        if psi < self.thresholds['low']:
            return DriftSeverity.NONE
        elif psi < self.thresholds['medium']:
            return DriftSeverity.LOW
        elif psi < self.thresholds['high']:
            return DriftSeverity.MEDIUM
        elif psi < self.thresholds['critical']:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class ModelMonitor:
    """Monitors model performance in production."""

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.predictions_log = []
        self.actuals_log = []
        self.timestamps = []

    def log_prediction(self, features: np.ndarray, prediction: int,
                       actual: Optional[int] = None):
        """Log a prediction for monitoring."""
        self.predictions_log.append(prediction)
        self.timestamps.append(datetime.now())
        if actual is not None:
            self.actuals_log.append(actual)

    def get_recent_accuracy(self, window_size: int = 100) -> Optional[float]:
        """Calculate accuracy over recent predictions with known actuals."""
        if len(self.actuals_log) < window_size:
            return None

        recent_preds = self.predictions_log[-window_size:]
        recent_actuals = self.actuals_log[-window_size:]

        return accuracy_score(recent_actuals, recent_preds)

    def get_prediction_distribution(self, window_size: int = 100) -> Dict:
        """Get distribution of recent predictions."""
        recent = self.predictions_log[-window_size:]
        unique, counts = np.unique(recent, return_counts=True)
        return dict(zip(unique.tolist(), (counts / len(recent)).tolist()))

    def detect_performance_degradation(self, baseline_accuracy: float,
                                       threshold: float = 0.05) -> Tuple[bool, float]:
        """Check if model performance has degraded."""
        current_accuracy = self.get_recent_accuracy()
        if current_accuracy is None:
            return False, 0.0

        degradation = baseline_accuracy - current_accuracy
        is_degraded = degradation > threshold

        return is_degraded, degradation


def simulate_production_data(baseline_data: pd.DataFrame,
                              drift_amount: float = 0.0) -> pd.DataFrame:
    """Simulate production data with optional drift."""
    production_data = baseline_data.copy()

    if drift_amount > 0:
        # Add drift to features
        for col in production_data.columns:
            noise = np.random.normal(drift_amount, drift_amount/2, len(production_data))
            production_data[col] = production_data[col] + noise

    return production_data


print("=" * 70)
print("Production Monitoring and Drift Detection")
print("=" * 70)


# [1] Setup: Train Baseline Model
print("\n[1] Setting Up Baseline Model...")
print("-" * 50)

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
with mlflow.start_run(run_name="baseline-model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"  Baseline accuracy: {baseline_accuracy:.4f}")


# [2] Initialize Drift Detector
print("\n[2] Initializing Drift Detector...")
print("-" * 50)

drift_detector = DataDriftDetector(X_train)
print(f"  Baseline established with {len(X_train)} samples")
print(f"  Features monitored: {list(X_train.columns)}")


# [3] Simulate Production Scenario - No Drift
print("\n[3] Scenario 1: Production Data (No Drift)...")
print("-" * 50)

production_data_clean = simulate_production_data(X_test, drift_amount=0.0)
drift_results_clean = drift_detector.detect_drift(production_data_clean)

print("  Drift Analysis Results:")
for result in drift_results_clean:
    print(f"    {result.feature}: PSI={result.drift_score:.4f}, "
          f"severity={result.severity.value}")


# [4] Simulate Production Scenario - With Drift
print("\n[4] Scenario 2: Production Data (With Drift)...")
print("-" * 50)

production_data_drifted = simulate_production_data(X_test, drift_amount=2.0)
drift_results_drifted = drift_detector.detect_drift(production_data_drifted)

print("  Drift Analysis Results:")
for result in drift_results_drifted:
    status_icon = {"none": "[OK]", "low": "[!]", "medium": "[!!]",
                   "high": "[!!!]", "critical": "[ALERT]"}[result.severity.value]
    print(f"    {status_icon} {result.feature}: PSI={result.drift_score:.4f}, "
          f"severity={result.severity.value}")


# [5] Log Drift Metrics to MLflow
print("\n[5] Logging Drift Metrics to MLflow...")
print("-" * 50)

with mlflow.start_run(run_name="drift-monitoring-run"):
    # Log overall drift status
    max_drift = max(r.drift_score for r in drift_results_drifted)
    max_severity = max(r.severity.value for r in drift_results_drifted)

    mlflow.log_metric("max_drift_score", max_drift)
    mlflow.log_param("max_drift_severity", max_severity)

    # Log per-feature drift
    for result in drift_results_drifted:
        mlflow.log_metric(f"drift_{result.feature}", result.drift_score)
        mlflow.log_metric(f"pvalue_{result.feature}", result.p_value)

    # Log feature statistics comparison
    drift_report = {
        'timestamp': datetime.now().isoformat(),
        'features': [
            {
                'name': r.feature,
                'drift_score': r.drift_score,
                'p_value': r.p_value,
                'severity': r.severity.value,
                'baseline_mean': r.baseline_mean,
                'current_mean': r.current_mean
            }
            for r in drift_results_drifted
        ]
    }

    # Save report as artifact
    with open("/tmp/drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)
    mlflow.log_artifact("/tmp/drift_report.json")

    mlflow.set_tag("monitoring_type", "drift_detection")
    mlflow.set_tag("drift_detected", str(max_drift > 0.1))

    print(f"  Drift metrics logged")
    print(f"  Max drift score: {max_drift:.4f}")


# [6] Model Performance Monitoring
print("\n[6] Simulating Performance Monitoring...")
print("-" * 50)

monitor = ModelMonitor("iris-model", "1")

# Simulate predictions with feedback
print("  Simulating 200 production predictions...")
for i in range(200):
    idx = i % len(X_test)
    features = X_test.iloc[idx].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    actual = y_test.iloc[idx]

    monitor.log_prediction(features, prediction, actual)

# Check performance
current_accuracy = monitor.get_recent_accuracy(window_size=100)
print(f"  Rolling accuracy (last 100): {current_accuracy:.4f}")

is_degraded, degradation = monitor.detect_performance_degradation(
    baseline_accuracy, threshold=0.05
)
print(f"  Performance degraded: {is_degraded} (delta: {degradation:.4f})")

# Prediction distribution
pred_dist = monitor.get_prediction_distribution()
print(f"  Prediction distribution: {pred_dist}")


# [7] Alerting Patterns
print("\n[7] Alerting Patterns...")
print("-" * 50)

def check_alerts(drift_results: List[DriftResult],
                 performance_degraded: bool,
                 prediction_distribution: Dict) -> List[Dict]:
    """Check for alert conditions."""
    alerts = []

    # Drift alerts
    for result in drift_results:
        if result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            alerts.append({
                'type': 'DATA_DRIFT',
                'severity': result.severity.value,
                'feature': result.feature,
                'message': f"High drift detected in {result.feature} (PSI={result.drift_score:.3f})"
            })

    # Performance alert
    if performance_degraded:
        alerts.append({
            'type': 'PERFORMANCE_DEGRADATION',
            'severity': 'high',
            'message': f"Model accuracy dropped below threshold"
        })

    # Distribution skew alert
    max_class_ratio = max(prediction_distribution.values()) if prediction_distribution else 0
    if max_class_ratio > 0.8:
        alerts.append({
            'type': 'PREDICTION_SKEW',
            'severity': 'medium',
            'message': f"Predictions heavily skewed to one class ({max_class_ratio:.1%})"
        })

    return alerts

alerts = check_alerts(drift_results_drifted, is_degraded, pred_dist)

print("  Alert Check Results:")
if alerts:
    for alert in alerts:
        print(f"    [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
else:
    print("    No alerts triggered")


# [8] Retraining Trigger Logic
print("\n[8] Retraining Trigger Logic...")
print("-" * 50)

def should_retrain(drift_results: List[DriftResult],
                   performance_degraded: bool,
                   days_since_last_train: int = 0) -> Tuple[bool, str]:
    """Determine if model should be retrained."""
    reasons = []

    # Check drift
    critical_drift = any(r.severity == DriftSeverity.CRITICAL for r in drift_results)
    high_drift_count = sum(1 for r in drift_results
                          if r.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL])

    if critical_drift:
        reasons.append("Critical drift detected")
    if high_drift_count >= 2:
        reasons.append(f"Multiple features ({high_drift_count}) showing high drift")

    # Check performance
    if performance_degraded:
        reasons.append("Performance below threshold")

    # Check time-based trigger
    if days_since_last_train > 30:
        reasons.append(f"Model age ({days_since_last_train} days) exceeds threshold")

    should_trigger = len(reasons) > 0
    return should_trigger, "; ".join(reasons) if reasons else "Model healthy"

retrain_needed, reason = should_retrain(drift_results_drifted, is_degraded, 45)
print(f"  Retraining recommended: {retrain_needed}")
print(f"  Reason: {reason}")


# [9] Log Monitoring Summary
print("\n[9] Logging Monitoring Summary...")
print("-" * 50)

with mlflow.start_run(run_name="monitoring-summary"):
    mlflow.log_param("monitoring_timestamp", datetime.now().isoformat())
    mlflow.log_param("retrain_recommended", retrain_needed)
    mlflow.log_param("retrain_reason", reason)

    mlflow.log_metric("current_accuracy", current_accuracy or 0)
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.log_metric("alert_count", len(alerts))

    mlflow.set_tag("monitoring_status", "needs_attention" if retrain_needed else "healthy")

    print("  Monitoring summary logged")

# Clean up
os.remove("/tmp/drift_report.json")


# [10] Monitoring Dashboard Patterns
print("\n[10] Monitoring Dashboard Patterns...")
print("-" * 50)

print("""
  Recommended Monitoring Dashboard Components:

  1. Real-time Metrics:
     - Prediction latency (p50, p95, p99)
     - Throughput (requests/second)
     - Error rate

  2. Model Performance:
     - Rolling accuracy (with confidence intervals)
     - Precision/Recall by class
     - Confusion matrix heatmap

  3. Data Quality:
     - Feature drift indicators
     - Missing value rates
     - Out-of-range value counts

  4. Alerts Panel:
     - Active alerts with severity
     - Alert history timeline
     - Acknowledgment status

  5. Business Metrics:
     - Prediction value distribution
     - Decision threshold impact
     - A/B test status

  Tools for Implementation:
  - Grafana + MLflow metrics export
  - Custom Streamlit dashboard
  - Evidently AI for drift visualization
  - WhyLabs for monitoring
""")


print("\n" + "=" * 70)
print("Monitoring and Drift Detection Complete!")
print("=" * 70)
print(f"""
  Key Capabilities Demonstrated:

  1. Data drift detection using PSI and KS tests
  2. Feature-level drift analysis
  3. Performance monitoring with rolling windows
  4. Alert condition checking
  5. Retraining trigger logic
  6. MLflow logging integration

  Enterprise Integration Points:

  - Schedule drift checks (hourly/daily)
  - Connect to alerting systems (PagerDuty, Slack)
  - Trigger automated retraining pipelines
  - Feed metrics to dashboards
  - Implement circuit breakers for bad models

  Production Checklist:
  [ ] Define baseline data snapshot
  [ ] Set drift thresholds per feature
  [ ] Configure alert channels
  [ ] Set up scheduled monitoring jobs
  [ ] Implement automatic fallback

  View at: {TRACKING_URI}
""")
print("=" * 70)
