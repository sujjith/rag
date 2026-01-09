"""
Phase 6.3: Champion-Challenger / A/B Testing (Enterprise Pattern)

This script demonstrates:
- Champion (production) vs Challenger (candidate) model comparison
- Traffic splitting simulation
- Online metrics collection and comparison
- Automated promotion based on statistical significance
- Safe rollback patterns

Run: python 03_champion_challenger.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import random

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-champion-challenger")

client = MlflowClient()
MODEL_NAME = "breast-cancer-classifier"


class ABTestManager:
    """
    Manages A/B testing between champion and challenger models.
    Simulates production traffic and tracks online metrics.
    """

    def __init__(self, champion_model, challenger_model, traffic_split: float = 0.1):
        """
        Args:
            champion_model: Current production model
            challenger_model: Candidate model to test
            traffic_split: Fraction of traffic to challenger (default 10%)
        """
        self.champion = champion_model
        self.challenger = challenger_model
        self.traffic_split = traffic_split

        # Metrics storage
        self.champion_predictions = []
        self.challenger_predictions = []
        self.champion_latencies = []
        self.challenger_latencies = []
        self.routing_decisions = []

    def route_request(self) -> str:
        """Route incoming request to champion or challenger."""
        if random.random() < self.traffic_split:
            return "challenger"
        return "champion"

    def predict(self, X) -> Tuple[np.ndarray, str]:
        """Route prediction and track metrics."""
        import time

        route = self.route_request()
        self.routing_decisions.append(route)

        if route == "champion":
            start = time.time()
            pred = self.champion.predict(X)
            latency = (time.time() - start) * 1000
            self.champion_latencies.append(latency)
            return pred, "champion"
        else:
            start = time.time()
            pred = self.challenger.predict(X)
            latency = (time.time() - start) * 1000
            self.challenger_latencies.append(latency)
            return pred, "challenger"

    def record_outcome(self, predictions, actuals, model_type: str):
        """Record prediction outcomes for analysis."""
        correct = (predictions == actuals).astype(int)
        if model_type == "champion":
            self.champion_predictions.extend(correct)
        else:
            self.challenger_predictions.extend(correct)

    def get_statistics(self) -> Dict:
        """Calculate A/B test statistics."""
        champion_accuracy = np.mean(self.champion_predictions) if self.champion_predictions else 0
        challenger_accuracy = np.mean(self.challenger_predictions) if self.challenger_predictions else 0

        # Statistical significance test
        if len(self.champion_predictions) > 10 and len(self.challenger_predictions) > 10:
            # Chi-squared test for proportions
            champion_correct = sum(self.champion_predictions)
            champion_total = len(self.champion_predictions)
            challenger_correct = sum(self.challenger_predictions)
            challenger_total = len(self.challenger_predictions)

            # Use proportions z-test
            from statsmodels.stats.proportion import proportions_ztest
            try:
                count = np.array([challenger_correct, champion_correct])
                nobs = np.array([challenger_total, champion_total])
                z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
            except:
                z_stat, p_value = 0, 1.0
        else:
            z_stat, p_value = 0, 1.0

        return {
            'champion_accuracy': champion_accuracy,
            'challenger_accuracy': challenger_accuracy,
            'champion_samples': len(self.champion_predictions),
            'challenger_samples': len(self.challenger_predictions),
            'champion_avg_latency_ms': np.mean(self.champion_latencies) if self.champion_latencies else 0,
            'challenger_avg_latency_ms': np.mean(self.challenger_latencies) if self.challenger_latencies else 0,
            'traffic_split_actual': len([r for r in self.routing_decisions if r == 'challenger']) / len(self.routing_decisions) if self.routing_decisions else 0,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'challenger_better': challenger_accuracy > champion_accuracy
        }


def simulate_production_traffic(ab_manager: ABTestManager, X, y, n_requests: int = 1000):
    """Simulate production traffic for A/B test."""
    indices = np.random.choice(len(X), size=n_requests, replace=True)

    for i, idx in enumerate(indices):
        X_sample = X.iloc[[idx]] if isinstance(X, pd.DataFrame) else X[[idx]]
        y_actual = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]

        pred, route = ab_manager.predict(X_sample)
        ab_manager.record_outcome(pred, [y_actual], route)

        # Progress update
        if (i + 1) % 200 == 0:
            stats = ab_manager.get_statistics()
            print(f"    Requests: {i+1}/{n_requests} | "
                  f"Champion: {stats['champion_accuracy']:.3f} ({stats['champion_samples']}) | "
                  f"Challenger: {stats['challenger_accuracy']:.3f} ({stats['challenger_samples']})")


# Clean up existing model
try:
    client.delete_registered_model(MODEL_NAME)
except:
    pass

print("=" * 70)
print("Champion-Challenger A/B Testing Framework")
print("=" * 70)

# Load data
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# [1] Train Champion Model (Current Production)
print("\n[1] Training Champion Model (Current Production)...")
print("-" * 50)

with mlflow.start_run(run_name="champion-model") as champion_run:
    champion_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    champion_model.fit(X_train, y_train)

    champion_accuracy = accuracy_score(y_test, champion_model.predict(X_test))
    champion_f1 = f1_score(y_test, champion_model.predict(X_test))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("role", "champion")
    mlflow.log_metric("accuracy", champion_accuracy)
    mlflow.log_metric("f1", champion_f1)

    mlflow.sklearn.log_model(champion_model, "model", registered_model_name=MODEL_NAME)
    mlflow.set_tag("model_role", "champion")

    print(f"  Champion trained: accuracy={champion_accuracy:.4f}, f1={champion_f1:.4f}")

import time
time.sleep(2)

# Promote champion to Production
champion_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
client.transition_model_version_stage(MODEL_NAME, champion_version, "Production")
print(f"  Champion promoted to Production (version {champion_version})")


# [2] Train Challenger Model (Candidate)
print("\n[2] Training Challenger Model (Candidate)...")
print("-" * 50)

with mlflow.start_run(run_name="challenger-model") as challenger_run:
    # Challenger uses a different algorithm
    challenger_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    challenger_model.fit(X_train, y_train)

    challenger_accuracy = accuracy_score(y_test, challenger_model.predict(X_test))
    challenger_f1 = f1_score(y_test, challenger_model.predict(X_test))

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("role", "challenger")
    mlflow.log_metric("accuracy", challenger_accuracy)
    mlflow.log_metric("f1", challenger_f1)

    mlflow.sklearn.log_model(challenger_model, "model", registered_model_name=MODEL_NAME)
    mlflow.set_tag("model_role", "challenger")

    print(f"  Challenger trained: accuracy={challenger_accuracy:.4f}, f1={challenger_f1:.4f}")

time.sleep(2)

# Promote challenger to Staging
challenger_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
client.transition_model_version_stage(MODEL_NAME, challenger_version, "Staging")
print(f"  Challenger promoted to Staging (version {challenger_version})")


# [3] Configure A/B Test
print("\n[3] Configuring A/B Test...")
print("-" * 50)

AB_CONFIG = {
    'traffic_split': 0.20,  # 20% to challenger
    'min_samples': 100,
    'significance_level': 0.05,
    'min_improvement': 0.01  # 1% minimum improvement required
}

print(f"  Traffic split: {AB_CONFIG['traffic_split']*100:.0f}% to challenger")
print(f"  Minimum samples: {AB_CONFIG['min_samples']}")
print(f"  Significance level: {AB_CONFIG['significance_level']}")
print(f"  Minimum improvement: {AB_CONFIG['min_improvement']*100:.1f}%")


# [4] Run A/B Test Simulation
print("\n[4] Running A/B Test Simulation...")
print("-" * 50)

ab_manager = ABTestManager(
    champion_model,
    challenger_model,
    traffic_split=AB_CONFIG['traffic_split']
)

# Simulate production traffic
print("  Simulating 1000 production requests...")
simulate_production_traffic(ab_manager, X_test, y_test, n_requests=1000)


# [5] Analyze Results
print("\n[5] A/B Test Results...")
print("-" * 50)

stats = ab_manager.get_statistics()

print(f"""
  Champion (Production):
    - Accuracy: {stats['champion_accuracy']:.4f}
    - Samples: {stats['champion_samples']}
    - Avg Latency: {stats['champion_avg_latency_ms']:.2f}ms

  Challenger (Staging):
    - Accuracy: {stats['challenger_accuracy']:.4f}
    - Samples: {stats['challenger_samples']}
    - Avg Latency: {stats['challenger_avg_latency_ms']:.2f}ms

  Statistical Analysis:
    - Actual traffic split: {stats['traffic_split_actual']*100:.1f}%
    - Z-statistic: {stats['z_statistic']:.4f}
    - P-value: {stats['p_value']:.4f}
    - Significant: {stats['significant']}
    - Challenger better: {stats['challenger_better']}
""")


# [6] Log A/B Test Results
print("\n[6] Logging A/B Test Results...")
print("-" * 50)

with mlflow.start_run(run_name="ab-test-results"):
    mlflow.log_param("champion_version", champion_version)
    mlflow.log_param("challenger_version", challenger_version)
    mlflow.log_param("traffic_split", AB_CONFIG['traffic_split'])
    mlflow.log_param("total_requests", len(ab_manager.routing_decisions))

    mlflow.log_metric("champion_accuracy", stats['champion_accuracy'])
    mlflow.log_metric("challenger_accuracy", stats['challenger_accuracy'])
    mlflow.log_metric("accuracy_improvement", stats['challenger_accuracy'] - stats['champion_accuracy'])
    mlflow.log_metric("p_value", stats['p_value'])

    mlflow.set_tag("test_type", "ab_test")
    mlflow.set_tag("significant", str(stats['significant']))
    mlflow.set_tag("challenger_better", str(stats['challenger_better']))

    # Log detailed results as artifact
    results_dict = {
        'config': AB_CONFIG,
        'statistics': stats,
        'timestamp': datetime.now().isoformat(),
        'champion_version': champion_version,
        'challenger_version': challenger_version
    }
    with open("/tmp/ab_test_results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    mlflow.log_artifact("/tmp/ab_test_results.json")

    print("  Results logged to MLflow")


# [7] Automated Promotion Decision
print("\n[7] Promotion Decision...")
print("-" * 50)

improvement = stats['challenger_accuracy'] - stats['champion_accuracy']
should_promote = (
    stats['significant'] and
    stats['challenger_better'] and
    improvement >= AB_CONFIG['min_improvement'] and
    stats['challenger_samples'] >= AB_CONFIG['min_samples']
)

if should_promote:
    print(f"  DECISION: PROMOTE CHALLENGER")
    print(f"  Reason: Challenger shows {improvement*100:.2f}% improvement with p-value {stats['p_value']:.4f}")

    # Promote challenger to Production
    client.transition_model_version_stage(
        MODEL_NAME, challenger_version, "Production",
        archive_existing_versions=True
    )
    print(f"  Challenger (v{challenger_version}) -> Production")
    print(f"  Champion (v{champion_version}) -> Archived")

    # Update tags
    client.set_model_version_tag(MODEL_NAME, challenger_version, "promotion_reason", "ab_test_winner")
    client.set_model_version_tag(MODEL_NAME, challenger_version, "ab_test_improvement", f"{improvement*100:.2f}%")

else:
    print(f"  DECISION: KEEP CHAMPION")
    reasons = []
    if not stats['significant']:
        reasons.append(f"not statistically significant (p={stats['p_value']:.4f})")
    if not stats['challenger_better']:
        reasons.append("challenger not better")
    if improvement < AB_CONFIG['min_improvement']:
        reasons.append(f"improvement {improvement*100:.2f}% < {AB_CONFIG['min_improvement']*100:.1f}%")
    if stats['challenger_samples'] < AB_CONFIG['min_samples']:
        reasons.append(f"insufficient samples ({stats['challenger_samples']})")

    print(f"  Reason: {', '.join(reasons)}")

    # Keep challenger in Staging for further testing or archive
    print(f"  Challenger (v{challenger_version}) remains in Staging")


# [8] Rollback Demonstration
print("\n[8] Rollback Pattern (for emergencies)...")
print("-" * 50)

print("""
  Emergency Rollback Steps:
  1. Identify problematic version in Production
  2. Find previous stable version in Archived
  3. Restore archived version to Production
  4. Archive problematic version

  Example code:
  ```python
  # Emergency rollback
  client.transition_model_version_stage(MODEL_NAME, "problematic_version", "Archived")
  client.transition_model_version_stage(MODEL_NAME, "previous_stable", "Production")
  ```
""")

# Clean up
os.remove("/tmp/ab_test_results.json")


print("\n" + "=" * 70)
print("A/B Testing Framework Complete!")
print("=" * 70)
print(f"""
  Key Enterprise Patterns Demonstrated:
  - Champion vs Challenger model comparison
  - Traffic splitting with configurable ratios
  - Online metrics collection and analysis
  - Statistical significance testing
  - Automated promotion/rejection decisions
  - Safe rollback patterns

  Production Considerations:
  - Use feature flags for traffic splitting
  - Implement circuit breakers for failures
  - Monitor latency degradation
  - Set up alerting for anomalies
  - Gradual rollout (10% -> 25% -> 50% -> 100%)

  View at: {TRACKING_URI}
""")
print("=" * 70)
