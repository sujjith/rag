"""
Phase 6.2: Model Validation Gates (Enterprise Pattern)

This script demonstrates:
- Pre-deployment validation checks
- Performance thresholds and quality gates
- Data quality validation
- Model behavior testing (smoke tests)
- Automated promotion/rejection decisions

Run: python 02_validation_gates.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from enum import Enum

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-validation-gates")

client = MlflowClient()
MODEL_NAME = "iris-validated-model"


class ValidationStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class ValidationResult:
    name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = None


class ModelValidator:
    """Enterprise model validation framework."""

    def __init__(self, model, X_test, y_test, validation_config: Dict):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.config = validation_config
        self.results: List[ValidationResult] = []

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        self.results = []

        # Run all validation checks
        self._validate_performance()
        self._validate_stability()
        self._validate_data_quality()
        self._validate_model_behavior()
        self._validate_prediction_distribution()
        self._validate_latency()

        # Determine overall pass/fail
        failed = [r for r in self.results if r.status == ValidationStatus.FAILED]
        warnings = [r for r in self.results if r.status == ValidationStatus.WARNING]

        overall_passed = len(failed) == 0
        return overall_passed, self.results

    def _validate_performance(self):
        """Check model meets minimum performance thresholds."""
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        min_accuracy = self.config.get('min_accuracy', 0.85)
        min_f1 = self.config.get('min_f1', 0.80)

        # Accuracy check
        if accuracy >= min_accuracy:
            self.results.append(ValidationResult(
                name="accuracy_threshold",
                status=ValidationStatus.PASSED,
                message=f"Accuracy {accuracy:.4f} >= {min_accuracy}",
                details={'accuracy': accuracy, 'threshold': min_accuracy}
            ))
        else:
            self.results.append(ValidationResult(
                name="accuracy_threshold",
                status=ValidationStatus.FAILED,
                message=f"Accuracy {accuracy:.4f} < {min_accuracy}",
                details={'accuracy': accuracy, 'threshold': min_accuracy}
            ))

        # F1 check
        if f1 >= min_f1:
            self.results.append(ValidationResult(
                name="f1_threshold",
                status=ValidationStatus.PASSED,
                message=f"F1 score {f1:.4f} >= {min_f1}",
                details={'f1': f1, 'threshold': min_f1}
            ))
        else:
            self.results.append(ValidationResult(
                name="f1_threshold",
                status=ValidationStatus.FAILED,
                message=f"F1 score {f1:.4f} < {min_f1}",
                details={'f1': f1, 'threshold': min_f1}
            ))

    def _validate_stability(self):
        """Check model prediction stability across multiple runs."""
        predictions = []
        for _ in range(5):
            pred = self.model.predict(self.X_test)
            predictions.append(pred)

        # Check if predictions are deterministic
        all_same = all(np.array_equal(predictions[0], p) for p in predictions)

        if all_same:
            self.results.append(ValidationResult(
                name="prediction_stability",
                status=ValidationStatus.PASSED,
                message="Model predictions are deterministic"
            ))
        else:
            self.results.append(ValidationResult(
                name="prediction_stability",
                status=ValidationStatus.WARNING,
                message="Model predictions vary between runs (may be expected for some models)"
            ))

    def _validate_data_quality(self):
        """Validate input data quality."""
        X = self.X_test

        # Check for NaN/Inf values
        has_nan = X.isna().any().any() if isinstance(X, pd.DataFrame) else np.isnan(X).any()
        has_inf = np.isinf(X.values if isinstance(X, pd.DataFrame) else X).any()

        if not has_nan and not has_inf:
            self.results.append(ValidationResult(
                name="data_quality",
                status=ValidationStatus.PASSED,
                message="No NaN or Inf values in test data"
            ))
        else:
            issues = []
            if has_nan:
                issues.append("NaN values found")
            if has_inf:
                issues.append("Inf values found")
            self.results.append(ValidationResult(
                name="data_quality",
                status=ValidationStatus.FAILED,
                message=f"Data quality issues: {', '.join(issues)}"
            ))

    def _validate_model_behavior(self):
        """Smoke tests for model behavior."""
        # Test 1: Model can handle single sample
        try:
            single_pred = self.model.predict(self.X_test.iloc[:1] if isinstance(self.X_test, pd.DataFrame)
                                            else self.X_test[:1])
            single_sample_ok = len(single_pred) == 1
        except Exception as e:
            single_sample_ok = False

        # Test 2: Model returns expected output shape
        predictions = self.model.predict(self.X_test)
        correct_shape = len(predictions) == len(self.y_test)

        # Test 3: Predictions are valid class labels
        unique_preds = set(predictions)
        unique_labels = set(self.y_test)
        valid_labels = unique_preds.issubset(unique_labels)

        all_passed = single_sample_ok and correct_shape and valid_labels

        if all_passed:
            self.results.append(ValidationResult(
                name="model_behavior",
                status=ValidationStatus.PASSED,
                message="All smoke tests passed"
            ))
        else:
            failures = []
            if not single_sample_ok:
                failures.append("single sample prediction failed")
            if not correct_shape:
                failures.append("output shape mismatch")
            if not valid_labels:
                failures.append(f"invalid predictions: {unique_preds - unique_labels}")
            self.results.append(ValidationResult(
                name="model_behavior",
                status=ValidationStatus.FAILED,
                message=f"Smoke test failures: {', '.join(failures)}"
            ))

    def _validate_prediction_distribution(self):
        """Check for prediction bias/imbalance."""
        predictions = self.model.predict(self.X_test)
        pred_counts = pd.Series(predictions).value_counts(normalize=True)

        # Check if any class dominates predictions (>80%)
        max_class_ratio = pred_counts.max()
        max_allowed_ratio = self.config.get('max_class_ratio', 0.80)

        if max_class_ratio <= max_allowed_ratio:
            self.results.append(ValidationResult(
                name="prediction_distribution",
                status=ValidationStatus.PASSED,
                message=f"Prediction distribution is balanced (max ratio: {max_class_ratio:.2f})",
                details={'distribution': pred_counts.to_dict()}
            ))
        else:
            self.results.append(ValidationResult(
                name="prediction_distribution",
                status=ValidationStatus.WARNING,
                message=f"Predictions may be biased (max ratio: {max_class_ratio:.2f} > {max_allowed_ratio})",
                details={'distribution': pred_counts.to_dict()}
            ))

    def _validate_latency(self):
        """Check prediction latency."""
        import time

        # Measure prediction time
        times = []
        for _ in range(10):
            start = time.time()
            _ = self.model.predict(self.X_test)
            times.append(time.time() - start)

        avg_latency_ms = np.mean(times) * 1000
        max_latency_ms = self.config.get('max_latency_ms', 100)

        if avg_latency_ms <= max_latency_ms:
            self.results.append(ValidationResult(
                name="latency",
                status=ValidationStatus.PASSED,
                message=f"Average latency {avg_latency_ms:.2f}ms <= {max_latency_ms}ms",
                details={'avg_latency_ms': avg_latency_ms}
            ))
        else:
            self.results.append(ValidationResult(
                name="latency",
                status=ValidationStatus.WARNING,
                message=f"Average latency {avg_latency_ms:.2f}ms > {max_latency_ms}ms",
                details={'avg_latency_ms': avg_latency_ms}
            ))


def validate_and_promote(model_name: str, version: str, validation_config: Dict) -> bool:
    """Validate a registered model and promote if passes."""
    print(f"\n  Validating {model_name} version {version}...")

    # Load model
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Load test data (in production, this would come from a validation dataset)
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run validation
    validator = ModelValidator(model, X_test, y_test, validation_config)
    passed, results = validator.validate_all()

    # Log validation results
    with mlflow.start_run(run_name=f"validation-{model_name}-v{version}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", version)
        mlflow.log_param("validation_passed", passed)

        for result in results:
            mlflow.log_param(f"check_{result.name}", result.status.value)
            if result.details:
                for k, v in result.details.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{result.name}_{k}", v)

        mlflow.set_tag("validation_type", "pre-deployment")
        mlflow.set_tag("overall_status", "PASSED" if passed else "FAILED")

    return passed, results


# Clean up existing model
try:
    client.delete_registered_model(MODEL_NAME)
except:
    pass

print("=" * 70)
print("Enterprise Model Validation Gates")
print("=" * 70)

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# [1] Train and Register Model
print("\n[1] Training and Registering Model...")
print("-" * 50)

with mlflow.start_run(run_name="model-for-validation"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name=MODEL_NAME
    )
    print(f"  Model trained with accuracy: {accuracy:.4f}")

import time
time.sleep(2)

# Get latest version
versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
model_version = versions[0].version
print(f"  Registered as {MODEL_NAME} version {model_version}")


# [2] Define Validation Configuration
print("\n[2] Validation Configuration...")
print("-" * 50)

validation_config = {
    'min_accuracy': 0.90,
    'min_f1': 0.85,
    'max_class_ratio': 0.80,
    'max_latency_ms': 100
}

print("  Quality gates:")
for key, value in validation_config.items():
    print(f"    {key}: {value}")


# [3] Run Validation
print("\n[3] Running Validation Gates...")
print("-" * 50)

passed, results = validate_and_promote(MODEL_NAME, model_version, validation_config)

print("\n  Validation Results:")
print("  " + "-" * 45)
for result in results:
    status_icon = {"PASSED": "[PASS]", "FAILED": "[FAIL]", "WARNING": "[WARN]"}[result.status.value]
    print(f"  {status_icon} {result.name}: {result.message}")
print("  " + "-" * 45)


# [4] Automated Promotion Decision
print("\n[4] Promotion Decision...")
print("-" * 50)

if passed:
    # Promote to Staging first
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Staging"
    )
    print(f"  APPROVED: Model promoted to Staging")
    print(f"  Next step: Manual review before Production")

    # In a real scenario, you might also:
    # - Send notification
    # - Create JIRA ticket
    # - Update deployment config
else:
    # Reject the model
    failed_checks = [r.name for r in results if r.status == ValidationStatus.FAILED]
    client.update_model_version(
        name=MODEL_NAME,
        version=model_version,
        description=f"REJECTED: Failed validation checks: {', '.join(failed_checks)}"
    )
    print(f"  REJECTED: Model failed validation")
    print(f"  Failed checks: {', '.join(failed_checks)}")


# [5] Test with Failing Model
print("\n[5] Testing with Intentionally Weak Model...")
print("-" * 50)

# Train a deliberately weak model
with mlflow.start_run(run_name="weak-model-for-validation"):
    weak_model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
    weak_model.fit(X_train, y_train)

    weak_accuracy = accuracy_score(y_test, weak_model.predict(X_test))
    mlflow.log_metric("accuracy", weak_accuracy)

    mlflow.sklearn.log_model(
        weak_model, "model",
        registered_model_name=MODEL_NAME
    )
    print(f"  Weak model trained with accuracy: {weak_accuracy:.4f}")

time.sleep(2)

versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
weak_version = versions[0].version

passed_weak, results_weak = validate_and_promote(MODEL_NAME, weak_version, validation_config)

print("\n  Weak Model Validation Results:")
print("  " + "-" * 45)
for result in results_weak:
    status_icon = {"PASSED": "[PASS]", "FAILED": "[FAIL]", "WARNING": "[WARN]"}[result.status.value]
    print(f"  {status_icon} {result.name}: {result.message}")
print("  " + "-" * 45)
print(f"\n  Expected: REJECTED (passed={passed_weak})")


# Summary
print("\n" + "=" * 70)
print("Validation Gates Summary")
print("=" * 70)
print(f"""
  Validation Framework Features:
  - Performance thresholds (accuracy, F1)
  - Data quality checks (NaN, Inf detection)
  - Model behavior smoke tests
  - Prediction distribution analysis
  - Latency benchmarking
  - Automated promotion/rejection

  Production Integration:
  - Integrate with CI/CD pipelines
  - Connect to notification systems
  - Add custom domain-specific validators
  - Implement rollback mechanisms

  View at: {TRACKING_URI}
""")
print("=" * 70)
