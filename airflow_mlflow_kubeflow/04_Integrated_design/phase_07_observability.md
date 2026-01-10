# Phase 07: Observability Stack

## Overview

Complete observability setup with Prometheus, Grafana, Loki, and Alertmanager for monitoring ML pipelines.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                         GRAFANA                                     │    │
│   │            (Visualization & Dashboards)                             │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                    │              │              │                           │
│          ┌─────────┴───┐   ┌─────┴─────┐   ┌───┴──────┐                    │
│          │             │   │           │   │          │                    │
│          ▼             ▼   ▼           ▼   ▼          ▼                    │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │
│   │ Prometheus │ │    Loki    │ │   Tempo    │ │Alertmanager│             │
│   │  (Metrics) │ │   (Logs)   │ │  (Traces)  │ │  (Alerts)  │             │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘             │
│          │              │              │              │                    │
│          └──────────────┼──────────────┼──────────────┘                    │
│                         │              │                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      DATA SOURCES                                    │  │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │
│   │  │ Airflow │ │ MLflow  │ │Kubeflow │ │ KServe  │ │  K8s    │       │  │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install Prometheus Stack

### Deploy with Helm

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create namespace
kubectl create namespace monitoring
```

Create `monitoring/prometheus-values.yaml`:

```yaml
# Prometheus Stack Values
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
    additionalScrapeConfigs:
      - job_name: 'mlflow'
        static_configs:
          - targets: ['mlflow.mlflow.svc.cluster.local:5000']
      - job_name: 'airflow'
        static_configs:
          - targets: ['airflow-statsd.airflow.svc.cluster.local:9102']
      - job_name: 'feast'
        static_configs:
          - targets: ['feast-feature-server.feast.svc.cluster.local:6566']
      - job_name: 'kserve'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names: ['kserve']
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true

grafana:
  enabled: true
  adminPassword: admin123
  persistence:
    enabled: true
    size: 10Gi
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: 'default'
          orgId: 1
          folder: 'MLOps'
          type: file
          disableDeletion: false
          editable: true
          options:
            path: /var/lib/grafana/dashboards/default
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-kube-prometheus-prometheus:9090
          isDefault: true
        - name: Loki
          type: loki
          url: http://loki:3100
        - name: Tempo
          type: tempo
          url: http://tempo:3100
  ingress:
    enabled: true
    ingressClassName: nginx
    hosts:
      - grafana.local

alertmanager:
  enabled: true
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 5Gi
  config:
    global:
      slack_api_url: '${SLACK_WEBHOOK_URL}'
    route:
      group_by: ['alertname', 'namespace']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      receiver: 'slack-notifications'
      routes:
        - match:
            severity: critical
          receiver: 'pagerduty'
        - match:
            severity: warning
          receiver: 'slack-notifications'
    receivers:
      - name: 'slack-notifications'
        slack_configs:
          - channel: '#ml-alerts'
            send_resolved: true
            title: '{{ .Status | toUpper }}: {{ .CommonAnnotations.summary }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: '${PAGERDUTY_KEY}'
            severity: '{{ .CommonLabels.severity }}'

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true
```

```bash
# Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --values monitoring/prometheus-values.yaml \
    --wait

# Verify
kubectl get pods -n monitoring
```

---

## Step 2: Install Loki (Log Aggregation)

Create `monitoring/loki-values.yaml`:

```yaml
loki:
  auth_enabled: false
  commonConfig:
    replication_factor: 1
  storage:
    type: filesystem
  persistence:
    enabled: true
    size: 50Gi

promtail:
  enabled: true
  config:
    clients:
      - url: http://loki:3100/loki/api/v1/push
    snippets:
      pipelineStages:
        - cri: {}
        - json:
            expressions:
              level: level
              msg: msg
        - labels:
            level:
        - match:
            selector: '{app=~"airflow.*"}'
            stages:
              - regex:
                  expression: '.*dag_id=(?P<dag_id>[^,]+).*'
              - labels:
                  dag_id:
        - match:
            selector: '{app=~"mlflow.*"}'
            stages:
              - regex:
                  expression: '.*experiment_id=(?P<experiment_id>[^,]+).*'
              - labels:
                  experiment_id:
```

```bash
# Add Grafana Helm repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Loki
helm install loki grafana/loki-stack \
    --namespace monitoring \
    --values monitoring/loki-values.yaml
```

---

## Step 3: Custom ML Metrics

### MLflow Metrics Exporter

Create `monitoring/exporters/mlflow_exporter.py`:

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import mlflow
from mlflow import MlflowClient
import time
import threading

# Define metrics
EXPERIMENTS_TOTAL = Gauge('mlflow_experiments_total', 'Total number of experiments')
RUNS_TOTAL = Gauge('mlflow_runs_total', 'Total number of runs', ['experiment_name', 'status'])
MODELS_TOTAL = Gauge('mlflow_registered_models_total', 'Total registered models')
MODEL_VERSIONS = Gauge('mlflow_model_versions', 'Model versions', ['model_name', 'stage'])

RUN_METRICS = Gauge('mlflow_run_metric', 'Run metrics', ['experiment_name', 'metric_name'])
RUN_DURATION = Histogram('mlflow_run_duration_seconds', 'Run duration', ['experiment_name'])

class MLflowExporter:
    def __init__(self, tracking_uri: str, port: int = 9101):
        self.client = MlflowClient(tracking_uri)
        self.port = port

    def collect_metrics(self):
        """Collect metrics from MLflow"""
        # Experiments
        experiments = self.client.search_experiments()
        EXPERIMENTS_TOTAL.set(len(experiments))

        for exp in experiments:
            # Runs by status
            for status in ['RUNNING', 'FINISHED', 'FAILED', 'KILLED']:
                runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"status = '{status}'"
                )
                RUNS_TOTAL.labels(
                    experiment_name=exp.name,
                    status=status
                ).set(len(runs))

            # Latest run metrics
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1,
                order_by=["end_time DESC"]
            )
            if runs:
                for key, value in runs[0].data.metrics.items():
                    RUN_METRICS.labels(
                        experiment_name=exp.name,
                        metric_name=key
                    ).set(value)

        # Registered models
        models = list(self.client.search_registered_models())
        MODELS_TOTAL.set(len(models))

        for model in models:
            for stage in ['None', 'Staging', 'Production', 'Archived']:
                versions = self.client.get_latest_versions(model.name, stages=[stage])
                MODEL_VERSIONS.labels(
                    model_name=model.name,
                    stage=stage
                ).set(len(versions))

    def run(self):
        """Start exporter"""
        start_http_server(self.port)
        print(f"MLflow exporter running on port {self.port}")

        while True:
            try:
                self.collect_metrics()
            except Exception as e:
                print(f"Error collecting metrics: {e}")
            time.sleep(30)

if __name__ == "__main__":
    exporter = MLflowExporter(
        tracking_uri="http://mlflow.mlflow.svc.cluster.local:5000",
        port=9101
    )
    exporter.run()
```

### Model Inference Metrics

Create `monitoring/exporters/inference_metrics.py`:

```python
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
PREDICTION_REQUESTS = Counter(
    'model_prediction_requests_total',
    'Total prediction requests',
    ['model_name', 'model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model_name', 'model_version'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Prediction confidence scores',
    ['model_name', 'class'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

DATA_DRIFT_SCORE = Gauge(
    'model_data_drift_score',
    'Data drift score',
    ['model_name', 'feature']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy_score',
    'Model accuracy on validation data',
    ['model_name', 'model_version']
)

def track_prediction(model_name: str, model_version: str):
    """Decorator to track prediction metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                PREDICTION_REQUESTS.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status='success'
                ).inc()

                # Track confidence if available
                if hasattr(result, 'confidence'):
                    PREDICTION_CONFIDENCE.labels(
                        model_name=model_name,
                        class=result.predicted_class
                    ).observe(result.confidence)

                return result

            except Exception as e:
                PREDICTION_REQUESTS.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status='error'
                ).inc()
                raise

            finally:
                duration = time.time() - start_time
                PREDICTION_LATENCY.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)

        return wrapper
    return decorator

# Usage example
class ChurnPredictor:
    def __init__(self):
        self.model_name = "churn-classifier"
        self.model_version = "v1.0"

    @track_prediction(model_name="churn-classifier", model_version="v1.0")
    def predict(self, features):
        # Prediction logic
        pass
```

---

## Step 4: Alerting Rules

Create `monitoring/alerts/ml-alerts.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-alerts
  namespace: monitoring
spec:
  groups:
    - name: ml-pipeline-alerts
      rules:
        # Airflow DAG Failures
        - alert: AirflowDAGFailure
          expr: airflow_dag_run_status{status="failed"} > 0
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Airflow DAG {{ $labels.dag_id }} failed"
            description: "DAG {{ $labels.dag_id }} has been failing for 5 minutes"

        # Long Running DAGs
        - alert: AirflowDAGRunningTooLong
          expr: airflow_dag_run_duration_seconds > 7200
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "DAG {{ $labels.dag_id }} running too long"
            description: "DAG has been running for over 2 hours"

        # MLflow Experiment Failures
        - alert: MLflowExperimentFailed
          expr: increase(mlflow_runs_total{status="FAILED"}[1h]) > 5
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Multiple MLflow runs failed"
            description: "More than 5 runs failed in the last hour"

        # Model Performance Degradation
        - alert: ModelAccuracyDegraded
          expr: model_accuracy_score < 0.8
          for: 15m
          labels:
            severity: critical
          annotations:
            summary: "Model {{ $labels.model_name }} accuracy degraded"
            description: "Model accuracy dropped below 80%"

        # High Prediction Latency
        - alert: HighPredictionLatency
          expr: histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m])) > 1
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "High prediction latency for {{ $labels.model_name }}"
            description: "95th percentile latency is above 1 second"

        # Data Drift Detection
        - alert: DataDriftDetected
          expr: model_data_drift_score > 0.3
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "Data drift detected for {{ $labels.model_name }}"
            description: "Feature {{ $labels.feature }} has drifted significantly"

        # Prediction Error Rate
        - alert: HighPredictionErrorRate
          expr: |
            sum(rate(model_prediction_requests_total{status="error"}[5m])) /
            sum(rate(model_prediction_requests_total[5m])) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High prediction error rate"
            description: "Error rate is above 5%"

        # KServe Pod Failures
        - alert: KServePodNotReady
          expr: kube_pod_status_ready{namespace="kserve", condition="true"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "KServe pod not ready"
            description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is not ready"

        # Feature Store Availability
        - alert: FeastServerDown
          expr: up{job="feast"} == 0
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "Feast feature server is down"
            description: "Feature server has been unreachable for 2 minutes"
```

---

## Step 5: Grafana Dashboards

### ML Pipeline Dashboard

Create `monitoring/dashboards/ml-pipeline-dashboard.json`:

```json
{
  "dashboard": {
    "title": "ML Pipeline Overview",
    "tags": ["mlops", "ml-pipeline"],
    "panels": [
      {
        "title": "Airflow DAG Status",
        "type": "stat",
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "sum(airflow_dag_run_status{status='success'})",
            "legendFormat": "Successful"
          }
        ]
      },
      {
        "title": "MLflow Runs",
        "type": "timeseries",
        "gridPos": {"x": 6, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum by (status) (mlflow_runs_total)",
            "legendFormat": "{{ status }}"
          }
        ]
      },
      {
        "title": "Model Prediction Latency",
        "type": "heatmap",
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum by (le) (rate(model_prediction_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "gauge",
        "gridPos": {"x": 12, "y": 8, "w": 6, "h": 4},
        "targets": [
          {
            "expr": "model_accuracy_score",
            "legendFormat": "{{ model_name }}"
          }
        ],
        "options": {
          "thresholds": {
            "steps": [
              {"color": "red", "value": 0},
              {"color": "yellow", "value": 0.7},
              {"color": "green", "value": 0.9}
            ]
          }
        }
      },
      {
        "title": "Prediction Requests",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 16, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "sum(rate(model_prediction_requests_total[5m])) by (model_name)",
            "legendFormat": "{{ model_name }}"
          }
        ]
      },
      {
        "title": "Data Drift Score",
        "type": "timeseries",
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 6},
        "targets": [
          {
            "expr": "model_data_drift_score",
            "legendFormat": "{{ model_name }} - {{ feature }}"
          }
        ]
      }
    ]
  }
}
```

### Load Dashboard

```bash
# Import dashboard via Grafana API
curl -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -d @monitoring/dashboards/ml-pipeline-dashboard.json \
    http://grafana.local/api/dashboards/db
```

---

## Step 6: Model Monitoring with Evidently

Create `monitoring/model_monitoring.py`:

```python
import evidently
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestAccuracyScore,
)
import pandas as pd
from prometheus_client import Gauge
import mlflow

# Prometheus metrics
DRIFT_SCORE = Gauge('model_drift_score', 'Overall drift score', ['model_name'])
COLUMN_DRIFT = Gauge('model_column_drift', 'Column drift', ['model_name', 'column'])
ACCURACY = Gauge('model_monitored_accuracy', 'Monitored accuracy', ['model_name'])

class ModelMonitor:
    def __init__(self, model_name: str, reference_data: pd.DataFrame):
        self.model_name = model_name
        self.reference_data = reference_data

    def check_data_drift(self, current_data: pd.DataFrame) -> dict:
        """Check for data drift"""
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        results = report.as_dict()

        # Update Prometheus metrics
        drift_score = results['metrics'][0]['result']['drift_share']
        DRIFT_SCORE.labels(model_name=self.model_name).set(drift_score)

        for col_result in results['metrics'][1]['result']['drift_by_columns']:
            COLUMN_DRIFT.labels(
                model_name=self.model_name,
                column=col_result['column_name']
            ).set(col_result['drift_score'])

        return results

    def run_tests(self, current_data: pd.DataFrame, predictions: pd.Series, actuals: pd.Series) -> bool:
        """Run monitoring tests"""
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=0.3),
            TestColumnDrift(column_name='feature_1'),
            TestAccuracyScore(gt=0.8),
        ])

        # Add prediction columns
        current_data_with_preds = current_data.copy()
        current_data_with_preds['prediction'] = predictions
        current_data_with_preds['target'] = actuals

        reference_with_preds = self.reference_data.copy()
        # Add reference predictions...

        test_suite.run(
            reference_data=reference_with_preds,
            current_data=current_data_with_preds
        )

        results = test_suite.as_dict()

        # Log to MLflow
        mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
        with mlflow.start_run(run_name=f"monitoring-{self.model_name}"):
            mlflow.log_dict(results, "monitoring_results.json")

        return all(test['status'] == 'SUCCESS' for test in results['tests'])

# Usage in Airflow DAG
def monitor_model(**context):
    monitor = ModelMonitor(
        model_name="churn-classifier",
        reference_data=pd.read_parquet("s3://data/reference.parquet")
    )

    current_data = pd.read_parquet("s3://data/production_predictions.parquet")

    # Check drift
    drift_results = monitor.check_data_drift(current_data)

    if drift_results['metrics'][0]['result']['drift_share'] > 0.3:
        # Trigger alert or retraining
        context['task_instance'].xcom_push(key='drift_detected', value=True)

    return drift_results
```

---

## Verification

```bash
#!/bin/bash
# verify_observability.sh

echo "=== Observability Stack Verification ==="

echo -e "\n1. Prometheus Status:"
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus

echo -e "\n2. Grafana Status:"
kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana

echo -e "\n3. Loki Status:"
kubectl get pods -n monitoring -l app.kubernetes.io/name=loki

echo -e "\n4. Alertmanager Status:"
kubectl get pods -n monitoring -l app.kubernetes.io/name=alertmanager

echo -e "\n5. Check Prometheus Targets:"
curl -s http://prometheus.local/api/v1/targets | jq '.data.activeTargets | length'

echo -e "\n6. Check Active Alerts:"
curl -s http://prometheus.local/api/v1/alerts | jq '.data.alerts'

echo -e "\n=== Verification Complete ==="
```

---

**Status**: Phase 07 Complete
**Features Covered**: Prometheus, Grafana, Loki, Alertmanager, ML Metrics, Model Monitoring
