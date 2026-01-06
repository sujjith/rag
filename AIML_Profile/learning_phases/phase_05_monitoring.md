# Phase 05: Monitoring & Observability

**Duration**: 2 weeks | **Prerequisites**: Phase 04 completed

---

## Learning Objectives

By the end of this phase, you will:
- [ ] Detect data and model drift with Evidently
- [ ] Profile data with Whylogs
- [ ] Set up Prometheus + Grafana monitoring
- [ ] Create ML-specific dashboards and alerts

---

## Week 1: ML Monitoring

### Day 1-3: Evidently AI

```bash
uv add evidently
```

```python
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

# Load reference and current data
reference_data = pd.read_csv("data/reference.csv")
current_data = pd.read_csv("data/current.csv")

# Define column mapping
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["feature_1", "feature_2", "feature_3"],
    categorical_features=["category"]
)

# Create data drift report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])
data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Save report
data_drift_report.save_html("reports/data_drift.html")

# Get metrics as dict
drift_results = data_drift_report.as_dict()
print(f"Dataset drift detected: {drift_results['metrics'][0]['result']['dataset_drift']}")
```

### Day 4-5: Evidently for Model Performance

```python
from evidently.metric_preset import ClassificationPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

# Classification report
classification_report = Report(metrics=[
    ClassificationPreset()
])
classification_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Test suite (pass/fail)
test_suite = TestSuite(tests=[
    DataStabilityTestPreset(),
])
test_suite.run(
    reference_data=reference_data,
    current_data=current_data
)

# Check if tests passed
if test_suite.as_dict()["summary"]["all_passed"]:
    print("All tests passed!")
else:
    print("Some tests failed!")
```

### Day 6-7: Whylogs Data Profiling

```bash
uv add whylogs
```

```python
import whylogs as why
import pandas as pd

# Profile data
df = pd.read_csv("data.csv")
profile = why.log(df)

# View profile
profile.view().to_pandas()

# Save profile
profile.write("profiles/data_profile.bin")

# Compare profiles
from whylogs.viz import NotebookProfileVisualizer

reference_profile = why.read("profiles/reference.bin")
current_profile = why.read("profiles/current.bin")

viz = NotebookProfileVisualizer()
viz.set_profiles(target_profile=current_profile, reference_profile=reference_profile)
viz.summary_drift_report()  # Shows drift summary
```

---

## Week 2: Infrastructure Monitoring

### Day 8-10: Prometheus + Grafana Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

```bash
docker-compose up -d
```

### Day 11-12: Instrument FastAPI with Prometheus

```bash
uv add prometheus-client
```

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

app = FastAPI()

# Define metrics
PREDICTIONS_TOTAL = Counter(
    'predictions_total', 
    'Total predictions made',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

PREDICTION_VALUE = Histogram(
    'prediction_value',
    'Prediction value distribution',
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        result = model.predict(request.features)
        
        # Record metrics
        PREDICTIONS_TOTAL.labels(model_version="v1", status="success").inc()
        PREDICTION_VALUE.observe(result["probability"])
        
    except Exception as e:
        PREDICTIONS_TOTAL.labels(model_version="v1", status="error").inc()
        raise e
    
    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
    
    return result

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Day 13-14: Grafana Dashboard

```json
// Example Grafana dashboard panel (JSON)
{
  "panels": [
    {
      "title": "Predictions per Minute",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(predictions_total[1m])"
      }]
    },
    {
      "title": "P95 Latency",
      "type": "stat",
      "targets": [{
        "expr": "histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))"
      }]
    },
    {
      "title": "Error Rate",
      "type": "gauge",
      "targets": [{
        "expr": "sum(rate(predictions_total{status='error'}[5m])) / sum(rate(predictions_total[5m]))"
      }]
    }
  ]
}
```

**Create dashboard with:**
- Request rate over time
- Latency percentiles (p50, p95, p99)
- Error rate
- Prediction distribution

---

## Milestone Checklist

- [ ] Evidently drift report generated
- [ ] Evidently test suite passing
- [ ] Whylogs profiles created
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboard created
- [ ] ML metrics exposed via `/metrics`
- [ ] Alerts configured

---

**Next Phase**: [Phase 06 - LLMOps & RAG](./phase_06_llmops.md)
