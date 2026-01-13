# Manual Python Pipeline Approach

## Overview

Replace Airflow DAGs with manual Python scripts for resource-constrained POC environments while keeping all other MLOps tools (DVC, Great Expectations, Feast, Kubeflow, MLflow, KServe, etc.)

```
Before:  Airflow → DVC → GE → Feast → Kubeflow → MLflow → KServe
After:   Python  → DVC → GE → Feast → Kubeflow → MLflow → KServe
```

---

## Design Decisions

| Decision | Choice |
|----------|--------|
| Execution mode | Each script runnable independently AND via orchestrator |
| Configuration | Single root `config.yaml` with sections per use case |
| Starting point | UC1 Churn Prediction (full end-to-end pipeline) |

---

## Directory Structure

```
04_usecases/python_pipeline/
│
├── config.yaml                                # Single config for all use cases
├── requirements.txt                           # Python dependencies
│
├── common/
│   ├── __init__.py
│   ├── config.py                              # Load & access config.yaml
│   ├── minio_client.py                        # MinIO/S3 helper functions
│   ├── k8s_client.py                          # Kubernetes helper functions
│   └── logger.py                              # Logging setup
│
├── uc1_churn_prediction/
│   ├── __init__.py
│   ├── uc1_01_data_ingestion_postgres.py      # Extract from PostgreSQL → MinIO
│   ├── uc1_02_data_versioning_dvc.py          # DVC add & push
│   ├── uc1_03_data_validation_ge.py           # Great Expectations validation
│   ├── uc1_04_feature_engineering_feast.py    # Feast apply & materialize
│   ├── uc1_05_model_training_kubeflow.py      # Trigger Kubeflow pipeline
│   ├── uc1_06_model_serving_kserve.py         # Deploy InferenceService
│   └── run_pipeline.py                        # Orchestrator - runs all steps
│
├── uc2_ab_testing/
│   ├── __init__.py
│   ├── uc2_01_deploy_canary_kserve.py         # Deploy canary version
│   ├── uc2_02_create_experiment_iter8.py      # Create Iter8 experiment
│   ├── uc2_03_monitor_experiment.py           # Monitor & check results
│   └── run_pipeline.py
│
├── uc3_drift_retraining/
│   ├── __init__.py
│   ├── uc3_01_drift_detection_evidently.py    # Check drift with Evidently
│   ├── uc3_02_trigger_retrain_kubeflow.py     # Trigger Kubeflow if drift
│   └── run_pipeline.py
│
├── uc4_data_lineage/
│   ├── __init__.py
│   ├── uc4_01_emit_lineage_openlineage.py     # Manual OpenLineage events
│   ├── uc4_02_query_lineage_marquez.py        # Query Marquez API
│   └── run_pipeline.py
│
├── uc5_fairness_docs/
│   ├── __init__.py
│   ├── uc5_01_fairness_analysis_wit.py        # What-If Tool analysis
│   ├── uc5_02_generate_model_card.py          # Model Card generation
│   └── run_pipeline.py
│
├── uc6_gitops_deployment/
│   ├── __init__.py
│   ├── uc6_01_promote_model_git.py            # Update manifest & push to Git
│   ├── uc6_02_sync_argocd.py                  # Trigger/monitor Argo CD sync
│   └── run_pipeline.py
│
└── run_all.py                                 # Master orchestrator (optional)
```

---

## Naming Convention

```
uc{N}_{step_number}_{description}_{tool}.py
```

Examples:
- `uc1_01_data_ingestion_postgres.py`
- `uc3_02_trigger_retrain_kubeflow.py`

---

## Use Case Mapping

| Use Case | Pipeline Steps | Tools |
|----------|----------------|-------|
| UC1: Churn Prediction | Ingest → Version → Validate → Features → Train → Serve | PostgreSQL, DVC, GE, Feast, Kubeflow, MLflow, KServe |
| UC2: A/B Testing | Deploy Canary → Create Experiment → Monitor | KServe, Iter8 |
| UC3: Drift Retraining | Detect Drift → Trigger Retrain | Evidently, Kubeflow |
| UC4: Data Lineage | Emit Events → Query Lineage | OpenLineage, Marquez |
| UC5: Fairness & Docs | Analyze Fairness → Generate Card | What-If Tool, Model Card Toolkit |
| UC6: GitOps Deploy | Promote Model → Sync ArgoCD | Git, Argo CD |

---

## Execution Modes

### Standalone (single step)
```bash
cd 04_usecases/python_pipeline
python uc1_churn_prediction/uc1_01_data_ingestion_postgres.py
```

### Orchestrated (full pipeline)
```bash
cd 04_usecases/python_pipeline
python uc1_churn_prediction/run_pipeline.py
```

### All use cases
```bash
cd 04_usecases/python_pipeline
python run_all.py
```

---

## Configuration (config.yaml)

```yaml
environment: development  # development | staging | production

# Database connections
postgres:
  host: postgresql.postgresql.svc.cluster.local
  port: 5432
  database: customers
  user: postgres
  password: postgres123

# Object storage
minio:
  endpoint: http://minio.minio.svc.cluster.local:9000
  access_key: minioadmin
  secret_key: minioadmin123
  bucket: dvc-storage

# MLOps tools
mlflow:
  tracking_uri: http://mlflow.mlflow.svc.cluster.local:5000

kubeflow:
  host: http://kubeflow.kubeflow.svc.cluster.local

feast:
  repo_path: /path/to/feature_repo

# Use case specific
uc1:
  raw_data_path: raw/customers.parquet
  model_name: churn-predictor
  namespace: models
```

---

## Implementation Phases

### Phase 1: UC1 Churn Prediction
- Core end-to-end pipeline
- Common utilities (config, logging)
- Full integration: PostgreSQL → DVC → GE → Feast → Kubeflow → MLflow → KServe

### Phase 2: UC2-UC3 (Model Operations)
- A/B testing with Iter8
- Drift detection and retraining

### Phase 3: UC4-UC6 (Governance & Deployment)
- Data lineage tracking
- Model fairness and documentation
- GitOps deployment

---

## Comparison: Airflow vs Manual Python

| Feature | Airflow | Manual Python |
|---------|---------|---------------|
| Task orchestration | Automatic DAG execution | Sequential function calls |
| Scheduling | Built-in (`@daily`, `@hourly`) | Manual run / cron |
| Retries | Automatic with `retries=N` | Manual retry logic |
| UI monitoring | Web UI for task status | Logs / print statements |
| Resource usage | High (scheduler, workers, DB, webserver) | Low (single Python process) |
| Complexity | High | Low |
| Production ready | Yes | POC/Development |

---

## When to Use Each Approach

**Use Manual Python when:**
- Running POC with limited resources
- Learning/experimenting with MLOps tools
- Local development and testing
- Simple one-off pipeline runs

**Use Airflow when:**
- Production deployment
- Scheduled recurring pipelines
- Complex DAG dependencies
- Team collaboration with UI visibility
- Built-in monitoring and alerting needed
