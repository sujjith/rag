# Enterprise Airflow Use Cases

A comprehensive collection of 35 DAGs across 12 phases covering enterprise Airflow patterns.

---

## Overview

| Phase | Topic | DAGs | Focus Area |
|-------|-------|:----:|------------|
| 1 | Fundamentals | 2 | Basic DAG structure |
| 2 | Scheduling | 2 | Cron & TaskFlow API |
| 3 | Operators | 3 | Bash, Python, Sensors |
| 4 | Control Flow | 4 | XCom, branching, triggers |
| 5 | Configuration | 4 | Connections, pools, groups |
| 6 | ML Integration | 2 | MLflow pipelines |
| 7 | Resilience | 3 | Retry, circuit breaker |
| 8 | Notifications | 3 | Email, Slack, routing |
| 9 | Integrations | 4 | DB, API, Cloud, Kafka |
| 10 | Data Quality | 3 | Validation, drift detection |
| 11 | Testing/CI-CD | 2 | Tests, deployment |
| 12 | Observability | 3 | Metrics, logging, health |

---

## Phase 1: Fundamentals

**Objective:** Understand basic Airflow concepts and DAG structure.

| DAG | Description |
|-----|-------------|
| `phase1_01_hello_airflow` | First DAG with EmptyOperator and linear dependencies |
| `phase1_02_parallel_tasks` | Fan-out/fan-in patterns, parallel execution |

**Key Concepts:**
- DAG definition with `with DAG()` context manager
- Task dependencies using `>>` operator
- `default_args` for common task settings
- Tags for organization

---

## Phase 2: Scheduling & TaskFlow

**Objective:** Master scheduling and modern TaskFlow API.

| DAG | Description |
|-----|-------------|
| `phase2_01_scheduled_dag` | Cron expressions, presets (@daily, @hourly) |
| `phase2_02_taskflow_api` | Decorator-based DAGs with `@dag` and `@task` |

**Key Concepts:**
- Schedule intervals: `@once`, `@hourly`, `@daily`, `@weekly`
- `catchup=False` to prevent backfilling
- `max_active_runs` for concurrency control
- Automatic XCom with TaskFlow

---

## Phase 3: Operators & Sensors

**Objective:** Use common operators and implement sensors.

| DAG | Description |
|-----|-------------|
| `phase3_01_bash_operator` | Shell commands, environment variables, templating |
| `phase3_02_python_operator` | Python functions, context access, return values |
| `phase3_03_sensors` | FileSensor, PythonSensor, TimeSensor |

**Key Concepts:**
- BashOperator with Jinja templating (`{{ ds }}`)
- PythonOperator with `op_kwargs` and `op_args`
- Sensor modes: `poke` vs `reschedule`
- `poke_interval` and `timeout` configuration

---

## Phase 4: Control Flow & Data Passing

**Objective:** Master XCom, branching, and dynamic tasks.

| DAG | Description |
|-----|-------------|
| `phase4_01_xcom_basics` | Push/pull data between tasks |
| `phase4_02_branching` | BranchPythonOperator, `@task.branch` |
| `phase4_03_trigger_rules` | all_success, one_failed, none_failed, etc. |
| `phase4_04_dynamic_tasks` | Task mapping with `expand()` and `partial()` |

**Key Concepts:**
- XCom for cross-task communication
- Branching for conditional execution
- Trigger rules for complex dependencies
- Dynamic task generation at runtime

---

## Phase 5: Configuration & Resources

**Objective:** Manage connections, variables, pools, and task organization.

| DAG | Description |
|-----|-------------|
| `phase5_01_connections_variables` | Secure credentials, configuration management |
| `phase5_02_pools_priority` | Resource pools, task priority weights |
| `phase5_03_task_groups` | Visual organization, nested groups |
| `phase5_04_callbacks` | Success/failure/retry callbacks, SLA monitoring |

**Key Concepts:**
- Connections for external system credentials
- Variables for configuration (simple and JSON)
- Pools for concurrency limiting
- Callbacks for custom handling

---

## Phase 6: ML Integration

**Objective:** Integrate Airflow with MLflow for ML pipelines.

| DAG | Description |
|-----|-------------|
| `phase6_01_mlflow_integration` | Experiment tracking, model logging, registry |
| `phase6_02_complete_ml_pipeline` | End-to-end: ingest, validate, train, evaluate, deploy |

**Key Concepts:**
- MLflow tracking URI configuration
- Parameter and metric logging
- Model registration and versioning
- Deployment decision logic

---

## Phase 7: Error Handling & Resilience

**Objective:** Build fault-tolerant pipelines with enterprise patterns.

| DAG | Description |
|-----|-------------|
| `phase7_01_retry_strategies` | Exponential backoff, timeout enforcement |
| `phase7_02_exception_handling` | Custom exceptions, graceful degradation, fallbacks |
| `phase7_03_circuit_breaker` | Circuit breaker pattern for external services |

**Key Concepts:**
- `retry_exponential_backoff=True` for smart retries
- `execution_timeout` to kill long-running tasks
- Custom exception types (TransientError, PermanentError)
- Circuit states: CLOSED, OPEN, HALF_OPEN
- Fallback strategies for degraded operation

---

## Phase 8: Notifications & Alerting

**Objective:** Implement multi-channel notifications with escalation.

| DAG | Description |
|-----|-------------|
| `phase8_01_email_notifications` | EmailOperator, failure callbacks, templates |
| `phase8_02_slack_notifications` | Slack webhooks, Block Kit formatting |
| `phase8_03_alert_routing` | Severity-based routing (P1-P4), team routing |

**Key Concepts:**
- SMTP configuration for email
- Slack webhook integration
- Alert severity levels:
  - P1: Page on-call, Slack #incidents, email leadership
  - P2: Slack alerts, email team
  - P3: Slack alerts only
  - P4: Log only
- Team-based routing using DAG tags

---

## Phase 9: External System Integration

**Objective:** Connect to databases, APIs, cloud platforms, and message queues.

| DAG | Description |
|-----|-------------|
| `phase9_01_database_integration` | PostgreSQL operations, transactions, quality checks |
| `phase9_02_api_integration` | REST APIs, OAuth2, pagination, rate limiting |
| `phase9_03_cloud_integration` | AWS S3, GCP BigQuery, Azure Blob, cross-cloud |
| `phase9_04_message_queues` | Kafka producer/consumer, event publishing, DLQ |

**Key Concepts:**
- PostgresHook for database operations
- HttpHook for API calls
- S3Hook, BigQueryHook, WasbHook for cloud storage
- Kafka integration for event-driven pipelines
- Dead letter queue handling

---

## Phase 10: Data Quality & Validation

**Objective:** Implement data validation, schema checks, and drift detection.

| DAG | Description |
|-----|-------------|
| `phase10_01_great_expectations` | Expectation suites, quality gates |
| `phase10_02_schema_validation` | JSON Schema, versioning, evolution |
| `phase10_03_data_profiling` | Statistical profiling, drift detection, alerting |

**Key Concepts:**
- Great Expectations-style validations:
  - `expect_column_to_exist`
  - `expect_column_values_to_not_be_null`
  - `expect_column_values_to_be_between`
  - `expect_table_row_count_to_be_between`
- Schema versioning and compatibility
- Data drift detection with thresholds
- Quality gates that fail pipelines

---

## Phase 11: Testing & CI/CD

**Objective:** Test DAGs and automate deployments.

| DAG | Description |
|-----|-------------|
| `phase11_01_dag_testing` | Testable patterns, unit test examples |
| `phase11_02_cicd_patterns` | CI validation, deployment gates, staging/prod |

**Key Concepts:**
- Separate business logic from operators for testability
- DAG validation tests (imports, cycles)
- Unit tests with pytest
- CI/CD pipeline stages:
  1. Lint (pylint, flake8, black)
  2. Test (pytest with coverage)
  3. Validate DAGs
  4. Deploy to staging
  5. Smoke tests
  6. Deploy to production

---

## Phase 12: Observability & Monitoring

**Objective:** Implement metrics, logging, tracing, and health checks.

| DAG | Description |
|-----|-------------|
| `phase12_01_metrics_monitoring` | Custom metrics, Prometheus format, SLA tracking |
| `phase12_02_logging_tracing` | Structured JSON logging, distributed tracing |
| `phase12_03_health_checks` | Component health, self-healing, escalation |

**Key Concepts:**
- Metric types: Counter, Gauge, Histogram
- Prometheus exposition format
- Structured logging with correlation IDs
- Distributed tracing with spans
- Health checks for:
  - Database connectivity
  - API availability
  - Storage accessibility
  - Message queue status
  - Airflow components
- Self-healing patterns with automatic remediation

---

## Quick Start

```bash
# Using Docker Compose
cd docker_compose
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d

# Access UI
# URL: http://localhost:8080
# Username: admin
# Password: admin
```

---

## Prerequisites by Phase

| Phase | Required Connections/Packages |
|-------|------------------------------|
| 1-5 | None (core Airflow) |
| 6 | `mlflow`, `scikit-learn` |
| 7 | None (core patterns) |
| 8 | SMTP connection, Slack webhook |
| 9 | `apache-airflow-providers-postgres`, `apache-airflow-providers-amazon`, `apache-airflow-providers-google`, `apache-airflow-providers-apache-kafka` |
| 10 | `great-expectations` (optional) |
| 11 | `pytest`, CI/CD platform |
| 12 | Prometheus (optional), logging backend |

---

## Directory Structure

```
dags/
├── airflow_usecases.md      # This file
├── phase1/                  # Fundamentals
│   ├── 01_hello_airflow.py
│   └── 02_parallel_tasks.py
├── phase2/                  # Scheduling
│   ├── 01_scheduled_dag.py
│   └── 02_taskflow_api.py
├── phase3/                  # Operators
│   ├── 01_bash_operator.py
│   ├── 02_python_operator.py
│   └── 03_sensors.py
├── phase4/                  # Control Flow
│   ├── 01_xcom_basics.py
│   ├── 02_branching.py
│   ├── 03_trigger_rules.py
│   └── 04_dynamic_tasks.py
├── phase5/                  # Configuration
│   ├── 01_connections_variables.py
│   ├── 02_pools_priority.py
│   ├── 03_task_groups.py
│   └── 04_callbacks.py
├── phase6/                  # ML Integration
│   ├── 01_mlflow_integration.py
│   └── 02_complete_ml_pipeline.py
├── phase7/                  # Resilience
│   ├── 01_retry_strategies.py
│   ├── 02_exception_handling.py
│   └── 03_circuit_breaker.py
├── phase8/                  # Notifications
│   ├── 01_email_notifications.py
│   ├── 02_slack_notifications.py
│   └── 03_alert_routing.py
├── phase9/                  # Integrations
│   ├── 01_database_integration.py
│   ├── 02_api_integration.py
│   ├── 03_cloud_integration.py
│   └── 04_message_queues.py
├── phase10/                 # Data Quality
│   ├── 01_great_expectations.py
│   ├── 02_schema_validation.py
│   └── 03_data_profiling.py
├── phase11/                 # Testing/CI-CD
│   ├── 01_dag_testing.py
│   └── 02_cicd_patterns.py
└── phase12/                 # Observability
    ├── 01_metrics_monitoring.py
    ├── 02_logging_tracing.py
    └── 03_health_checks.py
```

---

## Learning Path

**Beginner (Phases 1-4):**
Start with fundamentals, understand DAG structure, scheduling, and control flow.

**Intermediate (Phases 5-6):**
Learn configuration management and ML integration.

**Advanced (Phases 7-9):**
Master resilience patterns, notifications, and external integrations.

**Enterprise (Phases 10-12):**
Implement data quality, testing, CI/CD, and observability.

---

## Production Checklist

- [ ] Retry strategies configured for all external calls
- [ ] Circuit breakers for unreliable services
- [ ] Email/Slack notifications for failures
- [ ] Alert routing based on severity
- [ ] Database connections using Airflow Connections
- [ ] Data validation before loading
- [ ] Schema versioning for data contracts
- [ ] Unit tests for business logic
- [ ] CI/CD pipeline for DAG deployment
- [ ] Metrics exported to monitoring system
- [ ] Structured logging with correlation IDs
- [ ] Health checks for dependencies
- [ ] SLA definitions and monitoring
