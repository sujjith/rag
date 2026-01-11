# Enterprise Airflow Use Cases

A comprehensive collection of 53 DAGs across 18 phases covering enterprise Airflow patterns.

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
| 13 | Multi-DAG Orchestration | 3 | Cross-DAG dependencies, triggering |
| 14 | Dataset-Driven Scheduling | 3 | Data-aware scheduling (2.4+) |
| 15 | DAG Factory & Templates | 3 | Config-driven DAG generation |
| 16 | Security & Secrets | 3 | Vault, RBAC, audit logging |
| 17 | Data Lineage & Governance | 3 | OpenLineage, catalogs, compliance |
| 18 | Advanced MLOps | 3 | Feature stores, A/B testing, retraining |

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

## Phase 13: Multi-DAG Orchestration

**Objective:** Implement cross-DAG dependencies, triggering patterns, and complex workflow orchestration.

| DAG | Description |
|-----|-------------|
| `phase13_01_external_task_sensor` | ExternalTaskSensor for upstream DAG completion, execution delta handling |
| `phase13_02_trigger_dag_run` | TriggerDagRunOperator, pass configuration, conditional triggering |
| `phase13_03_dag_dependencies` | Complex multi-DAG pipelines, shared state patterns, dependency graphs |

**Key Concepts:**
- `ExternalTaskSensor` with `execution_date_fn` for flexible date matching
- `TriggerDagRunOperator` with `conf` parameter for dynamic configuration
- Cross-DAG XCom communication patterns
- Dependency graph visualization
- Handling timezone and schedule alignment between DAGs
- Patterns for parent-child DAG relationships
- Failure propagation across DAG boundaries

---

## Phase 14: Dataset-Driven Scheduling (Airflow 2.4+)

**Objective:** Implement data-aware scheduling using Airflow Datasets for event-driven pipelines.

| DAG | Description |
|-----|-------------|
| `phase14_01_dataset_producer` | Define and emit Dataset events, outlet declarations |
| `phase14_02_dataset_consumer` | Schedule based on Dataset updates, multi-dataset dependencies |
| `phase14_03_dataset_patterns` | Fan-out patterns, conditional dataset triggers, dataset-driven ML pipelines |

**Key Concepts:**
- `Dataset` definition with URI patterns
- Producer DAGs with `outlets` parameter
- Consumer DAGs with `schedule=[dataset1, dataset2]`
- Dataset UI visualization
- Combining time-based and data-based scheduling
- Dataset events for:
  - File arrivals (S3, GCS, local)
  - Database table updates
  - API data refreshes
  - ML model retraining triggers
- Cross-team data contracts using Datasets

---

## Phase 15: DAG Factory & Templates

**Objective:** Generate DAGs dynamically from configuration files for scalable, maintainable pipelines.

| DAG | Description |
|-----|-------------|
| `phase15_01_yaml_dag_factory` | YAML-driven DAG generation, template substitution |
| `phase15_02_jinja_templates` | Jinja2 templated DAGs, environment-specific configurations |
| `phase15_03_dynamic_dag_patterns` | Multi-tenant DAGs, parameterized pipelines, inheritance patterns |

**Key Concepts:**
- Configuration-driven DAG generation patterns:
  ```yaml
  dag_id: etl_customer_data
  schedule: "@daily"
  tasks:
    - id: extract
      operator: PythonOperator
      callable: extract_data
    - id: transform
      dependencies: [extract]
  ```
- `dag-factory` library integration
- Jinja2 templates for DAG code
- Environment variable substitution
- Multi-tenant patterns (one config, many DAGs)
- Inheritance and composition patterns
- Validation of configuration files
- Version control for DAG configurations
- Dynamic task generation vs dynamic DAG generation trade-offs

---

## Phase 16: Security & Secrets Management

**Objective:** Implement enterprise security patterns including secrets management, RBAC, and audit logging.

| DAG | Description |
|-----|-------------|
| `phase16_01_secrets_backend` | HashiCorp Vault, AWS Secrets Manager, Azure Key Vault integration |
| `phase16_02_rbac_patterns` | Role-based access control, DAG-level permissions, team isolation |
| `phase16_03_audit_compliance` | Audit logging, compliance reports, access tracking |

**Key Concepts:**
- Secrets Backend configuration:
  - HashiCorp Vault: `VaultBackend`
  - AWS Secrets Manager: `SecretsManagerBackend`
  - GCP Secret Manager: `CloudSecretManagerBackend`
- Secrets rotation patterns without DAG restarts
- RBAC configuration:
  - DAG-level permissions
  - Resource-based access control
  - Team/department isolation
- Audit logging:
  - Task execution audit trail
  - Variable/Connection access logging
  - User action tracking
- Compliance patterns:
  - PII data handling
  - SOC2 audit requirements
  - GDPR data lineage
- Secure credential injection to tasks
- Encryption at rest and in transit

---

## Phase 17: Data Lineage & Governance

**Objective:** Track data lineage, integrate with data catalogs, and implement governance workflows.

| DAG | Description |
|-----|-------------|
| `phase17_01_openlineage` | OpenLineage integration, automatic lineage capture, custom facets |
| `phase17_02_data_catalog` | DataHub/Amundsen/Unity Catalog integration, metadata publishing |
| `phase17_03_governance_workflows` | Data classification, retention policies, access reviews |

**Key Concepts:**
- OpenLineage integration:
  - Automatic lineage from operators (Spark, BigQuery, Snowflake)
  - Custom lineage events with facets
  - Lineage visualization in Marquez
- Data Catalog integration patterns:
  - Publish dataset metadata to DataHub
  - Sync Airflow DAGs to catalog
  - Tag-based data discovery
- Governance workflows:
  - Data classification tagging (PII, Confidential, Public)
  - Retention policy enforcement
  - Access review automation
- Impact analysis:
  - Downstream dependency tracking
  - Change propagation alerts
  - Schema change notifications
- Data contracts:
  - Schema validation at boundaries
  - SLA enforcement between teams
  - Breaking change detection

---

## Phase 18: Advanced MLOps

**Objective:** Implement production ML operations including feature stores, A/B testing, and automated retraining.

| DAG | Description |
|-----|-------------|
| `phase18_01_feature_store` | Feast/Tecton integration, feature computation, online/offline serving |
| `phase18_02_ab_testing` | Experiment orchestration, traffic splitting, statistical analysis |
| `phase18_03_auto_retraining` | Drift-triggered retraining, champion/challenger, model promotion |

**Key Concepts:**
- Feature Store integration:
  - Feast feature definitions and materialization
  - Batch feature computation pipelines
  - Online feature serving triggers
  - Feature freshness monitoring
- A/B Testing orchestration:
  - Experiment configuration management
  - Traffic allocation and routing
  - Statistical significance calculation
  - Winner deployment automation
- Automated Retraining:
  - Drift detection triggers (data drift, model drift)
  - Champion/Challenger patterns
  - Shadow mode deployment
  - Gradual rollout (canary → production)
- Model lifecycle:
  ```
  Train → Validate → Register → Stage → Canary → Production
  ```
- Integration with ML platforms:
  - MLflow model registry
  - SageMaker endpoints
  - Vertex AI deployment
  - Seldon/KServe orchestration
- Feedback loop pipelines:
  - Ground truth collection
  - Performance monitoring
  - Bias detection and alerting

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
| 13 | None (core Airflow 2.0+) |
| 14 | Airflow 2.4+ (Dataset support) |
| 15 | `dag-factory`, `pyyaml`, `jinja2` |
| 16 | `apache-airflow-providers-hashicorp`, `apache-airflow-providers-amazon` (Secrets Manager) |
| 17 | `openlineage-airflow`, `datahub-airflow-plugin` (optional) |
| 18 | `feast`, `mlflow`, `scipy` (statistical tests) |

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
├── phase12/                 # Observability
│   ├── 01_metrics_monitoring.py
│   ├── 02_logging_tracing.py
│   └── 03_health_checks.py
├── phase13/                 # Multi-DAG Orchestration
│   ├── 01_external_task_sensor.py
│   ├── 02_trigger_dag_run.py
│   └── 03_dag_dependencies.py
├── phase14/                 # Dataset-Driven Scheduling
│   ├── 01_dataset_producer.py
│   ├── 02_dataset_consumer.py
│   └── 03_dataset_patterns.py
├── phase15/                 # DAG Factory & Templates
│   ├── 01_yaml_dag_factory.py
│   ├── 02_jinja_templates.py
│   ├── 03_dynamic_dag_patterns.py
│   └── configs/
│       └── dag_definitions.yaml
├── phase16/                 # Security & Secrets
│   ├── 01_secrets_backend.py
│   ├── 02_rbac_patterns.py
│   └── 03_audit_compliance.py
├── phase17/                 # Data Lineage & Governance
│   ├── 01_openlineage.py
│   ├── 02_data_catalog.py
│   └── 03_governance_workflows.py
└── phase18/                 # Advanced MLOps
    ├── 01_feature_store.py
    ├── 02_ab_testing.py
    └── 03_auto_retraining.py
```

---

## Learning Path

**Beginner (Phases 1-4):**
Start with fundamentals, understand DAG structure, scheduling, and control flow.

**Intermediate (Phases 5-6):**
Learn configuration management and ML integration.

**Advanced (Phases 7-9):**
Master resilience patterns, notifications, and external integrations.

**Enterprise Foundation (Phases 10-12):**
Implement data quality, testing, CI/CD, and observability.

**Enterprise Orchestration (Phases 13-15):**
Master multi-DAG patterns, dataset-driven scheduling, and scalable DAG generation with factories.

**Enterprise Security & Governance (Phases 16-17):**
Implement secrets management, RBAC, audit logging, data lineage, and governance workflows.

**Enterprise MLOps (Phase 18):**
Build production ML operations with feature stores, A/B testing, and automated retraining pipelines.

---

## Production Checklist

### Core Operations (Phases 1-12)
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

### Orchestration (Phases 13-15)
- [ ] Cross-DAG dependencies documented
- [ ] ExternalTaskSensor timeouts configured
- [ ] Dataset producers/consumers mapped
- [ ] DAG factory configurations validated
- [ ] Multi-tenant isolation verified

### Security & Governance (Phases 16-17)
- [ ] Secrets backend configured (Vault/AWS/GCP)
- [ ] No hardcoded credentials in DAGs
- [ ] RBAC roles defined per team
- [ ] Audit logging enabled
- [ ] OpenLineage integration active
- [ ] Data classification tags applied
- [ ] Retention policies automated

### MLOps (Phase 18)
- [ ] Feature store materialization scheduled
- [ ] Model drift detection thresholds set
- [ ] Champion/Challenger framework in place
- [ ] A/B test statistical significance defined
- [ ] Automated retraining triggers configured
- [ ] Model rollback procedures documented
