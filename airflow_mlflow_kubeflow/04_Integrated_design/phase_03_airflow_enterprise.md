# Phase 03: Apache Airflow Enterprise Setup

## Overview

Complete Apache Airflow setup with enterprise features including KubernetesExecutor, RBAC, custom operators, sensors, dynamic DAGs, and production-ready configurations.

---

## Airflow Feature Matrix

| Feature | Description | Status |
|---------|-------------|--------|
| KubernetesExecutor | Run tasks in K8s pods | Included |
| CeleryExecutor | Distributed task execution | Included |
| RBAC | Role-based access control | Included |
| OAuth/LDAP | Enterprise authentication | Included |
| Custom Operators | Domain-specific operators | Included |
| Sensors | Wait for external events | Included |
| TaskGroups | Organize complex DAGs | Included |
| Dynamic DAGs | Programmatic DAG generation | Included |
| XCom | Task communication | Included |
| Pools | Resource management | Included |
| SLA Monitoring | Service level agreements | Included |
| Alerting | Slack, PagerDuty, Email | Included |
| Connections | Secure credential management | Included |
| Variables | Configuration management | Included |
| DAG Versioning | Git-based DAG management | Included |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AIRFLOW ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │  Webserver  │     │  Scheduler  │     │  Triggerer  │              │
│   │   (UI/API)  │     │  (DAG runs) │     │  (Deferral) │              │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘              │
│          │                   │                    │                      │
│          └───────────────────┼────────────────────┘                      │
│                              │                                           │
│                              ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                     METADATA DATABASE                             │  │
│   │                       (PostgreSQL)                                │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│          ┌───────────────────┼───────────────────┐                      │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   Worker    │     │   Worker    │     │   Worker    │              │
│   │  (K8s Pod)  │     │  (K8s Pod)  │     │  (K8s Pod)  │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                      EXTERNAL SERVICES                            │  │
│   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │  │
│   │  │ Redis  │  │ MinIO  │  │ MLflow │  │Kubeflow│  │ Feast  │     │  │
│   │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘     │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Deploy Airflow with Helm

### Create Values File

Create `airflow/helm/values.yaml`:

```yaml
# Apache Airflow Helm Values - Enterprise Configuration

# Airflow version
airflowVersion: "2.8.0"
defaultAirflowRepository: apache/airflow
defaultAirflowTag: "2.8.0"

# Executor configuration
executor: "KubernetesExecutor"

# Enable all components
webserver:
  replicas: 2
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"
  service:
    type: ClusterIP
  defaultUser:
    enabled: true
    role: Admin
    username: admin
    email: admin@example.com
    password: admin123

scheduler:
  replicas: 2
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"

triggerer:
  enabled: true
  replicas: 1
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"

workers:
  replicas: 3
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"
  persistence:
    enabled: true
    size: 10Gi

# Flower (Celery monitoring)
flower:
  enabled: true

# Redis for Celery
redis:
  enabled: true
  persistence:
    enabled: true
    size: 5Gi

# PostgreSQL
postgresql:
  enabled: true
  auth:
    postgresPassword: airflow123
    database: airflow
  persistence:
    enabled: true
    size: 20Gi

# Git-sync for DAGs
dags:
  persistence:
    enabled: true
    size: 10Gi
  gitSync:
    enabled: true
    repo: https://github.com/your-org/airflow-dags.git
    branch: main
    subPath: dags
    wait: 60

# Logs
logs:
  persistence:
    enabled: true
    size: 20Gi

# Webserver configuration
webserverConfig: |
  from flask_appbuilder.security.manager import AUTH_OAUTH
  from airflow.www.security import AirflowSecurityManager

  AUTH_TYPE = AUTH_OAUTH
  AUTH_USER_REGISTRATION = True
  AUTH_USER_REGISTRATION_ROLE = "Viewer"

  OAUTH_PROVIDERS = [{
      'name': 'google',
      'token_key': 'access_token',
      'icon': 'fa-google',
      'remote_app': {
          'api_base_url': 'https://www.googleapis.com/oauth2/v2/',
          'client_kwargs': {'scope': 'email profile'},
          'access_token_url': 'https://oauth2.googleapis.com/token',
          'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
          'request_token_url': None,
          'client_id': '${GOOGLE_CLIENT_ID}',
          'client_secret': '${GOOGLE_CLIENT_SECRET}',
      }
  }]

# Airflow configuration
config:
  core:
    dags_are_paused_at_creation: "True"
    load_examples: "False"
    executor: "KubernetesExecutor"
    parallelism: 32
    dag_concurrency: 16
    max_active_runs_per_dag: 5
  kubernetes:
    namespace: "airflow"
    worker_container_repository: "apache/airflow"
    worker_container_tag: "2.8.0"
    delete_worker_pods: "True"
    delete_worker_pods_on_failure: "False"
  webserver:
    rbac: "True"
    expose_config: "False"
    authenticate: "True"
    auth_backend: "airflow.contrib.auth.backends.password_auth"
  scheduler:
    catchup_by_default: "False"
    dag_dir_list_interval: 60
    min_file_process_interval: 30
  celery:
    worker_concurrency: 8
    flower_basic_auth: "admin:admin123"
  smtp:
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_starttls: "True"
    smtp_ssl: "False"
    smtp_user: "airflow@example.com"
    smtp_password: "${SMTP_PASSWORD}"
    smtp_mail_from: "airflow@example.com"
  logging:
    remote_logging: "True"
    remote_base_log_folder: "s3://airflow-logs/"
    remote_log_conn_id: "aws_default"

# Extra environment variables
extraEnv:
  - name: AIRFLOW__CORE__FERNET_KEY
    valueFrom:
      secretKeyRef:
        name: airflow-fernet-key
        key: fernet-key
  - name: MLFLOW_TRACKING_URI
    value: "http://mlflow.mlflow.svc.cluster.local:5000"
  - name: KUBEFLOW_HOST
    value: "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

# Extra Python packages
extraPipPackages:
  - apache-airflow-providers-amazon==8.13.0
  - apache-airflow-providers-google==10.12.0
  - apache-airflow-providers-cncf-kubernetes==7.11.0
  - apache-airflow-providers-postgres==5.8.0
  - apache-airflow-providers-redis==3.5.0
  - apache-airflow-providers-slack==8.4.0
  - apache-airflow-providers-http==4.8.0
  - mlflow==2.9.0
  - kfp==2.5.0
  - great-expectations==0.18.0
  - feast==0.35.0
  - boto3==1.34.0

# Service account for K8s
serviceAccount:
  create: true
  name: airflow

# RBAC
rbac:
  create: true

# Ingress
ingress:
  web:
    enabled: true
    ingressClassName: nginx
    hosts:
      - name: airflow.local
        tls:
          enabled: true
          secretName: airflow-tls
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"

# Metrics for Prometheus
metrics:
  statsd:
    enabled: true
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
```

### Deploy Airflow

```bash
# Add Airflow Helm repo
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Create namespace
kubectl create namespace airflow

# Create Fernet key secret
FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
kubectl create secret generic airflow-fernet-key \
    --namespace airflow \
    --from-literal=fernet-key=$FERNET_KEY

# Deploy Airflow
helm install airflow apache-airflow/airflow \
    --namespace airflow \
    --values airflow/helm/values.yaml \
    --timeout 10m \
    --wait

# Verify deployment
kubectl get pods -n airflow
kubectl get svc -n airflow
kubectl get ingress -n airflow
```

---

## Step 2: Custom Operators

### MLflow Operator

Create `airflow/plugins/operators/mlflow_operators.py`:

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional, Dict, Any
import mlflow
from mlflow import MlflowClient


class MLflowLogMetricsOperator(BaseOperator):
    """Operator to log metrics to MLflow"""

    template_fields = ['run_id', 'metrics']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_id: Optional[str] = None,
        metrics: Dict[str, float] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.metrics = metrics or {}
        self.run_name = run_name
        self.tags = tags or {}

    def execute(self, context):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        if self.run_id:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metrics(self.metrics)
        else:
            with mlflow.start_run(run_name=self.run_name, tags=self.tags):
                mlflow.log_metrics(self.metrics)
                return mlflow.active_run().info.run_id


class MLflowModelTransitionOperator(BaseOperator):
    """Operator to transition MLflow model stages"""

    template_fields = ['model_name', 'version', 'stage']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.version = version
        self.stage = stage
        self.archive_existing = archive_existing

    def execute(self, context):
        client = MlflowClient(self.tracking_uri)
        client.transition_model_version_stage(
            name=self.model_name,
            version=self.version,
            stage=self.stage,
            archive_existing_versions=self.archive_existing
        )
        self.log.info(f"Transitioned {self.model_name} v{self.version} to {self.stage}")


class MLflowLoadModelOperator(BaseOperator):
    """Operator to load and validate MLflow model"""

    template_fields = ['model_uri']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        model_uri: str,
        validate_signature: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.model_uri = model_uri
        self.validate_signature = validate_signature

    def execute(self, context):
        mlflow.set_tracking_uri(self.tracking_uri)
        model = mlflow.pyfunc.load_model(self.model_uri)

        # Validate model has signature
        if self.validate_signature:
            if model.metadata.signature is None:
                raise ValueError(f"Model {self.model_uri} has no signature")

        self.log.info(f"Successfully loaded model: {self.model_uri}")
        return {"model_uri": self.model_uri, "has_signature": True}
```

### Kubeflow Operator

Create `airflow/plugins/operators/kubeflow_operators.py`:

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional, Dict, Any
import kfp
from kfp import Client


class KubeflowPipelineOperator(BaseOperator):
    """Operator to trigger Kubeflow pipelines"""

    template_fields = ['pipeline_name', 'arguments', 'experiment_name']

    @apply_defaults
    def __init__(
        self,
        host: str,
        pipeline_name: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        experiment_name: str = "Default",
        arguments: Dict[str, Any] = None,
        run_name: Optional[str] = None,
        wait_for_completion: bool = True,
        timeout: int = 3600,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.host = host
        self.pipeline_name = pipeline_name
        self.pipeline_id = pipeline_id
        self.experiment_name = experiment_name
        self.arguments = arguments or {}
        self.run_name = run_name
        self.wait_for_completion = wait_for_completion
        self.timeout = timeout

    def execute(self, context):
        client = Client(host=self.host)

        # Get or create experiment
        experiment = client.create_experiment(
            name=self.experiment_name,
            namespace="kubeflow"
        )

        # Get pipeline
        if self.pipeline_id:
            pipeline = client.get_pipeline(self.pipeline_id)
        else:
            pipelines = client.list_pipelines(filter=f'name="{self.pipeline_name}"')
            if not pipelines.pipelines:
                raise ValueError(f"Pipeline {self.pipeline_name} not found")
            pipeline = pipelines.pipelines[0]

        # Create run
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=self.run_name or f"{self.pipeline_name}-{context['ts']}",
            pipeline_id=pipeline.pipeline_id,
            params=self.arguments
        )

        self.log.info(f"Started Kubeflow run: {run.run_id}")

        if self.wait_for_completion:
            result = client.wait_for_run_completion(run.run_id, timeout=self.timeout)
            if result.run.status != "Succeeded":
                raise Exception(f"Pipeline run failed: {result.run.status}")

        return {"run_id": run.run_id, "status": result.run.status if self.wait_for_completion else "Running"}


class KubeflowExperimentOperator(BaseOperator):
    """Operator to manage Kubeflow experiments"""

    template_fields = ['experiment_name', 'description']

    @apply_defaults
    def __init__(
        self,
        host: str,
        experiment_name: str,
        description: Optional[str] = None,
        namespace: str = "kubeflow",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.host = host
        self.experiment_name = experiment_name
        self.description = description
        self.namespace = namespace

    def execute(self, context):
        client = Client(host=self.host)

        experiment = client.create_experiment(
            name=self.experiment_name,
            description=self.description,
            namespace=self.namespace
        )

        self.log.info(f"Created/retrieved experiment: {experiment.experiment_id}")
        return {"experiment_id": experiment.experiment_id, "name": self.experiment_name}
```

### Great Expectations Operator

Create `airflow/plugins/operators/great_expectations_operators.py`:

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional, Dict, Any, List
import great_expectations as gx
from great_expectations.core.batch import BatchRequest


class GreatExpectationsOperator(BaseOperator):
    """Operator to run Great Expectations validations"""

    template_fields = ['batch_request', 'expectation_suite_name']

    @apply_defaults
    def __init__(
        self,
        context_root_dir: str,
        datasource_name: str,
        data_asset_name: str,
        expectation_suite_name: str,
        batch_request: Optional[Dict] = None,
        fail_task_on_validation_failure: bool = True,
        return_json_dict: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.context_root_dir = context_root_dir
        self.datasource_name = datasource_name
        self.data_asset_name = data_asset_name
        self.expectation_suite_name = expectation_suite_name
        self.batch_request = batch_request
        self.fail_task_on_validation_failure = fail_task_on_validation_failure
        self.return_json_dict = return_json_dict

    def execute(self, context):
        # Initialize GX context
        gx_context = gx.get_context(context_root_dir=self.context_root_dir)

        # Create batch request
        batch_request = BatchRequest(
            datasource_name=self.datasource_name,
            data_connector_name="default_inferred_data_connector_name",
            data_asset_name=self.data_asset_name,
            **(self.batch_request or {})
        )

        # Run validation
        checkpoint_result = gx_context.run_checkpoint(
            checkpoint_name="default_checkpoint",
            validations=[{
                "batch_request": batch_request,
                "expectation_suite_name": self.expectation_suite_name
            }]
        )

        # Check results
        validation_success = checkpoint_result.success

        if not validation_success and self.fail_task_on_validation_failure:
            raise Exception("Data validation failed!")

        self.log.info(f"Validation {'passed' if validation_success else 'failed'}")

        if self.return_json_dict:
            return checkpoint_result.to_json_dict()

        return validation_success
```

---

## Step 3: Sensors

### Custom Sensors

Create `airflow/plugins/sensors/ml_sensors.py`:

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional
import mlflow
from mlflow import MlflowClient
import requests


class MLflowModelStageSensor(BaseSensorOperator):
    """Sensor that waits for model to reach a specific stage"""

    template_fields = ['model_name', 'target_stage']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        target_stage: str,
        version: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.target_stage = target_stage
        self.version = version

    def poke(self, context):
        client = MlflowClient(self.tracking_uri)

        if self.version:
            mv = client.get_model_version(self.model_name, str(self.version))
            return mv.current_stage == self.target_stage
        else:
            # Check latest version
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                return False
            latest = max(versions, key=lambda x: int(x.version))
            return latest.current_stage == self.target_stage


class MLflowRunCompleteSensor(BaseSensorOperator):
    """Sensor that waits for MLflow run to complete"""

    template_fields = ['run_id']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        run_id: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.run_id = run_id

    def poke(self, context):
        client = MlflowClient(self.tracking_uri)
        run = client.get_run(self.run_id)
        return run.info.status in ["FINISHED", "FAILED", "KILLED"]


class KubeflowPipelineRunSensor(BaseSensorOperator):
    """Sensor that waits for Kubeflow pipeline run to complete"""

    template_fields = ['run_id']

    @apply_defaults
    def __init__(
        self,
        host: str,
        run_id: str,
        target_statuses: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.host = host
        self.run_id = run_id
        self.target_statuses = target_statuses or ["Succeeded", "Failed", "Error"]

    def poke(self, context):
        from kfp import Client
        client = Client(host=self.host)
        run = client.get_run(self.run_id)
        return run.run.status in self.target_statuses


class DataQualitySensor(BaseSensorOperator):
    """Sensor that checks data quality before proceeding"""

    template_fields = ['data_path', 'expectations']

    @apply_defaults
    def __init__(
        self,
        data_path: str,
        min_rows: int = 100,
        max_null_percentage: float = 0.1,
        required_columns: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.min_rows = min_rows
        self.max_null_percentage = max_null_percentage
        self.required_columns = required_columns or []

    def poke(self, context):
        import pandas as pd

        try:
            df = pd.read_csv(self.data_path)

            # Check row count
            if len(df) < self.min_rows:
                self.log.info(f"Insufficient rows: {len(df)} < {self.min_rows}")
                return False

            # Check null percentage
            null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if null_pct > self.max_null_percentage:
                self.log.info(f"Too many nulls: {null_pct:.2%} > {self.max_null_percentage:.2%}")
                return False

            # Check required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                self.log.info(f"Missing columns: {missing_cols}")
                return False

            return True

        except Exception as e:
            self.log.error(f"Error checking data: {e}")
            return False


class ModelPerformanceSensor(BaseSensorOperator):
    """Sensor that checks model performance meets threshold"""

    template_fields = ['model_uri', 'metric_name', 'threshold']

    @apply_defaults
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = ">=",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison

    def poke(self, context):
        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return False

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{self.metric_name} DESC"],
            max_results=1
        )

        if runs.empty:
            return False

        metric_value = runs.iloc[0][f"metrics.{self.metric_name}"]

        if self.comparison == ">=":
            return metric_value >= self.threshold
        elif self.comparison == ">":
            return metric_value > self.threshold
        elif self.comparison == "<=":
            return metric_value <= self.threshold
        elif self.comparison == "<":
            return metric_value < self.threshold
        elif self.comparison == "==":
            return metric_value == self.threshold

        return False
```

---

## Step 4: TaskGroups and Dynamic DAGs

### TaskGroups Example

Create `airflow/dags/ml_pipeline_taskgroups.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ml-alerts@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def extract_data(source: str, **context):
    print(f"Extracting data from {source}")
    return {"source": source, "rows": 1000}

def transform_data(transformation: str, **context):
    print(f"Applying transformation: {transformation}")
    return {"transformation": transformation, "success": True}

def train_model(model_type: str, **context):
    print(f"Training {model_type} model")
    return {"model": model_type, "accuracy": 0.95}

def evaluate_model(model_type: str, **context):
    print(f"Evaluating {model_type} model")
    return {"model": model_type, "passed": True}

with DAG(
    'ml_pipeline_with_taskgroups',
    default_args=default_args,
    description='ML Pipeline with TaskGroups',
    schedule_interval='0 0 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'taskgroups'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Data Extraction TaskGroup
    with TaskGroup(group_id='data_extraction') as extraction_group:
        sources = ['customer_data', 'transaction_data', 'product_data']
        extract_tasks = []

        for source in sources:
            task = PythonOperator(
                task_id=f'extract_{source}',
                python_callable=extract_data,
                op_kwargs={'source': source}
            )
            extract_tasks.append(task)

    # Data Transformation TaskGroup
    with TaskGroup(group_id='data_transformation') as transform_group:
        transformations = ['clean', 'normalize', 'feature_engineering']

        prev_task = None
        for transform in transformations:
            task = PythonOperator(
                task_id=f'transform_{transform}',
                python_callable=transform_data,
                op_kwargs={'transformation': transform}
            )
            if prev_task:
                prev_task >> task
            prev_task = task

    # Model Training TaskGroup
    with TaskGroup(group_id='model_training') as training_group:
        models = ['random_forest', 'gradient_boosting', 'neural_network']

        for model in models:
            with TaskGroup(group_id=f'{model}_pipeline') as model_group:
                train = PythonOperator(
                    task_id='train',
                    python_callable=train_model,
                    op_kwargs={'model_type': model}
                )
                evaluate = PythonOperator(
                    task_id='evaluate',
                    python_callable=evaluate_model,
                    op_kwargs={'model_type': model}
                )
                train >> evaluate

    end = EmptyOperator(task_id='end')

    # Define dependencies
    start >> extraction_group >> transform_group >> training_group >> end
```

### Dynamic DAG Generation

Create `airflow/dags/dynamic_dag_factory.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import yaml
import os

# DAG configuration file
DAG_CONFIGS = """
dags:
  - name: customer_churn_daily
    schedule: "0 2 * * *"
    model: churn_classifier
    data_source: s3://data/customers/
    features:
      - tenure
      - monthly_charges
      - total_charges

  - name: fraud_detection_hourly
    schedule: "0 * * * *"
    model: fraud_detector
    data_source: s3://data/transactions/
    features:
      - amount
      - merchant_category
      - time_since_last_txn

  - name: recommendation_weekly
    schedule: "0 0 * * 0"
    model: recommender
    data_source: s3://data/interactions/
    features:
      - user_id
      - item_id
      - rating
"""

def create_ml_dag(config: dict) -> DAG:
    """Factory function to create ML DAGs from config"""

    dag_id = f"ml_pipeline_{config['name']}"

    default_args = {
        'owner': 'ml-team',
        'depends_on_past': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=f"ML Pipeline for {config['name']}",
        schedule_interval=config['schedule'],
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['ml', 'dynamic', config['model']],
    )

    with dag:
        def load_data(data_source: str, **context):
            print(f"Loading data from {data_source}")
            return {"rows": 10000}

        def preprocess(features: list, **context):
            print(f"Preprocessing features: {features}")
            return {"features_processed": len(features)}

        def train(model_name: str, **context):
            print(f"Training model: {model_name}")
            return {"model": model_name, "trained": True}

        def evaluate(**context):
            print("Evaluating model")
            return {"accuracy": 0.92}

        def deploy(model_name: str, **context):
            print(f"Deploying model: {model_name}")
            return {"deployed": True}

        start = EmptyOperator(task_id='start')

        load = PythonOperator(
            task_id='load_data',
            python_callable=load_data,
            op_kwargs={'data_source': config['data_source']}
        )

        preprocess_task = PythonOperator(
            task_id='preprocess',
            python_callable=preprocess,
            op_kwargs={'features': config['features']}
        )

        train_task = PythonOperator(
            task_id='train_model',
            python_callable=train,
            op_kwargs={'model_name': config['model']}
        )

        evaluate_task = PythonOperator(
            task_id='evaluate_model',
            python_callable=evaluate
        )

        deploy_task = PythonOperator(
            task_id='deploy_model',
            python_callable=deploy,
            op_kwargs={'model_name': config['model']}
        )

        end = EmptyOperator(task_id='end')

        start >> load >> preprocess_task >> train_task >> evaluate_task >> deploy_task >> end

    return dag


# Generate DAGs from config
configs = yaml.safe_load(DAG_CONFIGS)

for dag_config in configs['dags']:
    dag = create_ml_dag(dag_config)
    globals()[dag.dag_id] = dag
```

---

## Step 5: XCom and Task Communication

Create `airflow/dags/xcom_patterns.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import XCom
from datetime import datetime, timedelta
import json

default_args = {
    'owner': 'ml-team',
    'retries': 1,
}

def produce_model_metrics(**context):
    """Produce metrics for downstream tasks"""
    metrics = {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.89,
        'f1_score': 0.90,
        'model_uri': 'models:/churn-classifier/1',
        'training_time': 120.5
    }

    # Push to XCom
    context['task_instance'].xcom_push(key='model_metrics', value=metrics)
    context['task_instance'].xcom_push(key='model_uri', value=metrics['model_uri'])

    return metrics

def evaluate_model_quality(**context):
    """Evaluate based on upstream metrics"""
    ti = context['task_instance']

    # Pull from XCom
    metrics = ti.xcom_pull(task_ids='train_model', key='model_metrics')

    # Evaluate
    threshold = 0.85
    passed = all([
        metrics['accuracy'] >= threshold,
        metrics['precision'] >= threshold,
        metrics['recall'] >= threshold
    ])

    result = {
        'passed': passed,
        'threshold': threshold,
        'metrics': metrics
    }

    ti.xcom_push(key='evaluation_result', value=result)
    return result

def register_model(**context):
    """Register model if evaluation passed"""
    ti = context['task_instance']

    evaluation = ti.xcom_pull(task_ids='evaluate_quality', key='evaluation_result')
    model_uri = ti.xcom_pull(task_ids='train_model', key='model_uri')

    if not evaluation['passed']:
        raise ValueError("Model did not pass quality checks")

    # Register model
    import mlflow
    mlflow.set_tracking_uri("http://mlflow.local:5000")

    result = mlflow.register_model(
        model_uri=model_uri,
        name="churn-classifier"
    )

    return {'registered_version': result.version}

def send_notification(**context):
    """Send notification with all results"""
    ti = context['task_instance']

    # Pull all results
    metrics = ti.xcom_pull(task_ids='train_model', key='model_metrics')
    evaluation = ti.xcom_pull(task_ids='evaluate_quality', key='evaluation_result')
    registration = ti.xcom_pull(task_ids='register_model')

    notification = {
        'dag_id': context['dag'].dag_id,
        'execution_date': str(context['execution_date']),
        'metrics': metrics,
        'evaluation': evaluation,
        'registration': registration
    }

    print(f"Sending notification: {json.dumps(notification, indent=2)}")
    # Send to Slack, email, etc.

with DAG(
    'ml_xcom_patterns',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    train = PythonOperator(
        task_id='train_model',
        python_callable=produce_model_metrics
    )

    evaluate = PythonOperator(
        task_id='evaluate_quality',
        python_callable=evaluate_model_quality
    )

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model
    )

    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule='all_done'  # Run regardless of upstream status
    )

    train >> evaluate >> register >> notify
```

---

## Step 6: Pools and Priority

Create `airflow/dags/resource_management.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# First, create pools via CLI or API:
# airflow pools set gpu_pool 2 "GPU resources pool"
# airflow pools set cpu_intensive 4 "CPU intensive tasks pool"
# airflow pools set api_calls 10 "External API calls pool"

default_args = {
    'owner': 'ml-team',
    'retries': 1,
}

def train_on_gpu(model_name: str, **context):
    """GPU-intensive training task"""
    print(f"Training {model_name} on GPU")
    return {"model": model_name, "device": "GPU"}

def cpu_intensive_preprocessing(**context):
    """CPU-intensive preprocessing"""
    print("Running CPU-intensive preprocessing")
    return {"preprocessed": True}

def call_external_api(endpoint: str, **context):
    """External API call with rate limiting"""
    print(f"Calling API: {endpoint}")
    return {"endpoint": endpoint, "status": "success"}

with DAG(
    'resource_managed_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')

    # GPU tasks with pool and priority
    gpu_train_1 = PythonOperator(
        task_id='gpu_train_model_1',
        python_callable=train_on_gpu,
        op_kwargs={'model_name': 'neural_network'},
        pool='gpu_pool',
        pool_slots=1,
        priority_weight=10,  # Higher priority
    )

    gpu_train_2 = PythonOperator(
        task_id='gpu_train_model_2',
        python_callable=train_on_gpu,
        op_kwargs={'model_name': 'transformer'},
        pool='gpu_pool',
        pool_slots=1,
        priority_weight=5,
    )

    # CPU intensive tasks
    preprocess_1 = PythonOperator(
        task_id='preprocess_dataset_1',
        python_callable=cpu_intensive_preprocessing,
        pool='cpu_intensive',
        pool_slots=1,
    )

    preprocess_2 = PythonOperator(
        task_id='preprocess_dataset_2',
        python_callable=cpu_intensive_preprocessing,
        pool='cpu_intensive',
        pool_slots=1,
    )

    # API calls with rate limiting
    api_calls = []
    for i in range(5):
        api_task = PythonOperator(
            task_id=f'api_call_{i}',
            python_callable=call_external_api,
            op_kwargs={'endpoint': f'/api/v1/resource/{i}'},
            pool='api_calls',
            pool_slots=1,
        )
        api_calls.append(api_task)

    end = EmptyOperator(task_id='end')

    # Dependencies
    start >> [preprocess_1, preprocess_2]
    preprocess_1 >> gpu_train_1
    preprocess_2 >> gpu_train_2
    [gpu_train_1, gpu_train_2] >> api_calls >> end
```

---

## Step 7: SLA and Alerting

Create `airflow/dags/sla_monitoring.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
from airflow.utils.email import send_email

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2),  # Default SLA for all tasks
}

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Callback when SLA is missed"""
    task_names = [task.task_id for task in task_list]
    blocking_names = [task.task_id for task in blocking_task_list] if blocking_task_list else []

    message = f"""
    SLA Miss Alert!

    DAG: {dag.dag_id}
    Tasks with SLA miss: {task_names}
    Blocking tasks: {blocking_names}
    SLAs: {slas}
    """

    # Send Slack notification
    send_slack_alert(message)

    # Send email
    send_email(
        to=['ml-team@example.com'],
        subject=f'SLA Miss: {dag.dag_id}',
        html_content=message
    )

def send_slack_alert(message: str):
    """Send alert to Slack"""
    import requests

    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    payload = {
        "text": message,
        "channel": "#ml-alerts"
    }
    requests.post(webhook_url, json=payload)

def task_failure_callback(context):
    """Callback on task failure"""
    task = context['task_instance']
    exception = context.get('exception')

    message = f"""
    Task Failed!

    DAG: {task.dag_id}
    Task: {task.task_id}
    Execution Date: {task.execution_date}
    Error: {str(exception)}
    Log URL: {task.log_url}
    """

    send_slack_alert(message)

def task_success_callback(context):
    """Callback on task success"""
    task = context['task_instance']

    # Log to monitoring
    print(f"Task {task.task_id} completed successfully")

def critical_task(**context):
    """Critical task with tight SLA"""
    import time
    time.sleep(10)  # Simulate work
    return {"status": "completed"}

with DAG(
    'sla_monitored_pipeline',
    default_args=default_args,
    description='Pipeline with SLA monitoring',
    schedule_interval='0 * * * *',  # Hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    sla_miss_callback=sla_miss_callback,
    tags=['ml', 'sla', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Task with specific SLA
    data_extraction = PythonOperator(
        task_id='extract_data',
        python_callable=critical_task,
        sla=timedelta(minutes=30),
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Critical task with tight SLA
    model_training = PythonOperator(
        task_id='train_model',
        python_callable=critical_task,
        sla=timedelta(hours=1),
        on_failure_callback=task_failure_callback,
    )

    # Task with default SLA
    model_deployment = PythonOperator(
        task_id='deploy_model',
        python_callable=critical_task,
        on_failure_callback=task_failure_callback,
    )

    end = EmptyOperator(task_id='end')

    start >> data_extraction >> model_training >> model_deployment >> end
```

---

## Step 8: Connections and Variables

### Setup via CLI

```bash
# Create connections
airflow connections add 'mlflow_default' \
    --conn-type 'http' \
    --conn-host 'mlflow.mlflow.svc.cluster.local' \
    --conn-port 5000

airflow connections add 'kubeflow_default' \
    --conn-type 'http' \
    --conn-host 'ml-pipeline.kubeflow.svc.cluster.local' \
    --conn-port 8888

airflow connections add 'postgres_ml' \
    --conn-type 'postgres' \
    --conn-host 'postgres-postgresql.mlflow.svc.cluster.local' \
    --conn-port 5432 \
    --conn-login 'mlflow' \
    --conn-password 'mlflow123' \
    --conn-schema 'mlflow'

airflow connections add 'minio_default' \
    --conn-type 'aws' \
    --conn-extra '{"aws_access_key_id": "minio", "aws_secret_access_key": "minio123", "endpoint_url": "http://minio.mlflow.svc.cluster.local:9000"}'

airflow connections add 'slack_webhook' \
    --conn-type 'http' \
    --conn-host 'hooks.slack.com' \
    --conn-password '/services/YOUR/WEBHOOK/URL'

# Create variables
airflow variables set ml_model_threshold 0.85
airflow variables set environment production
airflow variables set ml_config '{"default_model": "random_forest", "max_training_time": 3600}'
```

### Use Connections in DAGs

```python
# airflow/dags/connection_usage.py
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from datetime import datetime

def use_connections(**context):
    """Example of using connections and variables"""

    # Get connection
    mlflow_conn = BaseHook.get_connection('mlflow_default')
    mlflow_uri = f"http://{mlflow_conn.host}:{mlflow_conn.port}"

    # Get variables
    threshold = float(Variable.get('ml_model_threshold', default_var=0.8))
    environment = Variable.get('environment', default_var='development')
    ml_config = Variable.get('ml_config', deserialize_json=True)

    print(f"MLflow URI: {mlflow_uri}")
    print(f"Threshold: {threshold}")
    print(f"Environment: {environment}")
    print(f"Config: {ml_config}")

    return {
        "mlflow_uri": mlflow_uri,
        "threshold": threshold,
        "environment": environment
    }

with DAG(
    'connection_usage_example',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    task = PythonOperator(
        task_id='use_connections',
        python_callable=use_connections
    )
```

---

## Step 9: Branching and Conditional Logic

Create `airflow/dags/branching_patterns.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

def evaluate_model_performance(**context):
    """Evaluate model and decide next steps"""
    import random

    # Simulate model evaluation
    accuracy = random.uniform(0.7, 1.0)
    context['task_instance'].xcom_push(key='accuracy', value=accuracy)

    if accuracy >= 0.95:
        return 'deploy_to_production'
    elif accuracy >= 0.85:
        return 'deploy_to_staging'
    elif accuracy >= 0.75:
        return 'retrain_with_more_data'
    else:
        return 'investigate_model'

def multi_condition_branch(**context):
    """Branch based on multiple conditions"""
    ti = context['task_instance']
    accuracy = ti.xcom_pull(task_ids='evaluate', key='accuracy')

    # Multiple branches can be returned as list
    branches = []

    if accuracy >= 0.9:
        branches.append('high_performance_path')
    if accuracy < 0.8:
        branches.append('low_performance_alert')

    branches.append('log_metrics')  # Always log

    return branches

with DAG(
    'model_deployment_branching',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')

    evaluate = BranchPythonOperator(
        task_id='evaluate',
        python_callable=evaluate_model_performance
    )

    deploy_prod = PythonOperator(
        task_id='deploy_to_production',
        python_callable=lambda: print("Deploying to production!")
    )

    deploy_staging = PythonOperator(
        task_id='deploy_to_staging',
        python_callable=lambda: print("Deploying to staging!")
    )

    retrain = PythonOperator(
        task_id='retrain_with_more_data',
        python_callable=lambda: print("Triggering retraining...")
    )

    investigate = PythonOperator(
        task_id='investigate_model',
        python_callable=lambda: print("Model needs investigation!")
    )

    # Join point - all branches converge here
    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success'
    )

    start >> evaluate >> [deploy_prod, deploy_staging, retrain, investigate] >> end
```

---

## Step 10: Complete Enterprise ML DAG

Create `airflow/dags/enterprise_ml_pipeline.py`:

```python
"""
Enterprise ML Pipeline DAG
Demonstrates all Airflow enterprise features
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
import json

# Import custom operators
from operators.mlflow_operators import (
    MLflowLogMetricsOperator,
    MLflowModelTransitionOperator
)
from operators.kubeflow_operators import KubeflowPipelineOperator
from sensors.ml_sensors import (
    DataQualitySensor,
    ModelPerformanceSensor,
    MLflowModelStageSensor
)

default_args = {
    'owner': 'ml-platform-team',
    'depends_on_past': False,
    'email': ['ml-platform@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'sla': timedelta(hours=4),
}

def get_config():
    """Get pipeline configuration"""
    return Variable.get('ml_pipeline_config', deserialize_json=True, default_var={
        'model_name': 'churn-classifier',
        'accuracy_threshold': 0.85,
        'data_source': 's3://data/customers/',
        'training_params': {
            'n_estimators': 100,
            'max_depth': 10
        }
    })

def extract_data(**context):
    """Extract data from source"""
    config = get_config()
    # Implementation
    return {'rows_extracted': 10000, 'source': config['data_source']}

def preprocess_data(**context):
    """Preprocess extracted data"""
    ti = context['task_instance']
    extraction_result = ti.xcom_pull(task_ids='extraction.extract')
    # Implementation
    return {'rows_processed': extraction_result['rows_extracted'], 'features': 50}

def train_model(**context):
    """Train ML model"""
    import mlflow
    from sklearn.ensemble import RandomForestClassifier

    config = get_config()
    mlflow_conn = BaseHook.get_connection('mlflow_default')
    mlflow.set_tracking_uri(f"http://{mlflow_conn.host}:{mlflow_conn.port}")

    with mlflow.start_run(run_name=f"training-{context['ts']}"):
        # Training logic
        mlflow.log_params(config['training_params'])
        mlflow.log_metric('accuracy', 0.92)

        # Log model
        model = RandomForestClassifier(**config['training_params'])
        mlflow.sklearn.log_model(model, "model")

        return {
            'run_id': mlflow.active_run().info.run_id,
            'accuracy': 0.92
        }

def evaluate_model_branch(**context):
    """Decide deployment strategy based on model performance"""
    ti = context['task_instance']
    training_result = ti.xcom_pull(task_ids='training.train')
    config = get_config()

    accuracy = training_result['accuracy']

    if accuracy >= 0.95:
        return 'deployment.deploy_production'
    elif accuracy >= config['accuracy_threshold']:
        return 'deployment.deploy_staging'
    else:
        return 'deployment.trigger_retraining'

def deploy_to_production(**context):
    """Deploy model to production"""
    ti = context['task_instance']
    training_result = ti.xcom_pull(task_ids='training.train')
    config = get_config()

    # Transition model to Production stage
    from mlflow import MlflowClient
    mlflow_conn = BaseHook.get_connection('mlflow_default')
    client = MlflowClient(f"http://{mlflow_conn.host}:{mlflow_conn.port}")

    # Get latest version
    versions = client.search_model_versions(f"name='{config['model_name']}'")
    latest = max(versions, key=lambda x: int(x.version))

    client.transition_model_version_stage(
        name=config['model_name'],
        version=latest.version,
        stage="Production",
        archive_existing_versions=True
    )

    return {'deployed_version': latest.version, 'stage': 'Production'}

def deploy_to_staging(**context):
    """Deploy model to staging"""
    # Similar to production but with Staging stage
    return {'stage': 'Staging'}

def trigger_retraining(**context):
    """Trigger model retraining"""
    # Trigger Kubeflow pipeline for retraining
    return {'action': 'retrain_triggered'}

def send_deployment_notification(**context):
    """Send deployment notification"""
    ti = context['task_instance']

    # Get results from all possible upstream tasks
    prod_result = ti.xcom_pull(task_ids='deployment.deploy_production')
    staging_result = ti.xcom_pull(task_ids='deployment.deploy_staging')
    retrain_result = ti.xcom_pull(task_ids='deployment.trigger_retraining')

    result = prod_result or staging_result or retrain_result

    message = {
        'dag_id': context['dag'].dag_id,
        'execution_date': str(context['execution_date']),
        'result': result
    }

    print(f"Deployment notification: {json.dumps(message, indent=2)}")
    # Send to Slack, email, etc.

with DAG(
    'enterprise_ml_pipeline',
    default_args=default_args,
    description='Enterprise ML Pipeline with all features',
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'enterprise', 'production'],
    doc_md="""
    # Enterprise ML Pipeline

    This DAG demonstrates a production-ready ML pipeline with:
    - Data extraction and validation
    - Feature engineering
    - Model training with MLflow tracking
    - Automated deployment based on performance
    - Monitoring and alerting

    ## Configuration
    Set `ml_pipeline_config` variable to customize behavior.
    """,
) as dag:

    start = EmptyOperator(task_id='start')

    # Data Quality Check
    data_quality_check = DataQualitySensor(
        task_id='data_quality_check',
        data_path='/data/customers/latest.csv',
        min_rows=1000,
        max_null_percentage=0.05,
        required_columns=['customer_id', 'tenure', 'churn'],
        mode='poke',
        poke_interval=60,
        timeout=1800,
    )

    # Data Extraction TaskGroup
    with TaskGroup(group_id='extraction') as extraction_group:
        extract = PythonOperator(
            task_id='extract',
            python_callable=extract_data,
            pool='cpu_intensive',
        )

        preprocess = PythonOperator(
            task_id='preprocess',
            python_callable=preprocess_data,
            pool='cpu_intensive',
        )

        extract >> preprocess

    # Training TaskGroup
    with TaskGroup(group_id='training') as training_group:
        train = PythonOperator(
            task_id='train',
            python_callable=train_model,
            pool='gpu_pool',
            priority_weight=10,
        )

        evaluate = BranchPythonOperator(
            task_id='evaluate',
            python_callable=evaluate_model_branch,
        )

        train >> evaluate

    # Deployment TaskGroup
    with TaskGroup(group_id='deployment') as deployment_group:
        deploy_prod = PythonOperator(
            task_id='deploy_production',
            python_callable=deploy_to_production,
        )

        deploy_staging = PythonOperator(
            task_id='deploy_staging',
            python_callable=deploy_to_staging,
        )

        retrain = PythonOperator(
            task_id='trigger_retraining',
            python_callable=trigger_retraining,
        )

    # Notification
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_deployment_notification,
        trigger_rule='none_failed_min_one_success',
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    # Define dependencies
    start >> data_quality_check >> extraction_group >> training_group
    training_group >> [deploy_prod, deploy_staging, retrain]
    [deploy_prod, deploy_staging, retrain] >> notify >> end
```

---

## Verification

```bash
#!/bin/bash
# verify_airflow.sh

echo "=== Airflow Verification ==="

echo -e "\n1. Airflow Pods:"
kubectl get pods -n airflow

echo -e "\n2. Airflow Services:"
kubectl get svc -n airflow

echo -e "\n3. Airflow Ingress:"
kubectl get ingress -n airflow

echo -e "\n4. List DAGs:"
kubectl exec -n airflow deployment/airflow-webserver -- airflow dags list

echo -e "\n5. List Pools:"
kubectl exec -n airflow deployment/airflow-webserver -- airflow pools list

echo -e "\n6. List Connections:"
kubectl exec -n airflow deployment/airflow-webserver -- airflow connections list

echo -e "\n=== Verification Complete ==="
```

---

## Next Steps

- **Phase 04**: Kubeflow Complete Setup
- **Phase 05**: Feature Store & Data Validation

---

**Status**: Phase 03 Complete
**Features Covered**: All Airflow enterprise features
