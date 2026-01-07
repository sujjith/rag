# Phase 05: Feature Store & Data Validation

## Overview

Setup Feast feature store for feature management and Great Expectations for data validation in the ML pipeline.

---

## Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEAST FEATURE STORE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     FEATURE REGISTRY                                 │   │
│   │              (PostgreSQL - Feature Definitions)                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│          ┌──────────────────────────┴──────────────────────────┐            │
│          │                                                      │            │
│          ▼                                                      ▼            │
│   ┌─────────────────┐                                 ┌─────────────────┐   │
│   │  OFFLINE STORE  │                                 │  ONLINE STORE   │   │
│   │   (S3/MinIO)    │──── Materialization ──────────►│    (Redis)      │   │
│   │  Historical     │                                 │  Low-latency    │   │
│   │  Features       │                                 │  Serving        │   │
│   └─────────────────┘                                 └─────────────────┘   │
│          │                                                      │            │
│          │                                                      │            │
│          ▼                                                      ▼            │
│   ┌─────────────────┐                                 ┌─────────────────┐   │
│   │ Training Data   │                                 │ Real-time       │   │
│   │ Generation      │                                 │ Inference       │   │
│   └─────────────────┘                                 └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Deploy Feast

### Install Feast

```bash
# Install Feast CLI
pip install 'feast[redis,postgres,aws]'

# Create Feast project
mkdir -p feast-repo
cd feast-repo
feast init customer_features
cd customer_features
```

### Configure Feast Repository

Create `feast-repo/customer_features/feature_store.yaml`:

```yaml
project: customer_features
provider: local
registry:
  registry_type: sql
  path: postgresql://feast:feast123@postgres-postgresql.mlflow.svc.cluster.local:5432/feast
online_store:
  type: redis
  connection_string: redis://redis-master.airflow.svc.cluster.local:6379
offline_store:
  type: file
  # For S3/MinIO:
  # type: s3
  # path: s3://feast/offline_store
entity_key_serialization_version: 2
```

### Define Feature Repository

Create `feast-repo/customer_features/features.py`:

```python
from datetime import timedelta
from feast import (
    Entity,
    Feature,
    FeatureView,
    FileSource,
    Field,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.types import Float32, Int64, String

# Define Entities
customer = Entity(
    name="customer_id",
    description="Customer identifier",
    value_type=ValueType.INT64,
)

# Define Data Sources
customer_stats_source = FileSource(
    name="customer_stats_source",
    path="s3://feast/data/customer_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transaction_stats_source = FileSource(
    name="transaction_stats_source",
    path="s3://feast/data/transaction_stats.parquet",
    timestamp_field="event_timestamp",
)

# Push source for real-time features
customer_push_source = PushSource(
    name="customer_push_source",
    batch_source=customer_stats_source,
)

# Request source for on-demand features
request_source = RequestSource(
    name="request_source",
    schema=[
        Field(name="current_balance", dtype=Float32),
    ],
)

# Define Feature Views
customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure_months", dtype=Int64),
        Field(name="total_purchases", dtype=Int64),
        Field(name="total_spend", dtype=Float32),
        Field(name="avg_purchase_value", dtype=Float32),
        Field(name="customer_segment", dtype=String),
        Field(name="loyalty_score", dtype=Float32),
    ],
    source=customer_stats_source,
    online=True,
    tags={"team": "customer-analytics"},
)

transaction_stats_fv = FeatureView(
    name="transaction_stats",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="txn_count_7d", dtype=Int64),
        Field(name="txn_count_30d", dtype=Int64),
        Field(name="txn_amount_7d", dtype=Float32),
        Field(name="txn_amount_30d", dtype=Float32),
        Field(name="avg_txn_amount_7d", dtype=Float32),
        Field(name="max_txn_amount_30d", dtype=Float32),
    ],
    source=transaction_stats_source,
    online=True,
    tags={"team": "fraud-detection"},
)

# On-demand Feature View
from feast import on_demand_feature_view
import pandas as pd

@on_demand_feature_view(
    sources=[customer_stats_fv, request_source],
    schema=[
        Field(name="balance_to_spend_ratio", dtype=Float32),
        Field(name="is_high_value", dtype=Int64),
    ],
)
def customer_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["balance_to_spend_ratio"] = inputs["current_balance"] / (inputs["total_spend"] + 1)
    df["is_high_value"] = (inputs["loyalty_score"] > 0.8).astype(int)
    return df
```

### Apply Feature Definitions

```bash
# Apply feature definitions
cd feast-repo/customer_features
feast apply

# Verify
feast feature-views list
feast entities list
```

---

## Step 2: Feast Operations

### Materialize Features to Online Store

```python
# feast_operations.py
from feast import FeatureStore
from datetime import datetime, timedelta

store = FeatureStore(repo_path="feast-repo/customer_features")

# Materialize features to online store
store.materialize_incremental(
    end_date=datetime.utcnow(),
    feature_views=["customer_stats", "transaction_stats"]
)

# Or materialize specific time range
store.materialize(
    start_date=datetime.utcnow() - timedelta(days=7),
    end_date=datetime.utcnow(),
)
```

### Get Training Data

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="feast-repo/customer_features")

# Entity DataFrame (customers to get features for)
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003, 1004, 1005],
    "event_timestamp": [datetime.utcnow()] * 5,
    "current_balance": [1000.0, 2500.0, 500.0, 3000.0, 750.0]  # For on-demand features
})

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_stats:tenure_months",
        "customer_stats:total_purchases",
        "customer_stats:total_spend",
        "customer_stats:loyalty_score",
        "transaction_stats:txn_count_7d",
        "transaction_stats:txn_amount_7d",
        "customer_derived_features:balance_to_spend_ratio",
        "customer_derived_features:is_high_value",
    ],
).to_df()

print(training_df)
```

### Get Online Features (Real-time Serving)

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feast-repo/customer_features")

# Get online features for inference
feature_vector = store.get_online_features(
    features=[
        "customer_stats:tenure_months",
        "customer_stats:loyalty_score",
        "transaction_stats:txn_count_7d",
        "transaction_stats:txn_amount_7d",
    ],
    entity_rows=[
        {"customer_id": 1001},
        {"customer_id": 1002},
    ],
).to_dict()

print(feature_vector)
```

### Push Real-time Features

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="feast-repo/customer_features")

# Push new feature values in real-time
push_df = pd.DataFrame({
    "customer_id": [1001],
    "tenure_months": [25],
    "total_purchases": [150],
    "total_spend": [5500.0],
    "avg_purchase_value": [36.67],
    "customer_segment": ["gold"],
    "loyalty_score": [0.85],
    "event_timestamp": [datetime.utcnow()],
    "created_timestamp": [datetime.utcnow()],
})

store.push("customer_push_source", push_df)
```

---

## Step 3: Feast Feature Server

### Deploy Feast Server on Kubernetes

Create `feast/kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feast-feature-server
  namespace: feast
spec:
  replicas: 2
  selector:
    matchLabels:
      app: feast-feature-server
  template:
    metadata:
      labels:
        app: feast-feature-server
    spec:
      containers:
      - name: feast
        image: feastdev/feature-server:0.35.0
        ports:
        - containerPort: 6566
        env:
        - name: FEAST_REPO_PATH
          value: /feast-repo
        - name: FEAST_SERVER_HOST
          value: "0.0.0.0"
        - name: FEAST_SERVER_PORT
          value: "6566"
        volumeMounts:
        - name: feast-repo
          mountPath: /feast-repo
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        readinessProbe:
          httpGet:
            path: /health
            port: 6566
          initialDelaySeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 6566
          initialDelaySeconds: 30
      volumes:
      - name: feast-repo
        configMap:
          name: feast-repo-config
---
apiVersion: v1
kind: Service
metadata:
  name: feast-feature-server
  namespace: feast
spec:
  selector:
    app: feast-feature-server
  ports:
  - port: 6566
    targetPort: 6566
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: feast-feature-server
  namespace: feast
spec:
  ingressClassName: nginx
  rules:
  - host: feast.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: feast-feature-server
            port:
              number: 6566
```

### Use Feast in Airflow DAG

```python
# airflow/dags/feast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def materialize_features(**context):
    """Materialize features to online store"""
    from feast import FeatureStore

    store = FeatureStore(repo_path="/feast-repo")
    store.materialize_incremental(end_date=datetime.utcnow())

    return {"status": "materialized"}

def get_training_features(**context):
    """Get training features from Feast"""
    from feast import FeatureStore
    import pandas as pd

    store = FeatureStore(repo_path="/feast-repo")

    # Get entity data
    entity_df = pd.read_csv("s3://data/training_entities.csv")
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # Get historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_stats:tenure_months",
            "customer_stats:total_spend",
            "customer_stats:loyalty_score",
            "transaction_stats:txn_count_7d",
        ],
    ).to_df()

    # Save for training
    training_df.to_parquet("s3://data/training_data.parquet")

    context['task_instance'].xcom_push(
        key='training_data_path',
        value="s3://data/training_data.parquet"
    )

    return {"rows": len(training_df)}

with DAG(
    'feast_feature_pipeline',
    default_args={'owner': 'ml-team'},
    schedule_interval='0 * * * *',  # Hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    materialize = PythonOperator(
        task_id='materialize_features',
        python_callable=materialize_features
    )

    get_features = PythonOperator(
        task_id='get_training_features',
        python_callable=get_training_features
    )

    materialize >> get_features
```

---

## Step 4: Great Expectations Data Validation

### Initialize Great Expectations

```bash
# Install Great Expectations
pip install great-expectations

# Initialize GX project
great_expectations init
```

### Configure Data Sources

Create `great_expectations/great_expectations.yml`:

```yaml
config_version: 3.0
datasources:
  pandas_datasource:
    class_name: Datasource
    module_name: great_expectations.datasource
    execution_engine:
      class_name: PandasExecutionEngine
      module_name: great_expectations.execution_engine
    data_connectors:
      default_inferred_data_connector:
        class_name: InferredAssetFilesystemDataConnector
        module_name: great_expectations.datasource.data_connector
        base_directory: /data
        default_regex:
          pattern: (.*)\.csv
          group_names:
            - data_asset_name
      default_runtime_data_connector:
        class_name: RuntimeDataConnector
        module_name: great_expectations.datasource.data_connector
        batch_identifiers:
          - default_identifier_name

  spark_datasource:
    class_name: Datasource
    module_name: great_expectations.datasource
    execution_engine:
      class_name: SparkDFExecutionEngine
      module_name: great_expectations.execution_engine
    data_connectors:
      default_runtime_data_connector:
        class_name: RuntimeDataConnector
        module_name: great_expectations.datasource.data_connector
        batch_identifiers:
          - default_identifier_name

config_variables_file_path: uncommitted/config_variables.yml
plugins_directory: plugins/

stores:
  expectations_store:
    class_name: ExpectationsStore
    store_backend:
      class_name: TupleFilesystemStoreBackend
      base_directory: expectations/

  validations_store:
    class_name: ValidationsStore
    store_backend:
      class_name: TupleFilesystemStoreBackend
      base_directory: uncommitted/validations/

  evaluation_parameter_store:
    class_name: EvaluationParameterStore

  checkpoint_store:
    class_name: CheckpointStore
    store_backend:
      class_name: TupleFilesystemStoreBackend
      base_directory: checkpoints/

expectations_store_name: expectations_store
validations_store_name: validations_store
evaluation_parameter_store_name: evaluation_parameter_store
checkpoint_store_name: checkpoint_store

data_docs_sites:
  local_site:
    class_name: SiteBuilder
    store_backend:
      class_name: TupleFilesystemStoreBackend
      base_directory: uncommitted/data_docs/local_site/
    site_index_builder:
      class_name: DefaultSiteIndexBuilder
```

### Create Expectation Suite

```python
# great_expectations/create_expectations.py
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

# Get context
context = gx.get_context()

# Create expectation suite
suite_name = "customer_data_suite"
suite = context.add_expectation_suite(expectation_suite_name=suite_name)

# Add expectations
expectations = [
    # Schema expectations
    ExpectationConfiguration(
        expectation_type="expect_table_columns_to_match_ordered_list",
        kwargs={
            "column_list": [
                "customer_id", "tenure_months", "monthly_charges",
                "total_charges", "contract_type", "churn"
            ]
        }
    ),

    # Row count expectations
    ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_be_between",
        kwargs={"min_value": 1000, "max_value": 1000000}
    ),

    # Column expectations
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "customer_id"}
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_unique",
        kwargs={"column": "customer_id"}
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "tenure_months", "min_value": 0, "max_value": 120}
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "monthly_charges", "min_value": 0, "max_value": 500}
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "contract_type",
            "value_set": ["Month-to-month", "One year", "Two year"]
        }
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={"column": "churn", "value_set": [0, 1]}
    ),

    # Distribution expectations
    ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={"column": "monthly_charges", "min_value": 50, "max_value": 80}
    ),
    ExpectationConfiguration(
        expectation_type="expect_column_proportion_of_unique_values_to_be_between",
        kwargs={"column": "customer_id", "min_value": 0.99, "max_value": 1.0}
    ),
]

for expectation in expectations:
    suite.add_expectation(expectation_configuration=expectation)

context.save_expectation_suite(expectation_suite=suite)
print(f"Created expectation suite: {suite_name}")
```

### Create Checkpoint

```python
# great_expectations/create_checkpoint.py
import great_expectations as gx

context = gx.get_context()

# Create checkpoint
checkpoint_config = {
    "name": "customer_data_checkpoint",
    "config_version": 1.0,
    "class_name": "Checkpoint",
    "run_name_template": "%Y%m%d-%H%M%S-customer-validation",
    "validations": [
        {
            "batch_request": {
                "datasource_name": "pandas_datasource",
                "data_connector_name": "default_inferred_data_connector",
                "data_asset_name": "customer_data",
            },
            "expectation_suite_name": "customer_data_suite",
        }
    ],
    "action_list": [
        {
            "name": "store_validation_result",
            "action": {"class_name": "StoreValidationResultAction"},
        },
        {
            "name": "store_evaluation_params",
            "action": {"class_name": "StoreEvaluationParametersAction"},
        },
        {
            "name": "update_data_docs",
            "action": {"class_name": "UpdateDataDocsAction"},
        },
        {
            "name": "send_slack_notification",
            "action": {
                "class_name": "SlackNotificationAction",
                "slack_webhook": "${SLACK_WEBHOOK}",
                "notify_on": "failure",
                "renderer": {
                    "module_name": "great_expectations.render.renderer.slack_renderer",
                    "class_name": "SlackRenderer",
                },
            },
        },
    ],
}

context.add_checkpoint(**checkpoint_config)
```

### Run Validation

```python
# great_expectations/run_validation.py
import great_expectations as gx
import pandas as pd

context = gx.get_context()

# Load data
df = pd.read_csv("/data/customer_data.csv")

# Run checkpoint
result = context.run_checkpoint(
    checkpoint_name="customer_data_checkpoint",
    batch_request={
        "runtime_parameters": {"batch_data": df},
        "batch_identifiers": {"default_identifier_name": "customer_batch"},
    },
)

# Check results
if not result.success:
    failed_expectations = []
    for validation_result in result.list_validation_results():
        for result_item in validation_result.results:
            if not result_item.success:
                failed_expectations.append({
                    "expectation": result_item.expectation_config.expectation_type,
                    "kwargs": result_item.expectation_config.kwargs,
                })

    print(f"Validation FAILED! Failed expectations: {failed_expectations}")
    raise Exception("Data validation failed")
else:
    print("Validation PASSED!")
```

---

## Step 5: Integrate with Airflow

Create `airflow/dags/data_validation_dag.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-quality-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def validate_source_data(**context):
    """Validate source data with Great Expectations"""
    import great_expectations as gx
    import pandas as pd

    context_gx = gx.get_context()

    # Load data
    df = pd.read_csv(context['params']['data_path'])

    # Run validation
    result = context_gx.run_checkpoint(
        checkpoint_name="customer_data_checkpoint",
        batch_request={
            "runtime_parameters": {"batch_data": df},
            "batch_identifiers": {"default_identifier_name": "airflow_batch"},
        },
    )

    # Store results
    context['task_instance'].xcom_push(key='validation_success', value=result.success)
    context['task_instance'].xcom_push(
        key='validation_statistics',
        value=result.get_statistics()
    )

    return result.success

def branch_on_validation(**context):
    """Branch based on validation results"""
    ti = context['task_instance']
    validation_success = ti.xcom_pull(task_ids='validate_data', key='validation_success')

    if validation_success:
        return 'proceed_to_feature_engineering'
    else:
        return 'send_alert_and_quarantine'

def feature_engineering(**context):
    """Feature engineering with Feast"""
    from feast import FeatureStore
    import pandas as pd

    store = FeatureStore(repo_path="/feast-repo")

    # Get validated data
    entity_df = pd.read_csv(context['params']['data_path'])
    entity_df["event_timestamp"] = datetime.utcnow()

    # Enrich with Feast features
    enriched_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_stats:loyalty_score",
            "transaction_stats:txn_count_7d",
        ],
    ).to_df()

    # Save enriched data
    output_path = "s3://data/enriched_data.parquet"
    enriched_df.to_parquet(output_path)

    context['task_instance'].xcom_push(key='enriched_data_path', value=output_path)

def validate_enriched_data(**context):
    """Validate enriched data before training"""
    import great_expectations as gx
    import pandas as pd

    ti = context['task_instance']
    enriched_path = ti.xcom_pull(task_ids='feature_engineering', key='enriched_data_path')

    context_gx = gx.get_context()
    df = pd.read_parquet(enriched_path)

    result = context_gx.run_checkpoint(
        checkpoint_name="enriched_data_checkpoint",
        batch_request={
            "runtime_parameters": {"batch_data": df},
            "batch_identifiers": {"default_identifier_name": "enriched_batch"},
        },
    )

    if not result.success:
        raise Exception("Enriched data validation failed!")

    return result.success

def quarantine_data(**context):
    """Move invalid data to quarantine"""
    import shutil

    data_path = context['params']['data_path']
    quarantine_path = f"s3://data/quarantine/{datetime.utcnow().isoformat()}"

    # Move to quarantine
    # In production, use boto3 for S3 operations
    print(f"Moving {data_path} to {quarantine_path}")

def send_alert(**context):
    """Send alert for failed validation"""
    ti = context['task_instance']
    stats = ti.xcom_pull(task_ids='validate_data', key='validation_statistics')

    message = f"""
    Data Validation Failed!

    Statistics: {stats}
    Data Path: {context['params']['data_path']}
    Execution Date: {context['execution_date']}
    """

    # Send to Slack, email, etc.
    print(message)

with DAG(
    'data_quality_pipeline',
    default_args=default_args,
    schedule_interval='0 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={'data_path': 's3://data/customer_data.csv'},
) as dag:

    start = EmptyOperator(task_id='start')

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_source_data
    )

    branch = BranchPythonOperator(
        task_id='branch_on_validation',
        python_callable=branch_on_validation
    )

    proceed = PythonOperator(
        task_id='proceed_to_feature_engineering',
        python_callable=feature_engineering
    )

    validate_enriched = PythonOperator(
        task_id='validate_enriched_data',
        python_callable=validate_enriched_data
    )

    quarantine = PythonOperator(
        task_id='send_alert_and_quarantine',
        python_callable=quarantine_data
    )

    alert = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,
        trigger_rule='all_done'
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success'
    )

    start >> validate >> branch
    branch >> proceed >> validate_enriched >> end
    branch >> quarantine >> alert >> end
```

---

## Verification

```bash
#!/bin/bash
# verify_feast_gx.sh

echo "=== Feature Store & Validation Verification ==="

echo -e "\n1. Feast Registry:"
feast feature-views list
feast entities list

echo -e "\n2. Great Expectations:"
great_expectations checkpoint list
great_expectations suite list

echo -e "\n3. Feast Server (if deployed):"
kubectl get pods -n feast

echo -e "\n=== Verification Complete ==="
```

---

**Status**: Phase 05 Complete
**Features Covered**: Feast Feature Store, Great Expectations Data Validation
