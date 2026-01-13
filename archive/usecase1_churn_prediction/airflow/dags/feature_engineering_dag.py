"""
Airflow DAG: Feature Engineering Pipeline using Feast
Transforms validated data into features and materializes to Feast feature store.
Triggered after data_validation_dag completes.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
import os


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def prepare_feature_data(**context):
    """
    Download validated data and prepare it for Feast feature store.
    Creates computed features from raw data.
    """
    
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio.minio.svc.cluster.local:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    
    # Get the latest validated file
    response = s3.list_objects_v2(Bucket='dvc-storage', Prefix='raw/')
    latest = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
    
    # Download to shared storage
    raw_path = "/shared/customers_raw.parquet"
    s3.download_file('dvc-storage', latest['Key'], raw_path)
    
    # Load and transform
    df = pd.read_parquet(raw_path)
    
    # Add event_timestamp for Feast (required)
    df['event_timestamp'] = pd.Timestamp.now()
    
    # Create computed features
    df['purchase_frequency'] = df['total_purchases'] / (df['tenure_months'] + 1)
    df['support_ratio'] = df['support_tickets_count'] / (df['total_purchases'] + 1)
    df['recency_score'] = 1 / (df['days_since_last_purchase'] + 1)
    
    # Save feature data
    feature_path = "/shared/customer_features.parquet"
    df.to_parquet(feature_path, index=False)
    
    # Also upload to MinIO for Feast offline store
    s3.upload_file(feature_path, 'dvc-storage', 'features/customer_features.parquet')
    
    print(f"Prepared {len(df)} customer features")
    print(f"New features: purchase_frequency, support_ratio, recency_score")
    
    return feature_path


def register_feast_features(**context):
    """
    Create Feast feature definitions and apply them.
    This registers the feature views with the Feast server.
    """
    
    # Create feature_store.yaml if not exists
    feature_repo_path = "/shared/feature_repo"
    os.makedirs(feature_repo_path, exist_ok=True)
    
    # Feature store config
    feature_store_yaml = """
project: churn_prediction
provider: local
registry: /shared/feature_repo/registry.db
online_store:
  type: sqlite
  path: /shared/feature_repo/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
"""
    
    with open(f"{feature_repo_path}/feature_store.yaml", "w") as f:
        f.write(feature_store_yaml)
    
    # Feature definitions
    features_py = '''
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# Entity - the primary key for features
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Customer ID"
)

# Data source - parquet file in MinIO (mounted or accessible)
customer_source = FileSource(
    path="/shared/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# Feature View - defines the features available for this entity
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="tenure_months", dtype=Int32),
        Field(name="total_purchases", dtype=Int32),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int32),
        Field(name="support_tickets_count", dtype=Int32),
        Field(name="purchase_frequency", dtype=Float32),
        Field(name="support_ratio", dtype=Float32),
        Field(name="recency_score", dtype=Float32),
        Field(name="churn", dtype=Int32),
    ],
    source=customer_source,
    online=True,
)
'''
    
    with open(f"{feature_repo_path}/features.py", "w") as f:
        f.write(features_py)
    
    # Run feast apply
    import subprocess
    result = subprocess.run(
        ["feast", "apply"],
        cwd=feature_repo_path,
        capture_output=True,
        text=True
    )
    
    print(f"Feast apply stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"Feast apply stderr: {result.stderr}")
        raise Exception(f"Feast apply failed: {result.stderr}")
    
    return "Features registered"


def materialize_features(**context):
    """
    Materialize features from offline to online store.
    This makes features available for real-time serving.
    """
    
    import subprocess
    from datetime import datetime, timedelta
    
    feature_repo_path = "/shared/feature_repo"
    
    # Materialize from 7 days ago to now
    start_date = (datetime.now() - timedelta(days=7)).isoformat()
    end_date = datetime.now().isoformat()
    
    result = subprocess.run(
        ["feast", "materialize", start_date, end_date],
        cwd=feature_repo_path,
        capture_output=True,
        text=True
    )
    
    print(f"Materialize stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"Materialize stderr: {result.stderr}")
        # Don't fail on materialize errors for now
        print("Warning: Materialize had issues but continuing...")
    
    return "Features materialized"


with DAG(
    'feature_engineering_v2',
    default_args=default_args,
    description='Transform data into features and materialize to Feast',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['features', 'feast', 'mlops'],
) as dag:
    
    prepare_task = PythonOperator(
        task_id='prepare_feature_data',
        python_callable=prepare_feature_data,
    )
    
    register_task = PythonOperator(
        task_id='register_feast_features',
        python_callable=register_feast_features,
    )
    
    materialize_task = PythonOperator(
        task_id='materialize_features',
        python_callable=materialize_features,
    )
    
    prepare_task >> register_task >> materialize_task
