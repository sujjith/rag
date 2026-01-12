"""
Airflow DAG: Customer Churn Data Ingestion Pipeline
Extracts customer data from PostgreSQL and uploads to MinIO object storage.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
from sqlalchemy import create_engine


# Default DAG arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def extract_customer_data(**context):
    """Pull customer data from PostgreSQL data warehouse"""
    
    # PostgreSQL connection (update with your credentials)
    db_url = "postgresql://postgres:postgres123@postgresql.postgresql.svc.cluster.local:5432/customers"
    engine = create_engine(db_url)
    
    # Extract customer data
    query = """
    SELECT 
        customer_id,
        age,
        gender,
        tenure_months,
        total_purchases,
        avg_order_value,
        days_since_last_purchase,
        support_tickets_count,
        churn
    FROM customers
    """
    
    df = pd.read_sql(query, engine)
    
    # Save to local path for next task
    output_path = "/tmp/customers.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"Extracted {len(df)} customer records")
    return output_path


def push_to_minio(**context):
    """Upload extracted data to MinIO object storage"""
    
    # Get the file path from previous task
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='extract_customer_data')
    
    # MinIO connection
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio.minio.svc.cluster.local:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"raw/customers_{timestamp}.parquet"
    
    # Upload to MinIO
    s3.upload_file(input_path, 'dvc-storage', s3_key)
    
    print(f"Uploaded to s3://dvc-storage/{s3_key}")
    return s3_key


def validate_upload(**context):
    """Verify the file was uploaded successfully"""
    
    ti = context['ti']
    s3_key = ti.xcom_pull(task_ids='push_to_minio')
    
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio.minio.svc.cluster.local:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    
    # Check if file exists
    response = s3.head_object(Bucket='dvc-storage', Key=s3_key)
    file_size = response['ContentLength']
    
    print(f"Validated: {s3_key} ({file_size} bytes)")
    return True


# Define the DAG
with DAG(
    'customer_churn_ingestion',
    default_args=default_args,
    description='Extract customer data from PostgreSQL and upload to MinIO',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['churn', 'ingestion', 'mlops'],
) as dag:
    
    extract_task = PythonOperator(
        task_id='extract_customer_data',
        python_callable=extract_customer_data,
    )
    
    upload_task = PythonOperator(
        task_id='push_to_minio',
        python_callable=push_to_minio,
    )
    
    validate_task = PythonOperator(
        task_id='validate_upload',
        python_callable=validate_upload,
    )
    
    # Define task dependencies
    extract_task >> upload_task >> validate_task
