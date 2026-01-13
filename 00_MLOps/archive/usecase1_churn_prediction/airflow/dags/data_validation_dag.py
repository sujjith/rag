"""
Airflow DAG: Data Validation Pipeline using Great Expectations
Validates customer data quality before it flows to feature engineering.
Triggered after data_ingestion_dag completes.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pandas as pd
import boto3


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def download_latest_data(**context):
    """Download the latest parquet file from MinIO for validation"""
    
    s3 = boto3.client(
        's3',
        endpoint_url='http://minio.minio.svc.cluster.local:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    
    # Get the latest file from raw/ prefix
    response = s3.list_objects_v2(Bucket='dvc-storage', Prefix='raw/')
    latest = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
    
    # Download to shared storage
    local_path = "/shared/customers_to_validate.parquet"
    s3.download_file('dvc-storage', latest['Key'], local_path)
    
    print(f"Downloaded {latest['Key']} for validation")
    return local_path


def validate_customer_data(**context):
    """Validate customer data using Great Expectations-style checks"""
    
    ti = context['ti']
    file_path = ti.xcom_pull(task_ids='download_latest_data')
    
    # Load the data
    df = pd.read_parquet(file_path)
    
    validation_results = {
        'total_rows': len(df),
        'checks_passed': 0,
        'checks_failed': 0,
        'errors': []
    }
    
    # Check 1: customer_id column exists and has no nulls
    if 'customer_id' not in df.columns:
        validation_results['errors'].append("FAIL: customer_id column missing")
        validation_results['checks_failed'] += 1
    elif df['customer_id'].isnull().any():
        validation_results['errors'].append("FAIL: customer_id has null values")
        validation_results['checks_failed'] += 1
    else:
        validation_results['checks_passed'] += 1
        print("PASS: customer_id exists and has no nulls")
    
    # Check 2: age is between 18 and 120
    if 'age' in df.columns:
        invalid_age = df[(df['age'] < 18) | (df['age'] > 120)]
        if len(invalid_age) > 0:
            validation_results['errors'].append(f"FAIL: {len(invalid_age)} rows with invalid age")
            validation_results['checks_failed'] += 1
        else:
            validation_results['checks_passed'] += 1
            print("PASS: age values are valid (18-120)")
    
    # Check 3: churn is binary (0 or 1)
    if 'churn' in df.columns:
        invalid_churn = df[~df['churn'].isin([0, 1])]
        if len(invalid_churn) > 0:
            validation_results['errors'].append(f"FAIL: {len(invalid_churn)} rows with invalid churn")
            validation_results['checks_failed'] += 1
        else:
            validation_results['checks_passed'] += 1
            print("PASS: churn values are binary (0/1)")
    
    # Check 4: No duplicate customer_ids
    if 'customer_id' in df.columns:
        duplicates = df['customer_id'].duplicated().sum()
        if duplicates > 0:
            validation_results['errors'].append(f"FAIL: {duplicates} duplicate customer_ids")
            validation_results['checks_failed'] += 1
        else:
            validation_results['checks_passed'] += 1
            print("PASS: No duplicate customer_ids")
    
    # Check 5: Numeric columns have no negative values (where applicable)
    numeric_cols = ['total_purchases', 'avg_order_value', 'days_since_last_purchase']
    for col in numeric_cols:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                validation_results['errors'].append(f"FAIL: {col} has {negatives} negative values")
                validation_results['checks_failed'] += 1
            else:
                validation_results['checks_passed'] += 1
                print(f"PASS: {col} has no negative values")
    
    # Summary
    print(f"\n=== Validation Summary ===")
    print(f"Total rows: {validation_results['total_rows']}")
    print(f"Checks passed: {validation_results['checks_passed']}")
    print(f"Checks failed: {validation_results['checks_failed']}")
    
    if validation_results['checks_failed'] > 0:
        for error in validation_results['errors']:
            print(error)
        raise Exception(f"Data validation failed with {validation_results['checks_failed']} errors!")
    
    return validation_results


def mark_data_validated(**context):
    """Mark data as validated and ready for feature engineering"""
    
    ti = context['ti']
    results = ti.xcom_pull(task_ids='validate_customer_data')
    
    print(f"Data validated successfully!")
    print(f"Rows: {results['total_rows']}, Checks passed: {results['checks_passed']}")
    
    # Could trigger next DAG here or update a status table
    return "validated"


with DAG(
    'data_validation',
    default_args=default_args,
    description='Validate customer data quality using Great Expectations-style checks',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['validation', 'quality', 'mlops'],
) as dag:
    
    # Wait for ingestion DAG to complete (optional - can also be triggered manually)
    # wait_for_ingestion = ExternalTaskSensor(
    #     task_id='wait_for_ingestion',
    #     external_dag_id='customer_churn_ingestion',
    #     external_task_id='validate_upload',
    #     timeout=3600,
    # )
    
    download_task = PythonOperator(
        task_id='download_latest_data',
        python_callable=download_latest_data,
    )
    
    validate_task = PythonOperator(
        task_id='validate_customer_data',
        python_callable=validate_customer_data,
    )
    
    complete_task = PythonOperator(
        task_id='mark_data_validated',
        python_callable=mark_data_validated,
    )
    
    # Trigger feature engineering DAG after validation passes
    trigger_features = TriggerDagRunOperator(
        task_id='trigger_feature_engineering',
        trigger_dag_id='feature_engineering',
        wait_for_completion=False,
    )
    
    download_task >> validate_task >> complete_task >> trigger_features
