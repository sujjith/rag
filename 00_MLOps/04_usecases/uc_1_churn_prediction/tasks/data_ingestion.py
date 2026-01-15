# uc_1_churn_prediction/tasks/data_ingestion.py
"""Data ingestion tasks: PostgreSQL extraction and MinIO upload."""

from prefect import task
import pandas as pd
from sqlalchemy import create_engine
import boto3
from datetime import datetime
from common.config import get_config


@task(name="Extract Customer Data", retries=2, retry_delay_seconds=60)
def extract_from_postgres() -> str:
    """
    Step 1: Extract customer data from PostgreSQL.

    Returns:
        str: Local path to the extracted parquet file
    """
    cfg = get_config()

    db_url = (
        f"postgresql://{cfg['postgres']['user']}:{cfg['postgres']['password']}@"
        f"{cfg['postgres']['host']}:{cfg['postgres']['port']}/{cfg['postgres']['database']}"
    )
    engine = create_engine(db_url)

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

    output_path = "/tmp/customers.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Extracted {len(df)} customer records to {output_path}")
    return output_path


@task(name="Upload to MinIO", retries=2)
def upload_to_minio(local_path: str) -> str:
    """
    Step 2: Upload extracted data to MinIO object storage.

    Args:
        local_path: Path to local parquet file

    Returns:
        str: S3 key of uploaded file
    """
    cfg = get_config()

    s3 = boto3.client(
        's3',
        endpoint_url=cfg['minio']['endpoint'],
        aws_access_key_id=cfg['minio']['access_key'],
        aws_secret_access_key=cfg['minio']['secret_key']
    )

    # Create bucket if not exists
    try:
        s3.head_bucket(Bucket=cfg['minio']['bucket'])
    except Exception:
        s3.create_bucket(Bucket=cfg['minio']['bucket'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f"raw/customers_{timestamp}.parquet"

    s3.upload_file(local_path, cfg['minio']['bucket'], s3_key)

    s3_uri = f"s3://{cfg['minio']['bucket']}/{s3_key}"
    print(f"Uploaded to {s3_uri}")
    return s3_key
