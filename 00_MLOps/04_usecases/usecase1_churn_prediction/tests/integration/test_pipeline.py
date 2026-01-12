"""
Integration tests for end-to-end data pipeline flow.
Tests the full ETL: PostgreSQL -> Parquet -> MinIO
"""

import pytest
import pandas as pd
import boto3
from sqlalchemy import create_engine
from datetime import datetime


class TestDataPipeline:
    """Test end-to-end data pipeline flow."""
    
    @pytest.fixture
    def db_engine(self, postgres_config):
        """Create database engine."""
        cfg = postgres_config
        db_url = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        return create_engine(db_url)
    
    @pytest.fixture
    def s3_client(self, minio_config):
        """Create S3 client for MinIO."""
        return boto3.client(
            's3',
            endpoint_url=minio_config['endpoint_url'],
            aws_access_key_id=minio_config['access_key'],
            aws_secret_access_key=minio_config['secret_key']
        )
    
    @pytest.fixture
    def bucket_name(self, minio_config):
        """Get bucket name from config."""
        return minio_config['bucket']
    
    def test_extract_from_postgres(self, db_engine):
        """Test extracting data from PostgreSQL."""
        df = pd.read_sql("SELECT * FROM customers", db_engine)
        assert len(df) > 0, "No data extracted"
        assert "customer_id" in df.columns
        print(f"✓ Extracted {len(df)} records from PostgreSQL")
    
    def test_transform_to_parquet(self, db_engine, tmp_path):
        """Test transforming data to Parquet format."""
        # Extract
        df = pd.read_sql("SELECT * FROM customers", db_engine)
        
        # Transform: Save to Parquet
        parquet_file = tmp_path / "customers.parquet"
        df.to_parquet(parquet_file, index=False)
        
        # Verify
        assert parquet_file.exists()
        df_read = pd.read_parquet(parquet_file)
        assert len(df_read) == len(df)
        print(f"✓ Transformed to Parquet: {parquet_file}")
    
    def test_load_to_minio(self, db_engine, s3_client, bucket_name, tmp_path):
        """Test loading Parquet file to MinIO."""
        # Extract
        df = pd.read_sql("SELECT * FROM customers", db_engine)
        
        # Transform
        parquet_file = tmp_path / "customers.parquet"
        df.to_parquet(parquet_file, index=False)
        
        # Load
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"raw/customers_{timestamp}.parquet"
        s3_client.upload_file(str(parquet_file), bucket_name, s3_key)
        
        # Verify
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        assert response['ContentLength'] > 0
        print(f"✓ Loaded to s3://{bucket_name}/{s3_key} ({response['ContentLength']} bytes)")
    
    def test_full_etl_pipeline(self, db_engine, s3_client, bucket_name, tmp_path):
        """Test complete ETL pipeline: PostgreSQL -> Parquet -> MinIO."""
        print("\n--- Starting Full ETL Pipeline Test ---")
        
        # EXTRACT: Get data from PostgreSQL
        df = pd.read_sql("SELECT * FROM customers", db_engine)
        assert len(df) > 0, "No data extracted"
        print(f"✓ EXTRACT: {len(df)} records from PostgreSQL")
        
        # TRANSFORM: Save to Parquet format
        parquet_file = tmp_path / "customers.parquet"
        df.to_parquet(parquet_file, index=False)
        assert parquet_file.exists()
        print(f"✓ TRANSFORM: Saved to Parquet ({parquet_file.stat().st_size} bytes)")
        
        # LOAD: Upload to MinIO
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"raw/customers_{timestamp}.parquet"
        s3_client.upload_file(str(parquet_file), bucket_name, s3_key)
        
        # Verify upload
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        assert response['ContentLength'] > 0
        print(f"✓ LOAD: Uploaded to s3://{bucket_name}/{s3_key}")
        
        print("\n✅ Full ETL Pipeline Test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
