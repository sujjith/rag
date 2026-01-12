"""
Integration tests for MinIO/S3 connectivity and operations.
"""

import pytest
import boto3
from datetime import datetime


class TestMinioConnection:
    """Test MinIO/S3 connectivity and operations."""
    
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
    
    def test_connection(self, s3_client):
        """Test basic MinIO connection."""
        response = s3_client.list_buckets()
        assert "Buckets" in response
        print(f"✓ MinIO connection successful, found {len(response['Buckets'])} buckets")
    
    def test_bucket_exists(self, s3_client, bucket_name):
        """Test that dvc-storage bucket exists."""
        response = s3_client.list_buckets()
        bucket_names = [b['Name'] for b in response['Buckets']]
        assert bucket_name in bucket_names, f"Bucket {bucket_name} not found"
        print(f"✓ Bucket '{bucket_name}' exists")
    
    def test_list_buckets(self, s3_client):
        """Test listing all available buckets."""
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        print(f"✓ Available buckets: {buckets}")
        assert len(buckets) > 0
    
    def test_upload_file(self, s3_client, bucket_name, tmp_path):
        """Test uploading a file to MinIO."""
        # Create test file
        test_file = tmp_path / "test_upload.txt"
        test_file.write_text("test content")
        
        # Upload
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"test/test_upload_{timestamp}.txt"
        s3_client.upload_file(str(test_file), bucket_name, s3_key)
        
        # Verify
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        assert response['ContentLength'] > 0
        print(f"✓ File uploaded successfully: {s3_key}")
        
        # Cleanup
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print("✓ Cleanup successful")
    
    def test_download_file(self, s3_client, bucket_name, tmp_path):
        """Test downloading a file from MinIO."""
        # Create and upload test file
        test_content = "download test content"
        test_file = tmp_path / "test_download.txt"
        test_file.write_text(test_content)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"test/test_download_{timestamp}.txt"
        s3_client.upload_file(str(test_file), bucket_name, s3_key)
        
        # Download
        download_path = tmp_path / "downloaded.txt"
        s3_client.download_file(bucket_name, s3_key, str(download_path))
        
        # Verify
        assert download_path.exists()
        assert download_path.read_text() == test_content
        print("✓ File downloaded and verified successfully")
        
        # Cleanup
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
