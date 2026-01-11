import boto3
from botocore.client import Config
import os

s3 = boto3.client('s3',
                    endpoint_url='http://localhost:30900',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin123',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

bucket_name = 'mlpipeline'

# Create bucket
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created.")
except Exception as e:
    print(f"Bucket '{bucket_name}' might exist: {e}")

# Upload file
file_path = 'data/raw/customers.csv'
object_name = 'customers.csv'
try:
    s3.upload_file(file_path, bucket_name, object_name)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{object_name}")
except Exception as e:
    print(f"Error uploading: {e}")
