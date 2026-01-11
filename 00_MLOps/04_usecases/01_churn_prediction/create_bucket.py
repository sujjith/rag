import boto3
from botocore.client import Config

s3 = boto3.client('s3',
                    endpoint_url='http://localhost:30900',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin123',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

bucket_name = 'dvc-data'

try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")
except Exception as e:
    print(f"Error creating bucket (might already exist): {e}")
