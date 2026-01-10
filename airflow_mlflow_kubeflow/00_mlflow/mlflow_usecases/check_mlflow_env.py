#!/usr/bin/env python3
"""
Helper script to verify MLflow connection and environment setup.

Run this before running any use case scripts to ensure everything is configured correctly.
"""
import os
import sys
from pathlib import Path

# Load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print("📄 Loading environment variables from .env file...")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
                print(f"   ✓ Set {key}")
else:
    print("⚠️  No .env file found. Using default configuration.")

# Check MLflow connection
try:
    import mlflow
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"\n🔗 MLflow Tracking URI: {tracking_uri}")
    
    # Try to connect
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    
    print(f"✅ Successfully connected to MLflow!")
    print(f"📊 Found {len(experiments)} experiment(s)")
    
    if experiments:
        print("\nExisting experiments:")
        for exp in experiments[:5]:  # Show first 5
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
    
    print("\n✨ Environment is ready! You can now run the use case scripts.")
    sys.exit(0)
    
except ImportError:
    print("\n❌ MLflow is not installed!")
    print("Run: uv sync")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Failed to connect to MLflow: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Check if MLflow server is running")
    print(f"2. Verify the MLFLOW_TRACKING_URI in .env file")
    print(f"3. Ensure port forwarding is active:")
    print(f"   kubectl port-forward -n mlflow svc/mlflow 5000:80 --address 0.0.0.0")
    sys.exit(1)
