
from kfp.client import Client
import time
import sys
import mlflow
import pandas as pd

# KFP Client
client = Client(host='http://localhost:3000')

print("Checking KFP runs...")
# Try to finding the run
try:
    response = client.list_runs(page_size=1, sort_by='created_at desc')
    runs = response.runs
    if not runs:
        print("No KFP runs found.")
        sys.exit(1)
        
    latest_run = runs[0]
    # Inspect attributes
    run_id = getattr(latest_run, 'run_id', getattr(latest_run, 'id', 'unknown'))
    state = getattr(latest_run, 'state', getattr(latest_run, 'status', 'unknown'))
    
    print(f"Latest Run ID: {run_id}, State: {state}")
except Exception as e:
    print(f"Error checking KFP runs: {e}")

# Poll MLflow
print("Polling MLflow for experiment 'churn_prediction_v1'...")
mlflow.set_tracking_uri("http://localhost:30502")

max_retries = 10
for i in range(max_retries):
    try:
        exp = mlflow.get_experiment_by_name("churn_prediction_v1")
        if exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            if not runs.empty:
                uri = runs.iloc[0].artifact_uri
                run_id = runs.iloc[0].run_id
                print(f"✅ Found MLflow run!")
                print(f"Run ID: {run_id}")
                print(f"Model URI: {uri}/model")
                # Save to file for next step
                with open("model_uri.txt", "w") as f:
                    f.write(f"{uri}/model")
                sys.exit(0)
            else:
                print(f"Experiment found, but no runs yet. (Attempt {i+1}/{max_retries})")
        else:
            print(f"Experiment not found yet. (Attempt {i+1}/{max_retries})")
    except Exception as e:
        print(f"MLflow check error: {e}")
    
    time.sleep(10)

print("❌ Timed out waiting for MLflow run.")
sys.exit(1)
