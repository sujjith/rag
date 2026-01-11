
import mlflow

mlflow.set_tracking_uri("http://localhost:30502")
experiment_name = "churn_prediction_v1"

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(experiment_name)

if not exp:
    print(f"Experiment '{experiment_name}' not found.")
    exit(1)

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    print("No runs found.")
    exit(1)

latest_run = runs[0]
artifact_uri = latest_run.info.artifact_uri
model_uri = f"{artifact_uri}/model"

print(f"Latest Run ID: {latest_run.info.run_id}")
print(f"Model URI: {model_uri}")
