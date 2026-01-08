"""
Phase 2.4: Submit Pipeline via Python Client

Demonstrates:
- Connecting to Kubeflow Pipelines
- Uploading pipelines programmatically
- Creating and monitoring runs

Prerequisites:
- Kubeflow running with port-forward active
- Pipeline YAML files compiled

Run:
  python 04_submit_pipeline.py
"""
import kfp
from kfp import compiler
import time
import sys


def connect_to_kubeflow(host: str = "http://localhost:8080"):
    """Connect to Kubeflow Pipelines."""
    try:
        # For accessing through Istio ingress
        client = kfp.Client(
            host=f"{host}/pipeline",
            # If authentication is needed:
            # cookies="authservice_session=<session_cookie>"
        )
        print(f"Connected to Kubeflow at {host}")
        return client
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure port-forward is running: ./scripts/04_port_forward.sh")
        print("2. Try accessing http://localhost:8080 in browser")
        return None


def list_pipelines(client):
    """List existing pipelines."""
    print("\nExisting Pipelines:")
    print("-" * 40)
    pipelines = client.list_pipelines(page_size=10)
    if pipelines.pipelines:
        for p in pipelines.pipelines:
            print(f"  - {p.display_name} (ID: {p.pipeline_id[:8]}...)")
    else:
        print("  No pipelines found")
    return pipelines


def upload_pipeline(client, pipeline_file: str, pipeline_name: str):
    """Upload a pipeline."""
    print(f"\nUploading pipeline: {pipeline_name}")
    try:
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_file,
            pipeline_name=pipeline_name,
            description=f"Uploaded from {pipeline_file}"
        )
        print(f"  Pipeline ID: {pipeline.pipeline_id}")
        return pipeline
    except Exception as e:
        print(f"  Upload failed: {e}")
        # Try to get existing pipeline
        pipelines = client.list_pipelines()
        for p in pipelines.pipelines or []:
            if p.display_name == pipeline_name:
                print(f"  Using existing pipeline: {p.pipeline_id}")
                return p
        return None


def create_run(client, pipeline_id: str, run_name: str, params: dict = None):
    """Create a pipeline run."""
    print(f"\nCreating run: {run_name}")

    # Create experiment if needed
    experiment_name = "default-experiment"
    try:
        experiment = client.create_experiment(name=experiment_name)
    except:
        experiment = client.get_experiment(experiment_name=experiment_name)

    # Create run
    run = client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        params=params or {}
    )

    print(f"  Run ID: {run.run_id}")
    print(f"  Status: {run.state}")
    return run


def wait_for_run(client, run_id: str, timeout: int = 300):
    """Wait for run to complete."""
    print(f"\nWaiting for run {run_id[:8]}... to complete")

    start_time = time.time()
    while time.time() - start_time < timeout:
        run = client.get_run(run_id=run_id)
        state = run.state

        if state in ["SUCCEEDED", "Succeeded"]:
            print(f"  Run completed successfully!")
            return True
        elif state in ["FAILED", "Failed", "ERROR", "Error"]:
            print(f"  Run failed: {state}")
            return False
        else:
            elapsed = int(time.time() - start_time)
            print(f"  Status: {state} ({elapsed}s elapsed)")
            time.sleep(10)

    print(f"  Timeout after {timeout}s")
    return False


def main():
    print("=" * 60)
    print("Kubeflow Pipeline Submission")
    print("=" * 60)

    # Connect
    client = connect_to_kubeflow()
    if not client:
        sys.exit(1)

    # List existing pipelines
    list_pipelines(client)

    # Example: Upload and run training pipeline
    pipeline_file = "training_pipeline.yaml"
    pipeline_name = "iris-training-v1"
    run_name = f"training-run-{int(time.time())}"

    # Check if pipeline file exists
    import os
    if not os.path.exists(pipeline_file):
        print(f"\nPipeline file not found: {pipeline_file}")
        print("Run: python 02_data_pipeline.py first")
        sys.exit(1)

    # Upload pipeline
    pipeline = upload_pipeline(client, pipeline_file, pipeline_name)
    if not pipeline:
        sys.exit(1)

    # Create run with parameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "test_size": 0.2
    }

    run = create_run(
        client,
        pipeline_id=pipeline.pipeline_id,
        run_name=run_name,
        params=params
    )

    # Wait for completion
    success = wait_for_run(client, run.run_id, timeout=300)

    # Get run details
    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)

    final_run = client.get_run(run_id=run.run_id)
    print(f"  Run ID: {final_run.run_id}")
    print(f"  Name: {run_name}")
    print(f"  Status: {final_run.state}")
    print(f"\n  View in UI: http://localhost:8080/_/pipeline/#/runs/details/{final_run.run_id}")

    print("=" * 60)


if __name__ == "__main__":
    main()
