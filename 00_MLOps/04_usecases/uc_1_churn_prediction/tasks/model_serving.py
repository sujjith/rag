# uc_1_churn_prediction/tasks/model_serving.py
"""Model serving tasks using KServe."""

from prefect import task
from kubernetes import client, config
from common.config import get_config
import time


@task(name="Deploy to KServe")
def deploy_to_kserve(model_uri: str, model_name: str = "churn-predictor") -> str:
    """
    Step 7: Deploy model to KServe.

    Args:
        model_uri: S3 URI to model artifact
        model_name: Name for the inference service

    Returns:
        str: Endpoint URL
    """
    cfg = get_config()
    namespace = cfg['kserve']['namespace']

    config.load_kube_config()
    api = client.CustomObjectsApi()

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "storageUri": model_uri
                }
            }
        }
    }

    # Check if exists
    try:
        api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name
        )
        # Update existing
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service
        )
        print(f"Updated InferenceService: {model_name}")
    except client.exceptions.ApiException as e:
        if e.status == 404:
            # Create new
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
            print(f"Created InferenceService: {model_name}")
        else:
            raise

    endpoint = f"{model_name}.{namespace}.svc.cluster.local"
    print(f"Model deployed at: {endpoint}")
    return endpoint


@task(name="Wait for KServe Ready")
def wait_for_kserve_ready(
    model_name: str = "churn-predictor",
    timeout: int = 300
) -> bool:
    """
    Wait for KServe InferenceService to be ready.

    Args:
        model_name: Name of inference service
        timeout: Timeout in seconds

    Returns:
        bool: True if ready
    """
    cfg = get_config()
    namespace = cfg['kserve']['namespace']

    config.load_kube_config()
    api = client.CustomObjectsApi()

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            isvc = api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name
            )

            status = isvc.get("status", {})
            conditions = status.get("conditions", [])

            for condition in conditions:
                if condition.get("type") == "Ready":
                    if condition.get("status") == "True":
                        print(f"InferenceService {model_name} is ready!")
                        return True

            print(f"Waiting for {model_name} to be ready...")
            time.sleep(10)

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(10)

    raise TimeoutError(f"InferenceService {model_name} not ready after {timeout}s")
