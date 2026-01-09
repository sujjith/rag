"""
MLflow Model Serving Example

This script demonstrates:
- Making predictions via the MLflow model server REST API
- Different input formats (JSON, CSV)
- Batch predictions
- Error handling

Prerequisites:
- MLflow platform running (./scripts/start.sh)
- Model registered and served (./scripts/serve-model.sh iris-classifier)
"""

import requests
import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# Configuration
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:5001")
INVOCATIONS_ENDPOINT = f"{MODEL_SERVER_URL}/invocations"


def predict_json_split(data):
    """
    Make predictions using split-oriented JSON format.
    Format: {"columns": [...], "data": [[...], [...]]}
    """
    payload = {
        "dataframe_split": {
            "columns": data.columns.tolist(),
            "data": data.values.tolist(),
        }
    }

    response = requests.post(
        INVOCATIONS_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    return response


def predict_json_records(data):
    """
    Make predictions using records-oriented JSON format.
    Format: [{"col1": val1, "col2": val2}, ...]
    """
    payload = {"dataframe_records": data.to_dict(orient="records")}

    response = requests.post(
        INVOCATIONS_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    return response


def predict_inputs(data):
    """
    Make predictions using simple inputs format.
    Format: {"inputs": [[val1, val2, ...], ...]}
    """
    payload = {"inputs": data.values.tolist()}

    response = requests.post(
        INVOCATIONS_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    return response


def check_server_health():
    """Check if the model server is running."""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    print("=" * 60)
    print("MLflow Model Serving Demo")
    print("=" * 60)

    # Check server health
    print(f"\nModel Server URL: {MODEL_SERVER_URL}")
    print("Checking server health...")

    if not check_server_health():
        print("\nERROR: Model server is not running!")
        print("Start it with: ./scripts/serve-model.sh iris-classifier Production")
        return

    print("Server is healthy!")

    # Load test data
    iris = load_iris()
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Create sample data
    sample_data = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.2, 2.9, 4.3, 1.3],  # versicolor
            [7.7, 3.0, 6.1, 2.3],  # virginica
        ],
        columns=feature_names,
    )

    print("\n" + "-" * 60)
    print("Sample Input Data:")
    print("-" * 60)
    print(sample_data.to_string(index=False))

    # Test different prediction methods
    print("\n" + "=" * 60)
    print("Testing Prediction Methods")
    print("=" * 60)

    # Method 1: JSON Split format
    print("\n[Method 1: dataframe_split format]")
    try:
        response = predict_json_split(sample_data)
        if response.status_code == 200:
            predictions = response.json()
            print(f"Status: OK")
            print(f"Predictions: {predictions}")
        else:
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 2: JSON Records format
    print("\n[Method 2: dataframe_records format]")
    try:
        response = predict_json_records(sample_data)
        if response.status_code == 200:
            predictions = response.json()
            print(f"Status: OK")
            print(f"Predictions: {predictions}")
        else:
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 3: Simple inputs format
    print("\n[Method 3: inputs format]")
    try:
        response = predict_inputs(sample_data)
        if response.status_code == 200:
            predictions = response.json()
            print(f"Status: OK")
            print(f"Predictions: {predictions}")

            # Map predictions to class names
            if "predictions" in predictions:
                pred_classes = [target_names[p] for p in predictions["predictions"]]
                print(f"Classes: {pred_classes}")
        else:
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Batch prediction demo
    print("\n" + "=" * 60)
    print("Batch Prediction Demo")
    print("=" * 60)

    # Create larger batch
    np.random.seed(42)
    batch_size = 100
    batch_data = pd.DataFrame(
        np.random.uniform(
            low=[4.0, 2.0, 1.0, 0.1],
            high=[8.0, 4.5, 7.0, 2.5],
            size=(batch_size, 4),
        ),
        columns=feature_names,
    )

    print(f"\nPredicting {batch_size} samples...")

    try:
        import time

        start_time = time.time()
        response = predict_inputs(batch_data)
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            predictions = response.json()
            pred_array = predictions.get("predictions", [])

            print(f"Status: OK")
            print(f"Time: {elapsed_time:.3f} seconds")
            print(f"Throughput: {batch_size / elapsed_time:.1f} predictions/sec")

            # Count predictions by class
            from collections import Counter

            class_counts = Counter(pred_array)
            print(f"\nPrediction distribution:")
            for class_id, count in sorted(class_counts.items()):
                print(f"  {target_names[class_id]}: {count}")
        else:
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # cURL examples
    print("\n" + "=" * 60)
    print("cURL Examples")
    print("=" * 60)

    print("\n# Single prediction:")
    print(f'''curl -X POST {INVOCATIONS_ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [[5.1, 3.5, 1.4, 0.2]]}}\'''')

    print("\n# Batch prediction:")
    print(f'''curl -X POST {INVOCATIONS_ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}}\'''')

    print("\n" + "=" * 60)
    print("Model Serving Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
