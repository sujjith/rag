"""
Phase 4.2: Test model serving endpoint

This script demonstrates:
- Making predictions via REST API
- Different input formats
- Batch predictions
- Error handling

Prerequisites:
1. Run 01_prepare_model.py
2. Start model server: ./scripts/serve-model.sh iris-serving-model Production

Run: python 02_test_serving.py
"""
import requests
import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import time

MODEL_SERVER = "http://localhost:5001"
ENDPOINT = f"{MODEL_SERVER}/invocations"

# Load sample data
iris = load_iris()
feature_names = list(iris.feature_names)
target_names = list(iris.target_names)


def check_server():
    """Check if model server is running."""
    try:
        response = requests.get(f"{MODEL_SERVER}/health", timeout=5)
        return response.ok
    except:
        return False


def predict_inputs(data):
    """Predict using inputs format."""
    payload = {"inputs": data}
    return requests.post(ENDPOINT, json=payload)


def predict_dataframe_split(df):
    """Predict using dataframe_split format."""
    payload = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }
    return requests.post(ENDPOINT, json=payload)


def predict_dataframe_records(df):
    """Predict using dataframe_records format."""
    payload = {"dataframe_records": df.to_dict(orient="records")}
    return requests.post(ENDPOINT, json=payload)


print("=" * 60)
print("MLflow Model Serving Tests")
print("=" * 60)

# Check server
print("\n[0] Checking model server...")
print("-" * 40)

if not check_server():
    print("  ERROR: Model server is not running!")
    print("\n  Start it with:")
    print("    ./scripts/serve-model.sh iris-serving-model Production")
    print("\n  Or:")
    print("    docker run -p 5001:5001 ...")
    exit(1)

print(f"  Server is running at {MODEL_SERVER}")

# Test samples
samples = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.2, 2.9, 4.3, 1.3],  # versicolor
    [7.7, 3.0, 6.1, 2.3],  # virginica
]
sample_df = pd.DataFrame(samples, columns=feature_names)

print("\n[Test Data]")
print("-" * 40)
print(sample_df.to_string(index=False))

# Test 1: inputs format (simplest)
print("\n" + "=" * 60)
print("[Test 1: 'inputs' format]")
print("-" * 40)

response = predict_inputs(samples)
print(f"Status: {response.status_code}")

if response.ok:
    result = response.json()
    predictions = result.get("predictions", result)
    print(f"Predictions: {predictions}")
    print(f"Classes: {[target_names[p] for p in predictions]}")
else:
    print(f"Error: {response.text}")

# Test 2: dataframe_split format
print("\n" + "=" * 60)
print("[Test 2: 'dataframe_split' format]")
print("-" * 40)

response = predict_dataframe_split(sample_df)
print(f"Status: {response.status_code}")

if response.ok:
    result = response.json()
    predictions = result.get("predictions", result)
    print(f"Predictions: {predictions}")
else:
    print(f"Error: {response.text}")

# Test 3: dataframe_records format
print("\n" + "=" * 60)
print("[Test 3: 'dataframe_records' format]")
print("-" * 40)

response = predict_dataframe_records(sample_df)
print(f"Status: {response.status_code}")

if response.ok:
    result = response.json()
    predictions = result.get("predictions", result)
    print(f"Predictions: {predictions}")
else:
    print(f"Error: {response.text}")

# Test 4: Single prediction
print("\n" + "=" * 60)
print("[Test 4: Single prediction]")
print("-" * 40)

single_sample = [[5.1, 3.5, 1.4, 0.2]]
response = predict_inputs(single_sample)

if response.ok:
    result = response.json()
    pred = result.get("predictions", result)[0]
    print(f"Input: {single_sample[0]}")
    print(f"Prediction: {pred} ({target_names[pred]})")

# Test 5: Batch prediction performance
print("\n" + "=" * 60)
print("[Test 5: Batch prediction (100 samples)]")
print("-" * 40)

np.random.seed(42)
batch = np.random.uniform(
    low=[4.0, 2.0, 1.0, 0.1],
    high=[8.0, 4.5, 7.0, 2.5],
    size=(100, 4)
).tolist()

start = time.time()
response = predict_inputs(batch)
elapsed = time.time() - start

if response.ok:
    result = response.json()
    predictions = result.get("predictions", result)

    print(f"Samples: {len(batch)}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {len(batch)/elapsed:.1f} predictions/sec")

    # Distribution
    from collections import Counter
    dist = Counter(predictions)
    print("\nDistribution:")
    for class_id, count in sorted(dist.items()):
        print(f"  {target_names[class_id]}: {count}")

# Test 6: Error handling
print("\n" + "=" * 60)
print("[Test 6: Error handling]")
print("-" * 40)

# Wrong number of features
print("\nWrong number of features:")
response = predict_inputs([[1, 2, 3]])  # Only 3 features
print(f"  Status: {response.status_code}")
if not response.ok:
    print(f"  Error (expected): {response.text[:100]}...")

# Invalid data type
print("\nInvalid data type:")
response = predict_inputs([["a", "b", "c", "d"]])
print(f"  Status: {response.status_code}")
if not response.ok:
    print(f"  Error (expected): {response.text[:100]}...")

# cURL examples
print("\n" + "=" * 60)
print("cURL Examples")
print("=" * 60)

print("\n# Single prediction:")
print(f'''curl -X POST {ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [[5.1, 3.5, 1.4, 0.2]]}}'
''')

print("# Multiple predictions:")
print(f'''curl -X POST {ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}}'
''')

print("# DataFrame format:")
print(f'''curl -X POST {ENDPOINT} \\
  -H "Content-Type: application/json" \\
  -d '{{
    "dataframe_split": {{
      "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
      "data": [[5.1, 3.5, 1.4, 0.2]]
    }}
  }}'
''')

print("=" * 60)
print("Model serving tests complete!")
print("=" * 60)
