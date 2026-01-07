# Phase 06: Model Serving (KServe & Seldon Core)

## Overview

Production model serving with KServe and Seldon Core for scalable, enterprise-grade inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODEL SERVING ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ISTIO INGRESS GATEWAY                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                    ┌────────────────┴────────────────┐                      │
│                    │                                 │                      │
│                    ▼                                 ▼                      │
│   ┌────────────────────────────┐    ┌────────────────────────────┐        │
│   │         KSERVE             │    │      SELDON CORE           │        │
│   │  ┌──────────────────────┐  │    │  ┌──────────────────────┐  │        │
│   │  │   Predictor          │  │    │  │   Predictor          │  │        │
│   │  │   - SKLearn          │  │    │  │   - Triton           │  │        │
│   │  │   - XGBoost          │  │    │  │   - TensorFlow       │  │        │
│   │  │   - TensorFlow       │  │    │  │   - PyTorch          │  │        │
│   │  │   - PyTorch          │  │    │  │   - Custom           │  │        │
│   │  └──────────────────────┘  │    │  └──────────────────────┘  │        │
│   │  ┌──────────────────────┐  │    │  ┌──────────────────────┐  │        │
│   │  │   Transformer        │  │    │  │   Router             │  │        │
│   │  │   - Pre-processing   │  │    │  │   - A/B Testing      │  │        │
│   │  │   - Post-processing  │  │    │  │   - Canary           │  │        │
│   │  └──────────────────────┘  │    │  │   - Shadow           │  │        │
│   │  ┌──────────────────────┐  │    │  └──────────────────────┘  │        │
│   │  │   Explainer          │  │    │  ┌──────────────────────┐  │        │
│   │  │   - SHAP             │  │    │  │   Explainer          │  │        │
│   │  │   - Anchors          │  │    │  │   - Alibi            │  │        │
│   │  └──────────────────────┘  │    │  └──────────────────────┘  │        │
│   └────────────────────────────┘    └────────────────────────────┘        │
│                    │                                 │                      │
│                    └────────────────┬────────────────┘                      │
│                                     │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      MODEL STORAGE                                   │   │
│   │   MinIO / S3 / GCS / Azure Blob / MLflow Registry                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install KServe

### Prerequisites

```bash
# Install Istio (if not already installed)
istioctl install --set profile=default -y

# Install Cert-Manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s

# Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-core.yaml

# Configure Knative with Istio
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/net-istio.yaml

# Configure DNS
kubectl patch configmap/config-domain \
    --namespace knative-serving \
    --type merge \
    --patch '{"data":{"example.com":""}}'
```

### Install KServe

```bash
# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve-runtimes.yaml

# Wait for KServe
kubectl wait --for=condition=ready pod -l control-plane=kserve-controller-manager -n kserve --timeout=300s

# Verify
kubectl get pods -n kserve
```

---

## Step 2: Deploy Models with KServe

### SKLearn Model

Create `kserve/sklearn-model.yaml`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-churn-classifier
  namespace: kserve
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    scaleTarget: 80
    scaleMetric: concurrency
    sklearn:
      storageUri: "s3://models/sklearn/churn-classifier"
      protocolVersion: v2
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "1"
          memory: "2Gi"
```

### XGBoost Model

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: xgboost-fraud-detector
  namespace: kserve
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 10
    xgboost:
      storageUri: "s3://models/xgboost/fraud-detector"
      protocolVersion: v2
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
```

### TensorFlow Model

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: tensorflow-recommender
  namespace: kserve
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 10
    tensorflow:
      storageUri: "s3://models/tensorflow/recommender"
      runtimeVersion: "2.14.0"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

### PyTorch Model

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pytorch-image-classifier
  namespace: kserve
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    pytorch:
      storageUri: "s3://models/pytorch/image-classifier"
      protocolVersion: v2
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

### MLflow Model

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: mlflow-churn-model
  namespace: kserve
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    model:
      modelFormat:
        name: mlflow
      storageUri: "s3://mlflow/1/abc123/artifacts/model"
      protocolVersion: v2
```

---

## Step 3: Advanced KServe Features

### Transformer (Pre/Post Processing)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-with-transformer
  namespace: kserve
spec:
  predictor:
    sklearn:
      storageUri: "s3://models/sklearn/churn"
  transformer:
    containers:
    - name: transformer
      image: my-registry/churn-transformer:latest
      env:
      - name: FEAST_SERVER
        value: "http://feast-feature-server.feast.svc.cluster.local:6566"
      resources:
        requests:
          cpu: "500m"
          memory: "512Mi"
```

Create transformer code `kserve/transformer/transformer.py`:

```python
import kserve
from typing import Dict, List
import numpy as np
from feast import FeatureStore

class ChurnTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.feast_store = FeatureStore(repo_path="/feast-repo")
        self.ready = False

    def load(self):
        self.ready = True

    def preprocess(self, inputs: Dict, headers: Dict = None) -> Dict:
        """Enrich inputs with Feast features"""
        customer_ids = inputs.get("instances", [])

        # Get features from Feast
        features = self.feast_store.get_online_features(
            features=[
                "customer_stats:tenure_months",
                "customer_stats:loyalty_score",
                "transaction_stats:txn_count_7d",
            ],
            entity_rows=[{"customer_id": cid} for cid in customer_ids],
        ).to_dict()

        # Format for predictor
        enriched_inputs = {
            "instances": [
                [
                    features["tenure_months"][i],
                    features["loyalty_score"][i],
                    features["txn_count_7d"][i],
                ]
                for i in range(len(customer_ids))
            ]
        }

        return enriched_inputs

    def postprocess(self, outputs: Dict, headers: Dict = None) -> Dict:
        """Add interpretation to predictions"""
        predictions = outputs.get("predictions", [])

        return {
            "predictions": predictions,
            "interpretations": [
                "High churn risk" if p > 0.7 else
                "Medium churn risk" if p > 0.4 else
                "Low churn risk"
                for p in predictions
            ]
        }

if __name__ == "__main__":
    transformer = ChurnTransformer(
        name="churn-transformer",
        predictor_host="sklearn-churn-classifier-predictor"
    )
    kserve.ModelServer().start([transformer])
```

### Explainer (Model Interpretability)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-with-explainer
  namespace: kserve
spec:
  predictor:
    sklearn:
      storageUri: "s3://models/sklearn/churn"
  explainer:
    alibi:
      type: AnchorTabular
      storageUri: "s3://models/explainers/churn-anchors"
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
```

### Canary Deployment

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-canary
  namespace: kserve
spec:
  predictor:
    canaryTrafficPercent: 20
    minReplicas: 2
    sklearn:
      storageUri: "s3://models/sklearn/churn-v2"
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
```

### A/B Testing with Traffic Split

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-ab-test
  namespace: kserve
  annotations:
    serving.kserve.io/enable-tag-routing: "true"
spec:
  predictor:
    sklearn:
      storageUri: "s3://models/sklearn/churn-v1"
---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-ab-test
  namespace: kserve
  annotations:
    serving.kserve.io/canaryTrafficPercent: "50"
spec:
  predictor:
    sklearn:
      storageUri: "s3://models/sklearn/churn-v2"
```

---

## Step 4: Install Seldon Core

```bash
# Add Seldon Helm repo
helm repo add seldon https://storage.googleapis.com/seldon-charts
helm repo update

# Create namespace
kubectl create namespace seldon-system

# Install Seldon Core
helm install seldon-core seldon/seldon-core-operator \
    --namespace seldon-system \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --set ambassador.enabled=false

# Wait for Seldon
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=seldon-core-operator -n seldon-system --timeout=300s
```

---

## Step 5: Deploy Models with Seldon Core

### Basic Deployment

Create `seldon/sklearn-deployment.yaml`:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: sklearn-churn
  namespace: seldon
spec:
  name: sklearn-churn
  predictors:
  - name: default
    replicas: 2
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: s3://models/sklearn/churn
      envSecretRefName: seldon-s3-secret
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
```

### Multi-Model Deployment with Router

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: multi-model-router
  namespace: seldon
spec:
  name: multi-model
  predictors:
  - name: default
    replicas: 2
    graph:
      name: router
      implementation: RANDOM_ABTEST
      parameters:
      - name: ratioA
        value: "0.5"
        type: FLOAT
      children:
      - name: model-a
        implementation: SKLEARN_SERVER
        modelUri: s3://models/sklearn/churn-v1
      - name: model-b
        implementation: SKLEARN_SERVER
        modelUri: s3://models/sklearn/churn-v2
```

### Canary Deployment with Seldon

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: canary-deployment
  namespace: seldon
spec:
  name: canary
  predictors:
  - name: default
    replicas: 3
    traffic: 80
    graph:
      name: classifier-v1
      implementation: SKLEARN_SERVER
      modelUri: s3://models/sklearn/churn-v1
  - name: canary
    replicas: 1
    traffic: 20
    graph:
      name: classifier-v2
      implementation: SKLEARN_SERVER
      modelUri: s3://models/sklearn/churn-v2
```

### Shadow Deployment (A/B Testing with Logging)

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: shadow-deployment
  namespace: seldon
spec:
  name: shadow
  predictors:
  - name: default
    replicas: 2
    traffic: 100
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: s3://models/sklearn/churn-v1
  - name: shadow
    replicas: 1
    shadow: true
    graph:
      name: classifier-shadow
      implementation: SKLEARN_SERVER
      modelUri: s3://models/sklearn/churn-v2
```

### Custom Python Model

Create `seldon/custom-model/Model.py`:

```python
import numpy as np
from typing import Dict, List, Union
import pickle
import mlflow

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.ready = False

    def load(self):
        """Load model from MLflow"""
        mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
        self.model = mlflow.pyfunc.load_model("models:/churn-classifier/Production")
        self.ready = True

    def predict(
        self,
        X: Union[np.ndarray, List],
        names: List[str] = None,
        meta: Dict = None
    ) -> Union[np.ndarray, List, Dict]:
        """Make predictions"""
        import pandas as pd

        # Convert to DataFrame
        if names:
            df = pd.DataFrame(X, columns=names)
        else:
            df = pd.DataFrame(X)

        # Predict
        predictions = self.model.predict(df)

        return {
            "predictions": predictions.tolist(),
            "model_version": "production"
        }

    def predict_proba(
        self,
        X: Union[np.ndarray, List],
        names: List[str] = None,
        meta: Dict = None
    ) -> Dict:
        """Return prediction probabilities"""
        import pandas as pd

        df = pd.DataFrame(X, columns=names) if names else pd.DataFrame(X)

        # Get probabilities
        probas = self.model.predict_proba(df)

        return {
            "probabilities": probas.tolist(),
            "classes": ["no_churn", "churn"]
        }
```

Deploy custom model:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: custom-churn-model
  namespace: seldon
spec:
  predictors:
  - name: default
    replicas: 2
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: s3://models/custom/churn
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: my-registry/custom-churn:latest
          env:
          - name: MLFLOW_TRACKING_URI
            value: "http://mlflow.mlflow.svc.cluster.local:5000"
```

---

## Step 6: Model Serving with Triton

### Deploy Triton with KServe

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: triton-ensemble
  namespace: kserve
spec:
  predictor:
    triton:
      storageUri: "s3://models/triton/ensemble"
      runtimeVersion: "23.10-py3"
      resources:
        requests:
          cpu: "4"
          memory: "8Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

### Triton Model Repository Structure

```
models/
├── preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── classifier/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── ensemble/
    ├── 1/
    └── config.pbtxt
```

---

## Step 7: Client SDK Usage

### KServe Client

```python
# kserve_client.py
import requests
import json

KSERVE_URL = "http://sklearn-churn-classifier.kserve.example.com/v2/models/sklearn-churn-classifier/infer"

def predict(instances: list) -> dict:
    """Make prediction using KServe inference protocol v2"""

    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [len(instances), len(instances[0])],
                "datatype": "FP32",
                "data": instances
            }
        ]
    }

    response = requests.post(
        KSERVE_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    return response.json()

# Usage
instances = [
    [24, 65.5, 1572.0, 1],
    [12, 45.0, 540.0, 0]
]
result = predict(instances)
print(f"Predictions: {result}")
```

### Seldon Client

```python
# seldon_client.py
import requests
import json

SELDON_URL = "http://sklearn-churn.seldon.svc.cluster.local/api/v1.0/predictions"

def predict(data: list, names: list = None) -> dict:
    """Make prediction using Seldon"""

    payload = {
        "data": {
            "ndarray": data
        }
    }

    if names:
        payload["data"]["names"] = names

    response = requests.post(
        SELDON_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    return response.json()

def explain(data: list) -> dict:
    """Get model explanation"""

    payload = {"data": {"ndarray": data}}

    response = requests.post(
        SELDON_URL.replace("/predictions", "/explain"),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    return response.json()

# Usage
data = [[24, 65.5, 1572.0, 1]]
names = ["tenure", "monthly_charges", "total_charges", "contract"]

prediction = predict(data, names)
explanation = explain(data)

print(f"Prediction: {prediction}")
print(f"Explanation: {explanation}")
```

---

## Step 8: Autoscaling Configuration

### KServe Autoscaling

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: autoscaled-model
  namespace: kserve
  annotations:
    autoscaling.knative.dev/target: "100"
    autoscaling.knative.dev/target-utilization-percentage: "70"
    autoscaling.knative.dev/window: "60s"
    autoscaling.knative.dev/panic-window-percentage: "10"
    autoscaling.knative.dev/panic-threshold-percentage: "200"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 100
    scaleMetric: concurrency
    sklearn:
      storageUri: "s3://models/sklearn/churn"
```

### Seldon HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: seldon-hpa
  namespace: seldon
spec:
  scaleTargetRef:
    apiVersion: machinelearning.seldon.io/v1
    kind: SeldonDeployment
    name: sklearn-churn
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Verification

```bash
#!/bin/bash
# verify_model_serving.sh

echo "=== Model Serving Verification ==="

echo -e "\n1. KServe InferenceServices:"
kubectl get inferenceservices -n kserve

echo -e "\n2. Seldon Deployments:"
kubectl get seldondeployments -n seldon

echo -e "\n3. KServe Pods:"
kubectl get pods -n kserve -l serving.kserve.io/inferenceservice

echo -e "\n4. Seldon Pods:"
kubectl get pods -n seldon

echo -e "\n5. Test KServe Endpoint:"
KSERVE_URL=$(kubectl get inferenceservice sklearn-churn-classifier -n kserve -o jsonpath='{.status.url}')
curl -X POST "$KSERVE_URL/v2/models/sklearn-churn-classifier/infer" \
    -H "Content-Type: application/json" \
    -d '{"inputs":[{"name":"input-0","shape":[1,4],"datatype":"FP32","data":[[24,65.5,1572.0,1]]}]}'

echo -e "\n=== Verification Complete ==="
```

---

**Status**: Phase 06 Complete
**Features Covered**: KServe, Seldon Core, Canary, A/B Testing, Autoscaling
