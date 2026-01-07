# Phase 04: Kubeflow Complete Setup

## Overview

Complete Kubeflow platform setup including Pipelines, Notebooks, Katib (hyperparameter tuning), Training Operators (TFJob, PyTorchJob), and KServe for model serving.

---

## Kubeflow Feature Matrix

| Component | Description | Status |
|-----------|-------------|--------|
| Kubeflow Pipelines | ML workflow orchestration | Included |
| Kubeflow Notebooks | JupyterHub integration | Included |
| Katib | Hyperparameter tuning | Included |
| TFJob | TensorFlow distributed training | Included |
| PyTorchJob | PyTorch distributed training | Included |
| MPI Operator | MPI-based training | Included |
| KServe | Model serving | Included |
| Metadata | ML metadata tracking | Included |
| Feature Store | Feast integration | Included |
| Central Dashboard | Unified UI | Included |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KUBEFLOW ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     CENTRAL DASHBOARD                                │   │
│   │              (Unified access to all components)                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│   ┌──────────┬──────────┬──────────┼──────────┬──────────┬──────────┐      │
│   │          │          │          │          │          │          │      │
│   ▼          ▼          ▼          ▼          ▼          ▼          ▼      │
│ ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐    │
│ │Pipeline││Notebook││ Katib  ││ TFJob  ││PyTorch ││ KServe ││Metadata│    │
│ │   s    ││   s    ││        ││        ││  Job   ││        ││        │    │
│ └────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘    │
│      │         │         │         │         │         │         │         │
│      └─────────┴─────────┴─────────┼─────────┴─────────┴─────────┘         │
│                                    │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      KUBERNETES CLUSTER                              │   │
│   │   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │   │
│   │   │ Istio  │  │  GPU   │  │ Storage│  │  RBAC  │  │ Network│       │   │
│   │   │        │  │ Nodes  │  │  Class │  │        │  │ Policy │       │   │
│   │   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install Full Kubeflow

### Prerequisites

```bash
# Ensure Kubernetes cluster has sufficient resources
# Minimum: 8 CPUs, 16GB RAM, 100GB storage

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Clone Kubeflow manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.8.0
```

### Install Kubeflow Components

```bash
# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s

# Install Istio
cd manifests/common/istio-1-17/istio-crds/base
kustomize build . | kubectl apply -f -
cd ../istio-namespace/base
kustomize build . | kubectl apply -f -
cd ../istio-install/base
kustomize build . | kubectl apply -f -

# Wait for Istio
kubectl wait --for=condition=ready pod -l app=istiod -n istio-system --timeout=300s

# Install Kubeflow namespace
cd manifests/common/kubeflow-namespace/base
kustomize build . | kubectl apply -f -

# Install Kubeflow Roles
cd ../../kubeflow-roles/base
kustomize build . | kubectl apply -f -

# Install Kubeflow Istio Resources
cd ../../istio-1-17/kubeflow-istio-resources/base
kustomize build . | kubectl apply -f -

# Install Kubeflow Pipelines
cd ../../../apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user
kustomize build . | kubectl apply -f -

# Install Katib
cd ../../../../katib/upstream/installs/katib-with-kubeflow
kustomize build . | kubectl apply -f -

# Install Central Dashboard
cd ../../../../centraldashboard/upstream/overlays/kserve
kustomize build . | kubectl apply -f -

# Install Notebooks
cd ../../../jupyter-web-app/upstream/overlays/istio
kustomize build . | kubectl apply -f -
cd ../../../../notebook-controller/upstream/overlays/kubeflow
kustomize build . | kubectl apply -f -

# Install Training Operators
cd ../../../../training-operator/upstream/overlays/kubeflow
kustomize build . | kubectl apply -f -

# Install KServe
cd ../../../../kserve/upstream/overlays/kubeflow
kustomize build . | kubectl apply -f -

# Install Profiles + KFAM
cd ../../../../profiles/upstream/overlays/kubeflow
kustomize build . | kubectl apply -f -

# Wait for all pods
kubectl wait --for=condition=ready pod --all -n kubeflow --timeout=600s
```

### Alternative: Single Command Install

```bash
# Install everything at once (takes longer)
cd manifests
while ! kustomize build example | kubectl apply -f -; do
    echo "Retrying..."
    sleep 10
done
```

---

## Step 2: Access Kubeflow Dashboard

```bash
# Port forward to access dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &

# Default credentials
# Email: user@example.com
# Password: 12341234

# Or create new user
cat <<EOF | kubectl apply -f -
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: ml-team
spec:
  owner:
    kind: User
    name: ml-team@example.com
EOF
```

---

## Step 3: Kubeflow Pipelines (KFP v2)

### Create Pipeline Components

Create `kubeflow/pipelines/components.py`:

```python
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    ClassificationMetrics,
    Artifact,
)

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def load_data(
    data_path: str,
    output_dataset: Output[Dataset]
):
    """Load and validate data"""
    import pandas as pd

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Save as parquet
    df.to_parquet(output_dataset.path)
    output_dataset.metadata["rows"] = len(df)
    output_dataset.metadata["columns"] = len(df.columns)


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def preprocess_data(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    target_column: str = "target"
):
    """Preprocess data for training"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import pickle

    df = pd.read_parquet(input_dataset.path)

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Combine
    processed_df = X.copy()
    processed_df[target_column] = y

    processed_df.to_parquet(output_dataset.path)
    output_dataset.metadata["features"] = len(X.columns)


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow", "mlflow", "boto3"]
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    target_column: str = "target",
    n_estimators: int = 100,
    max_depth: int = 10,
    mlflow_tracking_uri: str = "",
    experiment_name: str = "kubeflow-training"
):
    """Train model and log to MLflow"""
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve, auc
    )
    import mlflow
    import mlflow.sklearn

    # Load data
    df = pd.read_parquet(input_dataset.path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Configure MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log to KFP
        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("precision", precision)
        metrics.log_metric("recall", recall)
        metrics.log_metric("f1_score", f1)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        classification_metrics.log_confusion_matrix(
            categories=["Class 0", "Class 1"],
            matrix=cm.tolist()
        )

        # Log ROC curve (for binary classification)
        if len(set(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            classification_metrics.log_roc_curve(fpr.tolist(), tpr.tolist())

        # Log to MLflow
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(model, "model")

        # Save model artifact
        with open(output_model.path, "wb") as f:
            pickle.dump(model, f)

        output_model.metadata["accuracy"] = accuracy
        output_model.metadata["mlflow_run_id"] = mlflow.active_run().info.run_id


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def evaluate_model(
    model: Input[Model],
    test_dataset: Input[Dataset],
    metrics: Output[Metrics],
    target_column: str = "target",
    accuracy_threshold: float = 0.8
) -> bool:
    """Evaluate model against threshold"""
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score

    # Load model
    with open(model.path, "rb") as f:
        clf = pickle.load(f)

    # Load test data
    df = pd.read_parquet(test_dataset.path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Evaluate
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("threshold", accuracy_threshold)
    metrics.log_metric("passed", int(accuracy >= accuracy_threshold))

    return accuracy >= accuracy_threshold


@component(
    base_image="python:3.10",
    packages_to_install=["mlflow", "boto3"]
)
def register_model(
    model: Input[Model],
    model_name: str,
    mlflow_tracking_uri: str,
    stage: str = "Staging"
):
    """Register model to MLflow registry"""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Register model
    run_id = model.metadata.get("mlflow_run_id")
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)

        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage
        )

        print(f"Registered {model_name} v{result.version} to {stage}")
```

### Create Complete Pipeline

Create `kubeflow/pipelines/training_pipeline.py`:

```python
from kfp import dsl
from kfp import compiler
from components import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    register_model
)

@dsl.pipeline(
    name="ML Training Pipeline",
    description="End-to-end ML training pipeline with MLflow integration"
)
def training_pipeline(
    data_path: str = "s3://data/training.csv",
    target_column: str = "target",
    n_estimators: int = 100,
    max_depth: int = 10,
    accuracy_threshold: float = 0.8,
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    model_name: str = "ml-model"
):
    # Load data
    load_task = load_data(data_path=data_path)

    # Preprocess
    preprocess_task = preprocess_data(
        input_dataset=load_task.outputs["output_dataset"],
        target_column=target_column
    )

    # Train model
    train_task = train_model(
        input_dataset=preprocess_task.outputs["output_dataset"],
        target_column=target_column,
        n_estimators=n_estimators,
        max_depth=max_depth,
        mlflow_tracking_uri=mlflow_tracking_uri
    )

    # Evaluate
    evaluate_task = evaluate_model(
        model=train_task.outputs["output_model"],
        test_dataset=preprocess_task.outputs["output_dataset"],
        target_column=target_column,
        accuracy_threshold=accuracy_threshold
    )

    # Conditional registration
    with dsl.Condition(evaluate_task.output == True, name="model-passed"):
        register_model(
            model=train_task.outputs["output_model"],
            model_name=model_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            stage="Staging"
        )


# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.yaml"
    )
```

### Run Pipeline

```python
from kfp import Client

# Connect to Kubeflow
client = Client(host="http://localhost:8080/pipeline")

# Create experiment
experiment = client.create_experiment(
    name="ml-experiments",
    namespace="kubeflow"
)

# Run pipeline
run = client.create_run_from_pipeline_func(
    training_pipeline,
    experiment_name="ml-experiments",
    run_name="training-run-001",
    arguments={
        "data_path": "s3://data/customers.csv",
        "n_estimators": 200,
        "max_depth": 15,
        "accuracy_threshold": 0.85
    }
)

print(f"Run ID: {run.run_id}")
```

---

## Step 4: Katib Hyperparameter Tuning

### Create Katib Experiment

Create `kubeflow/katib/hyperparameter_tuning.yaml`:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: ml-hyperparameter-tuning
  namespace: kubeflow
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: accuracy
    additionalMetricNames:
      - f1_score
      - precision
  algorithm:
    algorithmName: bayesianoptimization
    algorithmSettings:
      - name: "random_state"
        value: "42"
  parallelTrialCount: 3
  maxTrialCount: 30
  maxFailedTrialCount: 5
  resumePolicy: FromVolume
  parameters:
    - name: n_estimators
      parameterType: int
      feasibleSpace:
        min: "50"
        max: "500"
        step: "50"
    - name: max_depth
      parameterType: int
      feasibleSpace:
        min: "3"
        max: "20"
    - name: min_samples_split
      parameterType: int
      feasibleSpace:
        min: "2"
        max: "20"
    - name: min_samples_leaf
      parameterType: int
      feasibleSpace:
        min: "1"
        max: "10"
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.001"
        max: "0.3"
  metricsCollectorSpec:
    collector:
      kind: StdOut
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: n_estimators
        description: Number of trees
        reference: n_estimators
      - name: max_depth
        description: Maximum depth of trees
        reference: max_depth
      - name: min_samples_split
        description: Minimum samples to split
        reference: min_samples_split
      - name: min_samples_leaf
        description: Minimum samples per leaf
        reference: min_samples_leaf
      - name: learning_rate
        description: Learning rate for boosting
        reference: learning_rate
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          metadata:
            annotations:
              sidecar.istio.io/inject: "false"
          spec:
            containers:
              - name: training-container
                image: python:3.10
                command:
                  - "python"
                  - "/app/train.py"
                args:
                  - "--n_estimators=${trialParameters.n_estimators}"
                  - "--max_depth=${trialParameters.max_depth}"
                  - "--min_samples_split=${trialParameters.min_samples_split}"
                  - "--min_samples_leaf=${trialParameters.min_samples_leaf}"
                  - "--learning_rate=${trialParameters.learning_rate}"
                resources:
                  requests:
                    cpu: "1"
                    memory: "2Gi"
                  limits:
                    cpu: "2"
                    memory: "4Gi"
                volumeMounts:
                  - name: training-code
                    mountPath: /app
            volumes:
              - name: training-code
                configMap:
                  name: training-code
            restartPolicy: Never
```

### Create Training Script ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-code
  namespace: kubeflow
data:
  train.py: |
    import argparse
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    def train(args):
        # Generate sample data
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=10,
            random_state=42
        )

        # Create model
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            learning_rate=args.learning_rate,
            random_state=42
        )

        # Cross-validation
        accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')

        # Print metrics (Katib will parse these)
        print(f"accuracy={accuracy_scores.mean():.4f}")
        print(f"f1_score={f1_scores.mean():.4f}")
        print(f"precision={precision_scores.mean():.4f}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_estimators", type=int, default=100)
        parser.add_argument("--max_depth", type=int, default=5)
        parser.add_argument("--min_samples_split", type=int, default=2)
        parser.add_argument("--min_samples_leaf", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        args = parser.parse_args()
        train(args)
```

```bash
# Apply experiment
kubectl apply -f kubeflow/katib/hyperparameter_tuning.yaml

# Monitor experiment
kubectl get experiments -n kubeflow
kubectl get trials -n kubeflow

# Get best parameters
kubectl get experiment ml-hyperparameter-tuning -n kubeflow -o yaml
```

---

## Step 5: Training Operators

### TFJob - TensorFlow Distributed Training

Create `kubeflow/training/tfjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: distributed-tensorflow-training
  namespace: kubeflow
spec:
  runPolicy:
    cleanPodPolicy: None
    backoffLimit: 3
  tfReplicaSpecs:
    Chief:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.14.0-gpu
              command:
                - python
                - /app/train.py
                - --epochs=10
                - --batch_size=32
              resources:
                requests:
                  cpu: "2"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
              volumeMounts:
                - name: training-data
                  mountPath: /data
                - name: model-output
                  mountPath: /models
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: model-output
              persistentVolumeClaim:
                claimName: model-output-pvc
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.14.0-gpu
              command:
                - python
                - /app/train.py
                - --epochs=10
                - --batch_size=32
              resources:
                requests:
                  cpu: "2"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
                limits:
                  nvidia.com/gpu: "1"
              volumeMounts:
                - name: training-data
                  mountPath: /data
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
    PS:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.14.0
              command:
                - python
                - /app/train.py
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
```

### PyTorchJob - PyTorch Distributed Training

Create `kubeflow/training/pytorchjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-pytorch-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
              command:
                - python
                - -m
                - torch.distributed.launch
                - --nproc_per_node=1
                - --nnodes=3
                - --node_rank=0
                - /app/train.py
              env:
                - name: MASTER_ADDR
                  value: "distributed-pytorch-training-master-0"
                - name: MASTER_PORT
                  value: "23456"
              resources:
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
                limits:
                  nvidia.com/gpu: "1"
              volumeMounts:
                - name: training-data
                  mountPath: /data
                - name: model-output
                  mountPath: /models
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: model-output
              persistentVolumeClaim:
                claimName: model-output-pvc
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
              command:
                - python
                - -m
                - torch.distributed.launch
                - --nproc_per_node=1
                - --nnodes=3
                - /app/train.py
              resources:
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
                limits:
                  nvidia.com/gpu: "1"
```

---

## Step 6: Kubeflow Notebooks

### Create Notebook Server

```yaml
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: ml-notebook
  namespace: kubeflow
  labels:
    app: ml-notebook
spec:
  template:
    spec:
      containers:
        - name: ml-notebook
          image: kubeflownotebookswg/jupyter-scipy:v1.8.0
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          volumeMounts:
            - name: workspace
              mountPath: /home/jovyan
            - name: data
              mountPath: /data
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow.mlflow.svc.cluster.local:5000"
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: notebook-workspace
        - name: data
          persistentVolumeClaim:
            claimName: shared-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: notebook-workspace
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Custom Notebook Image

Create `kubeflow/notebooks/Dockerfile`:

```dockerfile
FROM kubeflownotebookswg/jupyter-scipy:v1.8.0

USER root

# Install additional packages
RUN pip install --no-cache-dir \
    mlflow==2.9.0 \
    kfp==2.5.0 \
    feast==0.35.0 \
    great-expectations==0.18.0 \
    evidently==0.4.0 \
    shap==0.44.0 \
    optuna==3.5.0 \
    xgboost==2.0.0 \
    lightgbm==4.2.0 \
    catboost==1.2.0

# Install VS Code extensions
RUN code-server --install-extension ms-python.python \
    && code-server --install-extension ms-toolsai.jupyter

USER jovyan

# Configure git
RUN git config --global user.email "ml-team@example.com" \
    && git config --global user.name "ML Team"
```

---

## Step 7: Pipeline Caching and Artifacts

### Enable Caching

```python
from kfp import dsl

@dsl.component(
    base_image="python:3.10"
)
def expensive_computation(input_data: str) -> str:
    """Component with caching enabled"""
    import time
    time.sleep(60)  # Simulate expensive computation
    return f"processed_{input_data}"

@dsl.pipeline(name="cached-pipeline")
def cached_pipeline(data: str):
    # Enable caching for this task
    task = expensive_computation(input_data=data)
    task.set_caching_options(enable_caching=True)

    # Disable caching
    task2 = expensive_computation(input_data=data)
    task2.set_caching_options(enable_caching=False)
```

### Artifact Management

```python
from kfp import dsl
from kfp.dsl import Artifact, Input, Output

@dsl.component
def create_artifact(
    data: str,
    output_artifact: Output[Artifact]
):
    """Create custom artifact"""
    with open(output_artifact.path, "w") as f:
        f.write(data)

    output_artifact.metadata["custom_field"] = "custom_value"
    output_artifact.metadata["data_hash"] = hash(data)

@dsl.component
def consume_artifact(
    input_artifact: Input[Artifact]
):
    """Consume artifact"""
    with open(input_artifact.path, "r") as f:
        data = f.read()

    print(f"Data: {data}")
    print(f"Metadata: {input_artifact.metadata}")
```

---

## Verification

```bash
#!/bin/bash
# verify_kubeflow.sh

echo "=== Kubeflow Verification ==="

echo -e "\n1. Kubeflow Pods:"
kubectl get pods -n kubeflow

echo -e "\n2. Kubeflow Services:"
kubectl get svc -n kubeflow

echo -e "\n3. Pipelines:"
kubectl get pods -n kubeflow -l app=ml-pipeline

echo -e "\n4. Katib:"
kubectl get experiments -n kubeflow
kubectl get suggestions -n kubeflow

echo -e "\n5. Training Operators:"
kubectl get tfjobs -n kubeflow
kubectl get pytorchjobs -n kubeflow

echo -e "\n6. Notebooks:"
kubectl get notebooks -n kubeflow

echo -e "\n7. KServe:"
kubectl get inferenceservices -n kubeflow

echo -e "\n=== Verification Complete ==="
```

---

## Next Steps

- **Phase 05**: Feature Store & Data Validation
- **Phase 06**: Model Serving (KServe, Seldon)

---

**Status**: Phase 04 Complete
**Features Covered**: All Kubeflow components
