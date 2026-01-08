# Kubeflow Learning Guide - Phased Implementation

## Overview

This guide breaks down Kubeflow into 6 learning phases.

| Phase | Topic | Outcome |
|-------|-------|---------|
| 1 | Setup & Basics | Kubeflow running on Minikube |
| 2 | Pipelines | Build and run ML pipelines |
| 3 | Notebooks | Interactive development environment |
| 4 | Katib | Automated hyperparameter tuning |
| 5 | Training Operators | Distributed training jobs |
| 6 | Integration | Connect with MLflow |

---

# Phase 1: Setup & Basic Concepts

## Objectives
- Install Minikube and prerequisites
- Deploy Kubeflow
- Navigate the Dashboard
- Understand namespaces and profiles

## 1.1 Prerequisites

```bash
# Run the prerequisites script
./scripts/01_install_prerequisites.sh

# Or install manually:

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube-linux-amd64 && sudo mv minikube-linux-amd64 /usr/local/bin/minikube

# kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

## 1.2 Setup Minikube

```bash
# Start Minikube with adequate resources
minikube start \
  --cpus=4 \
  --memory=8192 \
  --disk-size=40g \
  --driver=docker \
  --kubernetes-version=v1.28.0

# Enable addons
minikube addons enable default-storageclass
minikube addons enable storage-provisioner
minikube addons enable metrics-server

# Verify
kubectl cluster-info
kubectl get nodes
```

## 1.3 Install Kubeflow

```bash
# Clone Kubeflow manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Checkout stable version
git checkout v1.11.0

# Install Kubeflow (takes 10-15 minutes)
while ! kustomize build example | kubectl apply -f -; do
  echo "Retrying..."
  sleep 10
done

# Wait for pods to be ready
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s
```

## 1.4 Access Dashboard

```bash
# Port forward the Istio ingress gateway
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &

# Open browser
echo "Dashboard: http://localhost:8080"
echo "Credentials: user@example.com / 12341234"
```

## 1.5 Understanding Kubeflow Concepts

### Profiles (Namespaces)
- Each user gets a **profile** (Kubernetes namespace)
- Isolates resources, experiments, and pipelines
- Default profile: `kubeflow-user-example-com`

### Components
- **Pipelines**: Workflow orchestration using Argo
- **Notebooks**: Jupyter servers for development
- **Katib**: Hyperparameter optimization
- **Training Operators**: Distributed training (TF, PyTorch)

### Check Component Status
```bash
# List all pods
kubectl get pods -n kubeflow

# Check specific components
kubectl get pods -n kubeflow -l app=ml-pipeline
kubectl get pods -n kubeflow -l app=jupyter-web-app
kubectl get pods -n kubeflow -l katib.kubeflow.org/component=controller
```

## Phase 1 Checklist

- [ ] Minikube running with 4+ CPUs, 8+ GB RAM
- [ ] Kubeflow installed and all pods Running
- [ ] Dashboard accessible at localhost:8080
- [ ] Can log in with default credentials

---

# Phase 2: Kubeflow Pipelines

## Objectives
- Understand pipeline concepts
- Write pipelines using KFP SDK
- Compile and upload pipelines
- Run and monitor pipelines

## 2.1 Install KFP SDK

```bash
pip install kfp==2.10.0
```

## 2.2 Pipeline Concepts

```
Pipeline
├── Components (reusable units)
│   ├── Input/Output definitions
│   ├── Container image
│   └── Command/Arguments
├── Tasks (component instances)
│   ├── Connected via inputs/outputs
│   └── Can have dependencies
└── Artifacts (data passed between tasks)
    ├── Datasets
    ├── Models
    └── Metrics
```

## 2.3 Your First Pipeline

Create `phase2/01_hello_pipeline.py`:

```python
"""
Phase 2.1: Hello World Pipeline

A simple pipeline that demonstrates basic KFP concepts.
"""
from kfp import dsl
from kfp import compiler

# Define a simple component using @dsl.component decorator
@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    """A component that says hello."""
    message = f"Hello, {name}!"
    print(message)
    return message

@dsl.component(base_image="python:3.11-slim")
def print_message(message: str):
    """A component that prints a message."""
    print(f"Received: {message}")

# Define the pipeline
@dsl.pipeline(
    name="hello-world-pipeline",
    description="A simple hello world pipeline"
)
def hello_pipeline(name: str = "Kubeflow"):
    # Create tasks from components
    hello_task = say_hello(name=name)
    print_task = print_message(message=hello_task.output)

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path="hello_pipeline.yaml"
    )
    print("Pipeline compiled to: hello_pipeline.yaml")
    print("\nUpload this file to Kubeflow Pipelines UI")
```

## 2.4 Data Processing Pipeline

Create `phase2/02_data_pipeline.py`:

```python
"""
Phase 2.2: Data Processing Pipeline

Pipeline with multiple steps and data passing.
"""
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Dataset, Model, Metrics

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def load_data(dataset: Output[Dataset]):
    """Load and save the iris dataset."""
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    df.to_csv(dataset.path, index=False)
    print(f"Data saved to {dataset.path}")
    print(f"Shape: {df.shape}")

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def preprocess_data(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    test_size: float = 0.2
):
    """Split data into train and test sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_dataset.path)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42
    )

    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_dataset: Input[Dataset],
    model: Output[Model],
    n_estimators: int = 100
):
    """Train a RandomForest model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(train_dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, model.path)
    print(f"Model saved to {model.path}")

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    model: Input[Model],
    test_dataset: Input[Dataset],
    metrics: Output[Metrics]
):
    """Evaluate the model."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    clf = joblib.load(model.path)

    df = pd.read_csv(test_dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = clf.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average="weighted")

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

@dsl.pipeline(
    name="iris-training-pipeline",
    description="Train and evaluate a model on Iris dataset"
)
def training_pipeline(
    n_estimators: int = 100,
    test_size: float = 0.2
):
    # Load data
    load_task = load_data()

    # Preprocess
    preprocess_task = preprocess_data(
        input_dataset=load_task.outputs["dataset"],
        test_size=test_size
    )

    # Train
    train_task = train_model(
        train_dataset=preprocess_task.outputs["train_dataset"],
        n_estimators=n_estimators
    )

    # Evaluate
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        test_dataset=preprocess_task.outputs["test_dataset"]
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.yaml"
    )
    print("Pipeline compiled to: training_pipeline.yaml")
```

## 2.5 Running Pipelines

### Option 1: UI Upload
1. Open Kubeflow Dashboard
2. Go to Pipelines → Upload Pipeline
3. Select compiled YAML file
4. Create Run

### Option 2: Python Client

```python
"""Submit pipeline via Python client."""
import kfp

# Connect to Kubeflow Pipelines
client = kfp.Client(host="http://localhost:8080/pipeline")

# Upload pipeline
pipeline = client.upload_pipeline(
    pipeline_package_path="training_pipeline.yaml",
    pipeline_name="iris-training"
)

# Create a run
run = client.create_run_from_pipeline_package(
    pipeline_file="training_pipeline.yaml",
    arguments={"n_estimators": 100, "test_size": 0.2},
    run_name="training-run-1"
)

print(f"Run ID: {run.run_id}")
```

## Phase 2 Checklist

- [ ] KFP SDK installed
- [ ] Created hello world pipeline
- [ ] Created data processing pipeline
- [ ] Compiled pipelines to YAML
- [ ] Ran pipeline via UI or client
- [ ] Viewed run results and artifacts

---

# Phase 3: Kubeflow Notebooks

## Objectives
- Create Jupyter notebook servers
- Use custom container images
- Access data and models from notebooks

## 3.1 Create Notebook Server via UI

1. Open Kubeflow Dashboard
2. Go to Notebooks → New Notebook
3. Configure:
   - Name: `my-notebook`
   - Image: `kubeflownotebookswg/jupyter-scipy:v1.11.0`
   - CPU: 1, Memory: 2Gi
4. Click Launch

## 3.2 Create Notebook via kubectl

Create `phase3/notebook-server.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: my-jupyter
  namespace: kubeflow-user-example-com
spec:
  template:
    spec:
      containers:
      - name: my-jupyter
        image: kubeflownotebookswg/jupyter-scipy:v1.11.0
        resources:
          requests:
            cpu: "0.5"
            memory: 1Gi
          limits:
            cpu: "2"
            memory: 4Gi
        volumeMounts:
        - name: workspace
          mountPath: /home/jovyan
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: my-jupyter-workspace
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-jupyter-workspace
  namespace: kubeflow-user-example-com
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

```bash
kubectl apply -f phase3/notebook-server.yaml
```

## 3.3 Custom Notebook Image

Create `phase3/Dockerfile.notebook`:

```dockerfile
FROM kubeflownotebookswg/jupyter-scipy:v1.11.0

USER root

# Install additional packages
RUN pip install --no-cache-dir \
    mlflow==3.8.1 \
    kfp==2.10.0 \
    xgboost \
    lightgbm

USER jovyan
```

Build and push:
```bash
docker build -t my-notebook:latest -f Dockerfile.notebook .
# For Minikube, load directly:
minikube image load my-notebook:latest
```

## 3.4 Notebook Pipeline Development

Example notebook workflow:
1. Develop pipeline components in notebook
2. Test locally with small data
3. Export to Python script
4. Compile and submit pipeline

## Phase 3 Checklist

- [ ] Created notebook server via UI
- [ ] Created notebook via kubectl
- [ ] Built custom notebook image
- [ ] Developed pipeline in notebook

---

# Phase 4: Katib (Hyperparameter Tuning)

## Objectives
- Understand Katib concepts
- Create HP tuning experiments
- Use different search algorithms
- Analyze results

## 4.1 Katib Concepts

```
Experiment
├── Objective (metric to optimize)
├── Search Algorithm (random, grid, bayesian, etc.)
├── Parameters (hyperparameters to tune)
├── Trial Template (how to run each trial)
└── Trials (individual runs)
    └── Metrics (results)
```

## 4.2 Simple HP Tuning Experiment

Create `phase4/01_random_search.yaml`:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: random-search-example
  namespace: kubeflow-user-example-com
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: random
  parallelTrialCount: 2
  maxTrialCount: 10
  maxFailedTrialCount: 3
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.001"
        max: "0.1"
    - name: num_layers
      parameterType: int
      feasibleSpace:
        min: "1"
        max: "5"
    - name: optimizer
      parameterType: categorical
      feasibleSpace:
        list:
          - adam
          - sgd
          - rmsprop
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: learningRate
        description: Learning rate
        reference: learning_rate
      - name: numLayers
        description: Number of layers
        reference: num_layers
      - name: optimizer
        description: Optimizer
        reference: optimizer
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training-container
                image: python:3.11-slim
                command:
                  - "python"
                  - "-c"
                  - |
                    import random
                    lr = ${trialParameters.learningRate}
                    layers = ${trialParameters.numLayers}
                    opt = "${trialParameters.optimizer}"

                    # Simulated training
                    accuracy = 0.7 + random.uniform(0, 0.25)
                    print(f"accuracy={accuracy:.4f}")
            restartPolicy: Never
```

Apply:
```bash
kubectl apply -f phase4/01_random_search.yaml
```

## 4.3 Grid Search

Create `phase4/02_grid_search.yaml`:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: grid-search-example
  namespace: kubeflow-user-example-com
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: grid
  parallelTrialCount: 3
  maxTrialCount: 12
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        list:
          - "0.01"
          - "0.05"
          - "0.1"
    - name: batch_size
      parameterType: int
      feasibleSpace:
        list:
          - "16"
          - "32"
          - "64"
          - "128"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: learningRate
        reference: learning_rate
      - name: batchSize
        reference: batch_size
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training-container
                image: python:3.11-slim
                command:
                  - "python"
                  - "-c"
                  - |
                    import random
                    lr = ${trialParameters.learningRate}
                    bs = ${trialParameters.batchSize}

                    # Simulated training
                    accuracy = 0.75 + (0.1 - lr) + (bs / 1000) + random.uniform(0, 0.1)
                    print(f"accuracy={accuracy:.4f}")
            restartPolicy: Never
```

## 4.4 Bayesian Optimization

Create `phase4/03_bayesian.yaml`:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: bayesian-example
  namespace: kubeflow-user-example-com
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: bayesianoptimization
    algorithmSettings:
      - name: "random_state"
        value: "42"
  parallelTrialCount: 2
  maxTrialCount: 15
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.0001"
        max: "0.1"
    - name: dropout
      parameterType: double
      feasibleSpace:
        min: "0.0"
        max: "0.5"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: learningRate
        reference: learning_rate
      - name: dropout
        reference: dropout
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training-container
                image: python:3.11-slim
                command:
                  - "python"
                  - "-c"
                  - |
                    import math
                    import random

                    lr = ${trialParameters.learningRate}
                    dropout = ${trialParameters.dropout}

                    # Simulated objective function
                    # Optimal around lr=0.01, dropout=0.2
                    accuracy = 0.95 - abs(math.log10(lr) + 2) * 0.1 - abs(dropout - 0.2) * 0.2
                    accuracy += random.uniform(-0.02, 0.02)
                    accuracy = max(0, min(1, accuracy))

                    print(f"accuracy={accuracy:.4f}")
            restartPolicy: Never
```

## 4.5 Monitor Experiments

```bash
# List experiments
kubectl get experiments -n kubeflow-user-example-com

# Get experiment details
kubectl describe experiment random-search-example -n kubeflow-user-example-com

# List trials
kubectl get trials -n kubeflow-user-example-com

# Get best trial
kubectl get experiment random-search-example -n kubeflow-user-example-com \
  -o jsonpath='{.status.currentOptimalTrial}'
```

## Phase 4 Checklist

- [ ] Understand Katib concepts
- [ ] Created random search experiment
- [ ] Created grid search experiment
- [ ] Created Bayesian optimization experiment
- [ ] Monitored trials and found best parameters

---

# Phase 5: Training Operators

## Objectives
- Run TensorFlow distributed training
- Run PyTorch distributed training
- Understand worker/parameter server patterns

## 5.1 TFJob (TensorFlow Training)

Create `phase5/01_tfjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: mnist-simple
  namespace: kubeflow-user-example-com
spec:
  tfReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.13.0
              command:
                - "python"
                - "-c"
                - |
                  import tensorflow as tf
                  import os

                  print(f"TF Version: {tf.__version__}")
                  print(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

                  # Simple training simulation
                  model = tf.keras.Sequential([
                      tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
                      tf.keras.layers.Dense(10, activation='softmax')
                  ])

                  model.compile(
                      optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                  )

                  # Simulated data
                  import numpy as np
                  x_train = np.random.rand(1000, 784)
                  y_train = np.random.randint(0, 10, 1000)

                  model.fit(x_train, y_train, epochs=3, batch_size=32)
                  print("Training completed!")
              resources:
                limits:
                  cpu: "1"
                  memory: "2Gi"
```

Apply:
```bash
kubectl apply -f phase5/01_tfjob.yaml

# Monitor
kubectl get tfjobs -n kubeflow-user-example-com
kubectl logs -n kubeflow-user-example-com -l job-name=mnist-simple-worker -f
```

## 5.2 PyTorchJob

Create `phase5/02_pytorchjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-simple
  namespace: kubeflow-user-example-com
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
              command:
                - "python"
                - "-c"
                - |
                  import torch
                  import torch.nn as nn
                  import torch.optim as optim

                  print(f"PyTorch Version: {torch.__version__}")
                  print(f"CUDA Available: {torch.cuda.is_available()}")

                  # Simple model
                  class SimpleNet(nn.Module):
                      def __init__(self):
                          super().__init__()
                          self.fc1 = nn.Linear(784, 128)
                          self.fc2 = nn.Linear(128, 10)

                      def forward(self, x):
                          x = torch.relu(self.fc1(x))
                          return self.fc2(x)

                  model = SimpleNet()
                  criterion = nn.CrossEntropyLoss()
                  optimizer = optim.Adam(model.parameters())

                  # Simulated training
                  for epoch in range(3):
                      x = torch.randn(32, 784)
                      y = torch.randint(0, 10, (32,))

                      optimizer.zero_grad()
                      output = model(x)
                      loss = criterion(output, y)
                      loss.backward()
                      optimizer.step()

                      print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

                  print("Training completed!")
              resources:
                limits:
                  cpu: "1"
                  memory: "2Gi"
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
              command:
                - "python"
                - "-c"
                - |
                  import torch
                  import time
                  print("Worker started")
                  time.sleep(60)
                  print("Worker finished")
              resources:
                limits:
                  cpu: "1"
                  memory: "2Gi"
```

## 5.3 Distributed Training with Multiple Workers

Create `phase5/03_distributed_pytorch.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed
  namespace: kubeflow-user-example-com
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
              env:
                - name: MASTER_PORT
                  value: "23456"
              command:
                - "python"
                - "-c"
                - |
                  import os
                  import torch
                  import torch.distributed as dist
                  import torch.nn as nn
                  import torch.optim as optim
                  from torch.nn.parallel import DistributedDataParallel

                  def setup():
                      dist.init_process_group(backend='gloo')

                  def cleanup():
                      dist.destroy_process_group()

                  def main():
                      setup()
                      rank = dist.get_rank()
                      world_size = dist.get_world_size()
                      print(f"Rank {rank}/{world_size} initialized")

                      # Simple model
                      model = nn.Linear(10, 10)
                      ddp_model = DistributedDataParallel(model)

                      optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

                      for epoch in range(3):
                          x = torch.randn(32, 10)
                          y = torch.randn(32, 10)

                          optimizer.zero_grad()
                          output = ddp_model(x)
                          loss = nn.MSELoss()(output, y)
                          loss.backward()
                          optimizer.step()

                          if rank == 0:
                              print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

                      cleanup()
                      print(f"Rank {rank} finished")

                  if __name__ == "__main__":
                      main()
              resources:
                limits:
                  cpu: "1"
                  memory: "2Gi"
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
              env:
                - name: MASTER_PORT
                  value: "23456"
              command:
                - "python"
                - "-c"
                - |
                  # Same code as master
                  import os
                  import torch
                  import torch.distributed as dist
                  import torch.nn as nn
                  import torch.optim as optim
                  from torch.nn.parallel import DistributedDataParallel

                  def setup():
                      dist.init_process_group(backend='gloo')

                  def cleanup():
                      dist.destroy_process_group()

                  def main():
                      setup()
                      rank = dist.get_rank()
                      world_size = dist.get_world_size()
                      print(f"Rank {rank}/{world_size} initialized")

                      model = nn.Linear(10, 10)
                      ddp_model = DistributedDataParallel(model)
                      optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

                      for epoch in range(3):
                          x = torch.randn(32, 10)
                          y = torch.randn(32, 10)
                          optimizer.zero_grad()
                          output = ddp_model(x)
                          loss = nn.MSELoss()(output, y)
                          loss.backward()
                          optimizer.step()

                      cleanup()
                      print(f"Rank {rank} finished")

                  if __name__ == "__main__":
                      main()
              resources:
                limits:
                  cpu: "1"
                  memory: "2Gi"
```

## Phase 5 Checklist

- [ ] Created and ran TFJob
- [ ] Created and ran PyTorchJob
- [ ] Understand distributed training patterns
- [ ] Monitored training job logs

---

# Phase 6: MLflow Integration

## Objectives
- Track Kubeflow experiments in MLflow
- Log models from pipelines
- Compare runs across platforms

## 6.1 Pipeline with MLflow Tracking

Create `phase6/01_mlflow_pipeline.py`:

```python
"""
Pipeline that logs to MLflow tracking server.
"""
from kfp import dsl
from kfp import compiler

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow", "scikit-learn", "pandas", "boto3"]
)
def train_with_mlflow(
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int
) -> float:
    """Train model and log to MLflow."""
    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("source", "kubeflow_pipeline")

        # Train
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    return accuracy

@dsl.pipeline(
    name="mlflow-integration-pipeline",
    description="Pipeline with MLflow tracking"
)
def mlflow_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-server.mlflow.svc.cluster.local:5000",
    experiment_name: str = "kubeflow-experiments",
    n_estimators: int = 100
):
    train_task = train_with_mlflow(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlflow_pipeline,
        package_path="mlflow_pipeline.yaml"
    )
```

## 6.2 Katib with MLflow

Create `phase6/02_katib_mlflow.yaml`:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: katib-mlflow-experiment
  namespace: kubeflow-user-example-com
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: random
  parallelTrialCount: 2
  maxTrialCount: 5
  parameters:
    - name: n_estimators
      parameterType: int
      feasibleSpace:
        min: "50"
        max: "200"
    - name: max_depth
      parameterType: int
      feasibleSpace:
        min: "3"
        max: "15"
  trialTemplate:
    primaryContainerName: training
    trialParameters:
      - name: nEstimators
        reference: n_estimators
      - name: maxDepth
        reference: max_depth
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training
                image: python:3.11-slim
                command:
                  - "sh"
                  - "-c"
                  - |
                    pip install mlflow scikit-learn pandas boto3 -q

                    python -c "
                    import mlflow
                    import mlflow.sklearn
                    from sklearn.datasets import load_iris
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import accuracy_score

                    n_estimators = ${trialParameters.nEstimators}
                    max_depth = ${trialParameters.maxDepth}

                    mlflow.set_tracking_uri('http://mlflow-server.mlflow.svc.cluster.local:5000')
                    mlflow.set_experiment('katib-experiments')

                    iris = load_iris()
                    X_train, X_test, y_train, y_test = train_test_split(
                        iris.data, iris.target, test_size=0.2, random_state=42
                    )

                    with mlflow.start_run():
                        mlflow.log_params({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'source': 'katib'
                        })

                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        model.fit(X_train, y_train)

                        accuracy = accuracy_score(y_test, model.predict(X_test))
                        mlflow.log_metric('accuracy', accuracy)
                        mlflow.sklearn.log_model(model, 'model')

                        print(f'accuracy={accuracy:.4f}')
                    "
            restartPolicy: Never
```

## Phase 6 Checklist

- [ ] Created pipeline with MLflow tracking
- [ ] Logged experiments from Kubeflow to MLflow
- [ ] Created Katib experiment with MLflow
- [ ] Compared runs across platforms

---

# Summary

## Complete Learning Path

| Phase | Topic | Key Takeaways |
|-------|-------|---------------|
| 1 | Setup | Minikube, Kubeflow install, Dashboard |
| 2 | Pipelines | KFP SDK, Components, Artifacts |
| 3 | Notebooks | Jupyter servers, Custom images |
| 4 | Katib | HP tuning algorithms, Experiments |
| 5 | Training | TFJob, PyTorchJob, Distributed |
| 6 | Integration | MLflow tracking, End-to-end |

## Quick Reference

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192 --disk-size=40g

# Access Dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

# Check Kubeflow pods
kubectl get pods -n kubeflow

# Run pipeline
kfp run submit -f pipeline.yaml

# Create Katib experiment
kubectl apply -f experiment.yaml

# Check training jobs
kubectl get tfjobs,pytorchjobs -n kubeflow-user-example-com
```
