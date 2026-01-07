# Phase 09: CI/CD & GitOps

## Overview

Complete CI/CD pipeline setup with GitHub Actions and GitOps with ArgoCD for automated ML deployments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CI/CD & GITOPS ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         GITHUB REPOSITORY                            │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│   │   │   ML Code   │  │   Configs   │  │ Kubernetes  │                │   │
│   │   │  (models)   │  │  (params)   │  │ (manifests) │                │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                    ┌────────────────┴────────────────┐                      │
│                    │                                 │                      │
│                    ▼                                 ▼                      │
│   ┌────────────────────────────┐    ┌────────────────────────────┐        │
│   │      GITHUB ACTIONS        │    │         ARGOCD             │        │
│   │   ┌──────────────────┐    │    │   ┌──────────────────┐    │        │
│   │   │  Build & Test    │    │    │   │   Sync & Deploy  │    │        │
│   │   │  Train Model     │    │    │   │   Health Check   │    │        │
│   │   │  Push Artifacts  │    │    │   │   Rollback       │    │        │
│   │   └──────────────────┘    │    │   └──────────────────┘    │        │
│   └────────────────────────────┘    └────────────────────────────┘        │
│                    │                                 │                      │
│                    └────────────────┬────────────────┘                      │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      KUBERNETES CLUSTER                              │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │   │
│   │   │ Airflow │ │ MLflow  │ │Kubeflow │ │ KServe  │ │Monitoring│      │   │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: GitHub Actions - ML Pipeline

### Model Training Workflow

Create `.github/workflows/ml-training.yaml`:

```yaml
name: ML Training Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'models/**'
      - 'data/**'
      - 'configs/**'
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model to train'
        required: true
        default: 'churn-classifier'
      experiment_name:
        description: 'MLflow experiment name'
        required: true
        default: 'github-actions-training'

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest flake8 black isort mypy

      - name: Lint with flake8
        run: flake8 models/ --count --select=E9,F63,F7,F82 --show-source

      - name: Check formatting with black
        run: black --check models/

      - name: Type check with mypy
        run: mypy models/ --ignore-missing-imports

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=models --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  data-validation:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Great Expectations
        run: pip install great-expectations pandas

      - name: Validate training data
        run: |
          python -c "
          import great_expectations as gx
          context = gx.get_context()
          result = context.run_checkpoint(checkpoint_name='training_data_checkpoint')
          if not result.success:
              raise Exception('Data validation failed!')
          "

  train-model:
    runs-on: ubuntu-latest
    needs: data-validation
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: |
          python models/train.py \
            --model-name ${{ github.event.inputs.model_name || 'churn-classifier' }} \
            --experiment-name ${{ github.event.inputs.experiment_name || 'github-actions-training' }} \
            --run-name "gh-${{ github.run_id }}"

      - name: Get run metrics
        id: metrics
        run: |
          python -c "
          import mlflow
          mlflow.set_tracking_uri('${{ env.MLFLOW_TRACKING_URI }}')
          experiment = mlflow.get_experiment_by_name('${{ github.event.inputs.experiment_name || 'github-actions-training' }}')
          runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
          accuracy = runs.iloc[0]['metrics.accuracy']
          run_id = runs.iloc[0]['run_id']
          print(f'::set-output name=accuracy::{accuracy}')
          print(f'::set-output name=run_id::{run_id}')
          "

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Model Training Results

              - **Run ID**: ${{ steps.metrics.outputs.run_id }}
              - **Accuracy**: ${{ steps.metrics.outputs.accuracy }}
              - **MLflow UI**: ${{ env.MLFLOW_TRACKING_URI }}/#/experiments
              `
            })

    outputs:
      run_id: ${{ steps.metrics.outputs.run_id }}
      accuracy: ${{ steps.metrics.outputs.accuracy }}

  evaluate-model:
    runs-on: ubuntu-latest
    needs: train-model
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install mlflow scikit-learn pandas

      - name: Evaluate model
        id: evaluate
        run: |
          python -c "
          import mlflow
          mlflow.set_tracking_uri('${{ env.MLFLOW_TRACKING_URI }}')

          accuracy = float('${{ needs.train-model.outputs.accuracy }}')
          threshold = 0.85

          passed = accuracy >= threshold
          print(f'::set-output name=passed::{str(passed).lower()}')

          if not passed:
              print(f'Model accuracy {accuracy} below threshold {threshold}')
          "

    outputs:
      passed: ${{ steps.evaluate.outputs.passed }}

  register-model:
    runs-on: ubuntu-latest
    needs: [train-model, evaluate-model]
    if: needs.evaluate-model.outputs.passed == 'true' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install MLflow
        run: pip install mlflow

      - name: Register model
        run: |
          python -c "
          import mlflow
          from mlflow import MlflowClient

          mlflow.set_tracking_uri('${{ env.MLFLOW_TRACKING_URI }}')
          client = MlflowClient()

          run_id = '${{ needs.train-model.outputs.run_id }}'
          model_uri = f'runs:/{run_id}/model'
          model_name = '${{ github.event.inputs.model_name || 'churn-classifier' }}'

          # Register model
          result = mlflow.register_model(model_uri, model_name)
          print(f'Registered model version: {result.version}')

          # Transition to Staging
          client.transition_model_version_stage(
              name=model_name,
              version=result.version,
              stage='Staging'
          )
          print(f'Transitioned to Staging')
          "

  update-deployment:
    runs-on: ubuntu-latest
    needs: register-model
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT_TOKEN }}

      - name: Update Kubernetes manifest
        run: |
          MODEL_VERSION=$(python -c "
          import mlflow
          from mlflow import MlflowClient
          mlflow.set_tracking_uri('${{ env.MLFLOW_TRACKING_URI }}')
          client = MlflowClient()
          versions = client.get_latest_versions('${{ github.event.inputs.model_name || 'churn-classifier' }}', stages=['Staging'])
          print(versions[0].version)
          ")

          # Update KServe manifest
          yq eval ".spec.predictor.model.storageUri = \"models:/churn-classifier/${MODEL_VERSION}\"" \
            -i kubernetes/kserve/churn-classifier.yaml

          # Update image tag
          yq eval ".spec.predictor.containers[0].image = \"ml-service:${{ github.sha }}\"" \
            -i kubernetes/kserve/churn-classifier.yaml

      - name: Commit and push changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add kubernetes/
          git commit -m "Update model deployment to version ${{ github.sha }}"
          git push
```

### Docker Build Workflow

Create `.github/workflows/docker-build.yaml`:

```yaml
name: Build and Push Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'docker/**'
      - 'models/**'
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    strategy:
      matrix:
        include:
          - context: ./docker/training
            image: training
          - context: ./docker/serving
            image: serving
          - context: ./docker/transformer
            image: transformer

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.image }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ${{ matrix.context }}
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.image }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## Step 2: Install ArgoCD

```bash
# Create namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=argocd-server -n argocd --timeout=300s

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

### Configure ArgoCD

Create `argocd/config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-cm
  namespace: argocd
data:
  url: https://argocd.local

  # Repository credentials
  repositories: |
    - url: https://github.com/your-org/mlops-platform
      passwordSecret:
        name: repo-secret
        key: password
      usernameSecret:
        name: repo-secret
        key: username

  # OIDC configuration
  oidc.config: |
    name: Keycloak
    issuer: https://keycloak.local/realms/mlops-platform
    clientID: argocd-client
    clientSecret: $oidc.keycloak.clientSecret
    requestedScopes: ["openid", "profile", "email", "groups"]

  # Resource tracking
  resource.customizations: |
    serving.kserve.io/InferenceService:
      health.lua: |
        hs = {}
        if obj.status ~= nil then
          if obj.status.conditions ~= nil then
            for i, condition in ipairs(obj.status.conditions) do
              if condition.type == "Ready" and condition.status == "True" then
                hs.status = "Healthy"
                hs.message = "InferenceService is ready"
                return hs
              end
            end
          end
        end
        hs.status = "Progressing"
        hs.message = "Waiting for InferenceService to be ready"
        return hs
```

---

## Step 3: ArgoCD Applications

### MLOps Platform Application

Create `argocd/applications/mlops-platform.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mlops-platform
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/mlops-platform.git
    targetRevision: main
    path: kubernetes
    directory:
      recurse: true
  destination:
    server: https://kubernetes.default.svc
    namespace: mlops
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### Model Serving Application

Create `argocd/applications/model-serving.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: model-serving
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "2"
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/mlops-platform.git
    targetRevision: main
    path: kubernetes/kserve
  destination:
    server: https://kubernetes.default.svc
    namespace: kserve
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
  # Health checks for ML models
  ignoreDifferences:
    - group: serving.kserve.io
      kind: InferenceService
      jsonPointers:
        - /spec/predictor/minReplicas
```

### ApplicationSet for Multi-Environment

Create `argocd/applicationsets/environments.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ml-models
  namespace: argocd
spec:
  generators:
    - matrix:
        generators:
          # Environments
          - list:
              elements:
                - env: staging
                  cluster: https://staging-cluster.local
                  namespace: ml-staging
                - env: production
                  cluster: https://production-cluster.local
                  namespace: ml-production
          # Models
          - list:
              elements:
                - model: churn-classifier
                  path: kubernetes/models/churn
                - model: fraud-detector
                  path: kubernetes/models/fraud
                - model: recommender
                  path: kubernetes/models/recommender
  template:
    metadata:
      name: '{{model}}-{{env}}'
      namespace: argocd
    spec:
      project: default
      source:
        repoURL: https://github.com/your-org/mlops-platform.git
        targetRevision: '{{env}}'
        path: '{{path}}'
        kustomize:
          namePrefix: '{{env}}-'
      destination:
        server: '{{cluster}}'
        namespace: '{{namespace}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

---

## Step 4: Canary Deployments with Argo Rollouts

### Install Argo Rollouts

```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

### Model Canary Deployment

Create `kubernetes/rollouts/model-rollout.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: churn-classifier-rollout
  namespace: kserve
spec:
  replicas: 5
  strategy:
    canary:
      canaryService: churn-classifier-canary
      stableService: churn-classifier-stable
      trafficRouting:
        istio:
          virtualService:
            name: churn-classifier-vsvc
            routes:
              - primary
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - setWeight: 25
        - pause: {duration: 5m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 75
        - pause: {duration: 10m}
      analysis:
        templates:
          - templateName: model-success-rate
        startingStep: 2
        args:
          - name: service-name
            value: churn-classifier-canary
  selector:
    matchLabels:
      app: churn-classifier
  template:
    metadata:
      labels:
        app: churn-classifier
    spec:
      containers:
        - name: model-server
          image: ghcr.io/your-org/serving:latest
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_NAME
              value: churn-classifier
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: mlflow-secret
                  key: tracking-uri
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: model-success-rate
  namespace: kserve
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(
              model_prediction_requests_total{
                service="{{args.service-name}}",
                status="success"
              }[5m]
            )) /
            sum(rate(
              model_prediction_requests_total{
                service="{{args.service-name}}"
              }[5m]
            ))
    - name: latency-p99
      interval: 1m
      successCondition: result[0] < 500
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(
                model_prediction_latency_seconds_bucket{
                  service="{{args.service-name}}"
                }[5m]
              )) by (le)
            ) * 1000
```

---

## Step 5: GitOps Repository Structure

```
mlops-platform/
├── .github/
│   └── workflows/
│       ├── ml-training.yaml
│       ├── docker-build.yaml
│       └── promote-model.yaml
├── kubernetes/
│   ├── base/
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   └── rbac.yaml
│   ├── overlays/
│   │   ├── staging/
│   │   │   ├── kustomization.yaml
│   │   │   └── patches/
│   │   └── production/
│   │       ├── kustomization.yaml
│   │       └── patches/
│   ├── airflow/
│   │   ├── kustomization.yaml
│   │   └── values.yaml
│   ├── mlflow/
│   │   ├── kustomization.yaml
│   │   └── deployment.yaml
│   ├── kubeflow/
│   │   └── kustomization.yaml
│   ├── kserve/
│   │   ├── kustomization.yaml
│   │   ├── churn-classifier.yaml
│   │   └── fraud-detector.yaml
│   └── monitoring/
│       ├── kustomization.yaml
│       └── dashboards/
├── argocd/
│   ├── applications/
│   └── applicationsets/
├── models/
│   ├── churn/
│   ├── fraud/
│   └── recommender/
├── configs/
│   ├── training/
│   └── serving/
└── tests/
    ├── unit/
    └── integration/
```

---

## Step 6: Promotion Workflow

Create `.github/workflows/promote-model.yaml`:

```yaml
name: Promote Model

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model to promote'
        required: true
      from_stage:
        description: 'Source stage'
        required: true
        default: 'Staging'
      to_stage:
        description: 'Target stage'
        required: true
        default: 'Production'

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install mlflow

      - name: Promote model
        run: |
          python -c "
          import mlflow
          from mlflow import MlflowClient

          mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')
          client = MlflowClient()

          model_name = '${{ github.event.inputs.model_name }}'
          from_stage = '${{ github.event.inputs.from_stage }}'
          to_stage = '${{ github.event.inputs.to_stage }}'

          # Get latest version from source stage
          versions = client.get_latest_versions(model_name, stages=[from_stage])
          if not versions:
              raise Exception(f'No model found in {from_stage}')

          version = versions[0].version

          # Archive existing production model
          if to_stage == 'Production':
              prod_versions = client.get_latest_versions(model_name, stages=['Production'])
              for v in prod_versions:
                  client.transition_model_version_stage(
                      name=model_name,
                      version=v.version,
                      stage='Archived'
                  )

          # Promote model
          client.transition_model_version_stage(
              name=model_name,
              version=version,
              stage=to_stage
          )

          print(f'Promoted {model_name} v{version} to {to_stage}')
          "

      - name: Update production manifest
        if: github.event.inputs.to_stage == 'Production'
        run: |
          VERSION=$(python -c "
          import mlflow
          from mlflow import MlflowClient
          mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')
          client = MlflowClient()
          versions = client.get_latest_versions('${{ github.event.inputs.model_name }}', stages=['Production'])
          print(versions[0].version)
          ")

          yq eval ".spec.predictor.model.storageUri = \"models:/${{ github.event.inputs.model_name }}/${VERSION}\"" \
            -i kubernetes/overlays/production/kserve/${{ github.event.inputs.model_name }}.yaml

      - name: Commit and push
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add kubernetes/
          git commit -m "Promote ${{ github.event.inputs.model_name }} to ${{ github.event.inputs.to_stage }}"
          git push

      - name: Create release tag
        if: github.event.inputs.to_stage == 'Production'
        run: |
          VERSION=$(date +%Y%m%d%H%M%S)
          git tag -a "release-${{ github.event.inputs.model_name }}-${VERSION}" -m "Production release"
          git push origin "release-${{ github.event.inputs.model_name }}-${VERSION}"
```

---

## Verification

```bash
#!/bin/bash
# verify_cicd.sh

echo "=== CI/CD & GitOps Verification ==="

echo -e "\n1. ArgoCD Status:"
kubectl get pods -n argocd

echo -e "\n2. ArgoCD Applications:"
kubectl get applications -n argocd

echo -e "\n3. Application Sync Status:"
argocd app list

echo -e "\n4. Argo Rollouts Status:"
kubectl get rollouts -A

echo -e "\n5. Recent Deployments:"
kubectl get pods -n kserve -l app=churn-classifier --sort-by=.metadata.creationTimestamp

echo -e "\n=== Verification Complete ==="
```

---

**Status**: Phase 09 Complete
**Features Covered**: GitHub Actions, ArgoCD, GitOps, Canary Deployments, Model Promotion
