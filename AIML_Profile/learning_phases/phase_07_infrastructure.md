# Phase 07: Production Infrastructure

**Duration**: 3 weeks | **Prerequisites**: Phase 06 completed

---

## Week 1: Docker

### Day 1-3: Containerize ML Model

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv sync

COPY src/ ./src/
COPY model.pkl .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0"]
```

```bash
docker build -t ml-api:v1 .
docker run -p 8000:8000 ml-api:v1
```

### Day 4-7: Multi-stage Builds

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv sync --no-dev

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY src/ ./src/
CMD [".venv/bin/python", "-m", "uvicorn", "src.app:app"]
```

---

## Week 2: Kubernetes

### Day 8-10: Minikube Setup

```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

minikube start
kubectl get nodes
```

### Day 11-14: Deploy ML Service

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
```

---

## Week 3: CI/CD

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: myregistry/ml-api:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: kubectl set image deployment/ml-api ml-api=myregistry/ml-api:${{ github.sha }}
```

---

## Milestone Checklist
- [ ] Docker image built
- [ ] Minikube running
- [ ] K8s deployment created
- [ ] GitHub Actions pipeline working
- [ ] Secrets managed properly

**Next**: [Phase 08](./phase_08_advanced.md)
