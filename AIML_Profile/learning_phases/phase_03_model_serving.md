# Phase 03: Model Serving & APIs

**Duration**: 2 weeks | **Prerequisites**: Phase 02 completed

---

## Learning Objectives

By the end of this phase, you will:
- [ ] Build production-ready ML APIs with FastAPI
- [ ] Package models with BentoML
- [ ] Scale inference with Ray Serve
- [ ] Handle async requests and batching

---

## Week 1: FastAPI for ML

### Day 1-2: FastAPI Basics

```bash
# Install dependencies
uv add fastapi uvicorn pydantic
```

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="ML Prediction API", version="1.0.0")

# Load model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Request/Response schemas
class PredictionRequest(BaseModel):
    features: list[float]
    
    class Config:
        json_schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uv run uvicorn app:app --reload
```

### Day 3-4: Add MLflow Model Loading

```python
from fastapi import FastAPI
import mlflow

app = FastAPI()

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/iris-classifier/Production")

@app.post("/predict")
async def predict(request: PredictionRequest):
    import pandas as pd
    df = pd.DataFrame([request.features])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
```

### Day 5-7: Advanced Features

```python
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import asyncio

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    app.state.model = load_model()
    yield
    # Shutdown: Cleanup
    del app.state.model

app = FastAPI(lifespan=lifespan)

# Background task for logging
def log_prediction(request, response):
    # Log to database/file
    save_to_db(request, response)

@app.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    result = app.state.model.predict(request.features)
    background_tasks.add_task(log_prediction, request, result)
    return result

# Batch endpoint
class BatchRequest(BaseModel):
    inputs: list[list[float]]

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    predictions = app.state.model.predict(request.inputs)
    return {"predictions": predictions.tolist()}
```

**Hands-on Exercise:**
1. Build complete prediction API
2. Add `/docs` auto-documentation
3. Add request validation
4. Implement batch endpoint
5. Test with `curl` and Swagger UI

---

## Week 2: BentoML & Ray Serve

### Day 8-10: BentoML

```bash
uv add bentoml scikit-learn
```

```python
# save_model.py
import bentoml
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)

# Save to BentoML
saved_model = bentoml.sklearn.save_model("iris_classifier", model)
print(f"Saved: {saved_model}")
```

```python
# service.py
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, JSON

# Load model
iris_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# Create service
svc = bentoml.Service("iris_service", runners=[iris_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_array: np.ndarray) -> dict:
    prediction = await iris_runner.predict.async_run(input_array)
    return {"prediction": prediction.tolist()}
```

```bash
# Serve locally
uv run bentoml serve service:svc --reload

# Build bento
uv run bentoml build

# Containerize
uv run bentoml containerize iris_service:latest
```

### Day 11-12: Ray Serve

```bash
uv add "ray[serve]"
```

```python
from ray import serve
from starlette.requests import Request
import pickle

@serve.deployment(num_replicas=2)
class IrisClassifier:
    def __init__(self):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
    
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        features = data["features"]
        prediction = self.model.predict([features])[0]
        return {"prediction": int(prediction)}

# Deploy
serve.run(IrisClassifier.bind(), route_prefix="/predict")
```

### Day 13-14: Scaling with Ray Serve

```python
from ray import serve

@serve.deployment(
    num_replicas=2,              # Number of replicas
    ray_actor_options={
        "num_cpus": 1,           # CPUs per replica
        "num_gpus": 0.5,         # GPUs per replica (fractional)
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    }
)
class ScalableModel:
    def __init__(self):
        self.model = load_model()
    
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def predict_batch(self, requests: list[Request]):
        # Process batch together
        features = [await r.json() for r in requests]
        predictions = self.model.predict(features)
        return predictions.tolist()
    
    async def __call__(self, request: Request):
        return await self.predict_batch(request)
```

**Hands-on Exercise:**
1. Save model with BentoML
2. Create BentoML service
3. Build and run Docker container
4. Deploy with Ray Serve
5. Test autoscaling with load

---

## Milestone Checklist

- [ ] FastAPI prediction endpoint working
- [ ] Swagger docs accessible at `/docs`
- [ ] Batch prediction implemented
- [ ] BentoML model saved and served
- [ ] BentoML Docker image created
- [ ] Ray Serve deployment running
- [ ] Autoscaling configured

---

## Comparison Summary

| Feature | FastAPI | BentoML | Ray Serve |
|---------|---------|---------|-----------|
| Ease of use | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Batching | Manual | Built-in | Built-in |
| Scaling | External | Docker/K8s | Built-in |
| Model packaging | Manual | Automatic | Manual |
| Best for | Simple APIs | Production packaging | High-scale |

---

**Next Phase**: [Phase 04 - Data Engineering](./phase_04_data_engineering.md)
