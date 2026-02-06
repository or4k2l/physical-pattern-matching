# Deployment Guide

## Overview

This guide covers deploying robust vision models for production use.

## Quick Deployment

### 1. Export Model

After training, export your model:

```python
from robust_vision.training.state import TrainStateWithEMA
from flax.training import checkpoints

# Load checkpoint
state = checkpoints.restore_checkpoint(
    ckpt_dir="./checkpoints/best_checkpoint_18",
    target=state
)

# Use EMA parameters for inference
inference_params = state.ema_params
```

### 2. Create Inference Function

```python
import jax
import jax.numpy as jnp
from robust_vision.models.cnn import ProductionCNN

model = ProductionCNN(n_classes=10)

@jax.jit
def predict(params, images):
    """JIT-compiled prediction function."""
    logits = model.apply(params, images, training=False)
    return jax.nn.softmax(logits)

# Use it
predictions = predict(inference_params, test_images)
```

## FastAPI Deployment

### Create API Server

Create `serve.py`:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import io

from robust_vision.models.cnn import ProductionCNN
from flax.training import checkpoints

app = FastAPI(title="Robust Vision API")

# Load model
model = ProductionCNN(n_classes=10)
state = checkpoints.restore_checkpoint(
    ckpt_dir="./checkpoints/best_checkpoint_18",
    target=state
)
params = state.ema_params

@jax.jit
def predict(images):
    logits = model.apply(params, images, training=False)
    return jax.nn.softmax(logits)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict class for uploaded image."""
    # Load and preprocess image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = jnp.array(image)[None, ...]  # Add batch dimension
    
    # Predict
    probs = predict(image)
    class_id = int(jnp.argmax(probs[0]))
    confidence = float(probs[0, class_id])
    
    return JSONResponse({
        "class": class_id,
        "confidence": confidence,
        "probabilities": probs[0].tolist()
    })

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Run Server

```bash
pip install fastapi uvicorn python-multipart
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Test API

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

## Docker Deployment

### Build Production Image

Create `Dockerfile.prod`:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn python-multipart

# Copy application
COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/
COPY serve.py /app/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
docker build -f Dockerfile.prod -t robust-vision-api:latest .

docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name robust-vision-api \
  robust-vision-api:latest
```

## Cloud Deployment

### AWS (ECS/Fargate)

1. **Push to ECR**:

```bash
aws ecr create-repository --repository-name robust-vision-api
docker tag robust-vision-api:latest <aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/robust-vision-api:latest
docker push <aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/robust-vision-api:latest
```

2. **Create ECS Task Definition**:

```json
{
  "family": "robust-vision-api",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/robust-vision-api:latest",
      "memory": 4096,
      "cpu": 2048,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

3. **Deploy to ECS**.

### Google Cloud (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/robust-vision-api

# Deploy
gcloud run deploy robust-vision-api \
  --image gcr.io/PROJECT_ID/robust-vision-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure (Container Instances)

```bash
# Build and push
az acr build --registry myregistry --image robust-vision-api:latest .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name robust-vision-api \
  --image myregistry.azurecr.io/robust-vision-api:latest \
  --cpu 2 \
  --memory 4 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --port 8000
```

## Kubernetes Deployment

### Create Deployment

`deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robust-vision-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: robust-vision-api
  template:
    metadata:
      labels:
        app: robust-vision-api
    spec:
      containers:
      - name: api
        image: robust-vision-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: robust-vision-api-service
spec:
  selector:
    app: robust-vision-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Deploy

```bash
kubectl apply -f deployment.yaml
kubectl get services
```

## Performance Optimization

### 1. JIT Compilation

Always use `@jax.jit` for inference:

```python
@jax.jit
def predict(params, images):
    return model.apply(params, images, training=False)
```

### 2. Batching

Process multiple images at once:

```python
# Better: Batch processing
batch_predictions = predict(params, batch_images)

# Avoid: One at a time
predictions = [predict(params, img[None, ...]) for img in images]
```

### 3. Model Quantization

Reduce model size and increase speed:

```python
# Convert to int8
quantized_params = jax.tree_map(
    lambda x: (x * 127).astype(jnp.int8),
    params
)
```

### 4. Caching

Cache preprocessing steps:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def preprocess_image(image_path):
    # ... preprocessing ...
    return processed_image
```

### 5. Async Processing

Use async for I/O operations:

```python
import asyncio

async def predict_batch(image_paths):
    images = await asyncio.gather(*[
        load_image_async(path) for path in image_paths
    ])
    return predict(params, jnp.array(images))
```

## Monitoring

### Health Checks

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": params is not None,
        "gpu_available": jax.devices()[0].platform == 'gpu'
    }
```

### Metrics

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
async def predict_image(file):
    prediction_counter.inc()
    # ... prediction logic ...
```

## Load Testing

Test your deployment:

```bash
# Install locust
pip install locust

# Create locustfile.py
from locust import HttpUser, task

class RobustVisionUser(HttpUser):
    @task
    def predict(self):
        with open("test_image.jpg", "rb") as f:
            self.client.post("/predict", files={"file": f})

# Run load test
locust -f locustfile.py --host http://localhost:8000
```

## Security

### 1. Authentication

Add API key authentication:

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    # ... prediction logic ...
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_image(request: Request, file: UploadFile = File(...)):
    # ... prediction logic ...
```

### 3. Input Validation

```python
from fastapi import HTTPException

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")
    
    # ... prediction logic ...
```

## Troubleshooting

### High Latency

1. Enable JIT compilation
2. Use batching
3. Check GPU utilization
4. Profile with JAX profiler

### Out of Memory

1. Reduce batch size
2. Use model quantization
3. Clear JAX cache periodically

### Cold Start Issues

Pre-warm the model:

```python
@app.on_event("startup")
async def startup():
    # Warm up model with dummy input
    dummy_input = jnp.ones((1, 32, 32, 3))
    predict(params, dummy_input)
```

## Next Steps

- Monitor production metrics
- Set up alerting
- Implement A/B testing
- Add model versioning
