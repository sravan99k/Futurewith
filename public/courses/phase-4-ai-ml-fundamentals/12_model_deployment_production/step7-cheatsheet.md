# Model Deployment & Production - Complete Cheatsheet

_Quick Reference Guide for AI/ML Model Deployment_

---

## 🚀 Quick Start Commands

### Essential Deployment Flow

```bash
# 1. Prepare your model
python train_model.py
python save_model.py --format=onnx

# 2. Create container
docker build -t my-model:v1.0 .

# 3. Deploy to cloud
aws sagemaker create-endpoint-config --config-name my-config
aws sagemaker update-endpoint --endpoint-name my-endpoint

# 4. Monitor
curl -X POST http://api.com/predict -d '{"data": [1,2,3]}'
```

---

## 🎯 MLOps Fundamentals

### Core MLOps Pipeline

```
Data → Training → Validation → Model Registry →
Deployment → Monitoring → Retraining → Feedback Loop
```

### Key Principles

- **Reproducibility**: Same data, same results
- **Version Control**: Track models, data, code
- **Automation**: CI/CD for ML pipelines
- **Monitoring**: Track model performance in production
- **Governance**: Compliance and security

### Model Lifecycle

1. **Development**: Experiment and build
2. **Validation**: Test and verify
3. **Deployment**: Release to production
4. **Monitoring**: Track performance
5. **Retraining**: Update based on feedback
6. **Archiving**: Deprecate old versions

---

## ☁️ Cloud Deployment

### AWS SageMaker

```bash
# Setup SageMaker
import boto3
import sagemaker

# Create model
sm_client = boto3.client('sagemaker')
model_response = sm_client.create_model(
    ModelName='my-model',
    PrimaryContainer={
        'Image': '683313888378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        'ModelDataUrl': 's3://bucket/model.tar.gz'
    }
)

# Create endpoint
endpoint_config = sm_client.create_endpoint_config(
    EndpointConfigName='my-config',
    ProductionVariants=[{
        'VariantName': 'primary',
        'ModelName': 'my-model',
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.t2.medium'
    }]
)
```

### Google Cloud Vertex AI

```bash
# Setup Vertex AI
from google.cloud import aiplatform

# Initialize
aiplatform.init(project='my-project', location='us-central1')

# Deploy model
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://bucket/model',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest'
)

# Create endpoint
endpoint = model.deploy(
    deployed_model_id='my-deployment',
    machine_type='n1-standard-2'
)
```

### Azure ML

```bash
# Setup Azure ML
from azureml.core import Workspace, Model

# Get workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    model_path='outputs/model.pkl',
    model_name='my-model',
    workspace=ws
)

# Deploy to AKS
from azureml.core.compute import ComputeTarget, AksCompute

# Create compute target
compute_target = ComputeTarget(
    workspace=ws,
    name='my-aks-cluster',
    type='AksCompute'
)

# Deploy model
service = Model.deploy(
    workspace=ws,
    name='my-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    compute_target=compute_target
)
```

---

## 🐳 Docker Containerization

### Basic Dockerfile for ML Model

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "serve.py"]
```

### Complete ML Service Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ ./models/
COPY src/ ./src/
COPY *.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "30", "app:app"]
```

### Docker Commands

```bash
# Build image
docker build -t my-ml-service:latest .

# Run container
docker run -p 8080:8080 -e MODEL_PATH=/models/model.pkl my-ml-service:latest

# Check logs
docker logs -f <container_id>

# Save/load image
docker save my-ml-service:latest > my-ml-service.tar
docker load < my-ml-service.tar

# Multi-stage build
docker build -t optimized-model:latest -f Dockerfile.prod .
```

---

## ☸️ Kubernetes Orchestration

### Kubernetes Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
        - name: ml-model
          image: my-ml-service:latest
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_PATH
              value: "/models/model.pkl"
            - name: LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

### Kubernetes Commands

```bash
# Apply configuration
kubectl apply -f deployment.yaml

# Scale deployment
kubectl scale deployment ml-model-deployment --replicas=5

# Check status
kubectl get pods
kubectl get services
kubectl get deployments

# View logs
kubectl logs -f deployment/ml-model-deployment

# Describe resource
kubectl describe pod <pod-name>

# Port forward for testing
kubectl port-forward service/ml-model-service 8080:80

# Delete resources
kubectl delete -f deployment.yaml
```

### Helm Charts for ML Services

```yaml
# Chart.yaml
apiVersion: v2
name: ml-model
description: Machine Learning Model Service
version: 0.1.0

# values.yaml
replicaCount: 2

image:
  repository: my-ml-service
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

---

## 🌐 API Development

### FastAPI Service Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production ML Model Service",
    version="1.0.0"
)

# Load model
model = joblib.load('models/model.pkl')

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != 10:
            raise HTTPException(status_code=400, detail="Expected 10 features")

        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()

        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            model_version="1.0.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction
@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    try:
        all_features = []
        for request in requests:
            all_features.append(request.features)

        features = np.array(all_features)
        predictions = model.predict(features)
        confidences = model.predict_proba(features).max(axis=1)

        return {
            "predictions": predictions.tolist(),
            "confidences": confidences.tolist(),
            "batch_size": len(requests)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Service Example

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Load model
model = joblib.load('models/model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            raise BadRequest("Missing 'features' in request body")

        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()

        return jsonify({
            "prediction": float(prediction),
            "confidence": float(confidence),
            "model_version": "1.0.0"
        })

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 📊 Monitoring Systems

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'Request duration')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions', ['prediction_type'])

# Expose metrics endpoint
start_http_server(8001)

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    REQUEST_DURATION.observe(time.time() - start_time)

    return response

# Custom metrics for ML
def update_model_metrics(accuracy, precision, recall, f1_score):
    MODEL_ACCURACY.set(accuracy)
    # Add more metrics as needed
```

### Logging and Monitoring

```python
import logging
import structlog
from pythonjsonlogger import jsonlogger

# Setup structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)

logger = logging.getLogger('ml-service')
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Use structlog for better JSON logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Example usage
def log_prediction(features, prediction, confidence, user_id=None):
    structlog.get_logger().info(
        "prediction_made",
        features_shape=len(features),
        prediction=prediction,
        confidence=confidence,
        user_id=user_id,
        timestamp=time.time()
    )
```

### Model Drift Detection

```python
import numpy as np
from scipy import stats
import joblib
from datetime import datetime, timedelta

class DriftDetector:
    def __init__(self, reference_data, alpha=0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.baseline_stats = self._compute_baseline_stats()

    def _compute_baseline_stats(self):
        return {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'median': np.median(self.reference_data, axis=0)
        }

    def detect_drift(self, new_data):
        drift_detected = False
        drift_details = {}

        for i in range(new_data.shape[1]):
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )

            if p_value < self.alpha:
                drift_detected = True
                drift_details[f'feature_{i}'] = {
                    'ks_stat': ks_stat,
                    'p_value': p_value,
                    'drift_detected': True
                }

        return drift_detected, drift_details

    def check_model_performance(self, recent_predictions, true_labels):
        accuracy = np.mean(recent_predictions == true_labels)

        if accuracy < 0.8:  # Threshold for retraining
            return True, {"accuracy": accuracy, "threshold": 0.8}

        return False, {"accuracy": accuracy, "threshold": 0.8}

# Usage
drift_detector = DriftDetector(reference_data=X_train)
drift_detected, details = drift_detector.detect_drift(X_recent)
```

---

## 📈 Scaling Strategies

### Horizontal Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Load Balancing

```python
# Load balancer configuration
import random
from typing import List

class ModelLoadBalancer:
    def __init__(self, model_instances: List[object]):
        self.instances = model_instances
        self.current_index = 0

    def get_next_instance(self):
        # Round-robin
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return instance

    def get_best_instance(self, load_metrics: List[float]):
        # Choose instance with lowest load
        min_load_index = np.argmin(load_metrics)
        return self.instances[min_load_index]

    def predict(self, features):
        instance = self.get_next_instance()
        return instance.predict(features)
```

### Auto-scaling Configuration

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-model-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: ml-model
        maxAllowed:
          cpu: 2
          memory: 4Gi
        minAllowed:
          cpu: 100m
          memory: 128Mi
```

---

## 🔧 Production Best Practices

### Model Versioning

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("production-models")

# Log model
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="production-model"
    )
    mlflow.log_param("accuracy", 0.95)

# Load specific version
client = MlflowClient()
latest_version = client.get_latest_versions("production-model", stages=["Production"])[0]
model_uri = f"models:/{latest_version.name}/{latest_version.version}"
model = mlflow.sklearn.load_model(model_uri)
```

### Environment Management

```python
# Config management
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str
    model_version: str
    api_version: str
    log_level: str
    max_requests_per_minute: int
    enable_metrics: bool = True
    enable_caching: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            model_path=os.getenv("MODEL_PATH", "/models/model.pkl"),
            model_version=os.getenv("MODEL_VERSION", "1.0.0"),
            api_version=os.getenv("API_VERSION", "v1"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1000")),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true"
        )

# Usage
config = ModelConfig.from_env()
```

### Error Handling and Resilience

```python
import time
import random
from functools import wraps
from typing import Callable, Any
import asyncio

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e

                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)

        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, timeout: int = 60):
    def decorator(func: Callable) -> Callable:
        failures = 0
        last_failure_time = None
        state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failures, last_failure_time, state

            if state == "OPEN":
                if time.time() - last_failure_time > timeout:
                    state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                if state == "HALF_OPEN":
                    state = "CLOSED"
                    failures = 0
                return result
            except Exception as e:
                failures += 1
                last_failure_time = time.time()

                if failures >= failure_threshold:
                    state = "OPEN"

                raise e

        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3)
@circuit_breaker(failure_threshold=5)
def predict_with_retry(features):
    return model.predict(features)
```

### Security Best Practices

```python
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# JWT authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Input validation and sanitization
import re
from typing import Union

def validate_input(features: list) -> bool:
    # Check data type
    if not isinstance(features, list):
        return False

    # Check length
    if len(features) != 10:
        return False

    # Check each feature is numeric
    for feature in features:
        if not isinstance(feature, (int, float)):
            return False
        if abs(feature) > 1e6:  # Reasonable bounds
            return False

    return True

# Rate limiting
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True

# Usage in API
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.post("/predict")
async def predict_with_auth(
    request: PredictionRequest,
    user_data = Depends(verify_token)
):
    client_id = user_data.get("sub", "anonymous")

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not validate_input(request.features):
        raise HTTPException(status_code=400, detail="Invalid input data")

    return predict(request.features)
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: ML Model CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=src/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t ${{ github.repository }}:latest .
          docker tag ${{ github.repository }}:latest ${{ github.repository }}:${{ github.sha }}

      - name: Deploy to staging
        run: |
          # Deploy to staging environment
          kubectl apply -f k8s/staging/

      - name: Run integration tests
        run: |
          # Test the deployed model
          python tests/integration/test_deployment.py

      - name: Deploy to production
        if: success()
        run: |
          # Deploy to production
          kubectl apply -f k8s/production/
```

### Model Validation Pipeline

```python
# Model validation before deployment
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

def validate_model_for_production(model_path: str, test_data: tuple) -> dict:
    """
    Validate model meets production requirements
    """
    X_test, y_test = test_data

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Check confidence scores
    max_proba = np.max(y_proba, axis=1)
    low_confidence_predictions = np.sum(max_proba < 0.7)

    validation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "low_confidence_count": int(low_confidence_predictions),
        "total_predictions": len(y_pred),
        "validation_passed": True
    }

    # Check thresholds
    if accuracy < 0.85:
        validation_results["validation_passed"] = False
        validation_results["error"] = f"Accuracy {accuracy:.3f} below threshold 0.85"

    if low_confidence_predictions / len(y_pred) > 0.1:
        validation_results["validation_passed"] = False
        validation_results["error"] = f"Too many low confidence predictions: {low_confidence_predictions}/{len(y_pred)}"

    logging.info(f"Model validation results: {validation_results}")
    return validation_results

# Usage
results = validate_model_for_production(
    "models/production_model.pkl",
    (X_test, y_test)
)

if not results["validation_passed"]:
    raise ValueError(f"Model validation failed: {results['error']}")
```

---

## 📋 Deployment Checklist

### Pre-Deployment Checklist

- [ ] Model performance meets requirements (accuracy, latency, throughput)
- [ ] Model bias and fairness testing completed
- [ ] Security testing (input validation, authentication, authorization)
- [ ] Load testing completed
- [ ] Monitoring and alerting configured
- [ ] Rollback plan documented and tested
- [ ] Documentation updated
- [ ] Team notification sent

### Production Readiness Checklist

- [ ] Health check endpoints implemented
- [ ] Graceful shutdown handling
- [ ] Error handling and logging
- [ ] Rate limiting implemented
- [ ] API versioning strategy
- [ ] Model versioning and rollback
- [ ] Performance monitoring
- [ ] Cost monitoring
- [ ] Disaster recovery plan
- [ ] Compliance and audit logging

### Security Checklist

- [ ] Input validation and sanitization
- [ ] Authentication and authorization
- [ ] HTTPS/TLS encryption
- [ ] Secrets management
- [ ] Network security policies
- [ ] Regular security updates
- [ ] Access logging and monitoring
- [ ] Data privacy compliance (GDPR, CCPA)

---

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### High Latency

```python
# Problem: Model predictions are too slow
# Solutions:
# 1. Use model optimization (ONNX, TensorRT)
# 2. Implement batch prediction
# 3. Add caching layer
# 4. Use GPU acceleration
# 5. Scale horizontally

import functools
import pickle
from collections import OrderedDict

class PredictionCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

# Use cache
cache = PredictionCache()

@app.post("/predict")
async def cached_predict(request: PredictionRequest):
    # Create cache key
    cache_key = hash(tuple(request.features))

    # Check cache
    result = cache.get(cache_key)
    if result is not None:
        return result

    # Compute if not cached
    prediction = model.predict([request.features])[0]
    result = {"prediction": float(prediction)}

    # Cache result
    cache.put(cache_key, result)
    return result
```

#### Memory Issues

```python
# Problem: Out of memory errors
# Solutions:
# 1. Use smaller batch sizes
# 2. Implement memory-efficient models
# 3. Use model quantization
# 4. Clear unnecessary data

import gc
import psutil
import threading

class MemoryMonitor:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.monitoring = False
        self.lock = threading.Lock()

    def start_monitoring(self):
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

    def _monitor(self):
        while self.monitoring:
            with self.lock:
                memory_percent = psutil.virtual_memory().percent / 100
                if memory_percent > self.threshold:
                    gc.collect()
                    print(f"Memory usage: {memory_percent:.1%}, performed garbage collection")
            time.sleep(10)

    def stop(self):
        self.monitoring = False

# Usage
monitor = MemoryMonitor(threshold=0.8)
monitor.start_monitoring()
```

#### Model Drift Detection

```python
# Automated drift detection and alerting
import numpy as np
from scipy import stats
import smtplib
from email.mime.text import MIMEText

class DriftAlertSystem:
    def __init__(self, reference_data, email_config):
        self.reference_data = reference_data
        self.email_config = email_config
        self.baseline_stats = self._compute_baseline()

    def _compute_baseline(self):
        return {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0)
        }

    def check_drift(self, new_data, threshold=0.05):
        drift_scores = []

        for i in range(new_data.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            drift_scores.append({
                'feature': i,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            })

        # Check if any significant drift
        significant_drift = [score for score in drift_scores if score['drift_detected']]

        if significant_drift:
            self._send_alert(significant_drift)

        return drift_scores

    def _send_alert(self, drift_scores):
        try:
            msg = MIMEText(f"Model drift detected!\nDrift scores: {drift_scores}")
            msg['Subject'] = "Model Drift Alert"
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            print("Drift alert sent successfully")
        except Exception as e:
            print(f"Failed to send drift alert: {e}")
```

---

## 📚 Quick Reference Commands

### Docker

```bash
# Build and run
docker build -t my-model:v1.0 .
docker run -p 8080:8080 my-model:v1.0

# Debugging
docker exec -it <container_id> /bin/bash
docker logs -f <container_id>
docker inspect <container_id>

# Cleanup
docker system prune -a
docker volume prune
```

### Kubernetes

```bash
# Deploy
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs -f <pod-name>

# Scale
kubectl scale deployment <name> --replicas=5
kubectl hpa describe <deployment>

# Debug
kubectl describe pod <pod-name>
kubectl exec -it <pod-name> -- /bin/bash
```

### Monitoring

```bash
# View metrics
curl http://<service>/metrics
kubectl port-forward <pod> 8001:8001

# View logs
kubectl logs -f deployment/<name>
tail -f /var/log/app.log
```

### Cloud Deployment

```bash
# AWS
aws sagemaker create-endpoint --endpoint-name <name>
aws sagemaker describe-endpoint --endpoint-name <name>

# GCP
gcloud ai endpoints predict <endpoint-id> --region=<region> --json-request=request.json

# Azure
az ml online-endpoint create --name <name>
az ml online-endpoint invoke --name <name> --request request.json
```

---

## 💡 Best Practices Summary

1. **Always validate inputs** - Never trust external data
2. **Monitor everything** - Metrics, logs, performance
3. **Version everything** - Code, models, data, configs
4. **Automate deployments** - CI/CD pipelines are essential
5. **Plan for failure** - Have rollback strategies
6. **Security first** - Authenticate, authorize, encrypt
7. **Performance matters** - Optimize for production
8. **Document everything** - Future you will thank you
9. **Test thoroughly** - Unit, integration, load testing
10. **Stay compliant** - Follow regulations and best practices

---

_This cheatsheet covers the essential aspects of deploying ML models to production. Keep it handy for quick reference during deployment projects!_
