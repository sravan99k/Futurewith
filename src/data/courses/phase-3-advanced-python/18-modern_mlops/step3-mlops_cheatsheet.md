# MLOps Cheatsheet: Quick Reference Guide

## Table of Contents

1. [Essential Commands](#essential-commands)
2. [Configuration Templates](#configuration-templates)
3. [Common Tools Quick Reference](#common-tools-quick-reference)
4. [Code Snippets](#code-snippets)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Best Practices Checklist](#best-practices-checklist)
7. [Performance Optimization](#performance-optimization)
8. [Security Checklist](#security-checklist)
9. [Monitoring Commands](#monitoring-commands)
10. [Deployment Commands](#deployment-commands)

## Essential Commands

### Docker Operations

```bash
# Build ML model serving container
docker build -t ml-model:latest .

# Run model serving container
docker run -p 8080:8080 -e MODEL_PATH=/app/model.pkl ml-model:latest

# Multi-stage build for optimized images
docker build -f Dockerfile.prod -t ml-model:prod .

# Docker Compose for ML stack
docker-compose up -d mlflow postgres redis

# View container logs
docker logs -f ml-model-container

# Execute in running container
docker exec commands -it ml-model-container bash

# Clean up unused images
docker system prune -f

# Save and load Docker images
docker save ml-model:latest | gzip > ml-model.tar.gz
docker load < ml-model.tar.gz
```

### Kubernetes Operations

```bash
# Apply Kubernetes manifests
kubectl apply -f ml-deployment.yaml

# Get pod status
kubectl get pods -l app=ml-model

# View pod logs
kubectl logs -f deployment/ml-model-serving

# Port forward for local testing
kubectl port-forward svc/ml-model-service 8080:80

# Scale deployment
kubectl scale deployment ml-model-serving --replicas=5

# Describe pod for debugging
kubectl describe pod ml-model-serving-xxx

# Delete resources
kubectl delete -f ml-deployment.yaml

# Apply changes and rollout new version
kubectl set image deployment/ml-model-serving ml-model=ml-model:v2.0
kubectl rollout status deployment/ml-model-serving

# Rollback deployment
kubectl rollout undo deployment/ml-model-serving

# View rollout history
kubectl rollout history deployment/ml-model-serving
```

### MLflow Commands

```bash
# Start MLflow server
mlflow server --backend-store-uri postgresql://user:pass@localhost/mlflow --default-artifact-root s3://mlflow-artifacts/

# Track experiment
mlflow.start_run()
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.95)
mlflow.sklearn.log_model(model, "model")

# Search experiments
mlflow.search_experiments(filter_string="attributes.status = 'ACTIVE'")

# Register model
mlflow.register_model("runs:/run_id/model", "model_name")

# Transition model stage
mlflow.transition_model_version_stage(
    name="model_name",
    version="1",
    stage="Production"
)

# Download model
mlflow.download_artifacts(run_id="run_id", artifact_path="model")
```

### DVC Commands

```bash
# Initialize DVC
dvc init

# Add data to version control
dvc add data/dataset.csv

# Track data pipeline
dvc stage add -n prepare_data \
    -d data/raw/dataset.csv -d scripts/prepare.py \
    -o data/prepared/dataset.csv \
    python scripts/prepare.py

# Reproduce pipeline
dvc repro

# Track data with specific version
dvc add data/dataset.csv -T "v1.0"

# Pull latest data
dvc pull

# Push changes to remote
dvc push

# Show data pipeline
dvc dag

# Compare experiments
dvc exp show

# Cache data
dvc cache dir
dvc cache info

# Check data integrity
dvc dag --outs
```

### Git Commands for ML

```bash
# Add large files to .gitignore
echo "*.pkl" >> .gitignore
echo "models/" >> .gitignore
echo "data/raw/" >> .gitignore

# Create feature branch for experiment
git checkout -b feature/new-algorithm

# Commit with descriptive message
git commit -m "feat: add XGBoost model with hyperparameter tuning"

# Tag model versions
git tag -a model-v1.0 -m "Initial model version"
git push origin model-v1.0

# Clean large files from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch models/model.pkl' \
  --prune-empty --tag-name-filter cat -- --all

# Large File Storage (LFS)
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

## Configuration Templates

### Docker Compose for ML Stack

```yaml
# docker-compose.yml
version: "3.8"

services:
  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
      - ./mlflow:/mlflow
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/
    command: >
      sh -c "pip install mlflow psycopg2-binary boto3 &&
             mlflow server 
             --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow 
             --default-artifact-root s3://mlflow-artifacts/ 
             --host 0.0.0.0 
             --port 5000"
    depends_on:
      - postgres

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  model-serving:
    build:
      context: .
      dockerfile: Dockerfile.serving
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models
    depends_on:
      - mlflow

volumes:
  mlflow-data:
  postgres-data:
  redis-data:
```

### Kubernetes Deployment

```yaml
# ml-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
  namespace: ml-production
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
          image: ml-model:v1.0
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_PATH
              value: "/app/models/model.pkl"
            - name: LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
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
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: ml-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-serving
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Helm Chart Structure

```yaml
# values.yaml
replicaCount: 3

image:
  repository: ml-model
  tag: "v1.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
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
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: ml-model.local
      paths:
        - path: /
          pathType: Prefix

# values-prod.yaml
replicaCount: 5

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  minReplicas: 5
  maxReplicas: 20

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: ml-model.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ml-model-tls
      hosts:
        - ml-model.company.com
```

### CI/CD Pipeline Configuration

```yaml
# .github/workflows/ml-ci-cd.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ --cov=src/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate data
        run: |
          python scripts/validate_data.py

      - name: Check data drift
        run: |
          python scripts/check_drift.py

  train-and-evaluate:
    needs: [test, data-validation]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        experiment: [baseline, enhanced, experimental]
    steps:
      - uses: actions/checkout@v3

      - name: Train model (${{ matrix.experiment }})
        run: |
          python scripts/train_model.py --experiment ${{ matrix.experiment }}

      - name: Evaluate model
        run: |
          python scripts/evaluate_model.py --model outputs/${{ matrix.experiment }}/model.pkl

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-${{ matrix.experiment }}
          path: outputs/${{ matrix.experiment }}/

  build-and-push:
    needs: [train-and-evaluate]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-model-serving \
            ml-model=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          kubectl rollout status deployment/ml-model-serving

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/ml-model-serving \
            ml-model=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          kubectl rollout status deployment/ml-model-serving
```

## Common Tools Quick Reference

### MLflow Quick Reference

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

# Track experiment
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "model")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    mlflow.log_metric("f1", f1_score(y_test, model.predict(X_test)))

# Search experiments
experiments = mlflow.search_experiments()
client = MlflowClient()
runs = client.search_runs(experiment_ids=["1"])

# Register model
client.register_model(
    model_uri="runs:/run_id/model",
    name="my_model"
)

# Transition model stage
client.transition_model_version_stage(
    name="my_model",
    version="1",
    stage="Production"
)
```

### DVC Quick Reference

```python
import dvc.api
import dvc.repo

# Get data
data = dvc.api.get_url('path/to/data.csv')

# Version data
with dvc.repo.Repo() as repo:
    repo.add('data/dataset.csv')
    repo.push()

# Reproduce pipeline
with dvc.repo.Repo() as repo:
    repo.reproduce('dvc.yaml')

# Get data from specific version
with dvc.repo.Repo() as repo:
    repo.checkout('v1.0')
    data = repo.get('data/dataset.csv')
```

### Weights & Biases Quick Reference

```python
import wandb

# Initialize
wandb.init(project="my-project", entity="my-username")

# Log parameters and metrics
wandb.config.update({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100
})

# Log metrics
wandb.log({"accuracy": 0.95, "loss": 0.1})

# Log artifacts
artifact = wandb.Artifact('model', type='model')
artifact.add_file('model.pkl')
wandb.log_artifact(artifact)

# Log plots
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(y_test, y_pred)})
```

### Kubernetes Python Client

```python
from kubernetes import client, config

# Load kube config
config.load_kube_config()

# Create API clients
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()

# Get deployments
deployments = apps_v1.list_namespaced_deployment(namespace="default")

# Scale deployment
apps_v1.patch_namespaced_deployment(
    name="ml-model-serving",
    namespace="default",
    body={"spec": {"replicas": 5}}
)

# Get pod logs
pod_name = "ml-model-serving-xxx"
logs = core_v1.read_namespaced_pod_log(
    name=pod_name,
    namespace="default"
)

# Create service
service = client.V1Service(
    api_version="v1",
    kind="Service",
    metadata=client.V1ObjectMeta(name="ml-model-service"),
    spec=client.V1ServiceSpec(
        selector={"app": "ml-model"},
        ports=[client.V1ServicePort(port=80, target_port=8080)]
    )
)

core_v1.create_namespaced_service(
    namespace="default",
    body=service
)
```

## Code Snippets

### FastAPI Model Serving

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# Global model variable
model = None

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.on_event("startup")
async def startup_event():
    global model
    model = joblib.load("model.pkl")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare input data
        input_data = np.array([[request.feature1, request.feature2]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].max()

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "features": ["feature1", "feature2"],
        "classes": model.classes_.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Data Validation with Great Expectations

```python
# validate_data.py
import great_expectations as ge
import pandas as pd

def validate_data(file_path):
    # Load data
    df = ge.read_csv(file_path)

    # Define expectations
    df.expect_column_to_exist("feature1")
    df.expect_column_values_to_be_between("feature1", min_value=0, max_value=100)
    df.expect_column_values_to_not_be_null("feature2")
    df.expect_column_values_to_be_in_set("feature3", ["A", "B", "C"])

    # Check results
    results = df.validate()

    if not results["success"]:
        print("Data validation failed!")
        for result in results["results"]:
            if not result["success"]:
                print(f"- {result['expectation_config']['expectation_type']}: {result['exception_info']}")
        return False

    print("Data validation passed!")
    return True

if __name__ == "__main__":
    validate_data("data/train.csv")
```

### Model Training Pipeline

```python
# train_model.py
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path, model_params):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model locally
        joblib.dump(model, "model.pkl")

        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/train.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42
    }

    train_model(args.data_path, model_params)
```

### Monitoring Setup

```python
# monitoring.py
import time
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Prometheus metrics
REQUEST_COUNT = Counter('ml_model_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('ml_model_request_duration_seconds', 'Request latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage')

class ModelMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()

    def log_request(self, method, endpoint, latency):
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_LATENCY.observe(latency)

    def update_system_metrics(self):
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

    def log_model_metrics(self, accuracy):
        MODEL_ACCURACY.set(accuracy)

    def start_prometheus_server(self, port=8000):
        start_http_server(port)
        self.logger.info(f"Prometheus metrics server started on port {port}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Start monitoring
    monitor = ModelMonitor()
    monitor.start_prometheus_server()

    # Update system metrics every 30 seconds
    while True:
        monitor.update_system_metrics()
        time.sleep(30)
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Model Performance Issues

```bash
# Check model drift
python scripts/check_drift.py --model-path model.pkl --reference-data data/reference.csv --current-data data/current.csv

# Validate input data
python scripts/validate_input.py --input-file input.json

# Test model locally
python -c "
import joblib
import pandas as pd
model = joblib.load('model.pkl')
test_data = pd.read_csv('test_data.csv')
predictions = model.predict(test_data)
print(f'Predictions: {predictions}')
"
```

#### Deployment Issues

```bash
# Check pod status
kubectl get pods -l app=ml-model

# View pod logs
kubectl logs -f deployment/ml-model-serving

# Check service endpoints
kubectl get endpoints ml-model-service

# Port forward for testing
kubectl port-forward svc/ml-model-service 8080:80

# Check resource usage
kubectl top pods -l app=ml-model
```

#### Data Pipeline Issues

```bash
# Check data validation
python scripts/validate_data.py --data-path data/raw/dataset.csv

# Reproduce DVC pipeline
dvc repro

# Check data lineage
dvc dag --outs

# Validate data schema
python -c "
import pandas as pd
df = pd.read_csv('data.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df.info())
"
```

#### Performance Issues

```bash
# Load test model serving
python scripts/load_test.py --endpoint http://localhost:8080 --concurrent-users 10 --requests-per-user 100

# Check system resources
htop
df -h
free -h

# Profile model inference
python -c "
import time
import joblib
model = joblib.load('model.pkl')
import numpy as np
test_data = np.random.rand(1000, 10)
start = time.time()
predictions = model.predict(test_data)
end = time.time()
print(f'Inference time for 1000 samples: {end-start:.3f}s')
print(f'Average time per sample: {(end-start)/1000*1000:.3f}ms')
"
```

### Debugging Commands

#### Container Debugging

```bash
# Execute shell in container
kubectl exec -it deployment/ml-model-serving -- /bin/bash

# Check container logs
docker logs -f container-id

# Copy files from container
kubectl cp pod-name:/app/logs /local/logs

# Check container resources
kubectl top pod pod-name
```

#### Network Debugging

```bash
# Test service connectivity
kubectl run test-pod --image=busybox --rm -it -- wget -qO- http://ml-model-service

# Check DNS resolution
kubectl exec -it deployment/ml-model-serving -- nslookup ml-model-service

# Test port connectivity
kubectl exec -it deployment/ml-model-serving -- nc -zv ml-model-service 80
```

#### Database Debugging

```bash
# Connect to PostgreSQL
psql -h postgres -U mlflow -d mlflow

# Check MLflow tables
\dt

# Query experiment data
SELECT * FROM experiments LIMIT 10;

# Check model versions
SELECT * FROM model_versions WHERE name = 'my_model';
```

## Best Practices Checklist

### Development Best Practices

- [ ] **Code Quality**
  - [ ] Use linting (flake8, black)
  - [ ] Write unit tests (pytest)
  - [ ] Maintain type hints
  - [ ] Document functions and classes
  - [ ] Follow PEP 8 style guide

- [ ] **Data Management**
  - [ ] Version control data with DVC
  - [ ] Validate data schemas
  - [ ] Document data sources
  - [ ] Monitor data quality
  - [ ] Handle missing values appropriately

- [ ] **Model Development**
  - [ ] Track experiments with MLflow
  - [ ] Use cross-validation
  - [ ] Monitor for overfitting
  - [ ] Document model assumptions
  - [ ] Include feature importance analysis

### Production Best Practices

- [ ] **Deployment**
  - [ ] Use containerization (Docker)
  - [ ] Implement health checks
  - [ ] Set resource limits
  - [ ] Use secrets management
  - [ ] Implement rollback procedures

- [ ] **Monitoring**
  - [ ] Monitor model performance
  - [ ] Track data drift
  - [ ] Monitor system resources
  - [ ] Set up alerting
  - [ ] Log predictions and inputs

- [ ] **Security**
  - [ ] Use authentication
  - [ ] Implement rate limiting
  - [ ] Encrypt data in transit
  - [ ] Secure API endpoints
  - [ ] Regular security updates

### MLOps Best Practices

- [ ] **Pipeline Automation**
  - [ ] Automate data validation
  - [ ] Automate model training
  - [ ] Automate testing
  - [ ] Automate deployment
  - [ ] Automate monitoring

- [ ] **Collaboration**
  - [ ] Use version control
  - [ ] Document processes
  - [ ] Share model artifacts
  - [ ] Review code changes
  - [ ] Maintain changelogs

- [ ] **Governance**
  - [ ] Track model lineage
  - [ ] Document decisions
  - [ ] Implement approval workflows
  - [ ] Maintain audit logs
  - [ ] Ensure compliance

## Performance Optimization

### Model Optimization

```python
# Model quantization for faster inference
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save optimized model
joblib.dump(model, "model_optimized.pkl")

# For deep learning models
# import torch.quantization as quantization
# quantized_model = quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
```

### Caching Strategies

```python
# Feature caching
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=1000)
def get_cached_features(user_id):
    # Expensive feature computation
    return compute_user_features(user_id)

# Model prediction caching
from cachetools import TTLCache

prediction_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL

def get_cached_prediction(features):
    cache_key = hash(str(features))
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]

    prediction = model.predict(features)
    prediction_cache[cache_key] = prediction
    return prediction
```

### Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX idx_experiment_name ON experiments(name);
CREATE INDEX idx_run_experiment_id ON runs(experiment_id);
CREATE INDEX idx_model_version_name_stage ON model_versions(name, stage);

-- Query optimization
EXPLAIN ANALYZE SELECT * FROM runs WHERE experiment_id = 1;

-- Partition large tables
CREATE TABLE runs_2023 PARTITION OF runs
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
```

## Security Checklist

### API Security

- [ ] **Authentication**
  - [ ] Implement JWT tokens
  - [ ] Use API keys
  - [ ] Set token expiration
  - [ ] Validate tokens
  - [ ] Secure token storage

- [ ] **Authorization**
  - [ ] Role-based access control
  - [ ] API endpoint protection
  - [ ] Resource-level permissions
  - [ ] Audit access logs
  - [ ] Regular access reviews

### Data Security

- [ ] **Data Protection**
  - [ ] Encrypt sensitive data
  - [ ] Use secure protocols (HTTPS)
  - [ ] Implement data masking
  - [ ] Secure data storage
  - [ ] Regular data backups

- [ ] **Privacy**
  - [ ] Anonymize PII data
  - [ ] Implement data retention policies
  - [ ] GDPR compliance
  - [ ] Data access logging
  - [ ] Regular privacy audits

### Infrastructure Security

- [ ] **Container Security**
  - [ ] Use minimal base images
  - [ ] Scan for vulnerabilities
  - [ ] Run as non-root user
  - [ ] Secure container registry
  - [ ] Regular security updates

- [ ] **Network Security**
  - [ ] Use network policies
  - [ ] Implement firewalls
  - [ ] Secure service mesh
  - [ ] VPN for remote access
  - [ ] Regular security scans

## Monitoring Commands

### Prometheus Queries

```promql
# Model request rate
rate(ml_model_requests_total[5m])

# Model latency
histogram_quantile(0.95, rate(ml_model_request_duration_seconds_bucket[5m]))

# Error rate
rate(ml_model_requests_total{status="error"}[5m]) / rate(ml_model_requests_total[5m])

# System CPU usage
system_cpu_usage_percent

# Memory usage
system_memory_usage_percent

# Model accuracy
ml_model_accuracy

# Data drift score
data_drift_score
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(ml_model_requests_total{status=\"error\"}[5m]) / rate(ml_model_requests_total[5m]) * 100",
            "legendFormat": "Error %"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alerting-rules.yml
groups:
  - name: ml-model-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(ml_model_requests_total{status="error"}[5m]) / rate(ml_model_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(ml_model_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy has dropped"
          description: "Current accuracy is {{ $value }}"

      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
```

## Deployment Commands

### Blue-Green Deployment

```bash
# Deploy to green environment
kubectl apply -f deployment-green.yaml

# Test green environment
kubectl run test-pod --image=curlimages/curl --rm -it -- \
  curl http://green-service/health

# Switch traffic to green
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor deployment
kubectl rollout status deployment/ml-model-serving-green

# Rollback if needed
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"blue"}}}'
```

### Canary Deployment

```bash
# Deploy canary version
kubectl apply -f deployment-canary.yaml

# Monitor canary metrics
kubectl logs -f deployment/ml-model-serving-canary

# Gradually increase traffic
kubectl patch service ml-model-service -p '{
  "spec": {
    "selector": {"canary": "enabled"}
  }
}'

# Promote canary to production
kubectl patch deployment ml-model-serving -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "ml-model",
          "image": "ml-model:canary"
        }]
      }
    }
  }
}'
```

### Helm Deployment

```bash
# Install chart
helm install ml-model ./helm-chart

# Upgrade deployment
helm upgrade ml-model ./helm-chart -f values-production.yaml

# Rollback to previous version
helm rollback ml-model 1

# Check deployment status
helm status ml-model

# List releases
helm list

# Uninstall
helm uninstall ml-model
```

This cheatsheet provides quick access to essential MLOps commands, configurations, and best practices. Keep it handy for daily operations and troubleshooting!
