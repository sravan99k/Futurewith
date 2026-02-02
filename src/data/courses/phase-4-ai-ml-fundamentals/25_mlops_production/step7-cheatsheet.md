# MLOps & Production ML - Cheatsheet

## Quick Reference Guide

## Model Development

### Data Processing Pipeline

```python
# Basic data processing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def process_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Handle categorical variables
    le = LabelEncoder()
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
```

### Model Training

```python
# Model training with tracking
import mlflow
import mlflow.sklearn

def train_model(model, X_train, y_train, model_name):
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)

        # Log parameters
        if hasattr(model, 'get_params'):
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return model
```

### Model Evaluation

```python
# Model evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    return metrics
```

## Model Deployment

### FastAPI Deployment

```python
# Basic FastAPI model serving
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(request: PredictionRequest):
    features = [[request.feature1, request.feature2, request.feature3]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
```

### Docker Deployment

```dockerfile
# Dockerfile for ML model
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Docker commands
docker build -t ml-model .
docker run -p 8000:8000 ml-model
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
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
          image: ml-model:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

## Model Monitoring

### Performance Monitoring

```python
# Model performance monitoring
import numpy as np
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actual_values = []

    def log_prediction(self, prediction, actual=None, confidence=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.actual_values.append(actual)

    def calculate_accuracy(self, window_size=100):
        if len(self.actual_values) < window_size:
            return None

        recent_actual = self.actual_values[-window_size:]
        recent_pred = self.predictions[-window_size:]

        accuracy = np.mean(np.array(recent_actual) == np.array(recent_pred))
        return accuracy

    def check_drift(self, X_new, X_reference, threshold=0.05):
        from scipy import stats

        # Simple statistical drift detection
        drift_detected = False
        for col in X_new.columns:
            if X_new[col].dtype in ['int64', 'float64']:
                statistic, p_value = stats.ks_2samp(
                    X_reference[col], X_new[col]
                )
                if p_value < threshold:
                    drift_detected = True
                    break

        return drift_detected
```

### Data Quality Checks

```python
# Data validation with Great Expectations
import great_expectations as ge

def validate_data(df, expectations):
    ge_df = ge.dataset.PandasDataset(df)

    for expectation in expectations:
        if expectation['type'] == 'column_to_exist':
            ge_df.expect_column_to_exist(expectation['column'])
        elif expectation['type'] == 'column_values_to_not_be_null':
            ge_df.expect_column_values_to_not_be_null(expectation['column'])
        elif expectation['type'] == 'column_values_to_be_between':
            ge_df.expect_column_values_to_be_between(
                expectation['column'],
                expectation['min_value'],
                expectation['max_value']
            )

    results = ge_df.validate()
    return results
```

## MLOps Automation

### MLflow Tracking

```python
# MLflow setup and tracking
import mlflow
import mlflow.sklearn

# Set experiment
mlflow.set_experiment("production_model_training")

# Track experiment
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train and evaluate model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Airflow DAG for ML Pipeline

```python
# Airflow DAG for ML pipeline
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    schedule_interval='@daily'
)

def extract_data():
    # Data extraction logic
    pass

def train_model():
    # Model training logic
    pass

def deploy_model():
    # Model deployment logic
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

extract_task >> train_task >> deploy_task
```

## CI/CD for ML

### GitHub Actions Workflow

```yaml
name: ML CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: |
          python -m pytest tests/
          python src/models/train_model.py

      - name: Model validation
        run: |
          python -c "import joblib; model = joblib.load('models/production_model.pkl'); print('Model validation passed')"

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: |
          docker build -t ml-model:${{ github.sha }} .
          docker push ml-model:${{ github.sha }}
```

### Deployment Strategies

#### Blue-Green Deployment

```python
def blue_green_deploy(new_model):
    # Deploy to green environment
    deploy_to_environment(new_model, 'green')

    # Run smoke tests
    if run_smoke_tests('green'):
        # Switch traffic
        switch_traffic('green')
        keep_blue_as_backup()
    else:
        # Rollback
        rollback()
        cleanup('green')
```

#### Canary Deployment

```python
def canary_deploy(model, traffic_percentage=10):
    # Deploy to canary
    deploy_to_canary(model)

    # Gradually increase traffic
    for percentage in [10, 25, 50, 100]:
        if percentage > traffic_percentage:
            break

        if monitor_canary_performance(percentage):
            print(f"Canary {percentage}% successful")
        else:
            rollback()
            break
```

## Model Registry

### MLflow Model Registry

```python
# Register model
mlflow.register_model(
    "runs:/<run_id>/model",
    "production_model"
)

# Load from registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_version = client.get_latest_version("production_model")
model = mlflow.sklearn.load_model(
    f"models:/{model_version.name}/{model_version.version}"
)
```

### Custom Model Registry

```python
class ModelRegistry:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.models = {}

    def register_model(self, name, model, metadata):
        version = self._get_next_version(name)
        self.models[f"{name}:{version}"] = {
            'model': model,
            'metadata': metadata,
            'status': 'staging'
        }

    def promote_model(self, name, version, env='production'):
        key = f"{name}:{version}"
        if key in self.models:
            self.models[key]['status'] = env
            self.models[key]['promoted_at'] = datetime.now()
```

## Feature Engineering

### Automated Feature Engineering

```python
# Feature engineering pipeline
import pandas as pd
import numpy as np

def engineer_features(df):
    # Date features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek

    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['target', 'id']:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log(df[col] + 1)

    # Interaction features
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['feature1_x_feature2'] = df['feature1'] * df['feature2']

    return df
```

### Feature Selection

```python
from sklearn.feature_selection import RFE, SelectKBest, f_classif

def select_features(X, y, n_features=10):
    # Recursive feature elimination
    estimator = RandomForestClassifier()
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    return selected_features.tolist()
```

## Model Validation

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def cross_validate_model(model, X, y, cv=5):
    # Standard cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Stratified for imbalanced data
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    return {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'stratified_mean': stratified_scores.mean()
    }
```

### A/B Testing for Models

```python
class ModelABTest:
    def __init__(self):
        self.user_assignments = {}

    def assign_user(self, user_id, experiment_name):
        hash_val = hash(f"{user_id}_{experiment_name}") % 100

        # 50/50 split
        if hash_val < 50:
            return 'control'
        else:
            return 'treatment'

    def get_model_for_user(self, user_id, experiment_name):
        assignment = self.assign_user(user_id, experiment_name)

        if assignment == 'control':
            return self.control_model
        else:
            return self.treatment_model
```

## Performance Optimization

### Model Compression

```python
# Model pruning
def prune_model(model, sparsity=0.5):
    import torch

    if hasattr(model, 'parameters'):
        for param in model.parameters():
            if param.dim() > 1:
                mask = torch.rand_like(param) > sparsity
                param.data *= mask

    return model

# Quantization
def quantize_model(model):
    import torch.quantization as quantization

    model.eval()
    quantized_model = quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    return quantized_model
```

### Inference Optimization

```python
class BatchedModel:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def predict(self, features_list):
        results = []
        for i in range(0, len(features_list), self.batch_size):
            batch = features_list[i:i + self.batch_size]
            batch_predictions = self.model.predict(batch)
            results.extend(batch_predictions)
        return results
```

## Security and Compliance

### Model Security

```python
# Model integrity verification
import hashlib
import pickle

def verify_model_integrity(model, expected_hash):
    model_bytes = pickle.dumps(model)
    actual_hash = hashlib.sha256(model_bytes).hexdigest()
    return actual_hash == expected_hash

def secure_model_loading(model_path, expected_hash):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not verify_model_integrity(model, expected_hash):
        raise SecurityError("Model integrity verification failed")

    return model
```

### Access Control

```python
def require_permission(permission):
    def decorator(func):
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user.has_permission(permission):
                raise PermissionError(f"User lacks permission: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission('model:read')
def get_model_info(model_id):
    return load_model_info(model_id)
```

## Troubleshooting

### Common Issues and Solutions

#### Model Drift Detection

```python
# Detect model drift using statistical tests
from scipy import stats

def detect_drift(train_data, new_data, alpha=0.05):
    drift_detected = False
    drift_details = {}

    for column in train_data.select_dtypes(include=[np.number]).columns:
        if column in new_data.columns:
            statistic, p_value = stats.ks_2samp(
                train_data[column], new_data[column]
            )

            if p_value < alpha:
                drift_detected = True
                drift_details[column] = {
                    'p_value': p_value,
                    'statistic': statistic
                }

    return drift_detected, drift_details
```

#### Performance Issues

```python
# Optimize model inference
def optimize_inference(model):
    # Enable Just-In-Time compilation for scikit-learn
    import numba
    from numba import jit

    @jit(nopython=True)
    def fast_predict(features):
        # Fast prediction logic
        return model.predict(features)

    return fast_predict

# Memory optimization
def reduce_memory_usage(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype in ['int64', 'float64']:
            # Downcast numerical types
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')

    return df
```

### Error Handling Patterns

```python
# Robust prediction function
def robust_predict(model, features, fallback_model=None):
    try:
        # Try primary model
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()

        if confidence < 0.5:  # Low confidence threshold
            if fallback_model:
                prediction = fallback_model.predict([features])[0]
                return prediction, 'fallback'
            else:
                return prediction, 'low_confidence'

        return prediction, 'success'

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if fallback_model:
            try:
                return fallback_model.predict([features])[0], 'error_fallback'
            except Exception as fallback_error:
                logger.error(f"Fallback prediction failed: {fallback_error}")
                raise
        else:
            raise
```

## Configuration Management

### Environment Configuration

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Model paths
    MODEL_PATH: str = os.getenv('MODEL_PATH', 'models/production_model.pkl')

    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///mlops.db')

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

    # Monitoring
    DRIFT_THRESHOLD: float = float(os.getenv('DRIFT_THRESHOLD', '0.05'))
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.6'))

    # Performance
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '32'))
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '4'))

config = Config()
```

### Logging Configuration

```python
# logging_config.py
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/mlops_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

# Use in main application
setup_logging()
logger = logging.getLogger(__name__)
```

## Quick Commands Reference

### Model Operations

```bash
# Train model
python src/models/train_model.py

# Validate model
python -c "import joblib; m = joblib.load('model.pkl'); print('Model loaded')"

# Deploy model
python src/deployment/deploy_model.py

# Monitor model
python monitoring/model_monitor.py
```

### Docker Commands

```bash
# Build image
docker build -t ml-model .

# Run container
docker run -p 8000:8000 ml-model

# Check logs
docker logs ml-container

# Scale deployment
docker-compose up --scale ml-api=3
```

### Kubernetes Commands

```bash
# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods

# Scale
kubectl scale deployment ml-model --replicas=5

# View logs
kubectl logs -l app=ml-model
```

### Airflow Commands

```bash
# Run DAG
airflow dags trigger ml_pipeline

# Check status
airflow dags list-runs -d ml_pipeline

# Test task
airflow tasks test ml_pipeline train_model 2023-01-01
```

### MLflow Commands

```bash
# Start MLflow UI
mlflow ui

# List experiments
mlflow experiments list

# Register model
mlflow models register -m runs:/<run_id>/model -n production_model

# Serve model
mlflow models serve -m models:/production_model/1
```

This cheatsheet provides quick reference for the most commonly used MLOps patterns, commands, and best practices. Keep it handy for daily development work!
