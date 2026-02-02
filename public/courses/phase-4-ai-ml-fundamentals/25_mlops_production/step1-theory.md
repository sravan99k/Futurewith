# MLOps & Production ML - Theory

## Table of Contents

1. [Introduction to MLOps](#introduction-to-mlops)
2. [MLOps Lifecycle](#mlops-lifecycle)
3. [Model Development](#model-development)
4. [Model Deployment](#model-deployment)
5. [Model Monitoring](#model-monitoring)
6. [ML Pipeline Automation](#ml-pipeline-automation)
7. [Model Versioning and Registry](#model-versioning-and-registry)
8. [Infrastructure and Orchestration](#infrastructure-and-orchestration)
9. [Model Serving Platforms](#model-serving-platforms)
10. [Continuous Integration/Continuous Deployment](#continuous-integrationcontinuous-deployment)
11. [Data and Model Validation](#data-and-model-validation)
12. [Performance Optimization](#performance-optimization)
13. [MLOps Tools and Platforms](#mlops-tools-and-platforms)
14. [Security and Compliance](#security-and-compliance)
15. [Best Practices](#best-practices)

## Introduction to MLOps

### What is MLOps?

MLOps (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows. It encompasses the entire lifecycle of ML models from development, training, deployment, and monitoring to maintenance.

### Key Objectives

- **Reliability**: Ensure ML systems are dependable and perform consistently
- **Scalability**: Handle growing data volumes and model complexity
- **Reproducibility**: Replicate results across environments
- **Automation**: Minimize manual intervention in ML workflows
- **Collaboration**: Enable seamless teamwork between data scientists and engineers

### MLOps vs DevOps

| Aspect         | DevOps                 | MLOps                             |
| -------------- | ---------------------- | --------------------------------- |
| Primary Focus  | Software deployment    | ML model deployment               |
| Key Components | Code, Infrastructure   | Data, Models, Code                |
| Testing        | Unit, Integration      | Data validation, Model validation |
| Monitoring     | System metrics         | Model performance, Data drift     |
| Deployment     | Stateless applications | Stateful ML models                |

### Benefits of MLOps

- **Faster Model Deployment**: Automated pipelines reduce time to production
- **Improved Model Quality**: Continuous monitoring and validation
- **Better Resource Utilization**: Efficient resource allocation and scaling
- **Enhanced Compliance**: Audit trails and governance
- **Reduced Technical Debt**: Better organization and documentation

## MLOps Lifecycle

### Phase 1: Data Management

- **Data Collection**: Gathering data from various sources
- **Data Validation**: Ensuring data quality and consistency
- **Feature Engineering**: Creating and maintaining features
- **Data Versioning**: Tracking changes to datasets

### Phase 2: Model Development

- **Experimentation**: Testing different algorithms and parameters
- **Model Training**: Training models on prepared data
- **Model Evaluation**: Assessing model performance
- **Hyperparameter Tuning**: Optimizing model parameters

### Phase 3: Model Validation

- **Cross-validation**: Ensuring model generalizability
- **A/B Testing**: Comparing model versions
- **Performance Benchmarking**: Comparing against baselines
- **Bias Detection**: Identifying unfair model behavior

### Phase 4: Model Deployment

- **Packaging**: Preparing models for production
- **Deployment Strategies**: Blue-green, canary, rolling deployments
- **Service Integration**: Connecting with existing systems
- **Load Testing**: Ensuring performance under load

### Phase 5: Monitoring and Maintenance

- **Performance Monitoring**: Tracking model accuracy over time
- **Data Drift Detection**: Identifying changes in input data
- **Model Retraining**: Periodic model updates
- **Incident Response**: Handling production issues

## Model Development

### Development Environment Setup

```python
# Virtual environment setup
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas scikit-learn mlflow wandb
pip install jupyter ipython
```

### Project Structure Best Practices

```
mlops_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── validate_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate_model.py
│   └── visualization/
├── tests/
├── models/
│   ├── trained/
│   └── archived/
├── notebooks/
├── .env
├── requirements.txt
├── setup.py
└── README.md
```

### Code Organization Principles

- **Modularity**: Separate components for data, features, models
- **Configuration Management**: Use config files for parameters
- **Logging**: Comprehensive logging throughout the pipeline
- **Error Handling**: Graceful failure and recovery mechanisms

### Development Workflow

1. **Feature Development**: Write features in isolation
2. **Model Training**: Train models with proper validation
3. **Model Evaluation**: Assess performance on multiple metrics
4. **Documentation**: Document code, experiments, and results
5. **Code Review**: Peer review for quality assurance

## Model Deployment

### Deployment Strategies

#### 1. Batch Prediction

- **Description**: Generate predictions periodically (daily, hourly)
- **Use Cases**: Recommendation systems, churn prediction
- **Advantages**: Simple, cost-effective, no real-time constraints
- **Implementation**: Scheduled jobs, data pipelines

```python
# Batch prediction example
import pandas as pd
from sklearn.model import Joblib

def batch_predict():
    # Load model
    model = Joblib.load('models/production_model.pkl')

    # Load data
    data = pd.read_csv('data/batch_predictions.csv')

    # Generate predictions
    predictions = model.predict(data)

    # Save results
    pd.DataFrame({'predictions': predictions}).to_csv(
        'output/predictions.csv', index=False
    )
```

#### 2. Real-time Serving

- **Description**: Serve predictions via API endpoints
- **Use Cases**: Fraud detection, recommendation engines
- **Advantages**: Immediate responses, dynamic personalization
- **Implementation**: REST APIs, gRPC services

```python
# Real-time serving with Flask
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    return jsonify({
        'prediction': int(prediction),
        'confidence': float(probability)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### 3. Stream Processing

- **Description**: Process data streams and generate predictions
- **Use Cases**: Real-time anomaly detection, IoT analytics
- **Advantages**: Low latency, handles continuous data
- **Implementation**: Kafka, Apache Flink, Spark Streaming

```python
# Stream processing with Kafka
from kafka import KafkaConsumer, KafkaProducer
import json

consumer = KafkaConsumer(
    'input_stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for message in consumer:
    data = message.value
    prediction = model.predict([data['features']])[0]

    result = {
        'prediction': prediction,
        'confidence': get_confidence(data['features']),
        'timestamp': data['timestamp']
    }

    producer.send('predictions', result)
```

### Deployment Platforms

#### Cloud Platforms

- **AWS SageMaker**: End-to-end ML platform
- **Google AI Platform**: Managed ML services
- **Azure ML**: Integrated ML and MLOps
- **Heroku**: Simple deployment for small applications

#### Containerization

- **Docker**: Package models and dependencies
- **Kubernetes**: Orchestrate containerized applications
- **AWS ECS/EKS**: Managed container services

```dockerfile
# Dockerfile for ML model
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "serve.py"]
```

#### Serverless Deployment

- **AWS Lambda**: Function-as-a-Service
- **Google Cloud Functions**: Event-driven compute
- **Azure Functions**: Serverless compute platform

## Model Monitoring

### Key Metrics for Monitoring

#### 1. Performance Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

#### 2. Business Metrics

- **Conversion Rate**: For recommendation systems
- **Customer Retention**: For churn prediction models
- **Revenue Impact**: Financial outcomes of predictions
- **User Engagement**: Click-through rates, time spent

#### 3. System Metrics

- **Latency**: Time to generate predictions
- **Throughput**: Predictions per second
- **Error Rate**: Failed predictions / Total predictions
- **Resource Usage**: CPU, memory, disk utilization

### Data Drift Detection

#### Types of Data Drift

- **Feature Drift**: Changes in feature distributions
- **Concept Drift**: Changes in relationship between features and target
- **Temporal Drift**: Changes over time
- **Population Drift**: Changes in population characteristics

#### Detection Methods

```python
# Statistical tests for drift detection
from scipy import stats
import numpy as np

def detect_feature_drift(feature1, feature2, threshold=0.05):
    """
    Detect drift using Kolmogorov-Smirnov test
    """
    statistic, p_value = stats.ks_2samp(feature1, feature2)
    return {
        'drift_detected': p_value < threshold,
        'p_value': p_value,
        'statistic': statistic
    }

def detect_distribution_shift(data1, data2):
    """
    Compare multiple features for distribution shifts
    """
    drift_results = {}
    for column in data1.columns:
        if data1[column].dtype in ['int64', 'float64']:
            drift_results[column] = detect_feature_drift(
                data1[column], data2[column]
            )
    return drift_results
```

### Monitoring Implementation

#### Traditional Monitoring

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting
- **ELK Stack**: Log aggregation and analysis
- **Datadog**: Application performance monitoring

#### ML-Specific Monitoring

```python
# Model performance monitoring
class ModelMonitor:
    def __init__(self, model, reference_data):
        self.model = model
        self.reference_data = reference_data
        self.performance_history = []

    def log_prediction(self, features, prediction, actual=None):
        """Log prediction for monitoring"""
        entry = {
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        self.prediction_log.append(entry)

    def calculate_metrics(self, time_window='1d'):
        """Calculate performance metrics for time window"""
        window_data = self.get_predictions_in_window(time_window)

        if len(window_data) > 0 and all('actual' in d for d in window_data):
            actual = [d['actual'] for d in window_data]
            predictions = [d['prediction'] for d in window_data]

            return {
                'accuracy': accuracy_score(actual, predictions),
                'precision': precision_score(actual, predictions, average='weighted'),
                'recall': recall_score(actual, predictions, average='weighted'),
                'f1_score': f1_score(actual, predictions, average='weighted')
            }
        return None
```

## ML Pipeline Automation

### Pipeline Components

- **Data Ingestion**: Automated data collection
- **Data Validation**: Quality checks and alerts
- **Feature Engineering**: Automated feature creation
- **Model Training**: Scheduled or triggered training
- **Model Validation**: Performance and bias checks
- **Deployment**: Automated model deployment
- **Monitoring**: Continuous performance tracking

### Workflow Orchestration

#### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Automated ML pipeline',
    schedule_interval='@daily',
    catchup=False
)

def extract_data():
    # Data extraction logic
    pass

def validate_data():
    # Data validation logic
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

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
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

extract_task >> validate_task >> train_task >> deploy_task
```

#### Kubeflow Pipelines

```python
import kfp
from kfp import dsl
from kfp.components import func_to_container_op

# Component definitions
@func_to_container_op
def load_data_op():
    # Data loading component
    pass

@func_to_container_op
def train_model_op(learning_rate: float, epochs: int):
    # Model training component
    pass

@dsl.pipeline(
    name='ML Pipeline',
    description='An example machine learning pipeline'
)
def ml_pipeline(learning_rate: float = 0.01, epochs: int = 10):
    # Define pipeline steps
    load_data = load_data_op()
    train_model = train_model_op(
        learning_rate, epochs
    ).after(load_data)
```

### CI/CD for ML

#### GitHub Actions Workflow

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline CI/CD

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
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          python -m pytest tests/

      - name: Data validation
        run: |
          python -m pytest tests/test_data_validation.py

      - name: Model testing
        run: |
          python -m pytest tests/test_model.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v2

      - name: Build and push Docker image
        run: |
          docker build -t ml-model:${{ github.sha }} .
          docker push ml-model:${{ github.sha }}

      - name: Deploy to staging
        run: |
          # Deployment commands
```

## Model Versioning and Registry

### Version Control Strategies

#### Model Versioning

- **Semantic Versioning**: MAJOR.MINOR.PATCH
- **Timestamp-based**: YYYYMMDD_HHMMSS
- **Git-based**: Commit hash + metadata
- **Experiment-based**: UUID for each training run

#### MLflow Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Start MLflow experiment
mlflow.set_experiment("customer_churn_prediction")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("requirements.txt")
```

#### DVC (Data Version Control)

```python
# Track data and models
dvc add data/raw/dataset.csv
dvc add models/trained/model.pkl

# Version control
git add data/raw/dataset.csv.dvc models/trained/model.pkl.dvc
git commit -m "Add initial dataset and trained model"

# Reproduce experiments
dvc repro
```

### Model Registry

#### MLflow Model Registry

```python
# Register model
mlflow.register_model(
    "runs:/<run_id>/model",
    "customer_churn_model"
)

# Load model from registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_version = client.get_latest_version("customer_churn_model")
model = mlflow.sklearn.load_model(
    f"models:/{model_version.name}/{model_version.version}"
)
```

#### Custom Model Registry

```python
class ModelRegistry:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.models = {}

    def register_model(self, model_name, model, metadata):
        """Register a new model"""
        version = self._get_next_version(model_name)
        model_info = {
            'version': version,
            'model': model,
            'metadata': metadata,
            'registered_at': datetime.now(),
            'status': 'staging'
        }
        self.models[f"{model_name}:{version}"] = model_info

    def promote_model(self, model_name, version, environment='production'):
        """Promote model to specific environment"""
        model_key = f"{model_name}:{version}"
        if model_key in self.models:
            self.models[model_key]['status'] = environment
            self.models[model_key]['promoted_at'] = datetime.now()

    def get_model(self, model_name, version=None):
        """Retrieve model by name and version"""
        if version is None:
            # Get latest production model
            for key, info in self.models.items():
                if key.startswith(f"{model_name}:") and info['status'] == 'production':
                    return info['model']
        else:
            return self.models.get(f"{model_name}:{version}")['model']
```

## Infrastructure and Orchestration

### Container Orchestration

#### Kubernetes for ML

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
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
            - containerPort: 5000
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          env:
            - name: MODEL_PATH
              value: "/models/production_model.pkl"
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
      targetPort: 5000
  type: LoadBalancer
```

#### Helm for ML Applications

```yaml
# values.yaml
replicaCount: 3

image:
  repository: ml-model
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 5000

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
  targetCPUUtilizationPercentage: 70
```

### Resource Management

#### GPU Management

```yaml
# GPU-enabled deployment
apiVersion: v1
kind: Pod
metadata:
  name: ml-model-gpu
spec:
  containers:
    - name: ml-model
      image: ml-model:gpu
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
```

#### Auto-scaling

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
  maxReplicas: 10
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

## Model Serving Platforms

### Model Serving Options

#### 1. Self-hosted Solutions

- **TensorFlow Serving**: Optimized for TensorFlow models
- **TorchServe**: PyTorch model serving framework
- **KServe**: Kubernetes-native model serving
- **Seldon Core**: MLOps deployment platform

#### 2. Cloud Services

- **AWS SageMaker Endpoints**: Managed model hosting
- **Google AI Platform Prediction**: Scalable model serving
- **Azure ML Endpoints**: Serverless model deployment

#### 3. Edge Deployment

- **TensorFlow Lite**: Mobile and edge devices
- **ONNX Runtime**: Cross-platform inference
- **OpenVINO**: Intel hardware optimization

### Serving Architecture Patterns

#### Microservices Pattern

```python
# Model service microservice
class ModelService:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def predict(self, features):
        """Make prediction"""
        return self.model.predict(features)

    def batch_predict(self, batch_features):
        """Batch prediction for efficiency"""
        return self.model.predict(batch_features)

    def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy", "model_loaded": self.model is not None}
```

#### Model Ensemble Pattern

```python
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict(self, features):
        """Weighted ensemble prediction"""
        predictions = [model.predict([features])[0] for model in self.models]
        return np.average(predictions, weights=self.weights)
```

## Continuous Integration/Continuous Deployment

### CI/CD Pipeline Design

#### Stages of ML CI/CD

1. **Code Integration**: Merge and test code changes
2. **Data Validation**: Check data quality and schema
3. **Model Training**: Train models with new code/data
4. **Model Validation**: Evaluate model performance
5. **Model Deployment**: Deploy to staging/production
6. **Monitoring**: Track performance and alerts

#### Deployment Strategies

##### Blue-Green Deployment

```python
# Blue-green deployment logic
def blue_green_deploy(new_model_version):
    # Deploy to green environment
    deploy_to_environment(new_model_version, 'green')

    # Run smoke tests
    if run_smoke_tests('green'):
        # Switch traffic to green
        switch_traffic('green')

        # Keep blue as backup
        maintain_backup('blue')
    else:
        # Rollback - keep blue active
        cleanup_environment('green')
        raise DeploymentError("Smoke tests failed")
```

##### Canary Deployment

```python
def canary_deploy(model_version, traffic_percentage=10):
    """Deploy to subset of traffic"""
    # Deploy model to canary environment
    deploy_to_environment(model_version, 'canary')

    # Gradually increase traffic
    for percentage in [10, 25, 50, 100]:
        if percentage > traffic_percentage:
            break

        # Monitor performance
        if monitor_canary_performance('canary', percentage):
            print(f"Canary {percentage}% successful")
        else:
            print(f"Canary {percentage}% failed - rolling back")
            rollback()
            break
```

### A/B Testing Framework

```python
class ABTestingFramework:
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}

    def create_experiment(self, name, variants, allocation):
        """Create A/B test experiment"""
        self.experiments[name] = {
            'variants': variants,
            'allocation': allocation,
            'results': {variant: [] for variant in variants}
        }

    def assign_user(self, user_id, experiment_name):
        """Assign user to experiment variant"""
        if experiment_name not in self.experiments:
            return None

        # Consistent assignment based on user ID hash
        hash_val = hash(f"{user_id}_{experiment_name}") % 100
        cumulative = 0

        for variant, allocation in self.experiments[experiment_name]['allocation'].items():
            cumulative += allocation
            if hash_val < cumulative:
                self.user_assignments[f"{user_id}_{experiment_name}"] = variant
                return variant

        return list(self.experiments[experiment_name]['allocation'].keys())[0]

    def get_model(self, user_id, experiment_name):
        """Get model for user based on A/B test"""
        variant = self.assign_user(user_id, experiment_name)
        return self.experiments[experiment_name]['variants'][variant]
```

## Data and Model Validation

### Data Validation

#### Schema Validation

```python
from great_expectations import validate

# Define expectations
suite = context.create_expectation_suite("data_suite")
suite.expect_column_to_exist("customer_id")
suite.expect_column_values_to_not_be_null("email")
suite.expect_column_values_to_match_regex("email", r".+@.+\..+")
suite.expect_column_values_to_be_between("age", 0, 120)

# Validate data
result = validate(df, expectation_suite=suite)
```

#### Statistical Validation

```python
def validate_data_distribution(train_data, new_data):
    """Validate distribution similarity"""
    validation_results = {}

    for column in train_data.select_dtypes(include=[np.number]).columns:
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(
            train_data[column], new_data[column]
        )

        validation_results[column] = {
            'ks_statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }

    return validation_results
```

### Model Validation

#### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer

def comprehensive_model_validation(model, X, y):
    """Perform comprehensive model validation"""

    # Cross-validation
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='accuracy'
    )

    # Stratified K-Fold for imbalanced data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(
        model, X, y, cv=skf, scoring='accuracy'
    )

    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'stratified_mean': stratified_scores.mean(),
        'stratified_std': stratified_scores.std()
    }
```

#### Bias Detection

```python
def detect_model_bias(model, X, y, sensitive_attribute):
    """Detect bias in model predictions"""

    # Group analysis
    groups = X[sensitive_attribute].unique()
    bias_metrics = {}

    for group in groups:
        group_mask = X[sensitive_attribute] == group
        X_group = X[group_mask]
        y_group = y[group_mask]

        predictions = model.predict(X_group)

        bias_metrics[group] = {
            'accuracy': accuracy_score(y_group, predictions),
            'precision': precision_score(y_group, predictions, average='weighted'),
            'recall': recall_score(y_group, predictions, average='weighted')
        }

    # Calculate fairness metrics
    fairness_scores = calculate_fairness_metrics(bias_metrics)

    return bias_metrics, fairness_scores
```

## Performance Optimization

### Model Optimization

#### Model Compression

```python
# Model pruning
def prune_model(model, sparsity=0.5):
    """Prune model weights to reduce size"""
    import torch

    if hasattr(model, 'parameters'):
        for param in model.parameters():
            if param.dim() > 1:  # Only prune weights, not biases
                mask = torch.rand_like(param) > sparsity
                param.data *= mask

    return model

# Quantization
def quantize_model(model):
    """Quantize model to reduce precision"""
    import torch.quantization as quantization

    model.eval()
    quantized_model = quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    return quantized_model
```

#### Feature Selection

```python
from sklearn.feature_selection import RFE, SelectKBest, f_classif

def select_features(X, y, n_features=10):
    """Select most important features"""

    # Recursive feature elimination
    estimator = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]

    return selected_features.tolist()
```

### Inference Optimization

#### Batch Processing

```python
class BatchedModel:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def predict(self, features_list):
        """Process predictions in batches"""
        results = []

        for i in range(0, len(features_list), self.batch_size):
            batch = features_list[i:i + self.batch_size]
            batch_predictions = self.model.predict(batch)
            results.extend(batch_predictions)

        return results
```

#### Caching

```python
from functools import lru_cache

class CachedModel:
    def __init__(self, model):
        self.model = model

    @lru_cache(maxsize=1000)
    def predict(self, features_tuple):
        """Cached predictions for identical inputs"""
        features = np.array(features_tuple)
        return self.model.predict([features])[0]
```

## MLOps Tools and Platforms

### Open Source Tools

#### MLflow

- **Tracking**: Experiments, parameters, metrics
- **Projects**: Packaging ML code
- **Models**: Model packaging and deployment
- **Model Registry**: Centralized model management

#### Kubeflow

- **Components**: Reusable ML components
- **Pipelines**: End-to-end ML workflows
- **Serving**: Model serving and management
- **Training**: Distributed training

#### DVC (Data Version Control)

- **Data Versioning**: Track datasets and models
- **Pipelines**: Data and ML pipeline management
- **Metrics**: Track and compare model metrics
- **Reproducibility**: Reproduce experiments

### Commercial Platforms

#### Weights & Biases

- **Experiment Tracking**: Comprehensive experiment management
- **Data Visualization**: Interactive dashboards
- **Collaboration**: Team collaboration features
- **Artifacts**: Model and data artifact management

#### Neptune.ai

- **Experiment Management**: Structured experiment tracking
- **Model Registry**: Centralized model management
- **Monitoring**: Production model monitoring
- **Comparison**: Model comparison and analysis

### Cloud Platforms

#### AWS SageMaker

- **SageMaker Studio**: Integrated ML development
- **Training**: Managed training jobs
- **Deployment**: Model endpoints and batch transform
- **Pipelines**: CI/CD for ML
- **Monitor**: Model monitoring and drift detection

#### Google Cloud AI Platform

- **Notebooks**: Managed Jupyter notebooks
- **Training**: Distributed training jobs
- **Prediction**: Real-time and batch prediction
- **Metadata**: ML metadata tracking

#### Azure ML

- **Designer**: Visual ML pipeline builder
- **Compute**: Managed compute clusters
- **Pipelines**: Automated ML workflows
- **Registry**: Model and dataset registry

## Security and Compliance

### Model Security

#### Model Integrity

```python
import hashlib
import pickle

def verify_model_integrity(model, expected_hash):
    """Verify model hasn't been tampered with"""
    model_bytes = pickle.dumps(model)
    actual_hash = hashlib.sha256(model_bytes).hexdigest()

    return actual_hash == expected_hash

def secure_model_loading(model_path, expected_hash):
    """Securely load model with verification"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not verify_model_integrity(model, expected_hash):
        raise SecurityError("Model integrity verification failed")

    return model
```

#### Adversarial Attack Detection

```python
def detect_adversarial_examples(model, X, threshold=0.1):
    """Detect potential adversarial examples"""

    # Use prediction confidence
    probabilities = model.predict_proba(X)
    max_probs = np.max(probabilities, axis=1)

    # Flag low-confidence predictions
    adversarial_mask = max_probs < threshold

    return adversarial_mask
```

### Data Privacy

#### Differential Privacy

```python
def add_laplace_noise(value, sensitivity, epsilon):
    """Add differential privacy noise"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def private_aggregate(values, epsilon=1.0):
    """Aggregate values with differential privacy"""
    if len(values) == 0:
        return 0

    mean_value = np.mean(values)
    # Assume sensitivity is 1 for bounded values
    private_mean = add_laplace_noise(mean_value, 1.0, epsilon)
    return private_mean
```

#### Secure Multi-party Computation

```python
def secure_model_inference(model, encrypted_features):
    """Perform inference on encrypted features"""
    # Homomorphic encryption for secure inference
    # This is a simplified example

    # Decrypt features (in real implementation, keep encrypted)
    features = decrypt_features(encrypted_features)

    # Perform inference
    prediction = model.predict([features])[0]

    # Return encrypted result
    return encrypt_result(prediction)
```

### Compliance and Governance

#### Audit Logging

```python
import logging
from datetime import datetime

class ModelAuditLogger:
    def __init__(self, log_file):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('model_audit')

    def log_prediction(self, user_id, features, prediction, model_version):
        """Log prediction request and result"""
        self.logger.info({
            'event_type': 'prediction',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'prediction': prediction,
            'features_hash': hash(tuple(features))
        })

    def log_model_access(self, user_id, action, model_version):
        """Log model access events"""
        self.logger.info({
            'event_type': 'model_access',
            'user_id': user_id,
            'action': action,  # 'load', 'deploy', 'delete'
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version
        })
```

#### Access Control

```python
from functools import wraps

def require_permission(permission):
    """Decorator to check user permissions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user.has_permission(permission):
                raise PermissionError(f"User lacks permission: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission('model:read')
def get_model_info(model_id):
    """Get model information (requires read permission)"""
    return load_model_info(model_id)

@require_permission('model:write')
def deploy_model(model_id, environment):
    """Deploy model (requires write permission)"""
    return deploy_to_environment(model_id, environment)
```

## Best Practices

### Development Best Practices

#### Code Quality

- **Linting**: Use flake8, black, isort for code formatting
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Clear docstrings and README files
- **Version Control**: Git for all code and configuration

#### Experiment Management

- **Reproducible Experiments**: Fix random seeds and environments
- **Configuration Management**: Use config files for parameters
- **Hyperparameter Tracking**: Log all parameters and results
- **Model Artifacts**: Save all relevant model artifacts

### Deployment Best Practices

#### Reliability

- **Health Checks**: Implement comprehensive health checks
- **Graceful Degradation**: Handle failures gracefully
- **Rollback Strategy**: Always have a rollback plan
- **Monitoring**: Comprehensive monitoring and alerting

#### Performance

- **Resource Optimization**: Right-size compute resources
- **Caching**: Implement intelligent caching strategies
- **Batching**: Use batching for throughput optimization
- **Load Testing**: Test under realistic load conditions

### Operational Best Practices

#### Maintenance

- **Regular Updates**: Keep dependencies and models updated
- **Security Patches**: Apply security patches promptly
- **Documentation**: Maintain up-to-date documentation
- **Team Training**: Ensure team has necessary skills

#### Communication

- **Incident Response**: Clear escalation procedures
- **Stakeholder Updates**: Regular status updates
- **Knowledge Sharing**: Document lessons learned
- **Cross-team Collaboration**: Foster collaboration between teams

This comprehensive theory guide covers all essential aspects of MLOps and Production ML, providing the foundation needed to build and maintain robust, scalable, and reliable machine learning systems in production environments.
