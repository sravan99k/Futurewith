# Cloud ML Infrastructure Cheatsheet

## AWS SageMaker

### Quick Commands

```bash
# Install SageMaker SDK
pip install sagemaker

# Initialize SageMaker session
import sagemaker
session = sagemaker.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()

# Upload data to S3
s3_client.upload_file('data.csv', 'your-bucket', 'data/data.csv')

# Create training job
estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1'
)
estimator.fit({'train': 's3://your-bucket/data/'})

# Deploy model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Delete endpoint
predictor.delete_endpoint()
```

### Essential SageMaker Classes

```python
# Core Classes
SKLearn                  # Scikit-learn estimator
TensorFlow               # TensorFlow estimator
PyTorch                  # PyTorch estimator
XGBoost                  # XGBoost estimator
RandomCutForest          # Anomaly detection
FactorizationMachines    # Recommendation systems

# Model Types
LinearLearner            # Linear regression/classification
KNN                      # K-nearest neighbors
PCA                      # Principal component analysis
KMeans                   # Clustering
```

### SageMaker Pipeline Steps

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Create pipeline
pipeline = Pipeline(
    name="ml-pipeline",
    parameters=[...],
    steps=[processing_step, training_step]
)

# Execute pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

## Google Cloud Platform (GCP)

### Quick Setup

```bash
# Install GCP SDK
pip install google-cloud-aiplatform

# Authenticate
gcloud auth application-default login
gcloud config set project your-project-id

# Initialize Vertex AI
from google.cloud import aiplatform
aiplatform.init(project="your-project", location="us-central1")
```

### Vertex AI Core Functions

```python
# Create dataset
dataset = aiplatform.Dataset.create(
    display_name="ml-dataset",
    labels={"team": "engineering"}
)

# AutoML training
training_job = aiplatform.AutoMLTabularTrainingJob(
    display_name="automl-job",
    optimization_prediction_type="classification"
)

model = training_job.run(
    dataset=dataset,
    target_column="target",
    budget_milli_node_hours=1000
)

# Custom training
job = aiplatform.CustomTrainingJob(
    display_name="custom-job",
    container_uri="gcr.io/project/image:tag"
)

model = job.run(
    dataset=dataset,
    replica_count=1,
    machine_type="n1-standard-4"
)

# Deploy model
endpoint = model.deploy(
    deployed_model_id=None,
    machine_type="n1-standard-2"
)

# Make prediction
predictions = endpoint.predict(instances=instances)
```

### GCP ML Services

```python
# Natural Language API
from google.cloud import language_v2
client = language_v2.LanguageServiceClient()
document = {"content": "Sample text", "type": "PLAIN_TEXT"}
response = client.analyze_sentiment(document=document)

# Vision API
from google.cloud import vision
client = vision.ImageAnnotatorClient()
response = client.label_detection(image=image)

# Translation API
from google.cloud import translate_v2
client = translate_v2.Client()
result = client.translate('Hello', target_language='es')
```

## Azure ML

### Quick Setup

```bash
# Install Azure ML SDK
pip install azureml-sdk

# Login to Azure
az login
az account set --subscription your-subscription-id

# Create workspace config
from azureml.core import Workspace
ws = Workspace.create(
    name="ml-workspace",
    subscription_id="your-subscription",
    resource_group="ml-rg",
    create_resource_group=True,
    location="eastus"
)
```

### Azure ML Core Components

```python
# Dataset registration
dataset = Dataset.register_pandas_dataframe(
    workspace=ws,
    name="ml-dataset",
    dataframe=df
)

# Compute targets
compute_target = ComputeTarget.create(
    workspace=ws,
    name="cpu-cluster",
    type="AmlCompute",
    size="Standard_DS2_v2",
    min_nodes=0,
    max_instances=2
)

# Experiments
experiment = Experiment(workspace=ws, name="ml-experiment")

# Model registration
model = Model.register(
    workspace=ws,
    model_name="ml-model",
    model_path="model.pkl"
)

# Model deployment
service = Model.deploy(
    workspace=ws,
    name="ml-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
```

## Kubernetes for ML

### Essential Commands

```bash
# Create namespace
kubectl create namespace ml-inference

# Apply deployment
kubectl apply -f ml-deployment.yaml

# Scale deployment
kubectl scale deployment ml-service --replicas=3

# Check status
kubectl get pods -n ml-inference
kubectl describe pod ml-service-xxx -n ml-inference

# Port forward for testing
kubectl port-forward service/ml-service 8080:80 -n ml-inference

# Get logs
kubectl logs deployment/ml-service -n ml-inference
```

### Kubernetes ML Resources

```yaml
# Deployment YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
        - name: ml-service
          image: ml-service:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
spec:
  selector:
    app: ml-service
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer
```

## Kubeflow Pipelines

### Component Definition

```python
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Input, Output, Component

@dsl.component
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    # Component logic here
    pass

@dsl.component
def train_model(
    train_data: Input[Dataset],
    model_output: Output[Model]
):
    # Training logic here
    pass

@dsl.component
def evaluate_model(
    model_input: Input[Model],
    test_data: Input[Dataset],
    metrics_output: Output[Metrics]
):
    # Evaluation logic here
    pass
```

### Pipeline Definition

```python
@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    preprocess_step = preprocess_data()
    train_step = train_model(train_data=preprocess_step.outputs['output_data'])
    eval_step = evaluate_model(
        model_input=train_step.outputs['model_output'],
        test_data=preprocess_step.outputs['test_data']
    )

# Compile pipeline
compiler.compile(
    pipeline_func=ml_pipeline,
    package_path="pipeline.yaml"
)
```

## MLflow

### Quick Setup

```python
import mlflow
import mlflow.sklearn

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("ml-experiment")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 100)

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry Operations

```python
# Register model
mlflow.register_model("runs:/run_id/model", "model_name")

# Get latest version
from mlflow.tracking import MlflowClient
client = MlflowClient()
latest_version = client.get_latest_version("model_name")

# Transition stage
client.transition_model_version_stage(
    name="model_name",
    version=latest_version.version,
    stage="Production"
)

# Load model for prediction
model = mlflow.sklearn.load_model("models:/model_name/production")
```

## Prometheus Monitoring

### Metrics Setup

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

# Start metrics server
start_http_server(8000)

# Record metrics
prediction_counter.inc()
prediction_latency.observe(0.1)
model_accuracy.set(0.85)
```

### Query Examples

```promql
# Prediction rate
rate(predictions_total[5m])

# Average latency
rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])

# Model accuracy
model_accuracy

# CPU usage
rate(container_cpu_usage_seconds_total{pod=~"ml-service.*"}[5m])
```

## Docker for ML Models

### Dockerfile Template

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### Docker Commands

```bash
# Build image
docker build -t ml-service:latest .

# Run container
docker run -p 8080:8080 ml-service:latest

# Check logs
docker logs -f ml-service

# Stop container
docker stop ml-service

# Remove container
docker rm ml-service
```

## Cost Optimization

### AWS Cost Monitoring

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')

# Get cost data
response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': '2024-01-01',
        'End': '2024-01-31'
    },
    Granularity='DAILY',
    Metrics=['BlendedCost'],
    GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
)
```

### GCP Cost Monitoring

```python
from google.cloud import billing_v1

# Get cost data
client = billing_v1.CloudBillingClient()
request = billing_v1.ListSkusRequest(
    parent=f"projects/project-id/billingAccounts/billing-account-id",
    time_range_start="2024-01-01T00:00:00Z",
    time_range_end="2024-01-31T23:59:59Z"
)
response = client.list_skus(request=request)
```

## Auto-scaling Configuration

### AWS Auto Scaling

```python
import boto3

autoscaling = boto3.client('autoscaling')

# Create auto scaling group
response = autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='ml-autoscaling-group',
    MinSize=0,
    MaxSize=10,
    DesiredCapacity=1,
    LaunchConfigurationName='ml-launch-config',
    TargetGroupARNs=['arn:aws:elasticloadbalancing:region:account:targetgroup/ml-tg/123456789']
)
```

### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Security Best Practices

### Container Security

```dockerfile
# Use minimal base image
FROM python:3.8-slim

# Create non-root user
RUN adduser --disabled-password --gecos '' mluser
USER mluser

# Use read-only filesystem
RUN chmod 444 /etc/passwd

# Set security context
SECURITY CONTEXT SETUID:SetGID:SetUID
```

### IAM Best Practices

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:CreateModel"
      ],
      "Resource": "arn:aws:sagemaker:*:*:training-job/*",
      "Condition": {
        "StringEquals": {
          "sagemaker:TrainingJobSourceArn": "arn:aws:sagemaker:*:*:processing-job/*"
        }
      }
    }
  ]
}
```

## Environment Variables

### Essential Variables

```bash
# AWS
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# GCP
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
export GOOGLE_CLOUD_PROJECT=your-project-id

# Azure
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_TENANT_ID=your-tenant-id
export AZURE_CLIENT_ID=your-client-id

# MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
```

## Troubleshooting Commands

### AWS SageMaker

```bash
# Check training job status
aws sagemaker describe-training-job --training-job-name job-name

# List training jobs
aws sagemaker list-training-jobs --max-items 10

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name endpoint-name
```

### Kubernetes

```bash
# Check resource usage
kubectl top pods
kubectl top nodes

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Describe resource
kubectl describe pod pod-name
kubectl describe deployment deployment-name

# Check logs
kubectl logs pod-name
kubectl logs -f deployment/deployment-name
```

### Docker

```bash
# Check container status
docker ps -a

# Check resource usage
docker stats

# Check container logs
docker logs container-name

# Debug container
docker exec -it container-name bash
```

## Common File Paths

### Configurations

```
/etc/aws/credentials          # AWS credentials
~/.aws/config                # AWS configuration
~/.gcloud/configurations/    # GCP configurations
~/.azure/clouds.config       # Azure configuration
~/.kube/config              # Kubernetes configuration
~/.mlflow/                  # MLflow configuration
```

### Data Directories

```
/data                       # Local data
/s3://bucket/data           # S3 data
/data/data.csv              # CSV data
/data/model.pkl             # Model files
/models/                    # Saved models
/logs/                      # Application logs
/metrics/                   # Metrics files
```

## Performance Optimization

### Memory Optimization

```python
# Data loading with generators
def data_generator(file_path, batch_size=32):
    with open(file_path, 'r') as file:
        while True:
            batch = []
            for _ in range(batch_size):
                line = file.readline()
                if not line:
                    break
                batch.append(parse_line(line))
            if batch:
                yield np.array(batch)
```

### GPU Configuration

```python
# PyTorch GPU setup
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# TensorFlow GPU setup
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass
```

## Testing and Validation

### Model Testing

```python
# Unit tests
def test_model_prediction():
    model = load_model('test_model.pkl')
    test_input = np.array([[1, 2, 3, 4, 5]])
    prediction = model.predict(test_input)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]  # Binary classification

# Integration tests
def test_api_endpoint():
    response = client.post('/predict', json={'data': [[1, 2, 3, 4, 5]]})
    assert response.status_code == 200
    assert 'predictions' in response.json()
```

## Quick Reference Tables

### Instance Types

| Service | Small           | Medium          | Large           |
| ------- | --------------- | --------------- | --------------- |
| AWS     | ml.t3.medium    | ml.m5.large     | ml.m5.xlarge    |
| GCP     | n1-standard-2   | n1-standard-4   | n1-standard-8   |
| Azure   | Standard_D2s_v3 | Standard_D4s_v3 | Standard_D8s_v3 |

### Resource Limits

| Resource | Small   | Medium  | Large   |
| -------- | ------- | ------- | ------- |
| CPU      | 2 cores | 4 cores | 8 cores |
| Memory   | 4GB     | 8GB     | 16GB    |
| Storage  | 20GB    | 50GB    | 100GB   |

### Key Commands Summary

```bash
# AWS
aws sagemaker create-training-job --cli-input-json file://training-job.json
aws sagemaker create-model --model-name ml-model --primary-container Image=container-uri

# GCP
gcloud ai models upload --region=us-central1 --display-name=ml-model --container-uri=container-uri
gcloud ai endpoints create --region=us-central1 --display-name=ml-endpoint

# Azure
az ml model create --name ml-model --path model.pkl --type mlflow_model
az ml online-endpoint create --name ml-endpoint --auth-mode key

# Kubernetes
kubectl apply -f deployment.yaml
kubectl get pods,svc,deploy
kubectl scale deployment ml-service --replicas=3

# Docker
docker build -t ml-service . && docker run -p 8080:8080 ml-service
docker images && docker ps && docker logs container-name

# MLflow
mlflow server --host localhost --port 5000
mlflow run experiments/experiment-name
mlflow models serve -m models:/model-name/production -p 8080
```

This cheatsheet provides quick reference for common cloud ML infrastructure tasks across AWS, GCP, Azure, and Kubernetes environments.
