# Cloud ML Infrastructure Practice Exercises

## AWS SageMaker Practice Exercises

### Exercise 1: SageMaker Notebook Instance Setup

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Setup
region = 'us-east-1'
bucket_name = 'your-s3-bucket-name'
role = get_execution_role()

print(f"Region: {region}")
print(f"Role ARN: {role}")
print(f"Bucket: {bucket_name}")

# Create S3 bucket (if not exists)
try:
    s3_client = boto3.client('s3')
    s3_client.create_bucket(Bucket=bucket_name)
    print(f"Created S3 bucket: {bucket_name}")
except s3_client.exceptions.BucketAlreadyOwnedByYou:
    print(f"S3 bucket {bucket_name} already exists")
```

### Exercise 2: Build and Deploy ML Model with SageMaker

```python
# Generate sample data
np.random.seed(42)
n_samples = 1000

# Create synthetic dataset
X = np.random.randn(n_samples, 5)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Upload data to S3
train_data = pd.DataFrame(X_train)
train_data['label'] = y_train

test_data = pd.DataFrame(X_test)
test_data['label'] = y_test

# Save locally then upload to S3
train_data.to_csv('train.csv', header=False, index=False)
test_data.to_csv('test.csv', header=False, index=False)

s3_client.upload_file('train.csv', bucket_name, 'data/train.csv')
s3_client.upload_file('test.csv', bucket_name, 'data/test.csv')

print("Data uploaded to S3")
```

### Exercise 3: SageMaker Training Job

```python
from sagemaker.sklearn.estimator import SKLearn

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3'
)

# Create training script
training_script = '''
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker arguments
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')

    # Model hyperparameters
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--solver', type=str, default='liblinear')

    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(f"{args.train}/train.csv", header=None)
    test_data = pd.read_csv(f"{args.test}/test.csv", header=None)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Train model
    model = LogisticRegression(max_iter=args.max_iter, solver=args.solver)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, f"{args.model_dir}/model.joblib")
'''

# Save training script
with open('train.py', 'w') as f:
    f.write(training_script)

# Start training job
sklearn_estimator.fit({
    'train': f's3://{bucket_name}/data/train.csv',
    'test': f's3://{bucket_name}/data/test.csv'
})

print("Training job completed")
```

### Exercise 4: SageMaker Model Deployment

```python
# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Test prediction
test_data_np = X_test[:10]  # First 10 test samples
predictions = predictor.predict(test_data_np)

print("Predictions for first 10 test samples:")
print(f"Predicted: {predictions}")
print(f"Actual: {y_test[:10]}")

# Cleanup
predictor.delete_endpoint()
print("Endpoint deleted")
```

## Google Cloud Platform Practice Exercises

### Exercise 5: Vertex AI Setup

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import LabelingJob
import pandas as pd

# Initialize Vertex AI
project_id = "your-project-id"
location = "us-central1"

aiplatform.init(project=project_id, location=location)

print(f"Vertex AI initialized for project: {project_id}")
print(f"Location: {location}")

# Create dataset for training
dataset_display_name = "ml-pipeline-dataset"
dataset = aiplatform.Dataset.create(
    display_name=dataset_display_name,
    labels={"team": "engineering"}
)

print(f"Dataset created: {dataset.display_name}")
```

### Exercise 6: AutoML Tabular Training

```python
# Create AutoML training job
training_job = aiplatform.AutoMLTabularTrainingJob(
    display_name="automl-tabular-job",
    optimization_prediction_type="classification",
    optimization_objective="minimize-log-loss",
    column_specs=[
        {"feature_name": "feature_0", "type": "NUMERICAL"},
        {"feature_name": "feature_1", "type": "NUMERICAL"},
        {"feature_name": "feature_2", "type": "NUMERICAL"},
        {"feature_name": "feature_3", "type": "NUMERICAL"},
        {"feature_name": "feature_4", "type": "NUMERICAL"},
    ]
)

# Run training job
model = training_job.run(
    dataset=dataset,
    target_column="label",
    budget_milli_node_hours=1000,  # 1 compute hour
)

print(f"Model created: {model.display_name}")
```

### Exercise 7: Model Endpoint Deployment

```python
# Deploy model to endpoint
endpoint = model.deploy(
    deployed_model_id=None,  # Let Vertex AI create
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Endpoint created: {endpoint.display_name}")

# Make prediction
test_instances = X_test[:5].tolist()
predictions = endpoint.predict(instances=test_instances)

print("Predictions:")
for i, prediction in enumerate(predictions.predictions):
    print(f"Sample {i+1}: {prediction}")
    print(f"Actual: {y_test[i]}")

# Cleanup
endpoint.undeploy_all()
endpoint.delete()
```

## Azure ML Practice Exercises

### Exercise 8: Azure ML Workspace Setup

```python
from azureml.core import Workspace, Experiment, Dataset, ComputeTarget
from azureml.core.environment import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.model import Model
import azureml.data

# Connect to workspace
subscription_id = "your-subscription-id"
resource_group = "your-resource-group"
workspace_name = "your-workspace-name"

workspace = Workspace(subscription_id, resource_group, workspace_name)

print(f"Connected to workspace: {workspace.name}")

# Create experiment
experiment = Experiment(workspace=workspace, name="ml-pipeline-experiment")
print(f"Experiment created: {experiment.name}")
```

### Exercise 9: Azure ML Dataset Registration

```python
# Create dataset from local data
train_data = pd.DataFrame(X_train)
train_data['label'] = y_train

# Register dataset
dataset = Dataset.register_pandas_dataframe(
    workspace=workspace,
    name="ml-dataset-train",
    dataframe=train_data,
    description="Training dataset for ML model"
)

print(f"Dataset registered: {dataset.name}")

# Create test dataset
test_data = pd.DataFrame(X_test)
test_data['label'] = y_test

test_dataset = Dataset.register_pandas_dataframe(
    workspace=workspace,
    name="ml-dataset-test",
    dataframe=test_data,
    description="Test dataset for ML model"
)

print(f"Test dataset registered: {test_dataset.name}")
```

### Exercise 10: Azure ML Training Job

```python
from azureml.core.script_run_config import ScriptRunConfig
from azureml.pipeline.steps import PythonScriptStep

# Create compute target
compute_target = ComputeTarget.create(
    workspace=workspace,
    name="ml-compute-cluster",
    type="AmlCompute",
    size="Standard_DS2_v2",
    min_nodes=0,
    max_nodes=2
)

compute_target.wait_for_completion(show_output=True)

# Create training script
training_script_content = '''
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from azureml.core import Run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--solver', type=str, default='liblinear')

    args = parser.parse_args()

    # Get current run
    run = Run.get_context()

    # Load data
    train_data = pd.read_csv(f"{args.data_folder}/ml-dataset-train.parquet")
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    # Train model
    model = LogisticRegression(max_iter=args.max_iter, solver=args.solver)
    model.fit(X_train, y_train)

    # Log metrics
    run.log("max_iter", args.max_iter)
    run.log("solver", args.solver)

    # Save model
    import joblib
    joblib.dump(model, "model.pkl")

    # Register model
    run.upload_file(name="outputs/model.pkl", path_or_stream="model.pkl")

    run.complete()
'''

# Save training script
with open('train_azure.py', 'w') as f:
    f.write(training_script_content)

# Create script run config
script_config = ScriptRunConfig(
    source_directory='.',
    script='train_azure.py',
    arguments=['--data-folder', dataset.as_mount(), '--max-iter', 100],
    compute_target=compute_target
)

# Submit experiment
run = experiment.submit(config=script_config)
print(f"Training job submitted: {run.id}")
run.wait_for_completion(show_output=True)

# Register model
model = Model.register(
    workspace=workspace,
    model_name="ml-logistic-regression",
    model_path="outputs/model.pkl",
    description="Logistic regression model trained on synthetic data"
)

print(f"Model registered: {model.name}")
```

## Kubernetes and MLOps Practice

### Exercise 11: Kubernetes ML Service Deployment

```yaml
# ml-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  labels:
    app: ml-service
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
          env:
            - name: MODEL_PATH
              value: "/models/model.pkl"
            - name: PREDICTION_THRESHOLD
              value: "0.5"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
spec:
  selector:
    app: ml-service
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```

### Exercise 12: ML Pipeline with Kubeflow

```python
import kfp
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import Input, Output, Artifact, Dataset

@dsl.component
def prepare_data(output_data: Output[Dataset]):
    """Prepare training data"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    data['label'] = y

    # Save to output
    data.to_parquet(output_data.path, index=False)

    print(f"Generated {len(data)} samples")

@dsl.component
def train_model(
    input_data: Input[Dataset],
    model_output: Output[Artifact]
):
    """Train ML model"""
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import joblib
    import os

    # Load data
    data = pd.read_parquet(input_data.path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(model, f"{model_output.path}/model.pkl")

    # Log accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

@dsl.component
def deploy_model(
    model_input: Input[Artifact],
    service_endpoint: Output[Artifact]
):
    """Deploy model to production"""
    import subprocess
    import os

    # Save endpoint URL (placeholder)
    with open(service_endpoint.path, 'w') as f:
        f.write("http://ml-service.default.svc.cluster.local:80/predict")

    print("Model deployed successfully")

@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    """Complete ML pipeline"""
    # Data preparation
    data_step = prepare_data()

    # Model training
    train_step = train_model(input_data=data_step.outputs['output_data'])

    # Model deployment
    deploy_step = deploy_model(model_input=train_step.outputs['model_output'])

# Compile pipeline
compiler.compile(
    pipeline_func=ml_pipeline,
    package_path="ml_pipeline.json"
)

print("Pipeline compiled successfully")
```

## Monitoring and Observability Practice

### Exercise 13: Model Monitoring with Prometheus

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np

# Define metrics
prediction_counter = Counter('ml_predictions_total', 'Total number of predictions')
prediction_latency = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('ml_model_accuracy', 'Current model accuracy')

# Start metrics server
start_http_server(8000)

# Prediction function with monitoring
def predict_with_monitoring(model, X):
    start_time = time.time()

    # Make prediction
    prediction = model.predict(X)

    # Record metrics
    prediction_counter.inc()
    prediction_latency.observe(time.time() - start_time)

    return prediction

# Example usage
import joblib
from sklearn.datasets import make_classification

# Load model
model = joblib.load('model.pkl')

# Generate test data
X_test, y_test = make_classification(n_samples=100, n_features=5)

# Make monitored predictions
for i, X_sample in enumerate(X_test):
    prediction = predict_with_monitoring(model, X_sample.reshape(1, -1))

    # Update accuracy gauge periodically
    if i % 50 == 0:
        # Calculate rolling accuracy
        accuracy = np.mean((model.predict(X_test[:i+1]) == y_test[:i+1]))
        model_accuracy.set(accuracy)

    time.sleep(0.1)

print("Monitoring started on port 8000")
```

### Exercise 14: Logging and Alerting Setup

```python
# monitoring/logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

class MLModelLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('ml_model')
        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            'ml_model.log', maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_prediction(self, model_name, input_data, prediction, latency, confidence=None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': model_name,
            'input_shape': input_data.shape,
            'prediction': prediction.tolist(),
            'latency_ms': latency * 1000,
            'confidence': confidence
        }

        self.logger.info(f"Prediction logged: {json.dumps(log_entry)}")

    def log_model_performance(self, model_name, accuracy, precision, recall, f1_score):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        self.logger.info(f"Performance logged: {json.dumps(log_entry)}")

    def log_error(self, model_name, error_message, error_type=None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': model_name,
            'error_message': error_message,
            'error_type': error_type
        }

        self.logger.error(f"Error logged: {json.dumps(log_entry)}")

# Usage example
logger = MLModelLogger()

# Simulate predictions with logging
import numpy as np
import time
from sklearn.datasets import make_classification

# Generate test data
X_test, y_test = make_classification(n_samples=50, n_features=5)
model_name = "logistic_regression_v1"

for i, (X_sample, y_true) in enumerate(zip(X_test, y_test)):
    start_time = time.time()

    # Simulate prediction (replace with actual model prediction)
    prediction = np.array([1 if np.random.random() > 0.5 else 0])
    latency = time.time() - start_time
    confidence = np.random.uniform(0.7, 0.99)

    # Log prediction
    logger.log_prediction(
        model_name=model_name,
        input_data=X_sample,
        prediction=prediction,
        latency=latency,
        confidence=confidence
    )

    time.sleep(0.1)

# Log performance metrics
logger.log_model_performance(
    model_name=model_name,
    accuracy=0.85,
    precision=0.82,
    recall=0.88,
    f1_score=0.85
)

print("Logging system active")
```

### Exercise 15: A/B Testing Implementation

```python
# ab_testing/framework.py
import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy import stats

class ABTestingFramework:
    def __init__(self, experiment_name, traffic_split=0.5):
        self.experiment_name = experiment_name
        self.traffic_split = traffic_split
        self.control_group = []
        self.treatment_group = []

    def assign_variant(self, user_id):
        """Assign user to control or treatment group"""
        # Simple hash-based assignment for reproducibility
        hash_value = hash(str(user_id)) % 100
        return 'treatment' if hash_value < (self.traffic_split * 100) else 'control'

    def log_conversion(self, user_id, variant, conversion_value=1, metadata=None):
        """Log user conversion event"""
        event = {
            'user_id': user_id,
            'variant': variant,
            'conversion': conversion_value,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }

        if variant == 'control':
            self.control_group.append(event)
        else:
            self.treatment_group.append(event)

    def calculate_results(self, confidence_level=0.05):
        """Calculate statistical significance of results"""
        if len(self.control_group) == 0 or len(self.treatment_group) == 0:
            return None

        # Extract conversion rates
        control_conversions = [event['conversion'] for event in self.control_group]
        treatment_conversions = [event['conversion'] for event in self.treatment_group]

        # Calculate statistics
        control_rate = np.mean(control_conversions)
        treatment_rate = np.mean(treatment_conversions)

        # Statistical test
        t_stat, p_value = stats.ttest_ind(treatment_conversions, control_conversions)

        # Confidence intervals
        control_ci = stats.t.interval(1 - confidence_level, len(control_conversions) - 1,
                                    loc=np.mean(control_conversions),
                                    scale=stats.sem(control_conversions))
        treatment_ci = stats.t.interval(1 - confidence_level, len(treatment_conversions) - 1,
                                      loc=np.mean(treatment_conversions),
                                      scale=stats.sem(treatment_conversions))

        results = {
            'experiment_name': self.experiment_name,
            'control_group_size': len(self.control_group),
            'treatment_group_size': len(self.treatment_group),
            'control_conversion_rate': control_rate,
            'treatment_conversion_rate': treatment_rate,
            'uplift': (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0,
            'p_value': p_value,
            'statistically_significant': p_value < confidence_level,
            'control_confidence_interval': control_ci,
            'treatment_confidence_interval': treatment_ci,
            'recommendation': 'Implement treatment' if p_value < confidence_level and treatment_rate > control_rate else 'Keep control'
        }

        return results

# Example usage
ab_framework = ABTestingFramework(
    experiment_name="model_v1_vs_v2",
    traffic_split=0.5
)

# Simulate user interactions
np.random.seed(42)
n_users = 1000

for user_id in range(n_users):
    # Assign variant
    variant = ab_framework.assign_variant(user_id)

    # Simulate conversion (treatment has 10% higher conversion rate)
    if variant == 'control':
        conversion = 1 if np.random.random() < 0.15 else 0
    else:
        conversion = 1 if np.random.random() < 0.165 else 0

    # Log conversion
    ab_framework.log_conversion(
        user_id=user_id,
        variant=variant,
        conversion=conversion,
        metadata={'page_version': variant}
    )

# Calculate results
results = ab_framework.calculate_results()
print("\nA/B Test Results:")
print(json.dumps(results, indent=2))

# Save results
with open('ab_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nA/B test results saved to ab_test_results.json")
```

### Exercise 16: Model Registry and Versioning

```python
# model_registry/manager.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime
import hashlib

class ModelRegistry:
    def __init__(self, tracking_uri="sqlite:///model_registry.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def log_model(self, model, model_name, X_train, y_train, X_test, y_test,
                  parameters=None, artifacts=None, metrics=None):
        """Log model with metadata and metrics"""

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        test_precision = precision_score(y_test, model.predict(X_test), average='weighted')
        test_recall = recall_score(y_test, model.predict(X_test), average='weighted')
        test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')

        # Create model version
        model_version = self._generate_version(model_name)

        with mlflow.start_run():
            # Log parameters
            if parameters:
                for key, value in parameters.items():
                    mlflow.log_param(key, value)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1_score", test_f1)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Save additional metadata
            metadata = {
                'model_name': model_name,
                'version': model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'metrics': {
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    'test_precision': float(test_precision),
                    'test_recall': float(test_recall),
                    'test_f1_score': float(test_f1)
                }
            }

            # Save metadata as artifact
            with open('model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            mlflow.log_artifact('model_metadata.json')

            run_id = mlflow.active_run().info.run_id
            print(f"Model {model_name} version {model_version} logged with run_id: {run_id}")

            return run_id, model_version

    def _generate_version(self, model_name):
        """Generate version string"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}"

    def get_latest_version(self, model_name):
        """Get latest version of model"""
        try:
            latest_version = self.client.get_latest_version(model_name)
            return latest_version
        except Exception as e:
            print(f"Error getting latest version: {e}")
            return None

    def promote_model(self, model_name, version, stage="Production"):
        """Promote model to production"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"Model {model_name} version {version} promoted to {stage}")
        except Exception as e:
            print(f"Error promoting model: {e}")

    def get_model_metrics(self, model_name, version=None):
        """Get metrics for specific model version"""
        try:
            if version is None:
                latest = self.get_latest_version(model_name)
                version = latest.version if latest else None

            if version:
                run = self.client.get_run(self.client.get_run(version).info.run_id)
                return run.data.metrics
        except Exception as e:
            print(f"Error getting model metrics: {e}")
        return None

# Usage example
registry = ModelRegistry()

# Generate sample data
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'logistic_reg_v1': LogisticRegression(random_state=42, max_iter=100),
    'logistic_reg_v2': LogisticRegression(random_state=42, max_iter=200, solver='liblinear')
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Parameters
    parameters = {
        'max_iter': model.max_iter,
        'solver': model.solver
    }

    # Log model
    run_id, version = registry.log_model(
        model=model,
        model_name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        parameters=parameters
    )

    results[name] = {
        'run_id': run_id,
        'version': version
    }

# Get latest versions and metrics
for name in models.keys():
    latest = registry.get_latest_version(name)
    if latest:
        metrics = registry.get_model_metrics(name)
        print(f"\n{name} - Version {latest.version}:")
        print(f"Stage: {latest.current_stage}")
        print(f"Metrics: {metrics}")
```

### Exercise 17: Real-time Prediction Service

```python
# prediction_service/app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('current_model_accuracy', 'Current model accuracy')

# Model cache
model_cache = {}

def load_model(model_path):
    """Load model from cache or disk"""
    if model_path not in model_cache:
        try:
            model_cache[model_path] = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    return model_cache[model_path]

def validate_input(data):
    """Validate input data"""
    try:
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, dict):
            data = np.array([list(data.values())])

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Check if data is numeric
        if not np.issubdtype(data.dtype, np.number):
            return False, "Input data must be numeric"

        # Check for NaN values
        if np.isnan(data).any():
            return False, "Input data contains NaN values"

        return True, data
    except Exception as e:
        return False, f"Invalid input format: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': len(model_cache) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    start_time = time.time()

    try:
        # Parse request
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract parameters
        model_path = request_data.get('model_path', 'model.pkl')
        input_data = request_data.get('data')

        if input_data is None:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate input
        is_valid, validated_data = validate_input(input_data)
        if not is_valid:
            return jsonify({'error': validated_data}), 400

        # Load model
        model = load_model(model_path)
        if model is None:
            return jsonify({'error': f'Model not found: {model_path}'}), 404

        # Make prediction
        predictions = model.predict(validated_data)

        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(validated_data)
        except AttributeError:
            probabilities = None

        # Calculate latency
        latency = time.time() - start_time

        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(latency)

        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'latency_seconds': latency,
            'timestamp': time.time()
        }

        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()

        logger.info(f"Prediction completed in {latency:.3f} seconds")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    model_path = request.args.get('model_path', 'model.pkl')

    try:
        model = load_model(model_path)
        if model is None:
            return jsonify({'error': f'Model not found: {model_path}'}), 404

        info = {
            'model_path': model_path,
            'model_type': type(model).__name__,
            'loaded': True,
            'cache_size': len(model_cache)
        }

        # Add model-specific info
        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()

        if hasattr(model, 'feature_importances_'):
            info['feature_importances'] = model.feature_importances_.tolist()

        return jsonify(info)

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    # Start metrics server
    start_http_server(8000)

    # Load default model
    try:
        joblib.dump(
            LogisticRegression(random_state=42).fit([[1, 2], [3, 4]], [0, 1]),
            'model.pkl'
        )
        logger.info("Created default model")
    except:
        pass

    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)

# Test the service
if __name__ == '__main__':
    # Create test client
    client = app.test_client()

    # Test health check
    response = client.get('/health')
    print("Health check:", response.get_json())

    # Test prediction
    test_data = {
        'model_path': 'model.pkl',
        'data': [[1, 2, 3, 4, 5]]
    }

    response = client.post('/predict',
                          json=test_data,
                          content_type='application/json')

    print("Prediction response:", response.get_json())
```

## Cost Optimization Practice

### Exercise 18: Cloud Cost Monitoring

```python
# cost_monitoring/cost_tracker.py
import boto3
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import json

class CloudCostTracker:
    def __init__(self, aws_profile='default'):
        self.aws_profile = aws_profile
        self.ce = boto3.client('ce', profile_name=aws_profile)
        self.ce_region = 'us-east-1'

    def get_cost_data(self, start_date, end_date, granularity='MONTHLY'):
        """Get cost data from Cost Explorer"""
        try:
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity=granularity,
                Metrics=['BlendedCost', 'UnblendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )

            return response

        except Exception as e:
            print(f"Error getting cost data: {e}")
            return None

    def analyze_ml_costs(self, start_date, end_date):
        """Analyze ML-related costs"""
        cost_data = self.get_cost_data(start_date, end_date, 'DAILY')

        if not cost_data:
            return None

        # Filter ML-related services
        ml_services = [
            'Amazon SageMaker',
            'Amazon EC2-Other',
            'Amazon CloudWatch',
            'Amazon S3',
            'Amazon VPC',
            'Amazon EKS',
            'Amazon ECR',
            'AWS Lambda',
            'AWS Fargate'
        ]

        ml_costs = []
        total_costs = []

        for result in cost_data['ResultsByTime']:
            date = result['TimePeriod']['Start']
            total_cost = float(result['Total']['BlendedCost']['Amount'])

            ml_cost = 0
            service_breakdown = {}

            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                service_breakdown[service] = cost

                if service in ml_services:
                    ml_cost += cost

            ml_costs.append({
                'date': date,
                'ml_cost': ml_cost,
                'total_cost': total_cost,
                'ml_percentage': (ml_cost / total_cost * 100) if total_cost > 0 else 0,
                'service_breakdown': service_breakdown
            })

            total_costs.append({
                'date': date,
                'total_cost': total_cost
            })

        return {
            'ml_costs': ml_costs,
            'total_costs': total_costs,
            'summary': self._calculate_summary(ml_costs)
        }

    def _calculate_summary(self, ml_costs):
        """Calculate cost summary"""
        if not ml_costs:
            return {}

        total_ml_cost = sum(cost['ml_cost'] for cost in ml_costs)
        total_cost = sum(cost['total_cost'] for cost in ml_costs)

        # Find peak cost day
        peak_cost = max(ml_costs, key=lambda x: x['ml_cost'])

        # Calculate trend
        costs = [cost['ml_cost'] for cost in ml_costs]
        if len(costs) > 1:
            trend = "increasing" if costs[-1] > costs[0] else "decreasing"
        else:
            trend = "stable"

        return {
            'total_ml_cost': total_ml_cost,
            'total_cost': total_cost,
            'average_daily_ml_cost': total_ml_cost / len(ml_costs),
            'ml_cost_percentage': (total_ml_cost / total_cost * 100) if total_cost > 0 else 0,
            'peak_cost_date': peak_cost['date'],
            'peak_cost_amount': peak_cost['ml_cost'],
            'trend': trend
        }

    def generate_cost_report(self, start_date, end_date, output_file='cost_report.json'):
        """Generate comprehensive cost report"""
        analysis = self.analyze_ml_costs(start_date, end_date)

        if analysis:
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)

            print(f"Cost report saved to {output_file}")
            return analysis
        return None

# Usage example
tracker = CloudCostTracker()

# Analyze last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

report = tracker.generate_cost_report(
    start_date=start_date,
    end_date=end_date,
    output_file='ml_cost_report.json'
)

if report:
    print("\nCost Analysis Summary:")
    summary = report['summary']
    print(f"Total ML Cost: ${summary['total_ml_cost']:.2f}")
    print(f"ML Cost Percentage: {summary['ml_cost_percentage']:.1f}%")
    print(f"Average Daily Cost: ${summary['average_daily_ml_cost']:.2f}")
    print(f"Trend: {summary['trend']}")
    print(f"Peak Cost Day: {summary['peak_cost_date']} (${summary['peak_cost_amount']:.2f})")
```

### Exercise 19: Auto-scaling Configuration

```python
# autoscaling/configuration.py
import json
from datetime import datetime

class AutoScalingConfig:
    def __init__(self, config_name):
        self.config_name = config_name
        self.config = {
            'config_name': config_name,
            'created_at': datetime.utcnow().isoformat(),
            'policies': {},
            'alarms': {},
            'target_groups': {}
        }

    def add_scaling_policy(self, policy_name, policy_type, target_value,
                          scaling_adjustment, adjustment_type='ChangeInCapacity',
                          cooldown_period=300):
        """Add scaling policy"""
        self.config['policies'][policy_name] = {
            'policy_type': policy_type,
            'target_value': target_value,
            'scaling_adjustment': scaling_adjustment,
            'adjustment_type': adjustment_type,
            'cooldown_period': cooldown_period
        }

    def add_cloudwatch_alarm(self, alarm_name, metric_name, namespace,
                            comparison_operator, threshold, evaluation_periods=2,
                            period=300, statistic='Average'):
        """Add CloudWatch alarm"""
        self.config['alarms'][alarm_name] = {
            'metric_name': metric_name,
            'namespace': namespace,
            'comparison_operator': comparison_operator,
            'threshold': threshold,
            'evaluation_periods': evaluation_periods,
            'period': period,
            'statistic': statistic
        }

    def configure_ml_workload(self, instance_type='ml.m5.large',
                            min_capacity=0, max_capacity=10,
                            target_cpu_utilization=70,
                            scale_out_cooldown=300, scale_in_cooldown=600):
        """Configure auto-scaling for ML workload"""

        # Scale out policy (CPU utilization high)
        self.add_scaling_policy(
            policy_name='scale_out_cpu',
            policy_type='TargetTrackingScaling',
            target_value=target_cpu_utilization,
            scaling_adjustment=0  # Target tracking doesn't need adjustment
        )

        # Scale in policy
        self.add_scaling_policy(
            policy_name='scale_in_idle',
            policy_type='SimpleScaling',
            target_value=10,  # Scale in if CPU < 10%
            scaling_adjustment=-1,
            adjustment_type='ChangeInCapacity',
            cooldown_period=scale_in_cooldown
        )

        # CloudWatch alarms
        self.add_cloudwatch_alarm(
            alarm_name='high_cpu_utilization',
            metric_name='CPUUtilization',
            namespace='AWS/EC2',
            comparison_operator='GreaterThanThreshold',
            threshold=80,
            evaluation_periods=2,
            period=300
        )

        self.add_cloudwatch_alarm(
            alarm_name='low_request_count',
            metric_name='RequestCountPerTarget',
            namespace='AWS/ApplicationELB',
            comparison_operator='LessThanThreshold',
            threshold=10,
            evaluation_periods=3,
            period=300
        )

        # Save configuration
        self.save_config(f'{self.config_name}_config.json')

        return self.config

    def save_config(self, filename):
        """Save configuration to file"""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Configuration saved to {filename}")

    def generate_terraform_config(self, output_file='autoscaling.tf'):
        """Generate Terraform configuration"""
        terraform_config = '''
resource "aws_launch_configuration" "ml_launch_config" {
  name          = "ml-launch-config"
  image_id      = "ami-0c02fb55956c7d316"  # Amazon Linux 2
  instance_type = "ml.m5.large"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "ml_autoscaling_group" {
  name                = "ml-autoscaling-group"
  launch_configuration = aws_launch_configuration.ml_launch_config.name
  min_size            = 0
  max_size            = 10
  desired_capacity    = 1

  target_group_arns = [aws_lb_target_group.ml_target_group.arn]

  dynamic "tag" {
    for_each = {
      Name = "ml-instance"
      Environment = "production"
      ManagedBy = "terraform"
    }
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

resource "aws_autoscaling_policy" "scale_out_cpu" {
  name                = "scale-out-cpu"
  autoscaling_group   = aws_autoscaling_group.ml_autoscaling_group.name
  policy_type         = "TargetTrackingScaling"

  target_tracking_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ASGAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "high-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Alarm when CPU exceeds 80%"

  alarm_actions = [
    aws_autoscaling_policy.scale_out_cpu.arn
  ]
}
'''

        with open(output_file, 'w') as f:
            f.write(terraform_config)

        print(f"Terraform configuration saved to {output_file}")

# Usage example
autoscaler = AutoScalingConfig('ml_workload')

# Configure for ML workload
config = autoscaler.configure_ml_workload(
    instance_type='ml.m5.large',
    min_capacity=0,
    max_capacity=10,
    target_cpu_utilization=70
)

# Generate Terraform configuration
autoscaler.generate_terraform_config('ml_autoscaling.tf')

print("\nAuto-scaling configuration completed")
print(f"Configured policies: {list(config['policies'].keys())}")
print(f"Configured alarms: {list(config['alarms'].keys())}")
```

### Exercise 20: Complete MLOps Pipeline Integration

```python
# mlops_pipeline/pipeline_orchestrator.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from datetime import datetime
import subprocess
import logging

class MLOpsOrchestrator:
    def __init__(self, experiment_name="ml_pipeline",
                 model_registry="sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = model_registry

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize MLflow
        mlflow.set_tracking_uri(model_registry)
        mlflow.set_experiment(experiment_name)

    def data_validation(self, data_path):
        """Validate input data"""
        try:
            data = pd.read_csv(data_path)

            # Basic validation checks
            validation_results = {
                'shape': data.shape,
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': int(data.duplicated().sum()),
                'data_types': data.dtypes.to_dict(),
                'validation_passed': True
            }

            # Check for required columns
            if 'target' in data.columns:
                validation_results['has_target'] = True
                validation_results['target_distribution'] = data['target'].value_counts().to_dict()
            else:
                validation_results['has_target'] = False
                validation_results['validation_passed'] = False

            # Save validation results
            with open('data_validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)

            self.logger.info(f"Data validation completed. Shape: {data.shape}")
            return validation_results

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return {'validation_passed': False, 'error': str(e)}

    def feature_engineering(self, data_path, save_path='processed_data.csv'):
        """Perform feature engineering"""
        try:
            data = pd.read_csv(data_path)

            # Example feature engineering
            # Handle missing values
            data = data.fillna(data.median(numeric_only=True))

            # Create additional features (example)
            if all(col in data.columns for col in ['feature_0', 'feature_1']):
                data['feature_sum'] = data['feature_0'] + data['feature_1']
                data['feature_ratio'] = data['feature_0'] / (data['feature_1'] + 1e-6)

            # Save processed data
            data.to_csv(save_path, index=False)

            self.logger.info(f"Feature engineering completed. Saved to {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return None

    def model_training(self, data_path, model_configs):
        """Train multiple models and compare"""
        results = {}

        try:
            data = pd.read_csv(data_path)

            # Separate features and target
            X = data.drop('target', axis=1)
            y = data['target']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            with mlflow.start_run():
                for model_name, config in model_configs.items():
                    self.logger.info(f"Training {model_name}")

                    # Create model based on config
                    if config['type'] == 'random_forest':
                        model = RandomForestClassifier(
                            n_estimators=config.get('n_estimators', 100),
                            max_depth=config.get('max_depth', 10),
                            random_state=42
                        )
                    elif config['type'] == 'logistic_regression':
                        model = LogisticRegression(
                            max_iter=config.get('max_iter', 1000),
                            random_state=42
                        )
                    else:
                        self.logger.warning(f"Unknown model type: {config['type']}")
                        continue

                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)

                    # Log parameters
                    mlflow.log_param('model_type', config['type'])
                    for param_name, param_value in config.items():
                        if param_name != 'type':
                            mlflow.log_param(param_name, param_value)

                    # Log metrics
                    mlflow.log_metric('accuracy', accuracy)

                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        model_name,
                        registered_model_name=f"{self.experiment_name}_{model_name}"
                    )

                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'run_id': mlflow.active_run().info.run_id
                    }

                    self.logger.info(f"{model_name} accuracy: {accuracy:.4f}")

            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model_info = results[best_model_name]

            self.logger.info(f"Best model: {best_model_name} with accuracy {best_model_info['accuracy']:.4f}")

            return {
                'results': results,
                'best_model': best_model_name,
                'best_accuracy': best_model_info['accuracy']
            }

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return None

    def model_validation(self, model, X_test, y_test):
        """Validate trained model"""
        try:
            y_pred = model.predict(X_test)

            validation_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Save validation results
            with open('model_validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)

            self.logger.info("Model validation completed")
            return validation_results

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return None

    def deploy_model(self, model_name, deployment_target='local'):
        """Deploy model to target environment"""
        try:
            if deployment_target == 'local':
                # Save model locally
                model_info = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
                import joblib
                joblib.dump(model_info, 'deployed_model.pkl')
                self.logger.info("Model deployed locally")

            elif deployment_target == 'docker':
                # Create Dockerfile
                dockerfile_content = '''
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model_service.py .
COPY deployed_model.pkl .

EXPOSE 8080

CMD ["python", "model_service.py"]
'''
                with open('Dockerfile', 'w') as f:
                    f.write(dockerfile_content)

                # Build Docker image
                subprocess.run(['docker', 'build', '-t', 'ml-model:latest', '.'], check=True)
                self.logger.info("Model deployed to Docker")

            return True

        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False

    def monitor_model(self, model_path='deployed_model.pkl'):
        """Setup model monitoring"""
        try:
            monitoring_script = '''
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class ModelMonitor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.logger = logging.getLogger(__name__)

    def log_prediction(self, input_data, prediction, confidence=None):
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'input_shape': input_data.shape,
            'prediction': prediction,
            'confidence': confidence
        }

        self.logger.info(f"Prediction logged: {event}")

    def detect_data_drift(self, reference_data, current_data, threshold=0.1):
        """Simple data drift detection"""
        ref_mean = np.mean(reference_data, axis=0)
        curr_mean = np.mean(current_data, axis=0)

        drift_score = np.mean(np.abs((curr_mean - ref_mean) / (ref_mean + 1e-6)))

        is_drift = drift_score > threshold

        return {
            'drift_score': drift_score,
            'is_drift': is_drift,
            'threshold': threshold
        }
'''

            with open('model_monitor.py', 'w') as f:
                f.write(monitoring_script)

            self.logger.info("Model monitoring setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Model monitoring setup failed: {e}")
            return False

    def run_full_pipeline(self, data_path='data.csv'):
        """Execute complete MLOps pipeline"""
        self.logger.info("Starting MLOps pipeline")

        try:
            # Step 1: Data validation
            validation_results = self.data_validation(data_path)
            if not validation_results.get('validation_passed', False):
                raise ValueError("Data validation failed")

            # Step 2: Feature engineering
            processed_data_path = self.feature_engineering(data_path)

            # Step 3: Model training
            model_configs = {
                'random_forest_v1': {
                    'type': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'logistic_regression_v1': {
                    'type': 'logistic_regression',
                    'max_iter': 1000
                }
            }

            training_results = self.model_training(processed_data_path, model_configs)

            # Step 4: Model deployment
            best_model_name = f"{self.experiment_name}_{training_results['best_model']}"
            deployment_success = self.deploy_model(best_model_name, 'local')

            # Step 5: Setup monitoring
            monitoring_success = self.monitor_model()

            pipeline_results = {
                'data_validation': validation_results,
                'feature_engineering': processed_data_path,
                'model_training': training_results,
                'deployment_success': deployment_success,
                'monitoring_success': monitoring_success,
                'pipeline_status': 'completed' if all([deployment_success, monitoring_success]) else 'partial'
            }

            # Save pipeline results
            with open('pipeline_results.json', 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)

            self.logger.info("MLOps pipeline completed successfully")
            return pipeline_results

        except Exception as e:
            self.logger.error(f"MLOps pipeline failed: {e}")
            return {'pipeline_status': 'failed', 'error': str(e)}

# Usage example
orchestrator = MLOpsOrchestrator(experiment_name="production_ml_pipeline")

# Create sample data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'feature_0': np.random.randn(n_samples),
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'target': (np.random.randn(n_samples) > 0).astype(int)
})

data.to_csv('data.csv', index=False)

# Run complete pipeline
results = orchestrator.run_full_pipeline('data.csv')

print("\nMLOps Pipeline Results:")
print(f"Status: {results.get('pipeline_status', 'unknown')}")
if 'model_training' in results:
    best_model = results['model_training']['best_model']
    best_accuracy = results['model_training']['best_accuracy']
    print(f"Best model: {best_model}")
    print(f"Best accuracy: {best_accuracy:.4f}")

print("\nPipeline completed successfully!")
```

## Summary

This practice guide covers comprehensive cloud ML infrastructure implementation including:

### Key Areas Covered:

1. **AWS SageMaker**: Training, deployment, and management
2. **Google Cloud Vertex AI**: AutoML and custom training
3. **Azure ML**: Workspace setup and pipeline creation
4. **Kubernetes**: Container orchestration for ML workloads
5. **Kubeflow**: ML pipeline orchestration
6. **Monitoring**: Prometheus metrics and alerting
7. **A/B Testing**: Model comparison and statistical validation
8. **Model Registry**: MLflow for model versioning
9. **Cost Optimization**: Cloud cost monitoring and analysis
10. **Auto-scaling**: Dynamic resource allocation

### Learning Outcomes:

- Hands-on experience with major cloud ML platforms
- Understanding of production ML deployment patterns
- Practical implementation of monitoring and observability
- Cost optimization strategies for cloud infrastructure
- End-to-end MLOps pipeline orchestration

### Next Steps:

- Apply these practices to real-world ML projects
- Customize configurations for specific use cases
- Integrate with existing development workflows
- Scale solutions based on production requirements
