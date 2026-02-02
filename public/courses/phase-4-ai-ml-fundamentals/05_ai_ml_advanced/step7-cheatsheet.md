# AI/ML Advanced Cheat Sheet

## Complete Reference Guide for Phase 6: AI/ML Advanced

_Your comprehensive quick reference for AI/ML advanced topics - from hardware to production deployment_

---

## 📋 Table of Contents

### 1. [Hardware & Infrastructure Quick Reference](#hardware-infrastructure)

### 2. [MLOps Essentials](#mlops-essentials)

### 3. [Real-World Application Patterns](#application-patterns)

### 4. [Ethics & Responsibility Guidelines](#ethics-guidelines)

### 5. [Performance Optimization Tips](#optimization-tips)

### 6. [Production Deployment Checklist](#deployment-checklist)

### 7. [Common Pitfalls & Solutions](#pitfalls-solutions)

### 8. [Quick Command Reference](#command-reference)

### 9. [Tools & Resources](#tools-resources)

---

## 1. Hardware & Infrastructure Quick Reference {#hardware-infrastructure}

### Hardware Selection Guide

#### Budget-Based Recommendations

```yaml
Student/Beginner ($500-1,500):
  CPU: Intel i5-12400 / AMD Ryzen 5 5600
  GPU: NVIDIA GTX 1660 Super / RTX 3050
  RAM: 16GB DDR4-3200
  Storage: 500GB SSD + 1TB HDD

Enthusiast ($1,500-3,000):
  CPU: Intel i5-12600K / AMD Ryzen 7 5700X
  GPU: NVIDIA RTX 3060 Ti / RTX 3070
  RAM: 32GB DDR4-3600
  Storage: 1TB NVMe SSD + 2TB HDD

Professional ($3,000-6,000):
  CPU: Intel i7-12700K / AMD Ryzen 7 5800X
  GPU: NVIDIA RTX 3080 / RTX 4070 Ti
  RAM: 64GB DDR4-3600
  Storage: 2TB NVMe SSD + 4TB HDD

Enterprise ($6,000+):
  CPU: Intel i9-12900K / AMD Ryzen 9 5900X
  GPU: NVIDIA RTX 3090 / RTX 4090
  RAM: 128GB DDR4/DDR5
  Storage: Multiple NVMe SSDs in RAID
```

#### GPU Memory Requirements by Model

```python
COMPUTER_VISION = {
    "ResNet-50": "6-8 GB VRAM",
    "ResNet-152": "8-12 GB VRAM",
    "YOLOv5 Large": "8-12 GB VRAM",
    "EfficientNet-B7": "12-16 GB VRAM",
    "StyleGAN": "12-20 GB VRAM"
}

NLP_MODELS = {
    "BERT-Base": "8-12 GB VRAM",
    "BERT-Large": "16-20 GB VRAM",
    "GPT-2 Large": "20-30 GB VRAM",
    "T5-Large": "20-30 GB VRAM"
}

LLMS = {
    "LLaMA-2 7B": "14-20 GB VRAM",
    "LLaMA-2 13B": "26-30 GB VRAM",
    "LLaMA-2 70B": "140+ GB VRAM (multi-GPU)"
}
```

### Infrastructure Setup Commands

#### Docker Environment Setup

```bash
# GPU-enabled Docker container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  tensorflow/tensorflow:latest-gpu-py3

# Jupyter with GPU support
docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  jupyter/tensorflow-notebook

# Multi-container setup with docker-compose
version: '3.8'
services:
  jupyter:
    image: tensorflow/tensorflow:latest-gpu-py3
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

#### Kubernetes GPU Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-training
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-training
  template:
    metadata:
      labels:
        app: ai-training
    spec:
      containers:
        - name: training-container
          image: tensorflow/tensorflow:latest-gpu-py3
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "4"
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "8"
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
```

#### System Monitoring Commands

```bash
# GPU monitoring
nvidia-smi
watch -n 1 nvidia-smi

# Python GPU monitoring
python -c "
import pynvml
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f'GPU {i}: {mem.used//1024//1024}MB/{mem.total//1024//1024}MB, {util.gpu}%')
"

# System resources
htop
iotop
df -h
free -h
```

### Cloud Cost Optimization

#### Spot Instance Configuration

```python
# AWS Spot Instance Auto-Scaling
import boto3

def create_spot_training_config():
    client = boto3.client('ec2')

    # Request spot instances with specific pricing
    response = client.request_spot_instances(
        LaunchSpecification={
            'ImageId': 'ami-0abcdef1234567890',  # GPU AMI
            'InstanceType': 'p3.2xlarge',
            'SecurityGroupIds': ['sg-0123456789abcdef0'],
            'UserData': '''#!/bin/bash
                pip install tensorflow torch
                git clone https://github.com/your-repo/training-script.git
                cd training-script && python train.py
            ''',
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 100,
                    'DeleteOnTermination': True
                }
            }]
        },
        SpotPrice='0.50',  # Set max price
        InstanceCount=4,
        Type='one-time'
    )

    return response
```

#### Cost Monitoring Script

```python
import boto3
import json
from datetime import datetime, timedelta

def monitor_cloud_costs(days=30):
    """Monitor and analyze cloud costs"""

    ce = boto3.client('ce')

    # Get cost data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.isoformat(),
            'End': end_date.isoformat()
        },
        Granularity='DAILY',
        Metrics=['BlendedCost'],
        GroupBy=[
            {
                'Type': 'DIMENSION',
                'Key': 'SERVICE'
            },
            {
                'Type': 'DIMENSION',
                'Key': 'USAGE_TYPE'
            }
        ]
    )

    total_cost = 0
    services = {}

    for result in response['ResultsByTime']:
        date = result['TimePeriod']['Start']
        daily_cost = float(result['Total']['BlendedCost']['Amount'])
        total_cost += daily_cost

        for group in result['Groups']:
            service = group['Keys'][0]
            usage_type = group['Keys'][1]
            cost = float(group['Metrics']['BlendedCost']['Amount'])

            if service not in services:
                services[service] = {}
            services[service][usage_type] = cost

    print(f"Total cost for last {days} days: ${total_cost:.2f}")

    # Identify high-cost services
    service_totals = {service: sum(costs.values())
                     for service, costs in services.items()}

    for service, total in sorted(service_totals.items(),
                                key=lambda x: x[1], reverse=True)[:5]:
        print(f"{service}: ${total:.2f}")

    return services

# Usage
costs = monitor_cloud_costs()
```

---

## 2. MLOps Essentials {#mlops-essentials}

### Model Versioning & Tracking

#### MLflow Setup & Commands

```bash
# Install MLflow
pip install mlflow

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Run experiment with tracking
mlflow run experiments/experiment_name.py
```

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model_with_tracking():
    # Set experiment
    mlflow.set_experiment("credit_risk_prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artifacts
        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("visualizations/confusion_matrix.png")

        print(f"Model accuracy: {accuracy}")
        return model, accuracy
```

#### DVC (Data Version Control) Setup

```bash
# Initialize DVC
dvc init

# Add data files
dvc add data/train_data.csv
dvc add models/model.pkl

# Setup remote storage (S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote add myremote-s3 s3://my-bucket/dvc-storage

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Create DVC pipeline
dvc stage add -n train \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py

# Run pipeline
dvc repro
```

#### Weights & Biases Integration

```python
import wandb
from wandb.keras import WandbCallback

# Initialize W&B
wandb.init(project="credit-risk-model", entity="your-username")

# Train with W&B tracking
def train_with_wandb():
    # Model configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_units": [128, 64, 32]
    }

    wandb.config.update(config)

    # Build model
    model = build_model(config)

    # Train with callbacks
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[
            WandbCallback(),
            wandb.keras.WandbModelCheckpoint("models/")
        ]
    )

    # Log evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test)
    wandb.log({"test_accuracy": test_acc})

    wandb.finish()
```

### CI/CD for ML Pipelines

#### GitHub Actions for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  train-and-evaluate:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python src/train.py --config config/model_config.yaml

      - name: Evaluate model
        run: |
          python src/evaluate.py --model-path models/latest

      - name: Deploy to staging
        if: success()
        run: |
          python scripts/deploy_staging.py
```

### Model Monitoring & Drift Detection

#### Production Monitoring Pipeline

```python
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import accuracy_score
import logging

class ModelMonitor:
    def __init__(self, model, reference_data, feature_names):
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = {}

        # Calculate reference statistics
        for feature in feature_names:
            self.reference_stats[feature] = {
                'mean': reference_data[feature].mean(),
                'std': reference_data[feature].std(),
                'min': reference_data[feature].min(),
                'max': reference_data[feature].max()
            }

    def detect_data_drift(self, new_data, threshold=0.05):
        """Detect if input data distribution has drifted"""

        drift_detected = {}

        for feature in self.feature_names:
            if feature in new_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[feature],
                    new_data[feature]
                )

                # Population Stability Index (PSI)
                reference_binned = pd.cut(
                    self.reference_data[feature],
                    bins=10,
                    retbins=True
                )[1]
                new_binned = pd.cut(new_data[feature], bins=reference_binned, retbins=True)[0]

                ref_counts = pd.Series(reference_binned[:-1]).value_counts()
                new_counts = new_binned.value_counts()

                # Calculate PSI
                psi = self.calculate_psi(ref_counts, new_counts)

                drift_detected[feature] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'psi': psi,
                    'is_drifted': p_value < threshold or psi > 0.2
                }

        return drift_detected

    def calculate_psi(self, expected, actual):
        """Calculate Population Stability Index"""
        # Avoid division by zero
        expected = expected + 1
        actual = actual + 1

        psi = (actual - expected) * np.log(actual / expected)
        return psi.sum()

    def detect_performance_drift(self, predictions, true_labels,
                               prediction_threshold=0.05):
        """Monitor model performance drift"""

        current_accuracy = accuracy_score(true_labels, predictions)

        # This assumes you track historical performance
        # In practice, you'd compare against a performance baseline
        baseline_accuracy = 0.85  # Replace with your baseline

        performance_drift = {
            'current_accuracy': current_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_drop': baseline_accuracy - current_accuracy,
            'is_drifted': current_accuracy < baseline_accuracy - prediction_threshold
        }

        return performance_drift

    def log_monitoring_results(self, drift_results, performance_results):
        """Log monitoring results"""

        logger = logging.getLogger(__name__)

        # Log data drift
        for feature, result in drift_results.items():
            if result['is_drifted']:
                logger.warning(f"Data drift detected in {feature}: "
                             f"KS p-value={result['p_value']:.4f}, "
                             f"PSI={result['psi']:.4f}")

        # Log performance drift
        if performance_results['is_drifted']:
            logger.error(f"Performance drift detected: "
                        f"Current accuracy={performance_results['current_accuracy']:.3f}, "
                        f"Expected={performance_results['baseline_accuracy']:.3f}")

        # Alert if necessary
        if any(result['is_drifted'] for result in drift_results.values()) or \
           performance_results['is_drifted']:
            self.trigger_alerts(drift_results, performance_results)

    def trigger_alerts(self, drift_results, performance_results):
        """Trigger alerts for significant drift"""

        # Implementation depends on your alerting system
        # Could be Slack, email, PagerDuty, etc.

        alert_message = {
            'severity': 'high' if performance_results['is_drifted'] else 'medium',
            'type': 'model_drift',
            'details': {
                'data_drift': drift_results,
                'performance_drift': performance_results
            }
        }

        # Send alert (implementation specific)
        print(f"ALERT: {alert_message}")

# Usage
def monitor_model_in_production():
    # Load current data and model
    model = load_model("models/production_model.pkl")
    new_data = load_recent_data("data/recent_batch.csv")

    # Initialize monitor
    monitor = ModelMonitor(
        model=model,
        reference_data=load_reference_data("data/reference_data.csv"),
        feature_names=['feature1', 'feature2', 'feature3']
    )

    # Detect drift
    drift_results = monitor.detect_data_drift(new_data)

    # Make predictions and check performance drift
    predictions = model.predict(new_data)
    true_labels = get_true_labels(new_data)  # If available
    performance_results = monitor.detect_performance_drift(predictions, true_labels)

    # Log results
    monitor.log_monitoring_results(drift_results, performance_results)
```

---

## 3. Real-World Application Patterns {#application-patterns}

### Architecture Patterns

#### Microservices for AI

```python
# FastAPI microservices architecture
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Recommendation Service
app_rec = FastAPI(title="Recommendation Service")

class UserPreferences(BaseModel):
    user_id: int
    category: str
    max_recommendations: int = 10

@app_rec.post("/recommend")
async def get_recommendations(prefs: UserPreferences):
    """Get personalized recommendations"""

    try:
        # Business logic for recommendations
        recommendations = generate_recommendations(
            user_id=prefs.user_id,
            category=prefs.category,
            max_items=prefs.max_recommendations
        )

        return {
            "status": "success",
            "recommendations": recommendations,
            "count": len(recommendations)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_rec.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recommendation"}

# Image Classification Service
app_cv = FastAPI(title="Computer Vision Service")

class ImageRequest(BaseModel):
    image_url: str
    confidence_threshold: float = 0.5

@app_cv.post("/classify")
async def classify_image(request: ImageRequest):
    """Classify image using trained model"""

    # Load model
    model = load_classification_model()

    # Preprocess image
    image = load_and_preprocess_image(request.image_url)

    # Make prediction
    predictions = model.predict(image)

    # Filter by confidence
    high_conf_predictions = [
        pred for pred in predictions
        if pred['confidence'] >= request.confidence_threshold
    ]

    return {
        "predictions": high_conf_predictions,
        "processing_time": time.time() - start_time
    }
```

#### Event-Driven Architecture

```python
# Event-driven model retraining
import asyncio
from typing import Dict, Any
import json

class EventDrivenMLPipeline:
    def __init__(self):
        self.event_handlers = {
            'data_drift_detected': self.handle_data_drift,
            'performance_degradation': self.handle_performance_issue,
            'new_data_available': self.handle_new_data,
            'model_expiration': self.handle_model_expiration
        }

    async def handle_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle events in the ML pipeline"""

        if event_type in self.event_handlers:
            await self.event_handlers[event_type](event_data)
        else:
            print(f"No handler for event type: {event_type}")

    async def handle_data_drift(self, event_data):
        """Handle data drift events"""

        print("Data drift detected - starting model retraining")

        # Collect new data
        new_data = await self.collect_new_data(
            timeframe=event_data.get('timeframe', '7d')
        )

        # Retrain model
        model = await self.retrain_model(new_data)

        # Validate model
        validation_results = await self.validate_model(model)

        if validation_results['passes']:
            # Deploy new model
            await self.deploy_model(model)
            await self.broadcast_event('model_updated', {
                'model_version': model.version,
                'performance_metrics': validation_results
            })
        else:
            await self.broadcast_event('model_retraining_failed', {
                'reason': 'validation_failed',
                'details': validation_results
            })

    async def handle_performance_issue(self, event_data):
        """Handle performance degradation"""

        print("Performance issue detected - investigating")

        # Analyze recent performance
        analysis = await self.analyze_performance_issues(
            timeframe='24h'
        )

        # Determine action
        if analysis['severity'] == 'critical':
            # Rollback to previous model
            await self.rollback_model()
            await self.broadcast_event('model_rolled_back', analysis)
        else:
            # Schedule retraining
            await self.schedule_retraining(delay='1h')

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to other services"""

        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Send to message queue (implementation specific)
        await self.message_queue.publish('ml_events', json.dumps(event))

# Event consumer
async def consume_events():
    pipeline = EventDrivenMLPipeline()

    while True:
        # Listen for events
        events = await pipeline.message_queue.consume('ml_events')

        for event_message in events:
            event = json.loads(event_message.body)
            await pipeline.handle_event(event['type'], event['data'])
```

### Scalability Patterns

#### Batch vs Real-Time Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import time

class ScalableMLPipeline:
    def __init__(self, batch_size=100, max_workers=4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.batch_queue = queue.Queue()
        self.real_time_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Start batch processor
        asyncio.create_task(self.batch_processor())

        # Start real-time processor
        asyncio.create_task(self.real_time_processor())

    async def batch_process(self, data_batch):
        """Process large batches of data"""

        print(f"Processing batch of {len(data_batch)} items")

        # Use efficient algorithms for batch processing
        model = await self.load_optimized_model('batch')

        results = []
        for item in data_batch:
            result = model.predict(item)
            results.append(result)

        # Batch save results
        await self.batch_save_results(results)

        return len(results)

    async def real_time_process(self, single_item):
        """Process single items with low latency"""

        print(f"Processing single item with ID: {single_item.get('id')}")

        # Use lightweight model for real-time processing
        model = await self.load_optimized_model('realtime')

        result = model.predict(single_item)

        # Immediate response
        await self.send_real_time_response(single_item['request_id'], result)

        return result

    async def batch_processor(self):
        """Continuously process batches"""

        while True:
            try:
                # Wait for batch
                batch = []
                start_time = time.time()

                while len(batch) < self.batch_size and time.time() - start_time < 30:
                    try:
                        item = self.batch_queue.get(timeout=1)
                        batch.append(item)
                    except queue.Empty:
                        continue

                if batch:
                    await self.batch_process(batch)

            except Exception as e:
                print(f"Batch processing error: {e}")
                await asyncio.sleep(5)

    async def real_time_processor(self):
        """Process real-time requests"""

        while True:
            try:
                # Get single item
                item = await asyncio.get_event_loop().run_in_executor(
                    None, self.real_time_queue.get, True, 1.0
                )

                if item:
                    await self.real_time_process(item)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time processing error: {e}")

    def submit_batch(self, data):
        """Submit data for batch processing"""
        self.batch_queue.put(data)

    def submit_realtime(self, data):
        """Submit data for real-time processing"""
        self.real_time_queue.put(data)

# Usage
pipeline = ScalableMLPipeline(batch_size=50, max_workers=8)

# Batch processing
for data_batch in data_generator():
    pipeline.submit_batch(data_batch)

# Real-time processing
for request in request_stream():
    pipeline.submit_realtime(request)
```

---

## 4. Ethics & Responsibility Guidelines {#ethics-guidelines}

### Fairness Assessment Framework

#### Bias Detection & Mitigation

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

class FairnessAssessment:
    def __init__(self, model, X_test, y_test, protected_attributes):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.protected_attributes = protected_attributes
        self.predictions = model.predict(X_test)

    def demographic_parity(self, attribute_name):
        """Calculate demographic parity difference"""

        protected_values = self.protected_attributes[attribute_name].unique()

        # Calculate positive prediction rates for each group
        group_rates = {}
        for value in protected_values:
            mask = self.protected_attributes[attribute_name] == value
            group_predictions = self.predictions[mask]
            positive_rate = np.mean(group_predictions)
            group_rates[value] = positive_rate

        # Calculate demographic parity difference
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)

        return {
            'group_rates': group_rates,
            'demographic_parity_difference': max_diff,
            'is_fair': max_diff < 0.1  # 10% threshold
        }

    def equalized_odds(self, attribute_name):
        """Calculate equalized odds difference"""

        protected_values = self.protected_attributes[attribute_name].unique()

        odds_differences = {}

        for value in protected_values:
            mask = self.protected_attributes[attribute_name] == value

            # True Positive Rate
            tp_mask = mask & (self.y_test == 1)
            if tp_mask.sum() > 0:
                tpr = np.mean(self.predictions[tp_mask])
            else:
                tpr = 0

            # False Positive Rate
            fp_mask = mask & (self.y_test == 0)
            if fp_mask.sum() > 0:
                fpr = np.mean(self.predictions[fp_mask])
            else:
                fpr = 0

            odds_differences[value] = {
                'tpr': tpr,
                'fpr': fpr
            }

        # Calculate differences between groups
        tprs = [group['tpr'] for group in odds_differences.values()]
        fprs = [group['fpr'] for group in odds_differences.values()]

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return {
            'group_metrics': odds_differences,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'is_fair': max(tpr_diff, fpr_diff) < 0.1
        }

    def equal_opportunity(self, attribute_name):
        """Calculate equal opportunity difference"""

        protected_values = self.protected_attributes[attribute_name].unique()

        tpr_differences = {}

        for value in protected_values:
            mask = self.protected_attributes[attribute_name] == value
            tp_mask = mask & (self.y_test == 1)

            if tp_mask.sum() > 0:
                tpr = np.mean(self.predictions[tp_mask])
            else:
                tpr = 0

            tpr_differences[value] = tpr

        # Calculate difference
        tprs = list(tpr_differences.values())
        max_tpr_diff = max(tprs) - min(tprs)

        return {
            'group_tprs': tpr_differences,
            'opportunity_difference': max_tpr_diff,
            'is_fair': max_tpr_diff < 0.1
        }

    def comprehensive_fairness_assessment(self, attribute_name):
        """Run all fairness metrics"""

        demographic_parity = self.demographic_parity(attribute_name)
        equalized_odds = self.equalized_odds(attribute_name)
        equal_opportunity = self.equal_opportunity(attribute_name)

        # Overall fairness decision
        is_fair = (demographic_parity['is_fair'] and
                  equalized_odds['is_fair'] and
                  equal_opportunity['is_fair'])

        return {
            'attribute': attribute_name,
            'is_fair_overall': is_fair,
            'demographic_parity': demographic_parity,
            'equalized_odds': equalized_odds,
            'equal_opportunity': equal_opportunity,
            'recommendations': self._generate_recommendations(
                demographic_parity, equalized_odds, equal_opportunity
            )
        }

    def _generate_recommendations(self, dp, eo, ep):
        """Generate fairness improvement recommendations"""

        recommendations = []

        if not dp['is_fair']:
            recommendations.append(
                "Consider reweighting training data to achieve demographic parity"
            )

        if not eo['is_fair']:
            recommendations.append(
                "Apply post-processing calibration to equalize false positive rates"
            )

        if not ep['is_fair']:
            recommendations.append(
                "Adjust decision threshold to ensure equal true positive rates"
            )

        return recommendations

# Usage
def assess_model_fairness():
    # Load model and data
    model = load_model("models/fairness_assessment_model.pkl")
    X_test, y_test, protected_attrs = load_test_data()

    # Initialize fairness assessment
    fairness = FairnessAssessment(model, X_test, y_test, protected_attrs)

    # Assess fairness for different protected attributes
    fairness_report = {}

    for attribute in ['gender', 'race', 'age_group']:
        if attribute in protected_attrs.columns:
            fairness_report[attribute] = fairness.comprehensive_fairness_assessment(
                attribute
            )

    return fairness_report

# Generate fairness report
report = assess_model_fairness()
print(json.dumps(report, indent=2, default=str))
```

### Privacy-Preserving ML

#### Differential Privacy Implementation

```python
import numpy as np
from scipy.stats import laplace
from sklearn.tree import DecisionTreeClassifier
import copy

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5):
        """
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Approximate differential privacy parameter
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = None

    def calculate_sensitivity(self, query_function, max_influence=1):
        """Calculate the sensitivity of a query function"""
        # For most ML operations, the sensitivity is bounded
        # by the maximum change in output when one record changes
        self.sensitivity = max_influence
        return self.sensitivity

    def add_noise_to_model(self, model, epsilon=None):
        """Add differential privacy noise to model parameters"""

        if epsilon is None:
            epsilon = self.epsilon

        # Calculate noise scale based on epsilon and sensitivity
        noise_scale = self.calculate_sensitivity(None) / epsilon

        # Create a copy of the model
        private_model = copy.deepcopy(model)

        # Add noise to model parameters (implementation depends on model type)
        if hasattr(private_model, 'feature_importances_'):
            # For tree-based models
            noise = np.random.laplace(0, noise_scale,
                                    size=private_model.feature_importances_.shape)
            private_model.feature_importances_ += noise

        elif hasattr(private_model, 'coef_'):
            # For linear models
            noise = np.random.laplace(0, noise_scale,
                                    size=private_model.coef_.shape)
            private_model.coef_ += noise

        return private_model

    def private_decision_tree_training(self, X, y, epsilon=None):
        """Train a decision tree with differential privacy"""

        if epsilon is None:
            epsilon = self.epsilon

        # Calculate sensitivity for decision tree splitting
        self.sensitivity = 1.0  # For Gini impurity or information gain

        # Base training
        base_model = DecisionTreeClassifier(random_state=42)
        base_model.fit(X, y)

        # Add differential privacy noise
        private_model = self.add_noise_to_model(base_model, epsilon)

        return private_model

    def private_federated_averaging(self, client_models, epsilon=None):
        """Apply differential privacy to federated learning averaging"""

        if epsilon is None:
            epsilon = self.epsilon

        # Calculate privacy noise for averaging
        # Each client contributes equally to the final model
        noise_scale = self.sensitivity / (epsilon * len(client_models))

        # Average model parameters with noise
        averaged_model = copy.deepcopy(client_models[0])

        for param_name in ['feature_importances_', 'tree_']:
            if hasattr(client_models[0], param_name):
                # Calculate weighted average
                param_sum = np.zeros_like(getattr(client_models[0], param_name))

                for model in client_models:
                    param_sum += getattr(model, param_name)

                # Add differential privacy noise
                param_sum += np.random.laplace(0, noise_scale, param_sum.shape)

                # Set averaged parameters
                setattr(averaged_model, param_name, param_sum / len(client_models))

        return averaged_model

# Differential Privacy in Practice
def train_private_model():
    """Example of training a model with differential privacy"""

    # Initialize differential privacy
    dp = DifferentialPrivacy(epsilon=1.0)  # Lower epsilon = more privacy

    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train private model
    private_model = dp.private_decision_tree_training(X_train, y_train)

    # Evaluate model
    accuracy = accuracy_score(y_test, private_model.predict(X_test))
    print(f"Private model accuracy: {accuracy}")

    return private_model
```

### Explainable AI Implementation

#### SHAP Integration

```python
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class ExplainableModel:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None

        self._setup_explainer()

    def _setup_explainer(self):
        """Setup SHAP explainer based on model type"""

        if isinstance(self.model, RandomForestClassifier):
            self.explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'coef_'):
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, self.X_train.sample(100)
            )

    def explain_prediction(self, instance, plot=True):
        """Generate explanation for a single prediction"""

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        if plot:
            # Create SHAP force plot
            plt.figure(figsize=(10, 6))

            if isinstance(shap_values, list):
                # Multi-class classification
                for i, sv in enumerate(shap_values):
                    if i == 1:  # Plot positive class
                        shap.force_plot(
                            self.explainer.expected_value[i],
                            sv[0],
                            self.X_train.columns,
                            matplotlib=True
                        )
                        plt.title(f"Feature Importance for Class {i}")
                        plt.tight_layout()
                        plt.show()
            else:
                # Binary classification or regression
                shap.force_plot(
                    self.explainer.expected_value,
                    shap_values[0],
                    self.X_train.columns,
                    matplotlib=True
                )
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.show()

        return shap_values

    def global_feature_importance(self, plot=True):
        """Calculate global feature importance"""

        # Get SHAP values for entire dataset
        self.shap_values = self.explainer.shap_values(self.X_train)

        if plot:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, self.X_train, show=False)
            plt.tight_layout()
            plt.show()

            # Bar plot of mean absolute SHAP values
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.shap_values,
                self.X_train,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.show()

        return self.shap_values

    def generate_explanation_report(self, instance_idx=0):
        """Generate comprehensive explanation report"""

        instance = self.X_train.iloc[instance_idx]
        prediction = self.model.predict(instance.reshape(1, -1))[0]

        # SHAP explanation
        shap_values = self.explain_prediction(instance, plot=False)

        # Feature contributions
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        feature_contributions = dict(zip(
            self.X_train.columns,
            shap_values[0]
        ))

        # Sort by absolute contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Generate report
        report = {
            'instance_id': instance_idx,
            'prediction': prediction,
            'feature_contributions': sorted_contributions,
            'positive_contributors': [
                (feat, contrib) for feat, contrib in sorted_contributions
                if contrib > 0
            ],
            'negative_contributors': [
                (feat, contrib) for feat, contrib in sorted_contributions
                if contrib < 0
            ],
            'most_influential_features': sorted_contributions[:5]
        }

        return report

# Usage
def create_explainable_model():
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create explainable model wrapper
    explainer = ExplainableModel(model, X_train)

    # Generate explanations
    explanation = explainer.generate_explanation_report(instance_idx=0)

    print(f"Model predicted: {explanation['prediction']}")
    print("Top 5 most influential features:")
    for feature, contribution in explanation['most_influential_features']:
        print(f"  {feature}: {contribution:.4f}")

    return explainer
```

---

## 5. Performance Optimization Tips {#optimization-tips}

### Model Optimization Techniques

#### Quantization for Model Efficiency

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.quantization as quantization

class QuantizationOptimizer:
    def __init__(self, model):
        self.model = model
        self.quantized_model = None

    def dynamic_quantization(self, inplace=True):
        """Apply dynamic quantization (weights only)"""

        quantized_model = quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
            inplace=inplace
        )

        self.quantized_model = quantized_model
        return quantized_model

    def static_quantization(self, calibration_data):
        """Apply static quantization (weights + activations)"""

        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(self.model, inplace=True)

        # Calibration with representative data
        print("Calibrating model...")
        with torch.no_grad():
            for batch in calibration_data:
                self.model(batch)

        # Convert to quantized model
        self.quantized_model = quantization.convert(self.model, inplace=True)

        return self.quantized_model

    def compare_model_sizes(self):
        """Compare model sizes before and after quantization"""

        # Save original model
        torch.save(self.model.state_dict(), 'original_model.pth')
        original_size = os.path.getsize('original_model.pth')

        # Save quantized model
        if self.quantized_model:
            torch.save(self.quantized_model.state_dict(), 'quantized_model.pth')
            quantized_size = os.path.getsize('quantized_model.pth')

            reduction = (original_size - quantized_size) / original_size * 100

            print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
            print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
            print(f"Size reduction: {reduction:.1f}%")

            return {
                'original_size_mb': original_size / 1024 / 1024,
                'quantized_size_mb': quantized_size / 1024 / 1024,
                'reduction_percent': reduction
            }

        return None

    def benchmark_inference_speed(self, test_loader, num_iterations=100):
        """Benchmark inference speed before and after quantization"""

        results = {}

        # Original model
        print("Benchmarking original model...")
        original_time = self._benchmark_model(self.model, test_loader, num_iterations)
        results['original'] = original_time

        # Quantized model
        if self.quantized_model:
            print("Benchmarking quantized model...")
            quantized_time = self._benchmark_model(
                self.quantized_model, test_loader, num_iterations
            )
            results['quantized'] = quantized_time

            speedup = original_time / quantized_time
            results['speedup'] = speedup

            print(f"Speed improvement: {speedup:.2f}x")

        return results

    def _benchmark_model(self, model, test_loader, num_iterations):
        """Helper method to benchmark model inference"""

        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= num_iterations:
                    break

                # Warm up
                if i == 0:
                    for _ in range(10):
                        _ = model(data)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    start_time = time.time()

                _ = model(data)

        elapsed_time = time.time() - start_time
        return elapsed_time / num_iterations

# Usage
def optimize_model_performance():
    # Load and prepare model
    model = load_model("models/your_model.pth")

    # Initialize optimizer
    optimizer = QuantizationOptimizer(model)

    # Apply quantization
    calibration_loader = DataLoader(calibration_dataset, batch_size=32)
    quantized_model = optimizer.dynamic_quantization()

    # Compare sizes and speeds
    size_comparison = optimizer.compare_model_sizes()
    speed_benchmark = optimizer.benchmark_inference_speed(test_loader)

    return quantized_model, size_comparison, speed_benchmark
```

#### Model Pruning Techniques

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model):
        self.model = model
        self.original_weights = {}

    def magnitude_pruning(self, amount=0.5):
        """Apply magnitude-based unstructured pruning"""

        print(f"Applying {amount*100}% magnitude pruning...")

        # Store original weights for reference
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.original_weights[name] = module.weight.data.clone()

                # Prune the weights
                prune.global_unstructured(
                    module.parameters(),
                    pruning_method=prune.L1Unstructured,
                    amount=amount,
                )

        # Make pruning permanent
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')

        return self.model

    def structured_pruning(self, pruning_ratio=0.2):
        """Apply structured pruning (remove entire neurons/channels)"""

        print(f"Applying {pruning_ratio*100}% structured pruning...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores (using weight magnitudes)
                importance = torch.abs(module.weight.data).sum(dim=1)

                # Determine number of neurons to prune
                num_to_prune = int(module.in_features * pruning_ratio)

                # Find indices of least important neurons
                _, indices = torch.topk(importance, num_to_prune, largest=False)

                # Create mask
                mask = torch.ones(module.in_features)
                mask[indices] = 0

                # Apply mask
                module.weight.data *= mask.unsqueeze(1)

            elif isinstance(module, nn.Conv2d):
                # Similar approach for convolutional layers
                importance = torch.abs(module.weight.data).sum(dim=(1, 2, 3))

                num_to_prune = int(module.out_channels * pruning_ratio)
                _, indices = torch.topk(importance, num_to_prune, largest=False)

                mask = torch.ones(module.out_channels)
                mask[indices] = 0

                # Apply mask
                module.weight.data *= mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return self.model

    def gradual_pruning(self, initial_sparsity=0.1, final_sparsity=0.9,
                       schedule='polynomial', epochs=100):
        """Apply gradual pruning schedule"""

        def polynomial_schedule(current_epoch, total_epochs, initial, final):
            if schedule == 'polynomial':
                return final + (initial - final) * (1 - current_epoch / total_epochs) ** 3
            return initial

        # This is a simplified implementation
        # In practice, you'd integrate this with your training loop

        for epoch in range(epochs):
            current_sparsity = polynomial_schedule(
                epoch, epochs, initial_sparsity, final_sparsity
            )

            print(f"Epoch {epoch+1}/{epochs}: Sparsity {current_sparsity:.3f}")

            # Apply pruning for this epoch
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.global_unstructured(
                        module.parameters(),
                        pruning_method=prune.L1Unstructured,
                        amount=current_sparsity,
                    )

        return self.model

    def measure_sparsity(self):
        """Measure model sparsity after pruning"""

        total_params = 0
        zero_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()

        sparsity_ratio = zero_params / total_params
        print(f"Model sparsity: {sparsity_ratio:.3f} ({zero_params}/{total_params} zero params)")

        return sparsity_ratio

# Pruning scheduler for integration with training
class PruningScheduler:
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.5):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.current_epoch = 0

    def update_pruning(self, epoch, total_epochs):
        """Update pruning amount based on current epoch"""

        self.current_epoch = epoch

        # Calculate target sparsity for this epoch
        if total_epochs > 1:
            progress = epoch / total_epochs
            target_sparsity = self.initial_sparsity + \
                (self.final_sparsity - self.initial_sparsity) * progress
        else:
            target_sparsity = self.final_sparsity

        # Apply pruning
        self._apply_pruning(target_sparsity)

        return target_sparsity

    def _apply_pruning(self, sparsity):
        """Apply target sparsity to model"""

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Remove previous pruning
                try:
                    prune.remove(module, 'weight')
                except:
                    pass

                # Apply new pruning
                prune.global_unstructured(
                    module.parameters(),
                    pruning_method=prune.L1Unstructured,
                    amount=sparsity,
                )
```

---

## 6. Production Deployment Checklist {#deployment-checklist}

### Pre-Deployment Checklist

```yaml
Technical Readiness:
  - [ ] Model achieves target accuracy on test data
  - [ ] Model passes bias and fairness tests
  - [ ] Inference latency meets SLA requirements
  - [ ] Model size fits within memory constraints
  - [ ] Error handling and edge cases tested
  - [ ] API endpoints documented and tested
  - [ ] Security review completed
  - [ ] Performance benchmarking completed

Data Readiness:
  - [ ] Training data quality validated
  - [ ] Feature engineering pipeline tested
  - [ ] Data preprocessing steps documented
  - [ ] Feature drift monitoring setup
  - [ ] Data lineage tracked

Infrastructure Readiness:
  - [ ] Containerized deployment tested
  - [ ] Auto-scaling configuration verified
  - [ ] Monitoring and alerting setup
  - [ ] Logging pipeline configured
  - [ ] Rollback strategy implemented
  - [ ] Load testing completed

Operational Readiness:
  - [ ] Runbook created and reviewed
  - [ ] On-call procedures documented
  - [ ] Incident response plan established
  - [ ] Model versioning system implemented
  - [ ] A/B testing framework ready

Compliance & Ethics:
  - [ ] Privacy impact assessment completed
  - [ ] Bias testing results reviewed
  - [ ] Model interpretability requirements met
  - [ ] Regulatory compliance verified
  - [ ] Data retention policies implemented
```

### Deployment Script Templates

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-service
  labels:
    app: ai-model-service
    version: v1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ai-model-service
  template:
    metadata:
      labels:
        app: ai-model-service
        version: v1.0
    spec:
      containers:
        - name: ai-model
          image: ai-model:latest
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_PATH
              value: "/models/model.pkl"
            - name: MAX_BATCH_SIZE
              value: "32"
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
              nvidia.com/gpu: 1
            limits:
              memory: "4Gi"
              cpu: "2000m"
              nvidia.com/gpu: 1
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
              mountPath: /models
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model-service
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer
```

#### Docker Deployment Script

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "app.py"]
```

#### Terraform Infrastructure

```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and networking
resource "aws_vpc" "ai_ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "ai-ml-vpc"
  }
}

resource "aws_subnet" "public_subnet" {
  count             = var.availability_zones
  vpc_id            = aws_vpc.ai_ml_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "ai-ml-public-subnet-${count.index + 1}"
  }
}

# EKS cluster
resource "aws_eks_cluster" "ai_ml_cluster" {
  name     = "ai-ml-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.21"

  vpc_config {
    subnet_ids = aws_subnet.public_subnet[*].id
  }
}

# Node group with GPU support
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.ai_ml_cluster.name
  node_group_name = "gpu-nodes"
  node_role       = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.public_subnet[*].id

  instance_types = ["p3.2xlarge"]  # GPU instance type

  scaling_config {
    desired_size = 2
    max_size     = 5
    min_size     = 1
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Name = "ai-ml-gpu-nodes"
  }
}

# RDS for model metadata
resource "aws_db_instance" "model_metadata" {
  identifier = "ai-ml-metadata"

  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = "aimlmeta"
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.rds_subnet_group.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "ai-ml-meta-final-snapshot"

  tags = {
    Name = "ai-ml-model-metadata"
  }
}
```

### Monitoring & Alerting Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: "ai-model-service"
    static_configs:
      - targets: ["ai-model-service:8080"]
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: "mlflow-tracking"
    static_configs:
      - targets: ["mlflow-tracking:5000"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

#### Alert Rules

```yaml
# alert_rules.yml
groups:
  - name: ai_model_alerts
    rules:
      - alert: ModelInferenceLatencyHigh
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High model inference latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ModelAccuracyLow
        expr: model_accuracy < 0.80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy has dropped below threshold"
          description: "Current accuracy: {{ $value }}"

      - alert: DataDriftDetected
        expr: psi_score > 0.2
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected in input features"
          description: "PSI score: {{ $value }}"

      - alert: GPUUtilizationHigh
        expr: gpu_utilization_percent > 90
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High GPU utilization detected"
          description: "GPU utilization is {{ $value }}%"
```

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "AI Model Performance Dashboard",
    "panels": [
      {
        "title": "Model Accuracy Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "Error %"
          }
        ]
      }
    ]
  }
}
```

---

## 7. Common Pitfalls & Solutions {#pitfalls-solutions}

### Data-Related Pitfalls

#### Data Leakage Prevention

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class DataLeakageChecker:
    def __init__(self, data, target_col, feature_cols):
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols

    def check_temporal_leakage(self, timestamp_col):
        """Check for temporal leakage in time series data"""

        # Sort by timestamp
        data_sorted = self.data.sort_values(timestamp_col)

        # Check if target contains future information
        target_values = data_sorted[self.target_col]

        # Detect if target is a future value
        # This is a simplified check - adjust based on your use case
        leakage_indicators = []

        for i in range(1, len(target_values)):
            # Check if target at time i could have been influenced by
            # features at time i+1 (future)
            future_target = target_values.iloc[i]
            current_target = target_values.iloc[i-1]

            if abs(future_target - current_target) > target_values.std():
                leakage_indicators.append(i)

        if leakage_indicators:
            print(f"⚠️  Potential temporal leakage detected at indices: {leakage_indicators[:10]}")
            return False
        else:
            print("✅ No temporal leakage detected")
            return True

    def check_feature_correlation(self, threshold=0.95):
        """Check for highly correlated features that might cause leakage"""

        feature_data = self.data[self.feature_cols]
        correlation_matrix = feature_data.corr()

        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        if high_correlations:
            print("⚠️  High correlations detected:")
            for corr in high_correlations[:10]:  # Show first 10
                print(f"  {corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")

            print("\nRecommendations:")
            print("- Remove one feature from highly correlated pairs")
            print("- Use dimensionality reduction techniques")
            print("- Apply feature selection methods")

        return high_correlations

    def check_target_leakage(self):
        """Check if any feature is too correlated with target"""

        if self.target_col not in self.data.columns:
            return True

        target_correlations = {}
        for feature in self.feature_cols:
            if feature in self.data.columns:
                corr = self.data[feature].corr(self.data[self.target_col])
                target_correlations[feature] = corr

        # Sort by absolute correlation
        sorted_correlations = sorted(
            target_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        highly_correlated = [(feat, corr) for feat, corr in sorted_correlations
                           if abs(corr) > 0.8]

        if highly_correlated:
            print("⚠️  Features highly correlated with target (potential leakage):")
            for feat, corr in highly_correlated:
                print(f"  {feat}: {corr:.3f}")

            print("\nRecommendations:")
            print("- Investigate these features for data leakage")
            print("- Consider if these features should be available at prediction time")

        return highly_correlated

    def suggest_preventive_measures(self):
        """Suggest measures to prevent data leakage"""

        measures = [
            "✅ Use time-aware splits for time series data",
            "✅ Exclude future information from training features",
            "✅ Remove highly correlated features with target",
            "✅ Use proper cross-validation techniques",
            "✅ Document data collection process and timing",
            "✅ Implement feature engineering pipeline with proper timing"
        ]

        print("🛡️  Data Leakage Prevention Measures:")
        for measure in measures:
            print(f"  {measure}")

        return measures

# Usage
def validate_data_quality():
    # Load your data
    data = pd.read_csv('your_data.csv')

    # Initialize leakage checker
    checker = DataLeakageChecker(
        data=data,
        target_col='target',
        feature_cols=['feature1', 'feature2', 'feature3']
    )

    # Run checks
    temporal_ok = checker.check_temporal_leakage('timestamp')
    correlation_ok = checker.check_feature_correlation()
    target_ok = checker.check_target_leakage()

    # Get suggestions
    measures = checker.suggest_preventive_measures()

    return {
        'temporal_leakage': temporal_ok,
        'feature_correlations': correlation_ok,
        'target_leakage': target_ok,
        'preventive_measures': measures
    }
```

#### Class Imbalance Solutions

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt

class ClassImbalanceHandler:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.original_distribution = Counter(y)

    def analyze_imbalance(self):
        """Analyze class imbalance"""

        total_samples = len(self.y)
        imbalance_ratios = {}

        # Calculate imbalance ratios
        max_count = max(self.original_distribution.values())
        for class_label, count in self.original_distribution.items():
            ratio = max_count / count
            imbalance_ratios[class_label] = ratio

        print("📊 Class Distribution:")
        for class_label, count in self.original_distribution.items():
            percentage = count / total_samples * 100
            print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")

        print("\n📈 Imbalance Analysis:")
        severe_classes = []
        for class_label, ratio in imbalance_ratios.items():
            if ratio > 10:
                severe_classes.append(class_label)
                print(f"  ⚠️  Class {class_label}: {ratio:.1f}:1 ratio (SEVERE)")
            elif ratio > 5:
                print(f"  ⚠️  Class {class_label}: {ratio:.1f}:1 ratio (MODERATE)")
            else:
                print(f"  ✅ Class {class_label}: {ratio:.1f}:1 ratio (ACCEPTABLE)")

        return {
            'distribution': self.original_distribution,
            'imbalance_ratios': imbalance_ratios,
            'severe_classes': severe_classes,
            'needs_treatment': len(severe_classes) > 0
        }

    def apply_smote(self, random_state=42):
        """Apply SMOTE oversampling"""

        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(self.X, self.y)

        new_distribution = Counter(y_resampled)

        print("🔄 SMOTE Resampling Results:")
        print(f"  Original: {self.original_distribution}")
        print(f"  After:    {new_distribution}")

        return X_resampled, y_resampled

    def apply_adasyn(self, random_state=42):
        """Apply ADASYN oversampling"""

        adasyn = ADASYN(random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(self.X, self.y)

        new_distribution = Counter(y_resampled)

        print("🔄 ADASYN Resampling Results:")
        print(f"  Original: {self.original_distribution}")
        print(f"  After:    {new_distribution}")

        return X_resampled, y_resampled

    def apply_combined_sampling(self, method='smoteenn', random_state=42):
        """Apply combined oversampling and undersampling"""

        if method == 'smoteenn':
            sampler = SMOTEENN(random_state=random_state)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")

        X_resampled, y_resampled = sampler.fit_resample(self.X, self.y)

        new_distribution = Counter(y_resampled)

        print(f"🔄 {method.upper()} Resampling Results:")
        print(f"  Original: {self.original_distribution}")
        print(f"  After:    {new_distribution}")

        return X_resampled, y_resampled

    def apply_cost_sensitive_learning(self, model_class):
        """Apply cost-sensitive learning approach"""

        # Calculate class weights
        total_samples = len(self.y)
        n_classes = len(self.original_distribution)

        class_weights = {}
        for class_label, count in self.original_distribution.items():
            class_weights[class_label] = total_samples / (n_classes * count)

        print("💰 Cost-Sensitive Learning Weights:")
        for class_label, weight in class_weights.items():
            print(f"  Class {class_label}: {weight:.3f}")

        # Create model with class weights
        model = model_class(class_weight=class_weights)

        return model, class_weights

    def evaluate_resampling_methods(self, model_class, test_size=0.2):
        """Evaluate different resampling methods"""

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )

        methods = {
            'baseline': (X_train, y_train),
            'smote': self.apply_smote_to_data(X_train, y_train),
            'adasyn': self.apply_adasyn_to_data(X_train, y_train),
            'undersample': self.apply_undersampling_to_data(X_train, y_train),
        }

        results = {}

        print("📊 Comparing Resampling Methods:")
        print("=" * 50)

        for method_name, (X_res, y_res) in methods.items():
            # Train model
            model = model_class(random_state=42)
            model.fit(X_res, y_res)

            # Evaluate
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            results[method_name] = {
                'f1_macro': report['macro avg']['f1-score'],
                'f1_weighted': report['weighted avg']['f1-score'],
                'support': Counter(y_res)
            }

            print(f"{method_name:12} - F1 Macro: {report['macro avg']['f1-score']:.3f}")

        return results

    def apply_smote_to_data(self, X, y):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)

    def apply_adasyn_to_data(self, X, y):
        adasyn = ADASYN(random_state=42)
        return adasyn.fit_resample(X, y)

    def apply_undersampling_to_data(self, X, y):
        undersampler = RandomUnderSampler(random_state=42)
        return undersampler.fit_resample(X, y)

# Usage
def handle_class_imbalance():
    # Load your imbalanced data
    X, y = load_imbalanced_data()

    # Initialize handler
    handler = ClassImbalanceHandler(X, y)

    # Analyze imbalance
    analysis = handler.analyze_imbalance()

    if analysis['needs_treatment']:
        # Try different approaches
        X_smote, y_smote = handler.apply_smote()
        X_combined, y_combined = handler.apply_combined_sampling('smoteenn')

        # Evaluate methods
        results = handler.evaluate_resampling_methods(
            model_class=RandomForestClassifier,
            test_size=0.2
        )

        # Choose best method
        best_method = max(results.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n🏆 Best method: {best_method[0]} (F1 Macro: {best_method[1]['f1_macro']:.3f})")

        return results
    else:
        print("✅ Class imbalance is within acceptable limits")
        return None
```

---

## 8. Quick Command Reference {#command-reference}

### Essential Terminal Commands

#### ML Environment Setup

```bash
# Create Python virtual environment
python -m venv ml-env
source ml-env/bin/activate  # Linux/Mac
# ml-env\Scripts\activate  # Windows

# Install ML libraries
pip install tensorflow pytorch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install mlflow wandb dvc
pip install jupyter notebook

# GPU setup verification
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Jupyter configuration
jupyter notebook --generate-config
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### Model Training & Evaluation

```bash
# Train with TensorFlow
python train.py --model-type=cnn --epochs=100 --batch-size=32

# Train with PyTorch
python train_pytorch.py --data-path=data/train.csv --epochs=50

# Cross-validation
python evaluate.py --cv-folds=5 --model=random_forest

# Hyperparameter tuning
python tune_hparams.py --search-type=random --n-trials=100

# Model comparison
python compare_models.py --models=rf,svm,xgb --dataset=breast_cancer
```

#### Data Management

```bash
# DVC commands
dvc init
dvc add data/raw/train.csv
dvc add models/model.pkl
dvc push
dvc pull
dvc repro  # Reproduce pipeline
dvc metrics show  # Show metrics

# Data processing
python preprocess.py --input=data/raw.csv --output=data/processed.csv
python feature_engineering.py --config=config/features.yaml

# Dataset split
python split_data.py --train-size=0.8 --val-size=0.1 --test-size=0.1
```

#### Model Deployment

```bash
# Docker build and run
docker build -t ai-model:latest .
docker run -p 8080:8080 ai-model:latest

# Kubernetes deployment
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl logs -f deployment/ai-model-service

# Model serving
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
flask run --host=0.0.0.0 --port=5000
```

#### Monitoring & Debugging

```bash
# System monitoring
htop  # Process monitor
nvidia-smi  # GPU monitoring
df -h  # Disk usage
free -h  # Memory usage

# Python debugging
python -m pdb script.py  # Debugger
python -c "import pyinstrument; pyinstrument.start()"  # Profiler

# MLflow tracking
mlflow server --host 0.0.0.0 --port 5000
mlflow experiments list
mlflow runs list --experiment-id 0

# TensorBoard
tensorboard --logdir=logs/ --host 0.0.0.0 --port 6006
```

### Python Code Snippets

#### Model Loading & Saving

```python
# PyTorch
import torch

# Save
torch.save(model.state_dict(), 'model.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)

# TensorFlow/Keras
import tensorflow as tf

# Save
model.save('model.h5')
tf.saved_model.save(model, 'saved_model/')

# Load
model = tf.keras.models.load_model('model.h5')
model = tf.saved_model.load('saved_model/')

# scikit-learn
from joblib import dump, load

# Save
dump(model, 'model.joblib')

# Load
model = load('model.joblib')
```

#### Performance Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

def evaluate_model(y_true, y_pred, y_proba=None):
    """Comprehensive model evaluation"""

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ROC AUC if probabilities available
    if y_proba is not None:
        try:
            auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr')
            print(f"ROC AUC: {auc_score:.4f}")
        except:
            print("ROC AUC: Cannot calculate (check if all classes are represented)")

    # Classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score if y_proba is not None else None
    }

# Usage
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
metrics = evaluate_model(y_test, y_pred, y_proba)
```

#### Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col, test_size=0.2):
    """Complete data preprocessing pipeline"""

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values
    X = X.fillna(X.mean())  # Numerical
    X = X.fillna(X.mode().iloc[0])  # Categorical

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Encode target if categorical
    if y.dtype == 'object':
        y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, le

# Usage
X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(df, 'target')
```

---

## 9. Tools & Resources {#tools-resources}

### Development Tools

#### Essential Python Libraries

```python
# Core ML Libraries
CORE_ML = {
    "scikit-learn": "General machine learning algorithms",
    "pandas": "Data manipulation and analysis",
    "numpy": "Numerical computing",
    "matplotlib": "Basic plotting",
    "seaborn": "Statistical visualization"
}

# Deep Learning Frameworks
DL_FRAMEWORKS = {
    "tensorflow": "Google's deep learning framework",
    "pytorch": "Facebook's deep learning framework",
    "keras": "High-level neural network API",
    "transformers": "State-of-the-art NLP models"
}

# Specialized Libraries
SPECIALIZED = {
    "opencv-python": "Computer vision",
    "nltk": "Natural language processing",
    "spacy": "Industrial-strength NLP",
    "gensim": "Topic modeling and similarity detection",
    "lightgbm": "Gradient boosting framework",
    "xgboost": "Extreme gradient boosting"
}

# MLOps Libraries
MLOPS = {
    "mlflow": "Machine learning lifecycle management",
    "wandb": "Experiment tracking and visualization",
    "dvc": "Data version control",
    "airflow": "Workflow orchestration",
    "kubernetes": "Container orchestration",
    "docker": "Containerization"
}
```

#### Recommended VS Code Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.pylint",
    "ms-python.black-formatter",
    "ms-toolsai.jupyter",
    "visualstudioexptteam.vscodeintellicode",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-docker",
    "github.copilot",
    "ms-vscode.vscode-markdown"
  ]
}
```

#### Environment Setup Scripts

```bash
#!/bin/bash
# setup-ml-env.sh

echo "🚀 Setting up ML Environment..."

# Create virtual environment
python -m venv ml-env
source ml-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core libraries
echo "📦 Installing core ML libraries..."
pip install numpy pandas matplotlib seaborn jupyter

# Install ML frameworks
echo "🧠 Installing ML frameworks..."
pip install scikit-learn tensorflow pytorch torchvision torchaudio

# Install MLOps tools
echo "🔧 Installing MLOps tools..."
pip install mlflow wandb dvc prefect

# Install development tools
echo "🛠️ Installing development tools..."
pip install black flake8 isort pytest pytest-cov

# Create project structure
echo "📁 Creating project structure..."
mkdir -p {data/{raw,processed},models,notebooks,tests,src,configs,logs}

# Save requirements
pip freeze > requirements.txt

echo "✅ Environment setup complete!"
echo "Activate with: source ml-env/bin/activate"
```

### Resource Links

#### Learning Resources

```yaml
Documentation:
  - Scikit-learn: https://scikit-learn.org/stable/user_guide.html
  - TensorFlow: https://www.tensorflow.org/guide
  - PyTorch: https://pytorch.org/docs/stable/index.html
  - MLOps: https://ml-ops.org/

Tutorials:
  - FastAI: https://course.fast.ai/
  - CS231n (Stanford): http://cs231n.stanford.edu/
  - Deep Learning Specialization (Coursera)
  - Machine Learning Yearning (Andrew Ng)

Tools:
  - Google Colab: https://colab.research.google.com/
  - Kaggle: https://www.kaggle.com/
  - Papers With Code: https://paperswithcode.com/
  - Hugging Face: https://huggingface.co/
```

#### Datasets for Practice

```yaml
Computer Vision:
  - MNIST: Digit recognition
  - CIFAR-10/100: Object classification
  - ImageNet: Large-scale image classification
  - COCO: Object detection and segmentation

Natural Language Processing:
  - IMDB Reviews: Sentiment analysis
  - AG News: News classification
  - SQuAD: Question answering
  - WikiText: Language modeling

Time Series:
  - Airline Passengers: Seasonal patterns
  - Stock Prices: Financial forecasting
  - Energy Consumption: Demand prediction

Recommendation Systems:
  - MovieLens: Movie ratings
  - Amazon Reviews: Product reviews
  - Last.fm: Music listening data
```

#### Production Deployment Resources

```yaml
Containerization:
  - Docker: https://docs.docker.com/
  - Kubernetes: https://kubernetes.io/docs/
  - Helm: https://helm.sh/docs/

Cloud Platforms:
  - AWS SageMaker: https://aws.amazon.com/sagemaker/
  - Google AI Platform: https://cloud.google.com/ai-platform
  - Azure ML: https://azure.microsoft.com/en-us/services/machine-learning/

Monitoring:
  - Prometheus: https://prometheus.io/docs/
  - Grafana: https://grafana.com/docs/
  - Weights & Biases: https://docs.wandb.com/
  - MLflow: https://mlflow.org/docs/

Ethics & Fairness:
  - AI Ethics Guidelines: https://www.oecd.ai/
  - Fairlearn: https://fairlearn.org/
  - AI Fairness 360: http://aif360.mybluemix.net/
```

---

## 🎯 Quick Reference Summary

### Hardware Minimum Requirements by Task

- **Beginner Learning**: RTX 3050, 16GB RAM, 8GB VRAM
- **Computer Vision**: RTX 3070, 32GB RAM, 12GB VRAM
- **NLP/LLM**: RTX 3080+, 64GB RAM, 24GB VRAM
- **Production**: RTX 4080+, 128GB RAM, Multiple GPUs

### Key MLOps Commands

```bash
# MLflow
mlflow run . -P epochs=100
mlflow models serve -m models/production_model

# DVC
dvc repro  # Reproduce pipeline
dvc metrics show  # Show metrics

# Model deployment
kubectl apply -f k8s-deployment.yaml
docker run -p 8080:8080 ai-model:latest
```

### Performance Optimization Checklist

- [ ] Enable mixed precision training (FP16)
- [ ] Use batch size for memory efficiency
- [ ] Apply model quantization (INT8)
- [ ] Implement gradient checkpointing
- [ ] Use data loaders with multiple workers
- [ ] Enable model parallelism for large models
- [ ] Apply gradient accumulation
- [ ] Use efficient optimizers (AdamW, LAMB)

### Ethics & Fairness Must-Checks

- [ ] Test for demographic parity
- [ ] Check equalized odds
- [ ] Verify equal opportunity
- [ ] Analyze feature importance
- [ ] Document model limitations
- [ ] Test across demographic groups
- [ ] Monitor for bias drift
- [ ] Implement human oversight

### Production Readiness Checklist

- [ ] Model accuracy ≥ target threshold
- [ ] Latency meets SLA requirements
- [ ] Error handling implemented
- [ ] Monitoring and alerting setup
- [ ] Rollback strategy prepared
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Load testing performed

---

_Last Updated: November 2025 | Version 1.0_

**Happy Learning! 🚀**

This cheat sheet provides comprehensive coverage of advanced AI/ML topics from hardware selection to production deployment. Use it as your quick reference guide for real-world AI/ML implementation and don't forget to bookmark it for easy access!
