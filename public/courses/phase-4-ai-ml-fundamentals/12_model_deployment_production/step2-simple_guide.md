# Model Deployment & Production Systems - Universal Guide

## Taking Your AI from Lab to Real World - Made Simple!

_Learn how to share your AI creations with the world - from your computer to helping millions of users!_

---

## 🎯 How to Use This Guide

### 📚 **For Absolute Beginners**

- Start with **"What is AI Deployment?"** - understand the basics
- Read **"Why Deploy Models?"** - learn why it matters
- Focus on **Simple Examples** - see real-world applications

### ⚡ **For Hands-On Practice**

- Try **Step-by-Step Deployment** - follow along with examples
- Use **Ready-to-Use Templates** - deploy without technical setup
- Practice with **Test Environments** - try before going live

### 🚀 **For Professional Development**

- Study **Advanced Strategies** - learn enterprise-level approaches
- Explore **Cloud Platforms** - let experts handle the hard parts
- Master **MLOps Practices** - learn to maintain AI systems

### 💡 **What You'll Learn**

- How to turn your AI model into something people can actually use
- Where to host your AI (from simple free options to complex systems)
- How to keep your AI running smoothly and fix problems
- How to make your AI handle lots of users at once
- How to monitor and improve your AI over time

### 📖 **Table of Contents**

#### **🚀 Getting Started**

1. [What is AI Deployment? - From Computer to Cloud](#1-introduction-to-model-deployment)
2. [Simple Ways to Share Your AI - Easy Deployment Methods](#2-deployment-strategies)

#### **☁️ Cloud Platforms (Let Experts Help)**

3. [Cloud Platforms Overview - Big Companies Offering AI Hosting](#3-cloud-platforms-overview)
4. [Amazon Web Services (AWS) - Popular Cloud Platform](#4-aws-sagemaker-deployment)
5. [Google Cloud Platform (GCP) - Google's AI Platform](#5-google-cloud-platform-gcp-vertex-ai)
6. [Microsoft Azure - Microsoft's Cloud Solution](#6-microsoft-azure-ml)

#### **🔧 Technical Deployment**

7. [Docker Containers - Packaging Your AI](#7-docker-containerization)
8. [Kubernetes - Managing Many AI Systems](#8-kubernetes-orchestration)
9. [API Development - Creating AI Interfaces](#12-api-development)

#### **🎯 Professional Practices**

10. [MLOps Practices - Maintaining AI Systems](#9-mlops-practices)
11. [Model Monitoring - Keeping Track of AI Performance](#10-model-monitoring--management)
12. [Scaling Strategies - Handling More Users](#11-scaling-strategies)

#### **🌟 Real-World Applications**

13. [Real-World Workflows - How Companies Actually Do It](#13-real-world-deployment-workflows)
14. [Security & Best Practices - Keeping AI Safe](#14-security--best-practices)
15. [Troubleshooting - Fixing Common Problems](#15-troubleshooting--optimization)

---

## 1. What is AI Deployment? - From Computer to Cloud 🌟

### **The Simple Answer**

Think of AI deployment like **opening a restaurant**:

- **Your AI Model** = Your secret recipe that you perfected
- **Training** = You practiced cooking the recipe at home
- **Deployment** = Opening the restaurant so customers can order your dish
- **Infrastructure** = The kitchen, tables, waiters, etc.
- **Customers** = People using your AI to solve their problems

### **Why Bother Deploying?**

**The Problem:** Right now, your AI is like a perfect recipe sitting in your kitchen notebook - only you know about it!

**The Solution:** Deployment shares your AI with the world so it can actually help people:

- **Before Deployment:** 📱 AI sits on your laptop doing nothing
- **After Deployment:** 🌍 AI helps thousands of people every day

### **Real-World Examples You Already Use**

#### **Instagram Filters** 📸

- **What happens:** Camera analyzes your face instantly
- **AI Deployment:** Instagram's servers receive your photo, run face-detection AI, apply filters, send result back
- **Speed:** All happens in milliseconds!

#### **Google Maps Directions** 🗺️

- **What happens:** App finds the best route to your destination
- **AI Deployment:** Your phone sends request to Google's servers, AI calculates route considering traffic, returns directions
- **Intelligence:** AI learned from millions of trips to optimize routes

#### **Netflix Recommendations** 🎬

- **What happens:** App suggests movies you might like
- **AI Deployment:** Netflix's AI analyzes your viewing history compared to millions of other users, recommends matches
- **Learning:** AI gets smarter with every recommendation

### **How AI Deployment Works - Step by Step**

#### **Step 1: Prepare Your AI** 📦

```
Your Trained Model + Instructions → Ready-to-Deploy Package
```

#### **Step 2: Choose Where to Host** ☁️

- **Option A:** Your own computer (simple, limited users)
- **Option B:** Cloud service like AWS or Google (professional, scalable)
- **Option C:** Your company's servers (custom, secure)

#### **Step 3: Set Up the Interface** 📡

```
User sends data → AI receives → AI processes → AI sends result back
```

#### **Step 4: Monitor and Improve** 📊

- Track how accurate AI is
- See how fast it responds
- Update AI when it makes mistakes

### **Two Types of AI Use - Timing Matters!**

#### **🕒 Real-Time AI (Instant Results)**

- **What:** AI responds immediately while you wait
- **Examples:** Face unlock, voice assistants, spell check
- **Speed Required:** Under 1 second
- **Why it matters:** Users get instant gratification

#### **📅 Batch Processing AI (Scheduled Results)**

- **What:** AI processes data in the background and delivers later
- **Examples:** Netflix recommendations, email spam filtering, daily weather reports
- **Speed Required:** Can take minutes to hours
- **Why it matters:** More thorough analysis possible

### **Common Deployment Challenges (And Why They Matter)**

1. **🚀 Speed Problem:** Users don't want to wait
   - **Solution:** Optimize code, use better computers

2. **💪 Reliability Problem:** AI must work 24/7
   - **Solution:** Backup systems, error handling

3. **📈 Growth Problem:** More users = more load
   - **Solution:** Automatic scaling (like adding more cashiers)

4. **📊 Quality Problem:** Track if AI is working well
   - **Solution:** Monitoring dashboards, user feedback

5. **🔄 Updates Problem:** AI needs to improve over time
   - **Solution:** Version control, gradual rollouts

6. **🔒 Security Problem:** Protect against bad actors
   - **Solution:** Authentication, data encryption

---

## 2. Deployment Strategies

### Blue-Green Deployment

**Concept**: Keep two identical environments - blue (current) and green (new). Switch traffic between them.

**Real-world Example**: Like changing lanes on a highway

- **Blue Lane**: Current traffic (existing model)
- **Green Lane**: New traffic (new model)
- **Switch**: Direct all traffic to green lane when ready
- **Rollback**: Instant switch back to blue if problems

**Code Example**:

```python
# Blue-Green Deployment Implementation
import os
import json
from datetime import datetime

class BlueGreenDeployment:
    def __init__(self, model_registry_path):
        self.model_registry = model_registry_path

    def get_current_environment(self):
        """Check which environment is currently serving"""
        with open(f"{self.model_registry}/current_env.txt", "r") as f:
            return f.read().strip()

    def switch_to_new_model(self, new_model_version):
        """Switch traffic to new model"""
        current_env = self.get_current_environment()
        new_env = "green" if current_env == "blue" else "blue"

        # Update model registry
        deployment_data = {
            "new_version": new_model_version,
            "deployment_time": datetime.now().isoformat(),
            "target_env": new_env
        }

        with open(f"{self.model_registry}/deployment.json", "w") as f:
            json.dump(deployment_data, f)

        # Switch load balancer (pseudo-code)
        self.update_load_balancer(new_env, new_model_version)

        print(f"Switched traffic to {new_env} environment with model {new_model_version}")

    def rollback(self):
        """Quick rollback to previous version"""
        current_env = self.get_current_environment()
        rollback_env = "blue" if current_env == "green" else "green"

        with open(f"{self.model_registry}/rollback.json", "w") as f:
            json.dump({"rollback_to": rollback_env}, f)

        print(f"Rolled back to {rollback_env} environment")

# Usage
deployment = BlueGreenDeployment("/models/registry/")
deployment.switch_to_new_model("v2.1.0")  # Deploy new model
# If issues arise:
deployment.rollback()  # Instant rollback
```

**Advantages**:

- Zero downtime during deployment
- Instant rollback capability
- Safe testing environment
- Easy verification before full switch

**Disadvantages**:

- Requires double infrastructure (2x cost)
- Database synchronization challenges
- More complex monitoring setup

### Canary Deployment

**Concept**: Release new model to small percentage of users first, then gradually increase.

**Real-world Example**: Like testing new coffee recipe

- **1%**: Give to coffee enthusiasts to try
- **10%**: Expand to regular customers
- **50%**: Half of all customers
- **100%**: All customers, only if feedback is good

**Code Example**:

```python
import random
import numpy as np
from typing import Dict, List

class CanaryDeployment:
    def __init__(self):
        self.traffic_allocation = {
            "current_model": 100.0,
            "new_model": 0.0
        }
        self.metrics_tracker = {
            "current_model": {"accuracy": [], "latency": []},
            "new_model": {"accuracy": [], "latency": []}
        }

    def assign_user_to_model(self, user_id: str) -> str:
        """Assign user to model based on traffic allocation"""
        allocation_percentage = random.uniform(0, 100)

        if allocation_percentage <= self.traffic_allocation["new_model"]:
            return "new_model"
        else:
            return "current_model"

    def gradually_increase_canary(self, increment_percentage: float = 5.0,
                                max_canary_percentage: float = 50.0):
        """Gradually increase traffic to new model"""
        current_canary = self.traffic_allocation["new_model"]
        new_canary = min(current_canary + increment_percentage, max_canary_percentage)

        self.traffic_allocation["new_model"] = new_canary
        self.traffic_allocation["current_model"] = 100.0 - new_canary

        print(f"Increased canary traffic to {new_canary}%")

    def make_prediction(self, model_input: Dict, user_id: str):
        """Make prediction using canary routing"""
        assigned_model = self.assign_user_to_model(user_id)

        if assigned_model == "current_model":
            prediction = self.current_model.predict(model_input)
            self.track_metric("current_model", prediction)
        else:
            prediction = self.new_model.predict(model_input)
            self.track_metric("new_model", prediction)

        return prediction, assigned_model

    def track_metric(self, model_name: str, prediction_result: Dict):
        """Track performance metrics for each model"""
        self.metrics_tracker[model_name]["accuracy"].append(prediction_result.get("confidence", 0.0))
        self.metrics_tracker[model_name]["latency"].append(prediction_result.get("response_time", 0.0))

    def should_promote_canary(self, metrics_threshold: float = 0.95) -> bool:
        """Determine if canary should be promoted based on metrics"""
        if self.traffic_allocation["new_model"] < 10.0:
            return False  # Not enough traffic yet

        new_model_metrics = self.metrics_tracker["new_model"]
        current_model_metrics = self.metrics_tracker["current_model"]

        if not new_model_metrics["accuracy"]:
            return False

        new_avg_accuracy = np.mean(new_model_metrics["accuracy"])
        current_avg_accuracy = np.mean(current_model_metrics["accuracy"])

        # Promote if new model is at least as good as current model
        return new_avg_accuracy >= (current_avg_accuracy * metrics_threshold)

# Usage
canary = CanaryDeployment()

# Gradual deployment
for day in range(10):
    # Serve traffic
    for request in get_requests_for_day(day):
        prediction, model_used = canary.make_prediction(request["input"], request["user_id"])
        log_prediction(request, prediction, model_used)

    # Evaluate and increase traffic
    if canary.should_promote_canary():
        canary.gradually_increase_canary()
```

**Advantages**:

- Early problem detection
- Gradual risk exposure
- Real user feedback
- Automatic rollback if metrics degrade

**Disadvantages**:

- More complex monitoring
- Inconsistent user experience during rollout
- Requires sophisticated routing logic

### A/B Testing Deployment

**Concept**: Compare two or more model versions by randomly assigning users to different versions.

**Real-world Example**: Like testing two different store layouts

- **Group A**: See new layout for 2 weeks
- **Group B**: Keep old layout for 2 weeks
- **Compare**: Which generates more sales?
- **Decision**: Choose the better layout

**Code Example**:

```python
import hashlib
import random
from collections import defaultdict
from typing import Dict, List

class ABTesting:
    def __init__(self, experiments: Dict):
        self.experiments = experiments  # {"experiment_name": {"versions": ["v1", "v2"], "weights": [50, 50]}}
        self.user_assignments = {}
        self.results = defaultdict(lambda: defaultdict(list))

    def get_user_hash(self, user_id: str) -> int:
        """Get consistent hash for user assignment"""
        return int(hashlib.md5(user_id.encode()).hexdigest(), 16)

    def assign_to_version(self, user_id: str, experiment_name: str) -> str:
        """Assign user to a specific version based on hash"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        user_hash = self.get_user_hash(user_id)
        versions = self.experiments[experiment_name]["versions"]
        weights = self.experiments[experiment_name]["weights"]

        # Create deterministic assignment based on hash
        normalized_hash = user_hash % 100

        # Map hash to version based on weights
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if normalized_hash < cumulative_weight:
                version = versions[i]
                self.user_assignments[user_id] = {"experiment": experiment_name, "version": version}
                return version

        # Fallback to last version
        return versions[-1]

    def get_model_for_user(self, user_id: str, experiment_name: str):
        """Get appropriate model for user based on assignment"""
        assignment = self.user_assignments.get(user_id, {})

        if assignment.get("experiment") != experiment_name:
            assignment = self.assign_to_version(user_id, experiment_name)
        else:
            assignment = assignment["version"]

        return f"model_{assignment}"

    def record_result(self, user_id: str, experiment_name: str,
                     metric_name: str, metric_value: float):
        """Record result for A/B testing analysis"""
        user_version = self.user_assignments.get(user_id, {}).get("version", "unknown")
        self.results[experiment_name][user_version].append({
            "metric": metric_name,
            "value": metric_value,
            "timestamp": datetime.now().isoformat()
        })

    def analyze_results(self, experiment_name: str, metric_name: str) -> Dict:
        """Analyze A/B test results using statistical tests"""
        results = self.results[experiment_name]
        version_names = list(results.keys())

        if len(version_names) < 2:
            return {"error": "Need at least 2 versions to compare"}

        # Get metric values for each version
        version_values = {}
        for version in version_names:
            version_values[version] = [
                r["value"] for r in results[version]
                if r["metric"] == metric_name
            ]

        # Calculate statistics
        analysis = {}
        for version, values in version_values.items():
            if values:
                analysis[version] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values),
                    "confidence_interval_95": 1.96 * np.std(values) / np.sqrt(len(values))
                }

        # Determine statistical significance
        if len(version_names) == 2:
            v1_values = version_values[version_names[0]]
            v2_values = version_values[version_names[1]]

            # Perform t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(v1_values, v2_values)

            analysis["statistical_test"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "winner": version_names[0] if np.mean(v1_values) > np.mean(v2_values) else version_names[1]
            }

        return analysis

# Usage
experiments = {
    "recommendation_model": {
        "versions": ["baseline", "improved_v2"],
        "weights": [50, 50]
    },
    "pricing_model": {
        "versions": ["current", "dynamic_pricing"],
        "weights": [70, 30]  # 70% get current, 30% get dynamic
    }
}

ab_test = ABTesting(experiments)

# Serve predictions
def serve_recommendation(user_id: str, user_preferences: Dict):
    model_name = ab_test.get_model_for_user(user_id, "recommendation_model")

    if model_name == "model_baseline":
        prediction = baseline_model.predict(user_preferences)
    else:
        prediction = improved_v2_model.predict(user_preferences)

    # Record user interaction for analysis
    ab_test.record_result(user_id, "recommendation_model", "click_rate",
                         prediction.get("clicked", False))

    return prediction

# Analyze results
results = ab_test.analyze_results("recommendation_model", "click_rate")
print("A/B Test Results:", results)
```

**Advantages**:

- Scientific comparison of model versions
- Clear statistical significance testing
- Can run multiple experiments simultaneously
- Data-driven decision making

**Disadvantages**:

- Requires statistical expertise
- Need to wait for statistical significance
- Potential revenue loss if new model is worse
- Complex user experience across experiments

---

## 3. Cloud Platforms Overview

### Why Use Cloud Platforms?

**Simple Answer**: Like renting an apartment vs buying a house

- **Cloud**: Rent servers when you need them (flexible, no maintenance)
- **Own Servers**: Buy and maintain servers (expensive, complex)

**Key Benefits**:

- **Scalability**: Automatic scaling up/down based on demand
- **Cost-effective**: Pay only for what you use
- **Reliability**: 99.9% uptime guarantees
- **Security**: Enterprise-grade security built-in
- **Global reach**: Deploy worldwide instantly

### Platform Comparison

| Feature           | AWS SageMaker | GCP Vertex AI | Azure ML |
| ----------------- | ------------- | ------------- | -------- |
| **Ease of Use**   | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐ |
| **Cost**          | ⭐⭐⭐        | ⭐⭐⭐⭐      | ⭐⭐⭐⭐ |
| **Integration**   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐      | ⭐⭐⭐⭐ |
| **ML Features**   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐      | ⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐      | ⭐⭐⭐   |

### Cost Estimation Framework

**Real-world Example**: E-commerce recommendation system

- **Daily requests**: 100,000 predictions/day
- **Model size**: 500MB
- **Processing time**: 50ms per prediction

**AWS SageMaker**:

```
Endpoints: $0.096/hour = $69/day
Predictions: $0.0003 per 1K requests = $30/day
Data transfer: $0.09/GB = $2/day
Total: ~$101/day
```

**GCP Vertex AI**:

```
Endpoints: $0.091/hour = $66/day
Predictions: $0.0003 per 1K requests = $30/day
Data transfer: $0.12/GB = $3/day
Total: ~$99/day
```

**Azure ML**:

```
Endpoints: $0.10/hour = $72/day
Predictions: $0.0004 per 1K requests = $40/day
Data transfer: $0.087/GB = $2/day
Total: ~$114/day
```

---

## 4. AWS SageMaker Deployment

### What is AWS SageMaker?

SageMaker is like a complete kitchen for machine learning:

- **Training Studio**: Like a well-equipped kitchen for cooking (training)
- **Jupyter Notebooks**: Your recipe notebook for experiments
- **Model Registry**: Pantry to store your best recipes (models)
- **Endpoint Deployment**: Restaurant kitchen to serve customers (predictions)
- **Auto Scaling**: More chefs when more customers arrive

### Setting Up SageMaker Environment

```python
import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get execution role (permissions for SageMaker)
role = get_execution_role()

# Set region
region = boto3.session.Session().region_name

print(f"Role: {role}")
print(f"Region: {region}")
print(f"Session: {sagemaker_session}")
```

### Training Models in SageMaker

```python
from sagemaker.sklearn.estimator import SKLearn
import os

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',  # Training instance type
    framework_version='0.23-1',
    py_version='py3',
    script_mode=True,
    hyperparameters={
        'epochs': 50,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# Training script (train.py)
# Example content for train.py
training_script = '''
import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--validation', type=str, default='/opt/ml/input/data/validation')

    args = parser.parse_args()

    # Load data
    train_data = np.loadtxt(f'{args.train}/train.csv', delimiter=',', skiprows=1)
    val_data = np.loadtxt(f'{args.validation}/validation.csv', delimiter=',', skiprows=1)

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]

    # Train model
    model = RandomForestClassifier(n_estimators=args.epochs, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f'Validation Accuracy: {accuracy:.4f}')

    # Save model
    joblib.dump(model, f'{args.model_dir}/model.joblib')

    print("Training completed successfully!")

if __name__ == '__main__':
    main()
'''

# Write training script
os.makedirs('sagemaker_code', exist_ok=True)
with open('sagemaker_code/train.py', 'w') as f:
    f.write(training_script)

# Start training job
sklearn_estimator.fit({
    'train': 's3://my-bucket/training-data/train/',
    'validation': 's3://my-bucket/training-data/validation/'
})
```

### Deploying Models to SageMaker Endpoints

```python
# Deploy model to real-time endpoint
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'  # Inference instance type
)

# Make predictions
import numpy as np

# Test data
test_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Iris flower features

# Get prediction
response = predictor.predict(test_data)
print("Prediction:", response)

# Clean up (delete endpoint)
# predictor.delete_endpoint()
```

### Creating Batch Transform Jobs

```python
from sagemaker.transformer import Transformer

# Create transformer for batch predictions
transformer = sklearn_estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://my-bucket/batch-predictions/',
    assemble_with='Line',
    output_content_type='text/csv'
)

# Run batch transform
transformer.transform(
    data='s3://my-bucket/batch-data/test-data.csv',
    content_type='text/csv',
    split_type='Line'
)
```

### SageMaker Pipelines for MLOps

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.parameters import ParameterInteger

# Define parameters
input_data = ParameterString(name="InputData")
model_approval = ParameterString(name="ModelApproval")

# Processing step (data preprocessing)
processor = sagemaker.sklearn.SKLearnProcessor(
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    instance_count=1
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    code='preprocess.py',
    inputs=[sagemaker.inputs.Input(
        source=input_data,
        destination='/opt/ml/processing/input'
    )],
    outputs=[
        sagemaker.outputs.Output(
            source='/opt/ml/processing/output/train',
            destination='s3://my-bucket/preprocessing/train'
        ),
        sagemaker.outputs.Output(
            source='/opt/ml/processing/output/validation',
            destination='s3://my-bucket/preprocessing/validation'
        )
    ]
)

# Training step
training_step = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={
        'train': sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri,
            content_type='text/csv'
        ),
        'validation': sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['validation_data'].S3Output.S3Uri,
            content_type='text/csv'
        )
    }
)

# Create pipeline
pipeline = Pipeline(
    name="MyMLPipeline",
    parameters=[input_data, model_approval],
    steps=[processing_step, training_step]
)

# Execute pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Pipeline execution started:", execution.arn)
```

### Model Monitoring and Debugging

```python
# Set up model monitoring
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Create monitor
monitor = DefaultModelMonitor(
    role=role,
    instance_type='ml.m5.xlarge'
)

# Create monitoring schedule
monitor.create_monitoring_schedule(
    endpoint_input=predictor.endpoint_name,
    output_s3_uri='s3://my-bucket/monitoring-results/',
    schedule_cron_expression='cron(0 * * * ? *)',  # Every hour
    instance_count=1
)

# Generate reports
monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/baseline-data/baseline.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri='s3://my-bucket/baseline-results/'
)
```

---

## 5. Google Cloud Platform (GCP) Vertex AI

### What is Vertex AI?

Vertex AI is like Google's smart assistant for machine learning:

- **AutoML**: Let Google automatically design the best model
- **Custom Training**: Bring your own code and models
- **Unified Platform**: All ML tasks in one place
- **Integration**: Works seamlessly with Google services

### Setting Up Vertex AI

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import pandas as pd

# Initialize Vertex AI
aiplatform.init(
    project='your-project-id',
    location='us-central1'
)

# Define experiment
experiment = aiplatform.Experiment.create(
    display_name='iris-classification-exp',
    description='Experiment for Iris flower classification'
)
```

### Training with AutoML

```python
# Create dataset
dataset = aiplatform.ImageDataset.create(
    display_name='iris-dataset',
    gcs_source=['gs://my-bucket/image-data/'],
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
)

# Wait for import to complete
dataset.wait()

# Create AutoML training job
job = aiplatform.AutoMLImageTrainingJob(
    display_name='iris-automl-training',
    optimization_prediction_type='classification',
    budget_milli_node_hours=8000,
    model_display_name='iris-classifier'
)

# Run training
model = job.run(
    dataset=dataset,
    model_display_name='iris-classifier-v2',
    training_fraction_split=0.8,
    validation_fraction_split=0.2,
    test_fraction_split=0.0,
    budget_milli_node_hours=8000
)
```

### Custom Training Jobs

```python
# Create custom training job
job = aiplatform.CustomJob(
    display_name='iris-custom-training',
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "gcr.io/your-project/iris-trainer:latest",
                "args": ["--epochs=100", "--learning-rate=0.001"]
            }
        }
    ]
)

# Run the job
model = job.run(
    dataset=dataset,
    model_display_name='iris-custom-model',
    replica_count=1
)
```

### Batch Prediction

```python
# Create batch prediction job
batch_prediction_job = model.batch_predict(
    job_display_name='iris-batch-prediction',
    gcs_source=['gs://my-bucket/input-data/batch-input.jsonl'],
    gcs_destination_prefix='gs://my-bucket/predictions/',
    instances_format='jsonl',
    predictions_format='jsonl'
)

# Wait for completion
batch_prediction_job.wait()
```

### Real-time Endpoints

```python
# Deploy model to endpoint
endpoint = model.deploy(
    endpoint_display_name='iris-endpoint',
    deployed_model_id='iris-model-v1',
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    traffic_split={"0": 100}
)

# Make prediction
response = endpoint.predict(
    instances=[
        {
            "content": "base64_image_content"
        }
    ]
)

print("Prediction:", response)
```

### Vertex AI Feature Store

```python
from google.cloud.aiplatform import Featurestore

# Create feature store
featurestore = Featurestore.create(
    featurestore_id="user-behavior-store",
    online_serving_config={
        "enabled": True
    }
)

# Create entity type
entity_type = featurestore.create_entity_type(
    entity_type_id="user",
    feature_specs=[
        {"feature_id": "age", "value_type": "INT64"},
        {"feature_id": "preferences", "value_type": "STRING"},
        {"feature_id": "purchase_history", "value_type": "STRING"}
    ]
)

# Online serving
featurestore.batch_read_feature_values(
    entity_ids=['user123', 'user456'],
    feature_ids=['age', 'preferences', 'purchase_history']
)
```

### Model Registry

```python
# Register model in registry
model_registry = aiplatform.ModelRegistry(
    registry_uri="google.us-central1",
    project="your-project-id"
)

# Register trained model
registered_model = model_registry.register_model(
    model_display_name='iris-classifier',
    model=model.resource_name,
    model_version='1.0.0',
    serving_container_ports=[8080],
    serving_container_predict_route='/predict',
    serving_container_health_route='/health'
)

# Create model version
model_version = registered_model.version("1.0.0")
```

### Pipeline Integration

```python
from google.cloud.aiplatform import pipeline_jobs

# Create pipeline job
pipeline_job = pipeline_jobs.PipelineJob(
    template_path='iris-training-pipeline.yaml',
    display_name='iris-pipeline-job',
    parameter_values={
        'project_id': 'your-project-id',
        'dataset_resource_name': dataset.resource_name,
        'model_display_name': 'iris-classifier-v3'
    }
)

# Run pipeline
pipeline_job.run()
```

---

## 6. Microsoft Azure ML

### What is Azure ML?

Azure ML is Microsoft's cloud platform for machine learning, like a complete factory for building and deploying AI models:

- **Designer**: Drag-and-drop interface for creating ML pipelines
- **AutoML**: Automatic model selection and hyperparameter tuning
- **Compute Clusters**: Scalable computing for training
- **Model Registry**: Centralized model management
- **Endpoints**: Real-time and batch prediction services

### Setting Up Azure ML

```python
from azureml.core import Workspace, Experiment, Dataset, ComputeTarget
from azureml.train.automl import AutoMLConfig
from azureml.core.model import Model
import azureml.core
from azureml.core.authentication import InteractiveLoginAuthentication

# Initialize authentication
auth = InteractiveLoginAuthentication()
subscription_id = "your-subscription-id"
resource_group = "your-resource-group"
workspace_name = "your-workspace-name"

# Connect to workspace
workspace = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    auth=auth
)

print(f"Azure ML Workspace: {workspace.name}")
print(f"Region: {workspace.location}")
print(f"Subscription: {workspace.subscription_id}")
```

### Creating Compute Targets

```python
from azureml.core.compute import AmlCompute, ComputeTarget

# Create compute cluster for training
compute_name = "gpu-cluster"
compute_config = AmlCompute.provisioning_configuration(
    vm_size="Standard_NC6",  # GPU instance
    min_nodes=0,
    max_nodes=4,
    idle_seconds_before_scaledown=300
)

compute_target = ComputeTarget.create(workspace, compute_name, compute_config)
compute_target.wait_for_completion(show_output=True)

print(f"Compute cluster {compute_name} is ready")
```

### Training with AutoML

```python
# Load training data
dataset = Dataset.get_by_name(workspace, name='iris-dataset')

# Configure AutoML
automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    compute_target=compute_target,
    training_data=dataset,
    validation_data=dataset,
    label_column_name='species',
    experiment_timeout_hours=1,
    max_concurrent_iterations=4,
    max_cores_per_iteration=1,
    n_cross_validations=5
)

# Run experiment
experiment = Experiment(workspace, 'iris-automl-experiment')
run = experiment.submit(automl_config, show_output=True)

# Get best model
best_run, fitted_model = run.get_output()
print(f"Best run ID: {best_run.id}")
print(f"Best model accuracy: {best_run.get_metrics()['accuracy']}")
```

### Custom Training with ScriptRunConfig

```python
from azureml.core import ScriptRunConfig
from azureml.core.environment import Environment

# Create environment
env = Environment.from_conda_specification(
    name='iris-env',
    file_path='./environment.yml'
)

# Register environment
env.register(workspace)

# Create training script
training_script = '''
import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--model', type=str, help='Path to save model')
    parser.add_argument('--n-estimators', type=int, default=100)

    args = parser.parse_args()

    # Load data
    data = np.loadtxt(args.data, delimiter=',', skiprows=1)
    X, y = data[:, :-1], data[:, -1]

    # Train model
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X, y)

    # Evaluate
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Training accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, args.model)

    print("Training completed!")

if __name__ == '__main__':
    main()
'''

# Write training script
with open('train.py', 'w') as f:
    f.write(training_script)

# Create run configuration
config = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=['--data', 'data/train.csv', '--model', 'outputs/model.joblib', '--n-estimators', '200'],
    environment=env,
    compute_target=compute_target
)

# Run experiment
experiment = Experiment(workspace, 'custom-iris-training')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
```

### Model Deployment

```python
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice

# Register model
model = Model.register(
    workspace=workspace,
    model_name='iris-classifier',
    model_path='outputs/model.joblib',
    model_framework='ScikitLearn',
    model_framework_version='0.23.1'
)

print(f"Model registered: {model.name}, version: {model.version}")

# Deploy to AKS
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig

# Create inference configuration
inference_config = InferenceConfig(
    environment=env,
    entry_script='score.py'
)

# Scoring script (score.py)
scoring_script = '''
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('iris-classifier')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        predictions = model.predict(data)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == '__main__':
    init()
'''

# Write scoring script
with open('score.py', 'w') as f:
    f.write(scoring_script)

# Configure deployment
aks_config = AksWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    enable_app_insights=True
)

# Deploy model
service = Model.deploy(
    workspace=workspace,
    name='iris-endpoint',
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=None
)

service.wait_for_deployment(show_output=True)

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
```

### Batch Inference Pipeline

```python
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Datastore

# Define outputs
output_data = PipelineData(
    name='predictions',
    datastore=workspace.get_default_datastore(),
    output_path_on_compute='predictions.csv'
)

# Create batch inference step
batch_step = PythonScriptStep(
    name='batch_inference',
    script_name='batch_predict.py',
    arguments=['--input-data', input_data, '--output-data', output_data],
    inputs=[input_data],
    outputs=[output_data],
    compute_target=compute_target,
    allow_reuse=False
)

# Create pipeline
pipeline = Pipeline(
    workspace=workspace,
    steps=[batch_step]
)

# Run pipeline
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
```

### Model Monitoring

```python
from azureml.core import Workspace
from azureml.interpret import ExplanationClient
from azureml.contrib.interpret.shap import TabularExplainer

# Create model explainer
explainer = TabularExplainer(
    model=best_run,
    initialization_examples=X_train,
    features=dataset.column_names[:-1],
    classes=dataset.column_names[-1]
)

# Get explanations
explanation = explainer.explain_local(X_test)
explanation = explanation.get_explanation_dict()

print(f"Model explanations generated: {len(explanation)} features explained")
```

---

## 7. Docker Containerization

### What is Docker?

Docker is like packing your entire application in a shipping container:

- **Consistent Environment**: Same setup everywhere (your laptop, staging, production)
- **Isolation**: Each app runs in its own container (like apartment units in building)
- **Portability**: Runs on any computer with Docker
- **Version Control**: Track changes to your environment

**Real-world Analogy**: Like moving houses with all your furniture packed in boxes - everything travels together.

### Docker Basics

```dockerfile
# Example Dockerfile for ML model serving
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

### Docker Requirements File

```
# requirements.txt
flask==2.3.2
gunicorn==20.1.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
psutil==5.9.5
prometheus-client==0.17.1
opencv-python==4.7.1.72
transformers==4.30.2
torch==2.0.1
tensorflow==2.13.0
```

### Multi-stage Dockerfile for Optimized Images

```dockerfile
# Multi-stage build for smaller final image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH
ENV PATH="/home/appuser/.local/bin:$PATH"

# Create app user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Copy application
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set environment
ENV PYTHONPATH="/home/appuser/.local/lib/python3.9/site-packages:$PYTHONPATH"
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["python", "app.py"]
```

### Flask Application with Docker

```python
# app.py - Production-ready Flask application
import os
import json
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import joblib
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'Request duration')

app = Flask(__name__)

# Global variables for model and metadata
model = None
model_metadata = {}

def load_model():
    """Load the ML model with error handling"""
    global model, model_metadata

    try:
        model_path = os.getenv('MODEL_PATH', 'models/model.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)

        # Load model metadata
        metadata_path = os.getenv('METADATA_PATH', 'models/metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model metadata: {model_metadata}")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate input data"""
    if not data:
        return False, "No data provided"

    if 'input' not in data:
        return False, "Missing 'input' key in request"

    input_array = data['input']
    if not isinstance(input_array, list):
        return False, "Input must be a list"

    try:
        np.array(input_array, dtype=np.float32)
    except (ValueError, TypeError):
        return False, "Input must be numeric values"

    return True, ""

def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """Preprocess input data"""
    input_array = np.array(data['input'], dtype=np.float32)

    # Reshape for single prediction
    if len(input_array.shape) == 1:
        input_array = input_array.reshape(1, -1)

    return input_array

@app.before_first_request
def startup():
    """Initialize the application"""
    logger.info("Starting ML application...")
    load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'unhealthy', 'error': 'Model not loaded'}), 503

    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat(),
        'model_metadata': model_metadata
    })

@app.route('/predict', methods=['POST'])
@REQUEST_DURATION.time()
def predict():
    """Prediction endpoint"""
    start_time = time.time()

    try:
        # Increment request counter
        REQUEST_COUNT.labels(method=request.method, endpoint='/predict').inc()

        # Validate input
        data = request.get_json()
        is_valid, error_message = validate_input(data)

        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Preprocess input
        input_data = preprocess_input(data)

        # Make prediction
        prediction = model.predict(input_data)

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            result = {
                'prediction': prediction.tolist(),
                'probability': probabilities.tolist(),
                'confidence': np.max(probabilities, axis=1).tolist()
            }
        else:
            result = {
                'prediction': prediction.tolist(),
                'confidence': 1.0
            }

        # Add metadata
        result.update({
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata.get('version', 'unknown'),
            'processing_time': time.time() - start_time
        })

        logger.info(f"Prediction successful: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    info = {
        'model_type': type(model).__name__,
        'model_metadata': model_metadata,
        'features': model_metadata.get('features', []),
        'classes': model_metadata.get('classes', []) if hasattr(model, 'classes_') else []
    }

    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Load model at startup
    load_model()

    # Run with Gunicorn in production
    if os.environ.get('USE_GUNICORN', 'False').lower() == 'true':
        logger.info("Using Gunicorn for production deployment")
    else:
        logger.info(f"Starting Flask development server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: "3.8"

services:
  ml-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/model.joblib
      - METADATA_PATH=/app/models/metadata.json
      - USE_GUNICORN=True
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro # Mount models directory (read-only)
      - ./logs:/app/logs # Mount logs directory
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    restart: unless-stopped

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

volumes:
  redis-data:
```

### Production Dockerfile with Security

```dockerfile
# Production-optimized Dockerfile
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user --prefix=/usr/local --upgrade pip && \
    pip install --no-cache-dir --user --prefix=/usr/local -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install security updates and runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Remove unnecessary files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -exec rm -rf {} + || true

# Set environment variables
ENV PYTHONPATH="/root/.local/lib/python3.9/site-packages:$PYTHONPATH" \
    PATH="/root/.local/bin:$PATH" \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "sync", "--timeout", "120", "--keep-alive", "5", "--max-requests", "1000", "--max-requests-jitter", "50", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
```

### Docker Commands Reference

```bash
# Build Docker image
docker build -t ml-app:latest .

# Run container locally
docker run -p 8080:8080 \
  -e MODEL_PATH=/app/models/model.joblib \
  -v $(pwd)/models:/app/models:ro \
  ml-app:latest

# Run with Docker Compose
docker-compose up -d

# View logs
docker logs -f ml-app

# Execute commands in running container
docker exec -it ml-app bash

# Inspect container
docker inspect ml-app

# Check resource usage
docker stats ml-app

# Clean up
docker-compose down
docker system prune -a
```

---

## 8. Kubernetes Orchestration

### What is Kubernetes?

Kubernetes is like an army of robot chefs that manage your restaurant:

- **Automated Deployment**: Automatically places your app on available servers
- **Self-Healing**: If a server crashes, automatically restarts your app elsewhere
- **Scaling**: Automatically adds more servers when you're busy
- **Load Balancing**: Distributes customers across available chefs

**Real-world Example**: Like Uber managing drivers - automatically assigns rides to available drivers and scales up during peak hours.

### Kubernetes Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │    Master Node  │    │  Worker Node 1  │
│                 │    │                 │    │                 │
│ - Ingress       │───▶│ - API Server    │───▶│ - Pod 1         │
│ - SSL Term      │    │ - Scheduler     │    │   (ML App)      │
│ - Route Rules   │    │ - Controller    │    │ - Pod 2         │
│                 │    │   Manager       │    │   (Monitoring)  │
└─────────────────┘    │ - etcd          │    └─────────────────┘
                       │   Database      │
                       └─────────────────┘
                              │
                       ┌─────────────────┐
                       │  Worker Node 2  │
                       │                 │
                       │ - Pod 1         │
                       │   (ML App)      │
                       │ - Pod 2         │
                       │   (Cache)       │
                       └─────────────────┘
```

### Kubernetes Deployment Manifest

```yaml
# ml-app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-app-deployment
  labels:
    app: ml-app
    version: v1
spec:
  replicas: 3 # Number of pod replicas
  selector:
    matchLabels:
      app: ml-app
  template:
    metadata:
      labels:
        app: ml-app
        version: v1
    spec:
      containers:
        - name: ml-app
          image: ml-app:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: MODEL_PATH
              value: "/app/models/model.joblib"
            - name: MODEL_VERSION
              value: "v1.2.3"
            - name: LOG_LEVEL
              value: "INFO"
            - name: REDIS_URL
              value: "redis://redis-service:6379"
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
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-storage-pvc
      restartPolicy: Always
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values:
                      - ml-app
              topologyKey: "kubernetes.io/hostname"
```

### Service Manifest

```yaml
# ml-app-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-app-service
  labels:
    app: ml-app
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
      name: http
  selector:
    app: ml-app
---
apiVersion: v1
kind: Service
metadata:
  name: ml-app-headless
  labels:
    app: ml-app
spec:
  clusterIP: None # Headless service for direct pod access
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
      name: http
  selector:
    app: ml-app
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - ml-app.yourdomain.com
      secretName: ml-app-tls
  rules:
    - host: ml-app.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-app-service
                port:
                  number: 80
```

### ConfigMap and Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-app-config
data:
  LOG_LEVEL: "INFO"
  MODEL_PATH: "/app/models/model.joblib"
  MAX_WORKERS: "4"
  TIMEOUT: "120"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-app-secret
type: Opaque
data:
  database-password: YWRtaW46cGFzc3dvcmQxMjM= # base64 encoded
  api-key: ZXl4YmFwaWtleTEyMzQ1Njc4OWFiY2Rl # base64 encoded
```

### Persistent Volume Claim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard # Adjust based on your cloud provider
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-app-deployment
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

### PodDisruptionBudget

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-app-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ml-app
```

### Kubernetes Deployment Script

```bash
#!/bin/bash
# deploy.sh - Kubernetes deployment script

set -e

NAMESPACE="ml-app"
IMAGE_TAG="latest"
REGISTRY="your-registry.com"

echo "🚀 Deploying ML App to Kubernetes..."

# Apply ConfigMaps and Secrets
echo "📝 Applying ConfigMaps and Secrets..."
kubectl apply -f configmap.yaml -n $NAMESPACE
kubectl apply -f secret.yaml -n $NAMESPACE

# Apply PersistentVolumeClaims
echo "💾 Applying PersistentVolumeClaims..."
kubectl apply -f pvc.yaml -n $NAMESPACE

# Apply Deployments
echo "🚀 Applying Deployments..."
kubectl apply -f ml-app-deployment.yaml -n $NAMESPACE

# Apply Services
echo "🌐 Applying Services..."
kubectl apply -f ml-app-service.yaml -n $NAMESPACE

# Apply Ingress
echo "🔗 Applying Ingress..."
kubectl apply -f ingress.yaml -n $NAMESPACE

# Apply HorizontalPodAutoscaler
echo "📈 Applying HorizontalPodAutoscaler..."
kubectl apply -f hpa.yaml -n $NAMESPACE

# Apply PodDisruptionBudget
echo "🛡️ Applying PodDisruptionBudget..."
kubectl apply -f pdb.yaml -n $NAMESPACE

# Wait for rollout
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/ml-app-deployment -n $NAMESPACE --timeout=300s

# Check status
echo "📊 Deployment Status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE
kubectl get hpa -n $NAMESPACE

echo "✅ Deployment completed successfully!"
```

### Monitoring and Observability

```yaml
# monitoring.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-app-monitor
  labels:
    app: ml-app
spec:
  selector:
    matchLabels:
      app: ml-app
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
---
apiVersion: logging.coreos.com/v1
kind: ClusterLogForwarder
metadata:
  name: ml-app-logging
spec:
  outputs:
    - name: elasticsearch-output
      type: elasticsearch
      url: https://elasticsearch.yourdomain.com:9200
  pipelines:
    - name: application-logs
      inputRefs:
        - application
      filterRefs:
        - json-filter
      outputRefs:
        - elasticsearch-output
```

### Blue-Green Deployment with Kubernetes

```bash
#!/bin/bash
# blue-green-deployment.sh

NAMESPACE="ml-app"
CURRENT_COLOR="blue"  # or "green"
NEW_COLOR="green"     # or "blue"

# Deploy new version with different color
sed "s/REPLACE_COLOR/$NEW_COLOR/g" deployment-blue.yaml | \
kubectl apply -f - -n $NAMESPACE

# Wait for new version to be ready
echo "⏳ Waiting for new version to be ready..."
kubectl rollout status deployment/ml-app-$NEW_COLOR -n $NAMESPACE

# Update service selector to point to new version
kubectl patch service ml-app-service -n $NAMESPACE \
  -p '{"spec":{"selector":{"app":"ml-app","color":"'$NEW_COLOR'"}}}'

echo "✅ Traffic switched to $NEW_COLOR version"

# Keep old version for quick rollback
echo "💡 Old $CURRENT_COLOR version kept for rollback"

# Rollback function
rollback() {
    echo "🔄 Rolling back to $CURRENT_COLOR version..."
    kubectl patch service ml-app-service -n $NAMESPACE \
      -p '{"spec":{"selector":{"app":"ml-app","color":"'$CURRENT_COLOR'"}}}'
    echo "✅ Rolled back to $CURRENT_COLOR version"
}
```

---

## 9. MLOps Practices

### What is MLOps?

MLOps (Machine Learning Operations) is like running a successful restaurant chain:

- **Consistency**: Same recipe quality everywhere
- **Efficiency**: Streamlined kitchen operations
- **Monitoring**: Track customer satisfaction and costs
- **Scaling**: Grow to multiple locations smoothly

**Key Components**:

- **Version Control**: Track changes to code, data, and models
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Track model performance and data drift
- **Governance**: Ensure compliance and security

### CI/CD Pipeline for ML

```yaml
# .github/workflows/ml-pipeline.yml
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
          pip install pytest pytest-cov black flake8

      - name: Lint code
        run: |
          black --check .
          flake8 .

      - name: Run tests
        run: |
          pytest tests/ --cov=src/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate data
        run: |
          python scripts/validate_data.py \
            --input-path data/raw/ \
            --schema-path schemas/data_schema.json \
            --validation-report data/validation_report.json

      - name: Upload validation results
        uses: actions/upload-artifact@v3
        with:
          name: data-validation-report
          path: data/validation_report.json

  model-training:
    needs: [test, data-validation]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Train model
        run: |
          python scripts/train_model.py \
            --config configs/training_config.yaml \
            --experiment-name ${{ github.sha }} \
            --model-registry s3://ml-models/training/${{ github.sha }}/

      - name: Register model
        run: |
          python scripts/register_model.py \
            --model-path s3://ml-models/training/${{ github.sha }}/model.joblib \
            --model-name "ml-app-${{ github.sha }}" \
            --version "1.0.0-${{ github.sha }}"

  docker-build:
    needs: [test]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
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
    needs: [docker-build, model-training]
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging
        run: |
          helm upgrade --install ml-app-staging ./charts/ml-app \
            --namespace staging \
            --set image.tag=${{ github.sha }} \
            --set model.version=${{ github.sha }} \
            --set environment=staging

  integration-tests:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    steps:
      - name: Run integration tests
        run: |
          python tests/integration/test_api.py \
            --base-url ${{ vars.STAGING_URL }} \
            --api-key ${{ secrets.STAGING_API_KEY }}

  deploy-production:
    needs: [deploy-staging, integration-tests]
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: |
          # Blue-green deployment
          ./scripts/deploy-blue-green.sh \
            --image-tag ${{ github.sha }} \
            --environment production
```

### Model Versioning with MLflow

```python
# model_versioning.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from datetime import datetime

class MLflowModelVersioning:
    def __init__(self, tracking_uri=None, experiment_name="ml-app-experiment"):
        self.tracking_uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = experiment_name

        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    def train_model(self, X_train, y_train, X_val, y_val,
                   model_params=None, run_name=None):
        """Train model and log to MLflow"""

        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'random_state': 42,
                'max_depth': 10
            }

        with mlflow.start_run(run_name=run_name) as run:
            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)

            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)

            # Log parameters
            mlflow.log_params(model_params)

            # Log metrics
            mlflow.log_metrics({
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_val_gap': train_accuracy - val_accuracy
            })

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="ml-app-model"
            )

            # Save model locally
            model_path = f"models/model_{run.info.run_id}.joblib"
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, "models")

            # Log data summary
            data_summary = {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'n_features': X_train.shape[1],
                'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
                'timestamp': datetime.now().isoformat()
            }

            with open("data_summary.json", "w") as f:
                json.dump(data_summary, f, indent=2)
            mlflow.log_artifact("data_summary.json")

            print(f"Model trained - Run ID: {run.info.run_id}")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")

            return run.info.run_id, model

    def register_model(self, run_id, model_name="ml-app-model", stage="Staging"):
        """Register model with MLflow Model Registry"""
        client = MlflowClient()

        try:
            # Create model version
            model_uri = f"runs:/{run_id}/model"
            mv = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=f"Model trained on {datetime.now().strftime('%Y-%m-%d')}"
            )

            # Transition to stage
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage
            )

            print(f"Model registered as {model_name} version {mv.version} in stage {stage}")
            return mv.version

        except Exception as e:
            print(f"Error registering model: {str(e)}")
            return None

    def load_production_model(self, model_name="ml-app-model"):
        """Load the latest production model"""
        client = MlflowClient()

        try:
            # Get latest model in Production stage
            latest_version = client.get_latest_versions(
                model_name,
                stages=["Production"]
            )

            if not latest_version:
                print("No production model found")
                return None

            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)

            print(f"Loaded production model version: {latest_version[0].version}")
            return model

        except Exception as e:
            print(f"Error loading production model: {str(e)}")
            return None

    def compare_models(self, model_name="ml-app-model", stages=["Staging", "Production"]):
        """Compare models in different stages"""
        client = MlflowClient()

        comparisons = {}
        for stage in stages:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                version_info = client.get_run(versions[0].run_id)
                comparisons[stage] = {
                    'version': versions[0].version,
                    'run_id': versions[0].run_id,
                    'metrics': version_info.data.metrics,
                    'params': version_info.data.params
                }

        return comparisons

# Usage example
def train_and_register_model():
    # Sample data (replace with your data)
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize MLflow manager
    mlflow_manager = MLflowModelVersioning(
        tracking_uri="http://localhost:5000",
        experiment_name="iris-classification"
    )

    # Train model
    run_id, model = mlflow_manager.train_model(
        X_train, y_train, X_val, y_val,
        model_params={'n_estimators': 200, 'max_depth': 8},
        run_name="iris-model-v1"
    )

    # Register model
    version = mlflow_manager.register_model(run_id, stage="Staging")

    # Promote to production (after approval)
    if version:
        mlflow_manager.register_model(run_id, stage="Production")

    # Load production model
    production
    production_model = mlflow_manager.load_production_model()

if __name__ == "__main__":
    train_and_register_model()
```

### Data Version Control with DVC

```python
# data_versioning.py
import dvc.api
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataVersionControl:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        os.chdir(repo_path)

    def version_data(self, data_path, version_tag):
        """Add data to DVC for version control"""
        import subprocess

        # Add data file to DVC
        subprocess.run(["dvc", "add", data_path])

        # Create tag for this version
        subprocess.run(["git", "tag", f"data-{version_tag}"])

        print(f"Data versioned with tag: data-{version_tag}")

    def load_versioned_data(self, version_tag):
        """Load data from specific version"""
        try:
            # Checkout specific version
            subprocess.run(["git", "checkout", f"data-{version_tag}"], check=True)
            subprocess.run(["dvc", "checkout"], check=True)

            print(f"Checked out data version: data-{version_tag}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error checking out version: {e}")
            return False

    def compare_data_versions(self, version1, version2):
        """Compare datasets between versions"""
        import subprocess

        # Checkout and load both versions
        v1_data = self._load_data_version(version1)
        v2_data = self._load_data_version(version2)

        if v1_data is not None and v2_data is not None:
            comparison = {
                'version1': version1,
                'version2': version2,
                'shape_v1': v1_data.shape,
                'shape_v2': v2_data.shape,
                'missing_values_v1': v1_data.isnull().sum().to_dict(),
                'missing_values_v2': v2_data.isnull().sum().to_dict(),
                'basic_stats_v1': v1_data.describe().to_dict(),
                'basic_stats_v2': v2_data.describe().to_dict()
            }

            return comparison

    def _load_data_version(self, version_tag):
        """Helper to load specific data version"""
        try:
            subprocess.run(["git", "checkout", f"data-{version_tag}"], check=True)
            subprocess.run(["dvc", "checkout"], check=True)

            # Load your data file (adjust path as needed)
            data_path = "data/processed/processed_data.csv"
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
            return None

        except:
            return None

# Usage
dvc_manager = DataVersionControl()

# Version data
dvc_manager.version_data("data/raw/raw_data.csv", "v1.0")

# Load specific version
dvc_manager.load_versioned_data("v1.0")

# Compare versions
comparison = dvc_manager.compare_data_versions("v1.0", "v2.0")
print(comparison)
```

### Model Monitoring and Drift Detection

```python
# model_monitoring.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ModelMonitoring:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.baseline_metrics = {}
        self.baseline_data = None
        self.drift_detector = IsolationForest(contamination=0.1, random_state=42)
        self.monitoring_data = []

    def set_baseline(self, baseline_data: pd.DataFrame, baseline_labels: pd.Series):
        """Set baseline for comparison"""
        self.baseline_data = baseline_data

        # Calculate baseline metrics
        predictions = self.model.predict(baseline_data)
        self.baseline_metrics = {
            'accuracy': accuracy_score(baseline_labels, predictions),
            'precision': precision_score(baseline_labels, predictions, average='weighted'),
            'recall': recall_score(baseline_labels, predictions, average='weighted'),
            'f1_score': f1_score(baseline_labels, predictions, average='weighted'),
            'n_samples': len(baseline_data)
        }

        # Train drift detector on baseline
        self.drift_detector.fit(baseline_data[self.feature_names])

        logger.info(f"Baseline set with {len(baseline_data)} samples")
        logger.info(f"Baseline accuracy: {self.baseline_metrics['accuracy']:.4f}")

    def monitor_prediction(self, data_point: Dict, prediction_result: Dict):
        """Monitor a single prediction for drift"""
        timestamp = datetime.now()

        # Extract features
        features = [data_point.get(feature, 0) for feature in self.feature_names]

        # Check for data drift
        drift_score = self._check_data_drift(features)

        # Log prediction
        monitoring_record = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction_result.get('prediction'),
            'confidence': prediction_result.get('confidence'),
            'drift_score': drift_score,
            'features': features
        }

        self.monitoring_data.append(monitoring_record)

        # Check for alerts
        alerts = self._check_alerts(monitoring_record)

        if alerts:
            logger.warning(f"Alerts triggered: {alerts}")

        return alerts

    def _check_data_drift(self, features: List[float]) -> float:
        """Check for data drift using isolation forest"""
        features_array = np.array(features).reshape(1, -1)

        # Get anomaly score (-1 for anomaly, 1 for normal)
        anomaly_score = self.drift_detector.decision_function(features_array)[0]

        # Convert to drift score (0-1, higher means more drift)
        drift_score = max(0, -anomaly_score)

        return drift_score

    def _check_alerts(self, record: Dict) -> List[str]:
        """Check for various types of alerts"""
        alerts = []

        # Data drift alert
        if record['drift_score'] > 0.5:
            alerts.append(f"Data drift detected (score: {record['drift_score']:.4f})")

        # Low confidence alert
        if record.get('confidence', 1.0) < 0.7:
            alerts.append(f"Low confidence prediction ({record['confidence']:.4f})")

        return alerts

    def analyze_performance_drift(self, current_data: pd.DataFrame,
                                current_labels: pd.Series) -> Dict:
        """Analyze performance drift over time"""
        current_predictions = self.model.predict(current_data)

        current_metrics = {
            'accuracy': accuracy_score(current_labels, current_predictions),
            'precision': precision_score(current_labels, current_predictions, average='weighted'),
            'recall': recall_score(current_labels, current_predictions, average='weighted'),
            'f1_score': f1_score(current_labels, current_predictions, average='weighted')
        }

        # Calculate drift
        metric_drift = {}
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                drift = (baseline_value - current_value) / baseline_value
                metric_drift[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'drift_percentage': drift * 100,
                    'drift_direction': 'degradation' if drift > 0.05 else 'improvement' if drift < -0.05 else 'stable'
                }

        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'n_samples': len(current_data),
            'metric_drift': metric_drift,
            'overall_status': self._assess_overall_status(metric_drift)
        }

    def _assess_overall_status(self, metric_drift: Dict) -> str:
        """Assess overall model health"""
        degradation_count = sum(
            1 for drift in metric_drift.values()
            if drift['drift_direction'] == 'degradation'
        )

        if degradation_count >= len(metric_drift) * 0.7:
            return "critical"
        elif degradation_count >= len(metric_drift) * 0.3:
            return "warning"
        else:
            return "healthy"

    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        if not self.monitoring_data:
            return {"error": "No monitoring data available"}

        # Analyze recent monitoring data
        recent_data = [d for d in self.monitoring_data
                      if datetime.fromisoformat(d['timestamp']) > datetime.now() - timedelta(days=7)]

        if recent_data:
            avg_drift_score = np.mean([d['drift_score'] for d in recent_data])
            avg_confidence = np.mean([d.get('confidence', 0) for d in recent_data])
            high_drift_count = sum(1 for d in recent_data if d['drift_score'] > 0.5)
        else:
            avg_drift_score = 0
            avg_confidence = 0
            high_drift_count = 0

        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.monitoring_data),
            'recent_predictions': len(recent_data),
            'baseline_metrics': self.baseline_metrics,
            'drift_analysis': {
                'average_drift_score': float(avg_drift_score),
                'high_drift_predictions': high_drift_count,
                'drift_percentage': (high_drift_count / len(recent_data)) * 100 if recent_data else 0
            },
            'confidence_analysis': {
                'average_confidence': float(avg_confidence),
                'low_confidence_predictions': sum(1 for d in recent_data if d.get('confidence', 1) < 0.7)
            },
            'recommendations': self._generate_recommendations(avg_drift_score, avg_confidence)
        }

        return report

    def _generate_recommendations(self, avg_drift_score: float, avg_confidence: float) -> List[str]:
        """Generate monitoring-based recommendations"""
        recommendations = []

        if avg_drift_score > 0.3:
            recommendations.append("Consider retraining model with recent data")
            recommendations.append("Investigate data pipeline for changes")

        if avg_confidence < 0.8:
            recommendations.append("Review feature engineering pipeline")
            recommendations.append("Consider ensemble methods for improved confidence")

        if len(recent_data) < 100:
            recommendations.append("Gather more prediction data for reliable monitoring")

        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges")

        return recommendations

# Usage example
def setup_model_monitoring():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    # Train baseline model
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_baseline, y_train, y_baseline = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Setup monitoring
    monitoring = ModelMonitoring(model, [f"feature_{i}" for i in range(X_baseline.shape[1])])
    monitoring.set_baseline(pd.DataFrame(X_baseline), pd.Series(y_baseline))

    # Monitor new predictions
    test_sample = X_baseline[0].tolist()
    prediction = model.predict([test_sample])[0]
    confidence = max(model.predict_proba([test_sample])[0])

    alerts = monitoring.monitor_prediction(
        data_point={f"feature_{i}": val for i, val in enumerate(test_sample)},
        prediction_result={'prediction': prediction, 'confidence': confidence}
    )

    # Generate report
    report = monitoring.generate_monitoring_report()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    setup_model_monitoring()
```

---

## 10. Model Monitoring & Management

### Prometheus Metrics Collection

```python
# prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import time
import psutil
import threading
from typing import Dict, Any

class ModelMetrics:
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'model_requests_total',
            'Total number of requests to the model',
            ['model_name', 'method', 'status']
        )

        self.request_duration = Histogram(
            'model_request_duration_seconds',
            'Time spent processing requests',
            ['model_name']
        )

        self.request_size = Histogram(
            'model_request_size_bytes',
            'Size of requests in bytes',
            ['model_name']
        )

        self.response_size = Histogram(
            'model_response_size_bytes',
            'Size of responses in bytes',
            ['model_name']
        )

        # Model performance metrics
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Prediction confidence scores',
            ['model_name', 'prediction_type']
        )

        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['model_name', 'version']
        )

        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.gpu_usage = Gauge('system_gpu_usage_percent', 'GPU usage percentage')

        # Business metrics
        self.business_score = Gauge(
            'model_business_score',
            'Business score from model predictions',
            ['model_name', 'user_segment']
        )

        # Start system metrics collection
        self._start_system_monitoring()

    def _start_system_monitoring(self):
        """Start background thread for system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)

                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.percent)

                    # GPU usage (if available)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_usage = gpus[0].load * 100
                            self.gpu_usage.set(gpu_usage)
                    except ImportError:
                        pass  # GPU monitoring not available

                    time.sleep(30)  # Collect every 30 seconds

                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(60)  # Wait longer on error

        # Start monitoring thread
        monitor_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        monitor_thread.start()

    def record_request(self, model_name: str, method: str, status: str,
                      duration: float, request_size: int, response_size: int):
        """Record request metrics"""
        self.request_count.labels(
            model_name=model_name,
            method=method,
            status=status
        ).inc()

        self.request_duration.labels(model_name=model_name).observe(duration)
        self.request_size.labels(model_name=model_name).observe(request_size)
        self.response_size.labels(model_name=model_name).observe(response_size)

    def record_prediction(self, model_name: str, confidence: float,
                         prediction_type: str = "classification"):
        """Record prediction confidence"""
        self.prediction_confidence.labels(
            model_name=model_name,
            prediction_type=prediction_type
        ).observe(confidence)

    def update_model_accuracy(self, model_name: str, version: str, accuracy: float):
        """Update model accuracy metric"""
        self.model_accuracy.labels(
            model_name=model_name,
            version=version
        ).set(accuracy)

    def record_business_score(self, model_name: str, user_segment: str, score: float):
        """Record business score"""
        self.business_score.labels(
            model_name=model_name,
            user_segment=user_segment
        ).set(score)

# Global metrics instance
metrics = ModelMetrics()

def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return generate_latest()
```

### ELK Stack for Logging

```python
# elk_logging.py
import logging
import json
import os
from datetime import datetime
from elasticsearch import Elasticsearch
from pythonjsonlogger import jsonlogger

class ELKLogger:
    def __init__(self, elasticsearch_host="localhost", elasticsearch_port=9200):
        # Setup Elasticsearch client
        self.es = Elasticsearch([f"http://{elasticsearch_host}:{elasticsearch_port}"])

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # JSON formatter for structured logging
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)

        # File handler for local backup
        file_handler = logging.FileHandler('/var/log/ml-app.log')
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)

    def log_prediction(self, user_id: str, model_name: str, input_data: Dict,
                      prediction: Any, confidence: float, processing_time: float):
        """Log prediction event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "prediction",
            "user_id": user_id,
            "model_name": model_name,
            "input_size": len(json.dumps(input_data)),
            "prediction": prediction,
            "confidence": confidence,
            "processing_time_seconds": processing_time,
            "environment": os.getenv("ENVIRONMENT", "development")
        }

        # Log to application
        self.logger.info("Prediction made", extra=log_entry)

        # Index in Elasticsearch
        try:
            self.es.index(
                index="ml-predictions",
                body=log_entry
            )
        except Exception as e:
            self.logger.error(f"Failed to index in Elasticsearch: {e}")

    def log_model_performance(self, model_name: str, version: str,
                            metrics: Dict, data_drift: Dict):
        """Log model performance metrics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_performance",
            "model_name": model_name,
            "model_version": version,
            "metrics": metrics,
            "data_drift": data_drift,
            "environment": os.getenv("ENVIRONMENT", "development")
        }

        self.logger.info("Model performance update", extra=log_entry)

        try:
            self.es.index(
                index="ml-performance",
                body=log_entry
            )
        except Exception as e:
            self.logger.error(f"Failed to index performance in Elasticsearch: {e}")

    def log_error(self, error_type: str, error_message: str,
                  context: Dict = None):
        """Log error event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "environment": os.getenv("ENVIRONMENT", "development")
        }

        self.logger.error("Application error", extra=log_entry)

        try:
            self.es.index(
                index="ml-errors",
                body=log_entry
            )
        except Exception as e:
            self.logger.error(f"Failed to index error in Elasticsearch: {e}")

# Elasticsearch index templates
elasticsearch_setup = '''
{
  "mappings": {
    "properties": {
      "timestamp": {"type": "date"},
      "event_type": {"type": "keyword"},
      "user_id": {"type": "keyword"},
      "model_name": {"type": "keyword"},
      "model_version": {"type": "keyword"},
      "prediction": {"type": "object"},
      "confidence": {"type": "float"},
      "processing_time_seconds": {"type": "float"},
      "environment": {"type": "keyword"}
    }
  }
}
'''

def setup_elasticsearch_indices():
    """Setup Elasticsearch indices for ML logging"""
    es = Elasticsearch(["localhost:9200"])

    # Create index for predictions
    if not es.indices.exists(index="ml-predictions"):
        es.indices.create(
            index="ml-predictions",
            body=json.loads(elasticsearch_setup)
        )

    # Create index for performance
    if not es.indices.exists(index="ml-performance"):
        es.indices.create(
            index="ml-performance",
            body=json.loads(elasticsearch_setup)
        )

    # Create index for errors
    if not es.indices.exists(index="ml-errors"):
        es.indices.create(
            index="ml-errors",
            body=json.loads(elasticsearch_setup)
        )

    print("Elasticsearch indices created successfully")

# Usage
elk_logger = ELKLogger()

# Log a prediction
elk_logger.log_prediction(
    user_id="user123",
    model_name="iris-classifier",
    input_data={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    prediction="setosa",
    confidence=0.95,
    processing_time=0.123
)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "ML Model Monitoring Dashboard",
    "tags": ["ml", "monitoring", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_requests_total[5m])",
            "legendFormat": "{{model_name}} - {{status}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile - {{model_name}}"
          },
          {
            "expr": "histogram_quantile(0.50, rate(model_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile - {{model_name}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 }
      },
      {
        "id": 3,
        "title": "Prediction Confidence",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_prediction_confidence_sum[5m]) / rate(model_prediction_confidence_count[5m])",
            "legendFormat": "Average Confidence - {{model_name}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 }
      },
      {
        "id": 4,
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "{{model_name}} v{{version}}"
          }
        ],
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 }
      },
      {
        "id": 5,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          { "expr": "system_cpu_usage_percent", "legendFormat": "CPU Usage %" },
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory Usage %"
          },
          { "expr": "system_gpu_usage_percent", "legendFormat": "GPU Usage %" }
        ],
        "gridPos": { "h": 8, "w": 24, "x": 0, "y": 16 }
      }
    ]
  }
}
```

---

## 11. Scaling Strategies

### Horizontal Scaling

```yaml
# hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-app-deployment
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
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
        - type: Pods
          value: 2
          periodSeconds: 60
```

### Vertical Pod Autoscaler

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-app-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: ml-app
        minAllowed:
          cpu: 100m
          memory: 128Mi
        maxAllowed:
          cpu: 4
          memory: 8Gi
        controlledResources: ["cpu", "memory"]
        controlledValues: RequestsAndLimits
```

### Load Balancer Configuration

```python
# load_balancer.py
import requests
import time
from typing import List, Dict, Optional
import random
import logging

logger = logging.getLogger(__name__)

class LoadBalancer:
    def __init__(self, backend_urls: List[str], algorithm: str = "round_robin"):
        self.backends = backend_urls
        self.algorithm = algorithm
        self.current_index = 0
        self.backend_weights = {url: 1 for url in backend_urls}
        self.backend_health = {url: True for url in backend_urls}
        self.request_counts = {url: 0 for url in backend_urls}

    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backends"""
        return [url for url, healthy in self.backend_health.items() if healthy]

    def select_backend(self) -> Optional[str]:
        """Select backend based on algorithm"""
        healthy_backends = self.get_healthy_backends()

        if not healthy_backends:
            logger.warning("No healthy backends available")
            return None

        if self.algorithm == "round_robin":
            return self._round_robin_select(healthy_backends)
        elif self.algorithm == "least_connections":
            return self._least_connections_select(healthy_backends)
        elif self.algorithm == "weighted":
            return self._weighted_select(healthy_backends)
        else:
            return random.choice(healthy_backends)

    def _round_robin_select(self, backends: List[str]) -> str:
        """Round robin selection"""
        backend = backends[self.current_index % len(backends)]
        self.current_index += 1
        return backend

    def _least_connections_select(self, backends: List[str]) -> str:
        """Select backend with least connections"""
        return min(backends, key=lambda url: self.request_counts.get(url, 0))

    def _weighted_select(self, backends: List[str]) -> str:
        """Weighted random selection"""
        weights = [self.backend_weights.get(url, 1) for url in backends]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(backends)

        random_weight = random.uniform(0, total_weight)
        current_weight = 0

        for i, backend in enumerate(backends):
            current_weight += weights[i]
            if random_weight <= current_weight:
                return backend

        return backends[-1]

    def make_request(self, endpoint: str, data: Dict,
                    headers: Dict = None) -> Optional[Dict]:
        """Make request through load balancer"""
        backend = self.select_backend()

        if not backend:
            return {"error": "No healthy backends available"}

        url = f"{backend}/{endpoint}"
        headers = headers or {"Content-Type": "application/json"}

        try:
            self.request_counts[backend] += 1
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()

            # Simulate varying response times based on load
            time.sleep(random.uniform(0.1, 0.5))

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Request to {backend} failed: {e}")

            # Mark backend as unhealthy (simple health check)
            self.backend_health[backend] = False

            # Schedule health check
            self._schedule_health_check(backend)

            # Retry with other backend
            return self.make_request(endpoint, data, headers)

        finally:
            self.request_counts[backend] = max(0, self.request_counts[backend] - 1)

    def _schedule_health_check(self, backend: str):
        """Schedule health check for backend"""
        def health_check():
            time.sleep(30)  # Wait 30 seconds before checking
            try:
                response = requests.get(f"{backend}/health", timeout=10)
                if response.status_code == 200:
                    self.backend_health[backend] = True
                    logger.info(f"Backend {backend} is now healthy")
                else:
                    self._schedule_health_check(backend)
            except:
                self._schedule_health_check(backend)

        import threading
        thread = threading.Thread(target=health_check, daemon=True)
        thread.start()

    def update_backend_weight(self, backend: str, weight: int):
        """Update backend weight for weighted algorithms"""
        if backend in self.backend_weights:
            self.backend_weights[backend] = weight
            logger.info(f"Updated weight for {backend}: {weight}")

    def get_stats(self) -> Dict:
        """Get load balancer statistics"""
        healthy_backends = self.get_healthy_backends()

        return {
            "total_backends": len(self.backends),
            "healthy_backends": len(healthy_backends),
            "algorithm": self.algorithm,
            "backend_stats": {
                url: {
                    "healthy": self.backend_health[url],
                    "weight": self.backend_weights[url],
                    "requests": self.request_counts[url]
                }
                for url in self.backends
            }
        }

# Usage
backend_urls = [
    "http://ml-app-1:8080",
    "http://ml-app-2:8080",
    "http://ml-app-3:8080"
]

load_balancer = LoadBalancer(backend_urls, algorithm="weighted")

# Update weights based on performance
load_balancer.update_backend_weight("http://ml-app-1:8080", 3)  # 3x weight
load_balancer.update_backend_weight("http://ml-app-2:8080", 2)  # 2x weight
load_balancer.update_backend_weight("http://ml-app-3:8080", 1)  # 1x weight (default)

# Make requests
response = load_balancer.make_request("predict", {"input": [5.1, 3.5, 1.4, 0.2]})
print("Response:", response)

# Check statistics
stats = load_balancer.get_stats()
print("Load Balancer Stats:", stats)
```

### Caching Strategies

```python
# caching.py
import redis
import json
import hashlib
import pickle
from typing import Any, Optional
import time
from functools import wraps

class ModelCache:
    def __init__(self, redis_host="localhost", redis_port=6379,
                 cache_ttl=3600, max_memory="256mb"):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self.cache_ttl = cache_ttl

    def _generate_cache_key(self, model_name: str, input_data: Any) -> str:
        """Generate cache key from model name and input data"""
        # Convert input to string and hash it
        input_str = json.dumps(input_data, sort_keys=True) if isinstance(input_data, dict) else str(input_data)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()

        return f"ml_cache:{model_name}:{input_hash}"

    def get(self, model_name: str, input_data: Any) -> Optional[Any]:
        """Get cached prediction"""
        cache_key = self._generate_cache_key(model_name, input_data)

        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                result = pickle.loads(cached_result)
                print(f"Cache hit for key: {cache_key}")
                return result

            print(f"Cache miss for key: {cache_key}")
            return None

        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None

    def set(self, model_name: str, input_data: Any, prediction: Any) -> bool:
        """Cache prediction result"""
        cache_key = self._generate_cache_key(model_name, input_data)

        try:
            # Serialize prediction
            serialized_result = pickle.dumps(prediction)

            # Store in Redis with TTL
            result = self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                serialized_result
            )

            print(f"Cached result for key: {cache_key}")
            return result

        except Exception as e:
            print(f"Cache storage error: {e}")
            return False

    def invalidate(self, pattern: str = "ml_cache:*") -> int:
        """Invalidate cache entries matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                print(f"Invalidated {deleted} cache entries")
                return deleted
            return 0

        except Exception as e:
            print(f"Cache invalidation error: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            keys = self.redis_client.keys("ml_cache:*")

            return {
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human"),
                "hit_rate": "N/A",  # Would need to track hits/misses separately
                "cache_ttl": self.cache_ttl
            }

        except Exception as e:
            return {"error": str(e)}

def cache_prediction(cache: ModelCache, model_name: str):
    """Decorator for caching model predictions"""
    def decorator(func):
        @wraps(func)
        def wrapper(input_data, *args, **kwargs):
            # Try to get from cache
            cached_result = cache.get(model_name, input_data)
            if cached_result is not None:
                return cached_result

            # Make prediction
            result = func(input_data, *args, **kwargs)

            # Cache result
            cache.set(model_name, input_data, result)

            return result
        return wrapper
    return decorator

# Advanced caching with multiple levels
class MultiLevelCache:
    def __init__(self, l1_memory_size=1000, l2_redis_ttl=3600):
        # L1: In-memory cache
        self.l1_cache = {}
        self.l1_max_size = l1_memory_size
        self.l1_access_times = {}

        # L2: Redis cache
        self.l2_cache = ModelCache(cache_ttl=l2_redis_ttl)

        self.hit_counts = {"l1": 0, "l2": 0, "miss": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get from multi-level cache"""
        current_time = time.time()

        # Check L1 cache first
        if key in self.l1_cache:
            self.l1_access_times[key] = current_time
            self.hit_counts["l1"] += 1
            return self.l1_cache[key]

        # Check L2 cache
        l2_result = self.l2_cache.get("multi_cache", key)
        if l2_result is not None:
            # Promote to L1
            self._promote_to_l1(key, l2_result)
            self.hit_counts["l2"] += 1
            return l2_result

        self.hit_counts["miss"] += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Set in both cache levels"""
        # Set in L1
        self._promote_to_l1(key, value)

        # Set in L2
        self.l2_cache.set("multi_cache", key, value)

    def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote item to L1 cache with LRU eviction"""
        if len(self.l1_cache) >= self.l1_max_size:
            self._evict_l1()

        self.l1_cache[key] = value
        self.l1_access_times[key] = time.time()

    def _evict_l1(self) -> None:
        """Evict least recently used item from L1"""
        if not self.l1_access_times:
            return

        # Find least recently used key
        lru_key = min(self.l1_access_times.keys(), key=lambda k: self.l1_access_times[k])

        del self.l1_cache[lru_key]
        del self.l1_access_times[lru_key]

    def get_stats(self) -> Dict:
        """Get multi-level cache statistics"""
        total_requests = sum(self.hit_counts.values())

        return {
            "l1_size": len(self.l1_cache),
            "l2_stats": self.l2_cache.get_stats(),
            "hit_rates": {
                "l1": (self.hit_counts["l1"] / total_requests) * 100 if total_requests > 0 else 0,
                "l2": (self.hit_counts["l2"] / total_requests) * 100 if total_requests > 0 else 0,
                "miss": (self.hit_counts["miss"] / total_requests) * 100 if total_requests > 0 else 0
            }
        }

# Usage
cache = ModelCache()

# Use decorator for automatic caching
@cache_prediction(cache, "iris-classifier")
def predict_iris(input_data):
    # Your model prediction logic here
    import numpy as np
    # Simulate prediction
    return {"prediction": "setosa", "confidence": 0.95}

# Make predictions (will be cached automatically)
result1 = predict_iris([5.1, 3.5, 1.4, 0.2])
result2 = predict_iris([5.1, 3.5, 1.4, 0.2])  # Will be served from cache

print("Results:", result1, result2)
print("Cache stats:", cache.get_stats())
```

---

## 12. API Development

### FastAPI Application with Production Features

```python
# fastapi_app.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import uvicorn
import asyncio
import logging
from datetime import datetime, timedelta
import json
from typing import List, Optional, Dict, Any
import time
from prometheus_client import generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model API",
    description="Production ML Model API with monitoring and security",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Security
security = HTTPBearer()

# Rate limiting (simple implementation)
rate_limits = {}
RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

async def rate_limit_check(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Rate limiting dependency"""
    client_id = credentials.credentials  # Assuming token-based auth

    current_time = time.time()

    # Clean old entries
    cutoff_time = current_time - RATE_LIMIT_WINDOW
    rate_limits[client_id] = [t for t in rate_limits.get(client_id, []) if t > cutoff_time]

    # Check limit
    if len(rate_limits.get(client_id, [])) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Record request
    if client_id not in rate_limits:
        rate_limits[client_id] = []
    rate_limits[client_id].append(current_time)

    return client_id

# Pydantic models
class PredictionRequest(BaseModel):
    input: List[float]
    metadata: Optional[Dict[str, Any]] = {}

    @validator('input')
    def validate_input(cls, v):
        if len(v) == 0:
            raise ValueError('Input cannot be empty')
        if len(v) > 1000:  # Prevent memory issues
            raise ValueError('Input too large')
        return v

class BatchPredictionRequest(BaseModel):
    inputs: List[List[float]]
    metadata: Optional[Dict[str, Any]] = {}

    @validator('inputs')
    def validate_batch_inputs(cls, v):
        if len(v) == 0:
            raise ValueError('No inputs provided')
        if len(v) > 100:  # Batch size limit
            raise ValueError('Batch too large')
        return v

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float
    model_version: str
    processing_time: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_version: str
    total_processing_time: float
    average_processing_time: float
    timestamp: str

# Global model variable (load your model here)
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Starting ML API...")

    # Load model (replace with your model loading logic)
    global model
    try:
        import joblib
        model = joblib.load("models/model.joblib")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ML Model API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "model_version": "1.0.0",
        "features": ["feature_1", "feature_2", "feature_3"],  # Update with your features
        "supported_operations": ["predict", "predict_proba"],
        "input_shape": getattr(model, "n_features_in_", "unknown"),
        "classes": getattr(model, "classes_", []) if hasattr(model, "classes_") else []
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(rate_limit_check)
):
    """Single prediction endpoint"""
    start_time = time.time()

    try:
        # Validate input
        input_array = request.input

        # Make prediction
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prediction = model.predict([input_array])[0]

        # Get confidence if available
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([input_array])[0]
            confidence = float(max(probabilities))

        processing_time = time.time() - start_time

        # Background task for logging (non-blocking)
        background_tasks.add_task(
            log_prediction,
            client_id=client_id,
            input_data=input_array,
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time
        )

        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version="1.0.0",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(rate_limit_check)
):
    """Batch prediction endpoint"""
    start_time = time.time()

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        inputs = request.inputs

        # Make batch prediction
        predictions = model.predict(inputs)

        # Get confidences if available
        confidences = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(inputs)
            confidences = [float(max(prob)) for prob in probabilities]
        else:
            confidences = [1.0] * len(inputs)

        # Build response
        prediction_results = []
        for i, (prediction, confidence) in enumerate(zip(predictions, confidences)):
            prediction_results.append({
                "index": i,
                "prediction": prediction,
                "confidence": confidence
            })

        total_processing_time = time.time() - start_time
        average_processing_time = total_processing_time / len(inputs)

        response = BatchPredictionResponse(
            predictions=prediction_results,
            model_version="1.0.0",
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            timestamp=datetime.now().isoformat()
        )

        # Background logging
        background_tasks.add_task(
            log_batch_prediction,
            client_id=client_id,
            input_count=len(inputs),
            processing_time=total_processing_time
        )

        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/version")
async def version():
    """Get API version"""
    return {"version": "1.0.0", "build_date": "2024-01-01"}

# Background tasks
async def log_prediction(client_id: str, input_data: List[float],
                        prediction: Any, confidence: float, processing_time: float):
    """Background task for prediction logging"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "event_type": "prediction",
            "input_size": len(input_data),
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time
        }

        # Log to file or external service
        logger.info(f"Prediction logged: {json.dumps(log_entry)}")

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

async def log_batch_prediction(client_id: str, input_count: int, processing_time: float):
    """Background task for batch prediction logging"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "event_type": "batch_prediction",
            "input_count": input_count,
            "processing_time": processing_time,
            "average_time_per_input": processing_time / input_count
        }

        logger.info(f"Batch prediction logged: {json.dumps(log_entry)}")

    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
```

### API Documentation and Testing

```python
# api_testing.py
import requests
import json
import time
from typing import Dict, List

class APITester:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def test_health(self) -> Dict:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def test_predict(self, input_data: List[float],
                    expected_status: int = 200) -> Dict:
        """Test prediction endpoint"""
        try:
            payload = {"input": input_data}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload
            )

            result = {
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": response.status_code == expected_status
            }

            if response.status_code == expected_status:
                result["data"] = response.json()
            else:
                result["error"] = response.text

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def test_batch_predict(self, inputs: List[List[float]],
                          expected_status: int = 200) -> Dict:
        """Test batch prediction endpoint"""
        try:
            payload = {"inputs": inputs}
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload
            )

            result = {
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": response.status_code == expected_status
            }

            if response.status_code == expected_status:
                result["data"] = response.json()
            else:
                result["error"] = response.text

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def test_load_performance(self, requests_count: int = 100,
                             concurrent_requests: int = 10) -> Dict:
        """Test API performance under load"""
        import threading
        import queue

        results = queue.Queue()

        def make_request():
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json={"input": [5.1, 3.5, 1.4, 0.2]}
                )
                end_time = time.time()

                results.put({
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                })
            except Exception as e:
                results.put({
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                })

        # Create threads
        threads = []
        requests_per_thread = requests_count // concurrent_requests

        for _ in range(concurrent_requests):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        response_times = []
        success_count = 0
        status_codes = {}

        while not results.empty():
            result = results.get()
            if result["success"]:
                response_times.append(result["response_time"])
                success_count += 1

            status_codes[result["status_code"]] = status_codes.get(result["status_code"], 0) + 1

        return {
            "total_requests": requests_count,
            "concurrent_requests": concurrent_requests,
            "total_time": total_time,
            "success_rate": (success_count / requests_count) * 100,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "requests_per_second": requests_count / total_time,
            "status_codes": status_codes
        }

    def run_full_test_suite(self) -> Dict:
        """Run complete test suite"""
        print("🚀 Running API Test Suite...")

        test_results = {}

        # Health check
        print("📊 Testing health endpoint...")
        test_results["health"] = self.test_health()

        # Model info
        print("ℹ️ Testing model info...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            test_results["model_info"] = {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "success": response.status_code == 200
            }
        except Exception as e:
            test_results["model_info"] = {"success": False, "error": str(e)}

        # Single prediction tests
        print("🎯 Testing single predictions...")
        test_cases = [
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.5, 3.0, 5.8, 2.2]
        ]

        test_results["single_predictions"] = []
        for i, input_data in enumerate(test_cases):
            result = self.test_predict(input_data)
            test_results["single_predictions"].append(result)
            print(f"  Test {i+1}: {'✅' if result.get('success') else '❌'}")

        # Batch prediction test
        print("📦 Testing batch predictions...")
        test_results["batch_prediction"] = self.test_batch_predict(test_cases)

        # Load test (smaller scale for demo)
        print("⚡ Testing performance...")
        test_results["load_test"] = self.test_load_performance(requests_count=50, concurrent_requests=5)

        return test_results

# Usage
def main():
    # Initialize tester
    tester = APITester("http://localhost:8080")  # Update with your API URL

    # Run test suite
    results = tester.run_full_test_suite()

    # Print results
    print("\n📋 Test Results Summary:")
    print(f"Health: {'✅' if results['health'].get('status') == 'success' else '❌'}")
    print(f"Model Info: {'✅' if results['model_info'].get('success') else '❌'}")
    print(f"Single Predictions: {sum(1 for r in results['single_predictions'] if r.get('success'))}/{len(results['single_predictions'])} ✅")
    print(f"Batch Prediction: {'✅' if results['batch_prediction'].get('success') else '❌'}")
    print(f"Load Test Success Rate: {results['load_test'].get('success_rate', 0):.1f}%")

    return results

if __name__ == "__main__":
    main()
```

---

## 13. Real-World Deployment Workflows

### Complete Deployment Pipeline

```yaml
# complete-deployment-pipeline.yaml
name: Complete ML Deployment Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 2 * * *" # Daily at 2 AM

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  KUBERNETES_NAMESPACE: ml-production

jobs:
  pre-deployment-checks:
    runs-on: ubuntu-latest
    outputs:
      should-deploy: ${{ steps.check.outputs.should-deploy }}
    steps:
      - uses: actions/checkout@v3

      - name: Run Pre-deployment Checks
        id: check
        run: |
          # Check if model has been updated
          git diff --name-only HEAD~1 HEAD | grep -q "models/" && echo "should-deploy=true" >> $GITHUB_OUTPUT || echo "should-deploy=false" >> $GITHUB_OUTPUT

          # Check data validation
          python scripts/validate_data.py --check-model-updates

          # Check test coverage
          COVERAGE=$(python -m pytest --cov=src --cov-report=json | jq '.totals.percent_covered')
          echo "coverage=$COVERAGE" >> $GITHUB_OUTPUT

          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "❌ Test coverage too low: $COVERAGE%"
            exit 1
          fi

  data-validation:
    needs: pre-deployment-checks
    runs-on: ubuntu-latest
    if: needs.pre-deployment-checks.outputs.should-deploy == 'true'
    steps:
      - uses: actions/checkout@v3

      - name: Validate Training Data
        run: |
          python scripts/validate_training_data.py \
            --data-path data/training/latest/ \
            --schema-path schemas/training_schema.json \
            --drift-check

      - name: Generate Data Quality Report
        run: |
          python scripts/generate_data_report.py \
            --output-path reports/data_quality_$(date +%Y%m%d).json

  model-training:
    needs: [pre-deployment-checks, data-validation]
    runs-on: ubuntu-latest
    if: needs.pre-deployment-checks.outputs.should-deploy == 'true'
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Train Model
        run: |
          python scripts/train_production_model.py \
            --config configs/production_config.yaml \
            --experiment-name "production-${{ github.sha }}" \
            --model-registry s3://ml-models/production/${{ github.sha }}/ \
            --tag-latest

      - name: Run Model Tests
        run: |
          python scripts/test_model_performance.py \
            --model-path s3://ml-models/production/${{ github.sha }}/model.joblib \
            --test-data-path data/testing/latest/ \
            --benchmark-baseline models/baseline.joblib

      - name: Model Explainability
        run: |
          python scripts/generate_model_explanations.py \
            --model-path s3://ml-models/production/${{ github.sha }}/model.joblib \
            --output-path reports/model_explanations_${{ github.sha }}.json

  security-scan:
    needs: pre-deployment-checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Security Scan
        uses: securecodewarrior/github-action-add-sarif@v1
        with:
          sarif-file: security-scan-results.sarif

      - name: Check Dependencies for Vulnerabilities
        run: |
          safety check --json --output safety-report.json || true

      - name: Docker Security Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: "sarif"
          output: "trivy-results.sarif"

  build-and-test:
    needs: [pre-deployment-checks, security-scan]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker Image
        run: |
          docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
          docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest .

      - name: Run Container Tests
        run: |
          docker run --rm ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} python -m pytest tests/integration/ -v

      - name: Container Performance Test
        run: |
          python scripts/benchmark_container_performance.py \
            --image ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --requests 1000 --concurrency 10

  deploy-staging:
    needs: [build-and-test, model-training]
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3

      - name: Configure Kubernetes
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

      - name: Deploy to Staging
        run: |
          helm upgrade --install ml-app-staging ./charts/ml-app \
            --namespace ${{ env.KUBERNETES_NAMESPACE }}-staging \
            --set image.tag=${{ github.sha }} \
            --set model.version=${{ github.sha }} \
            --set environment=staging \
            --set replicaCount=2

      - name: Wait for Deployment
        run: |
          kubectl rollout status deployment/ml-app-staging \
            --namespace ${{ env.KUBERNETES_NAMESPACE }}-staging \
            --timeout=300s

      - name: Run Staging Tests
        run: |
          python scripts/test_deployment.py \
            --environment staging \
            --base-url https://ml-staging.yourdomain.com \
            --model-version ${{ github.sha }}

  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: End-to-End Tests
        run: |
          python scripts/run_e2e_tests.py \
            --environment staging \
            --test-scenarios "smoke,performance,security"

      - name: A/B Test Setup
        run: |
          python scripts/setup_ab_test.py \
            --new-model-version ${{ github.sha }} \
            --traffic-split "80/20" \
            --duration "24h"

  performance-benchmark:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Performance Benchmarks
        run: |
          python scripts/performance_benchmark.py \
            --model-version ${{ github.sha }} \
            --test-duration 30m \
            --load-pattern "ramp-up" \
            --metrics "latency,throughput,error-rate"

      - name: Compare with Baseline
        run: |
          python scripts/compare_performance.py \
            --current-model ${{ github.sha }} \
            --baseline-model production \
            --threshold-latency 0.1 \
            --threshold-throughput 0.05

  deploy-production:
    needs: [integration-tests, performance-benchmark]
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Configure Production Kubernetes
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}

      - name: Blue-Green Deployment
        run: |
          ./scripts/deploy-blue-green.sh \
            --new-version ${{ github.sha }} \
            --namespace ${{ env.KUBERNETES_NAMESPACE }} \
            -- Canary deployment first, then switch

      - name: Health Check
        run: |
          kubectl get pods -n ${{ env.KUBERNETES_NAMESPACE }}
          kubectl get services -n ${{ env.KUBERNETES_NAMESPACE }}

          # Wait for health check
          python scripts/wait_for_health.py \
            --url https://ml.yourdomain.com/health \
            --timeout 300s

      - name: Traffic Switch
        run: |
          ./scripts/switch-traffic.sh \
            --from-version previous \
            --to-version ${{ github.sha }}

  post-deployment-validation:
    needs: deploy-production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Production Smoke Tests
        run: |
          python scripts/smoke_tests_production.py \
            --environment production \
            --duration 5m

      - name: Monitor Initial Performance
        run: |
          python scripts/monitor_initial_performance.py \
            --model-version ${{ github.sha }} \
            --monitoring-duration 15m

      - name: Alert Slack
        if: success()
        uses: 8398a7/action-slack@v3
        with:
          status: success
          text: "🚀 ML Model deployment successful! Version: ${{ github.sha }}"

      - name: Alert Slack on Failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          text: "❌ ML Model deployment failed! Version: ${{ github.sha }}"

  rollback-if-needed:
    needs: post-deployment-validation
    runs-on: ubuntu-latest
    if: failure()
    steps:
      - name: Automatic Rollback
        run: |
          ./scripts/rollback-deployment.sh \
            --reason "post-deployment validation failed" \
            --previous-version production

      - name: Notify Rollback
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          channel: "#alerts"
          custom_payload: |
            {
              "text": "🚨 Auto-rollback initiated for ML model",
              "attachments": [{
                "color": "danger",
                "fields": [{
                  "title": "Version",
                  "value": "${{ github.sha }}",
                  "short": true
                },{
                  "title": "Reason",
                  "value": "Post-deployment validation failed",
                  "short": true
                }]
              }]
            }
```

### Model Registry and Governance

```python
# model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ModelVersion:
    version: str
    model_uri: str
    stage: str
    run_id: str
    description: str
    metrics: Dict
    created_at: str
    approval_status: str

class ModelRegistry:
    def __init__(self, registry_uri: str, model_name: str):
        self.client = MlflowClient(tracking_uri=registry_uri)
        self.model_name = model_name
        self.registry_uri = registry_uri

    def register_model_from_run(self, run_id: str,
                               version: str,
                               description: str = "",
                               tags: Dict[str, str] = None) -> str:
        """Register model from MLflow run"""
        try:
            # Get run info
            run_info = self.client.get_run(run_id)

            # Create model version
            model_uri = f"runs:/{run_id}/model"

            # Register model
            mv = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
                description=description,
                tags=tags or {}
            )

            print(f"Model registered as {self.model_name} version {mv.version}")
            return mv.version

        except Exception as e:
            print(f"Error registering model: {e}")
            raise

    def transition_to_stage(self, version: str,
                          stage: str,
                          comment: str = "") -> bool:
        """Transition model version to specific stage"""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )

            # Log transition
            self._log_stage_transition(version, stage, comment)
            return True

        except Exception as e:
            print(f"Error transitioning model: {e}")
            return False

    def approve_model(self, version: str,
                     reviewer: str,
                     comments: str = "") -> bool:
        """Approve model for production"""
        try:
            # Update approval status
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="approval_status",
                value="approved"
            )

            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="reviewer",
                value=reviewer
            )

            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="approval_comments",
                value=comments
            )

            # Auto-transition to production if criteria met
            if self._check_production_criteria(version):
                self.transition_to_stage(version, "Production",
                                       f"Auto-approved by {reviewer}")

            return True

        except Exception as e:
            print(f"Error approving model: {e}")
            return False

    def _check_production_criteria(self, version: str) -> bool:
        """Check if model meets production criteria"""
        try:
            # Get model version info
            mv = self.client.get_model_version(name=self.model_name, version=version)

            # Check approval status
            if mv.tags.get("approval_status") != "approved":
                return False

            # Get run metrics
            run_info = self.client.get_run(mv.run_id)
            metrics = run_info.data.metrics

            # Define criteria (adjust based on your requirements)
            criteria = {
                "accuracy": 0.85,  # Minimum accuracy
                "precision": 0.80,  # Minimum precision
                "recall": 0.75     # Minimum recall
            }

            # Check criteria
            for metric, threshold in criteria.items():
                if metric in metrics and metrics[metric] < threshold:
                    print(f"Model fails criteria: {metric} = {metrics[metric]} < {threshold}")
                    return False

            return True

        except Exception as e:
            print(f"Error checking production criteria: {e}")
            return False

    def get_latest_versions(self, stages: List[str] = None) -> List[ModelVersion]:
        """Get latest versions for specified stages"""
        if stages is None:
            stages = ["Staging", "Production"]

        versions = []
        for stage in stages:
            latest_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=[stage]
            )

            for mv in latest_versions:
                run_info = self.client.get_run(mv.run_id)
                version = ModelVersion(
                    version=mv.version,
                    model_uri=mv.source,
                    stage=stage,
                    run_id=mv.run_id,
                    description=mv.description,
                    metrics=run_info.data.metrics,
                    created_at=mv.creation_timestamp,
                    approval_status=mv.tags.get("approval_status", "unknown")
                )
                versions.append(version)

        return versions

    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        try:
            # Get both versions
            mv1 = self.client.get_model_version(name=self.model_name, version=version1)
            mv2 = self.client.get_model_version(name=self.model_name, version=version2)

            # Get run info
            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)

            # Compare metrics
            comparison = {
                "version1": version1,
                "version2": version2,
                "metrics_comparison": {},
                "stage1": mv1.current_stage,
                "stage2": mv2.current_stage,
                "timestamps": {
                    "version1": mv1.creation_timestamp,
                    "version2": mv2.creation_timestamp
                }
            }

            # Compare common metrics
            common_metrics = set(run1.data.metrics.keys()) & set(run2.data.metrics.keys())

            for metric in common_metrics:
                val1 = run1.data.metrics[metric]
                val2 = run2.data.metrics[metric]

                comparison["metrics_comparison"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "improvement": ((val2 - val1) / val1) * 100 if val1 != 0 else float('inf'),
                    "winner": "version2" if val2 > val1 else "version1"
                }

            return comparison

        except Exception as e:
            print(f"Error comparing versions: {e}")
            return {}

    def _log_stage_transition(self, version: str, stage: str, comment: str):
        """Log stage transition for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "version": version,
            "action": f"transitioned to {stage}",
            "comment": comment
        }

        # In production, you might want to log this to a database or audit system
        print(f"Stage transition logged: {json.dumps(log_entry, indent=2)}")

# Usage example
def manage_model_lifecycle():
    """Example of managing model lifecycle"""

    # Initialize registry
    registry = ModelRegistry(
        registry_uri="http://localhost:5000",
        model_name="iris-classifier"
    )

    # Register new model
    version = registry.register_model_from_run(
        run_id="abc123",
        version="1.0.0",
        description="Iris classification model v1.0.0",
        tags={"team": "ml-engineering", "environment": "production"}
    )

    # Wait for manual review and approval
    # registry.approve_model(version, "john.doe@company.com", "Model meets all criteria")

    # Auto-promote to production if criteria met
    if registry._check_production_criteria(version):
        registry.transition_to_stage(version, "Production", "Auto-promoted")

    # Get current versions
    current_versions = registry.get_latest_versions()
    for v in current_versions:
        print(f"Stage: {v.stage}, Version: {v.version}, Metrics: {v.metrics}")

    # Compare with previous version
    if len(current_versions) > 1:
        comparison = registry.compare_versions(current_versions[0].version, current_versions[1].version)
        print("Version Comparison:", json.dumps(comparison, indent=2))

if __name__ == "__main__":
    manage_model_lifecycle()
```

### Disaster Recovery and Backup

```python
# disaster_recovery.py
import boto3
import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DisasterRecovery:
    def __init__(self, aws_region: str = "us-east-1"):
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.ec2_client = boto3.client('ec2', region_name=aws_region)
        self.rds_client = boto3.client('rds', region_name=aws_region)

    def backup_model(self, model_path: str, model_name: str,
                    backup_s3_bucket: str) -> str:
        """Backup model to S3 with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_key = f"models/{model_name}/backup_{timestamp}"

        try:
            # Upload model to S3
            self.s3_client.upload_file(
                model_path,
                backup_s3_bucket,
                backup_key
            )

            # Store metadata
            metadata = {
                "model_name": model_name,
                "original_path": model_path,
                "backup_timestamp": timestamp,
                "s3_key": backup_key,
                "file_size": os.path.getsize(model_path)
            }

            metadata_key = f"models/{model_name}/metadata_{timestamp}.json"
            self.s3_client.put_object(
                Bucket=backup_s3_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata)
            )

            logger.info(f"Model backed up to s3://{backup_s3_bucket}/{backup_key}")
            return backup_key

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

    def restore_model(self, backup_s3_bucket: str, backup_key: str,
                     restore_path: str) -> bool:
        """Restore model from S3 backup"""
        try:
            # Download model
            self.s3_client.download_file(
                backup_s3_bucket,
                backup_key,
                restore_path
            )

            # Verify restore
            if os.path.exists(restore_path) and os.path.getsize(restore_path) > 0:
                logger.info(f"Model restored from s3://{backup_s3_bucket}/{backup_key}")
                return True
            else:
                logger.error("Restore verification failed")
                return False

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def create_infrastructure_backup(self, namespace: str,
                                   backup_s3_bucket: str) -> Dict:
        """Backup Kubernetes configurations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Export Kubernetes resources
            backup_info = {
                "timestamp": timestamp,
                "namespace": namespace,
                "resources": {}
            }

            # Export deployments
            import subprocess
            try:
                result = subprocess.run([
                    "kubectl", "get", "deployment",
                    "-n", namespace, "-o", "yaml"
                ], capture_output=True, text=True, check=True)

                backup_info["resources"]["deployments"] = result.stdout

            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not export deployments: {e}")

            # Export services
            try:
                result = subprocess.run([
                    "kubectl", "get", "service",
                    "-n", namespace, "-o", "yaml"
                ], capture_output=True, text=True, check=True)

                backup_info["resources"]["services"] = result.stdout

            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not export services: {e}")

            # Export configmaps
            try:
                result = subprocess.run([
                    "kubectl", "get", "configmap",
                    "-n", namespace, "-o", "yaml"
                ], capture_output=True, text=True, check=True)

                backup_info["resources"]["configmaps"] = result.stdout

            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not export configmaps: {e}")

            # Upload backup
            backup_key = f"infrastructure/{namespace}/backup_{timestamp}.json"
            self.s3_client.put_object(
                Bucket=backup_s3_bucket,
                Key=backup_key,
                Body=json.dumps(backup_info, indent=2)
            )

            logger.info(f"Infrastructure backed up to s3://{backup_s3_bucket}/{backup_key}")
            return {"backup_key": backup_key, "timestamp": timestamp}

        except Exception as e:
            logger.error(f"Infrastructure backup failed: {e}")
            raise

    def simulate_disaster_recovery(self, namespace: str,
                                 backup_s3_bucket: str,
                                 new_namespace: str) -> Dict:
        """Simulate disaster recovery process"""
        logger.info("Starting disaster recovery simulation...")

        recovery_steps = []
        start_time = datetime.now()

        try:
            # Step 1: Identify latest backup
            recovery_steps.append({"step": "identify_backup", "status": "starting"})

            backup_files = self.s3_client.list_objects_v2(
                Bucket=backup_s3_bucket,
                Prefix=f"infrastructure/{namespace}/"
            )

            if 'Contents' not in backup_files:
                raise Exception("No backup files found")

            # Get latest backup
            latest_backup = max(
                backup_files['Contents'],
                key=lambda x: x['LastModified']
            )

            recovery_steps.append({
                "step": "identify_backup",
                "status": "completed",
                "backup_file": latest_backup['Key']
            })

            # Step 2: Download backup
            recovery_steps.append({"step": "download_backup", "status": "starting"})

            backup_content = self.s3_client.get_object(
                Bucket=backup_s3_bucket,
                Key=latest_backup['Key']
            )['Body'].read().decode('utf-8')

            backup_info = json.loads(backup_content)

            recovery_steps.append({
                "step": "download_backup",
                "status": "completed"
            })

            # Step 3: Restore infrastructure
            recovery_steps.append({"step": "restore_infrastructure", "status": "starting"})

            # Create new namespace
            subprocess.run([
                "kubectl", "create", "namespace", new_namespace
            ], check=True)

            # Apply resources (simplified - in practice you'd parse and apply YAML)
            logger.info(f"Created namespace: {new_namespace}")

            recovery_steps.append({
                "step": "restore_infrastructure",
                "status": "completed"
            })

            # Step 4: Verify recovery
            recovery_steps.append({"step": "verify_recovery", "status": "starting"})

            # Check if pods are running
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", new_namespace
            ], capture_output=True, text=True)

            recovery_steps.append({
                "step": "verify_recovery",
                "status": "completed"
            })

            # Calculate recovery time
            recovery_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "success",
                "recovery_time_seconds": recovery_time,
                "steps": recovery_steps,
                "new_namespace": new_namespace
            }

        except Exception as e:
            recovery_steps.append({
                "step": "recovery_failed",
                "status": "error",
                "error": str(e)
            })

            return {
                "status": "failed",
                "error": str(e),
                "steps": recovery_steps
            }

    def setup_monitoring_alerts(self, monitoring_config: Dict) -> bool:
        """Setup monitoring alerts for disaster recovery"""
        try:
            # This would integrate with your monitoring system (e.g., CloudWatch, PagerDuty)
            alerts_config = {
                "model_performance_degradation": {
                    "threshold": 0.1,  # 10% accuracy drop
                    "duration": "5m"
                },
                "high_error_rate": {
                    "threshold": 0.05,  # 5% error rate
                    "duration": "2m"
                },
                "response_time_degradation": {
                    "threshold": 1000,  # 1 second response time
                    "duration": "3m"
                },
                "infrastructure_failure": {
                    "check": "health_endpoint",
                    "threshold": 0,  # Complete failure
                    "duration": "1m"
                }
            }

            # Log alert configuration (in production, you'd set up actual alerts)
            logger.info(f"Monitoring alerts configured: {json.dumps(alerts_config, indent=2)}")

            return True

        except Exception as e:
            logger.error(f"Failed to setup monitoring alerts: {e}")
            return False

# Usage example
def setup_disaster_recovery():
    """Setup disaster recovery for ML deployment"""

    dr = DisasterRecovery(aws_region="us-east-1")

    # Backup current model
    backup_key = dr.backup_model(
        model_path="models/current_model.joblib",
        model_name="iris-classifier",
        backup_s3_bucket="ml-backups-production"
    )

    # Backup infrastructure
    infra_backup = dr.create_infrastructure_backup(
        namespace="ml-production",
        backup_s3_bucket="ml-backups-production"
    )

    # Setup monitoring alerts
    dr.setup_monitoring_alerts({})

    # Simulate disaster recovery
    recovery_result = dr.simulate_disaster_recovery(
        namespace="ml-production",
        backup_s3_bucket="ml-backups-production",
        new_namespace="ml-recovery"
    )

    print("Disaster Recovery Setup Complete:")
    print(f"Model Backup: {backup_key}")
    print(f"Infrastructure Backup: {infra_backup}")
    print(f"Recovery Test: {recovery_result}")

if __name__ == "__main__":
    setup_disaster_recovery()
```

---

## 14. Security & Best Practices

### API Security Implementation

```python
# security.py
import hashlib
import hmac
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from functools import wraps
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer()

class SecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = 3600  # 1 hour
        self.rate_limits = {}
        self.max_requests_per_hour = 1000

    def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
            "iat": datetime.utcnow()
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def authenticate(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
        """Authenticate user from bearer token"""
        if not credentials.credentials:
            raise HTTPException(status_code=401, detail="No token provided")

        return self.verify_token(credentials.credentials)

    def check_permission(self, required_permission: str):
        """Check if user has required permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(user_data: Dict = Depends(self.authenticate), *args, **kwargs):
                permissions = user_data.get("permissions", [])

                if required_permission not in permissions and "admin" not in permissions:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission '{required_permission}' required"
                    )

                return await func(user_data=user_data, *args, **kwargs)
            return wrapper
        return decorator

    def rate_limit_check(self, user_id: str) -> bool:
        """Check rate limiting for user"""
        current_time = int(time.time())
        hour_window = current_time // 3600

        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}

        if hour_window not in self.rate_limits[user_id]:
            self.rate_limits[user_id][hour_window] = 0

        # Check if over limit
        if self.rate_limits[user_id][hour_window] >= self.max_requests_per_hour:
            return False

        # Increment counter
        self.rate_limits[user_id][hour_window] += 1

        return True

class InputValidator:
    """Input validation for security"""

    @staticmethod
    def validate_input_size(data: str, max_size: int = 10000) -> bool:
        """Validate input size"""
        if len(data) > max_size:
            raise HTTPException(status_code=413, detail="Input too large")
        return True

    @staticmethod
    def validate_input_content(data: str) -> bool:
        """Validate input content for malicious patterns"""
        malicious_patterns = [
            "<script", "javascript:", "vbscript:", "data:",  # XSS
            "../../", "..\\", "\\..\\",  # Path traversal
            "eval(", "exec(", "system(",  # Code injection
        ]

        data_lower = data.lower()
        for pattern in malicious_patterns:
            if pattern in data_lower:
                raise HTTPException(status_code=400, detail="Invalid input content detected")

        return True

    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize input by removing dangerous characters"""
        import re

        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', data)

        # Remove potentially dangerous HTML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)

        return sanitized.strip()

class AuditLogger:
    """Audit logging for security compliance"""

    def __init__(self, log_file: str = "/var/log/ml-audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")

        # Setup file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_api_call(self, user_id: str, endpoint: str,
                    method: str, status_code: int,
                    client_ip: str, user_agent: str):
        """Log API call for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_call",
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "client_ip": client_ip,
            "user_agent": user_agent
        }

        self.logger.info(json.dumps(log_entry))

    def log_model_access(self, user_id: str, model_name: str,
                        action: str, success: bool, details: Dict = None):
        """Log model access for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_access",
            "user_id": user_id,
            "model_name": model_name,
            "action": action,
            "success": success,
            "details": details or {}
        }

        self.logger.info(json.dumps(log_entry))

    def log_security_event(self, event_type: str, user_id: str,
                          description: str, severity: str = "medium"):
        """Log security events"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "security_event",
            "security_event_type": event_type,
            "user_id": user_id,
            "description": description,
            "severity": severity
        }

        self.logger.warning(json.dumps(log_entry))

# Initialize security components
security_manager = SecurityManager(secret_key="your-secret-key-here")
input_validator = InputValidator()
audit_logger = AuditLogger()

# Security middleware
def security_middleware():
    """Apply security middleware"""
    from fastapi import Request

    async def security_check(request: Request, call_next):
        # Log request
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")

        # Basic rate limiting (simplified)
        if not security_manager.rate_limit_check(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        response = await call_next(request)

        # Log response
        logger.info(f"Request processed: {request.url} - Status: {response.status_code}")

        return response

    return security_check

# Usage in FastAPI application
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response

# Protected endpoints with security
@app.post("/secure/predict")
@security_manager.check_permission("predict")
async def secure_predict(
    request: PredictionRequest,
    user_data: Dict = Depends(security_manager.authenticate),
    rate_limited: bool = Depends(lambda: security_manager.rate_limit_check(user_data["user_id"]))
):
    """Secure prediction endpoint with authentication and permissions"""

    if not rate_limited:
        audit_logger.log_security_event(
            "rate_limit_exceeded",
            user_data["user_id"],
            "User exceeded rate limit"
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Validate and sanitize input
    input_data = [input_validator.sanitize_input(str(x)) for x in request.input]
    input_validator.validate_input_size(json.dumps(input_data))
    input_validator.validate_input_content(json.dumps(input_data))

    # Log model access
    audit_logger.log_model_access(
        user_id=user_data["user_id"],
        model_name="iris-classifier",
        action="predict",
        success=True,
        details={"input_size": len(input_data)}
    )

    # Make prediction (your existing prediction logic)
    # ...

    return prediction_result
```

### Data Encryption and Privacy

```python
# encryption.py
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Union

class DataEncryption:
    """Data encryption for sensitive information"""

    def __init__(self, password: str = None):
        self.password = password or os.environ.get('ENCRYPTION_KEY', 'default-key')
        self.salt = os.environ.get('ENCRYPTION_SALT', b'default-salt')
        self.fernet = self._create_fernet()

    def _create_fernet(self) -> Fernet:
        """Create Fernet instance with derived key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        return Fernet(key)

    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string"""
        if isinstance(data, str):
            data = data.encode()

        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded encrypted data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

class PrivacyPreservingML:
    """Privacy-preserving machine learning techniques"""

    def __init__(self):
        self.encryption = DataEncryption()

    def differential_privacy_noise(self, data: list, epsilon: float = 1.0) -> list:
        """Add differential privacy noise to data"""
        import numpy as np

        # Calculate sensitivity (simplified - adjust based on your use case)
        sensitivity = 1.0

        # Calculate noise scale
        noise_scale = sensitivity / epsilon

        # Add Laplace noise
        noise = np.random.laplace(0, noise_scale, len(data))
        noisy_data = [max(0, val + noise_val) for val, noise_val in zip(data, noise)]

        return noisy_data

    def federated_averaging_weights(self, model_weights_list: list,
                                  privacy_budget: float = 1.0) -> dict:
        """Aggregate model weights with differential privacy"""
        import numpy as np

        # Average weights across participants
        avg_weights = {}
        for key in model_weights_list[0].keys():
            avg_weights[key] = np.mean([weights[key] for weights in model_weights_list], axis=0)

        # Add noise to protect individual participants
        if privacy_budget > 0:
            noise_scale = 1.0 / privacy_budget

            for key in avg_weights:
                noise = np.random.laplace(0, noise_scale, avg_weights[key].shape)
                avg_weights[key] = avg_weights[key] + noise

        return avg_weights

    def secure_multiparty_computation(self, participant_data: list) -> dict:
        """Simulate secure multiparty computation"""
        # This is a simplified simulation - real MPC would use cryptographic protocols
        import numpy as np

        # Each participant encrypts their data
        encrypted_shares = []
        for data in participant_data:
            encrypted_data = self.encryption.encrypt_data(str(data))
            # Split into shares (simplified - real implementation would use proper secret sharing)
            shares = [encrypted_data[i:i+len(encrypted_data)//3] for i in range(0, len(encrypted_data), len(encrypted_data)//3)]
            encrypted_shares.append(shares)

        # Aggregate encrypted shares
        aggregated_result = {"aggregated_shares": [], "participants": len(participant_data)}

        for i in range(len(encrypted_shares[0])):
            # Combine shares from all participants for each feature
            combined_share = "".join([shares[i] for shares in encrypted_shares])
            aggregated_result["aggregated_shares"].append(combined_share)

        return aggregated_result

# Usage examples
def implement_privacy_features():
    """Implement privacy-preserving features"""

    privacy_ml = PrivacyPreservingML()

    # Differential privacy
    sensitive_data = [100, 200, 150, 175, 125]
    noisy_data = privacy_ml.differential_privacy_noise(sensitive_data, epsilon=0.5)
    print(f"Original data: {sensitive_data}")
    print(f"Privacy-protected data: {noisy_data}")

    # Federated learning
    client_weights = [
        {"layer1": [0.1, 0.2, 0.3]},
        {"layer1": [0.15, 0.25, 0.35]},
        {"layer1": [0.12, 0.22, 0.32]}
    ]

    aggregated = privacy_ml.federated_averaging_weights(client_weights, privacy_budget=2.0)
    print(f"Aggregated weights: {aggregated}")

    # Secure multiparty computation
    participant_data = [
        [100, 200, 150],
        [120, 180, 160],
        [110, 190, 155]
    ]

    secure_result = privacy_ml.secure_multiparty_computation(participant_data)
    print(f"Secure computation result: {secure_result}")

if __name__ == "__main__":
    implement_privacy_features()
```

---

## 15. Troubleshooting & Optimization

### Performance Monitoring and Optimization

```python
# performance_optimization.py
import psutil
import time
import gc
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and optimize model performance"""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.prediction_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_usage = []
        self.monitoring_active = False
        self.performance_stats = {}

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring"""
        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.cpu_usage.append(cpu_percent)

                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.append({
                        'timestamp': datetime.now(),
                        'percent': memory.percent,
                        'available': memory.available,
                        'used': memory.used
                    })

                    # GPU usage (if available)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            self.gpu_usage.append({
                                'timestamp': datetime.now(),
                                'gpu_id': gpus[0].id,
                                'load': gpus[0].load * 100,
                                'memory_used': gpus[0].memoryUsed,
                                'memory_total': gpus[0].memoryTotal
                            })
                    except ImportError:
                        pass  # GPU monitoring not available

                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started for {self.model_name}")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")

    def profile_prediction_function(self, input_data: Any, profile_memory: bool = True):
        """Profile prediction function using cProfile"""
        profiler = cProfile.Profile()

        # Enable memory profiling if requested
        if profile_memory:
            import tracemalloc
            tracemalloc.start()

        # Profile the prediction
        profiler.enable()
        start_time = time.time()

        try:
            prediction = self.model.predict(input_data)

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_data)
            else:
                probabilities = None

            prediction_time = time.time() - start_time

        finally:
            profiler.disable()

            if profile_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                peak_memory_mb = peak / (1024 * 1024)
            else:
                peak_memory_mb = 0

        # Generate profiling report
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions

        profile_report = {
            "prediction_time": prediction_time,
            "peak_memory_mb": peak_memory_mb,
            "profiling_output": s.getvalue()
        }

        return prediction, probabilities, profile_report

    def optimize_batch_processing(self, inputs: List[Any],
                                batch_size: int = 100) -> Dict:
        """Optimize batch processing with different strategies"""
        results = {}

        # Strategy 1: Single large batch
        start_time = time.time()
        predictions_1 = self.model.predict(inputs)
        time_1 = time.time() - start_time

        # Strategy 2: Multiple small batches
        start_time = time.time()
        predictions_2 = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_predictions = self.model.predict(batch)
            predictions_2.extend(batch_predictions)
        time_2 = time.time() - start_time

        # Strategy 3: Parallel processing
        start_time = time.time()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_predictions = []
            futures = []

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                future = executor.submit(self.model.predict, batch)
                futures.append(future)

            for future in futures:
                parallel_predictions.extend(future.result())
        time_3 = time.time() - start_time

        results["strategies"] = {
            "single_batch": {
                "time": time_1,
                "predictions": predictions_1.tolist()
            },
            "small_batches": {
                "time": time_2,
                "predictions": predictions_2
            },
            "parallel_batches": {
                "time": time_3,
                "predictions": parallel_predictions
            }
        }

        # Find best strategy
        times = [time_1, time_2, time_3]
        best_idx = times.index(min(times))
        strategy_names = ["single_batch", "small_batches", "parallel_batches"]
        results["best_strategy"] = strategy_names[best_idx]
        results["best_time"] = times[best_idx]

        return results

    def garbage_collection_optimization(self):
        """Optimize memory with garbage collection"""
        # Force garbage collection
        gc.collect()

        # Get memory statistics before and after
        memory_before = psutil.virtual_memory().percent

        # Force multiple GC cycles
        for _ in range(3):
            gc.collect()

        memory_after = psutil.virtual_memory().percent

        return {
            "memory_before_gc": memory_before,
            "memory_after_gc": memory_after,
            "memory_freed": memory_before - memory_after
        }

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.prediction_times:
            return {"error": "No prediction times recorded"}

        import numpy as np

        # Calculate statistics
        avg_prediction_time = np.mean(self.prediction_times)
        p95_prediction_time = np.percentile(self.prediction_times, 95)
        p99_prediction_time = np.percentile(self.prediction_times, 99)

        # Memory statistics
        if self.memory_usage:
            avg_memory = np.mean([m['percent'] for m in self.memory_usage])
            peak_memory = max([m['percent'] for m in self.memory_usage])
        else:
            avg_memory = 0
            peak_memory = 0

        # CPU statistics
        if self.cpu_usage:
            avg_cpu = np.mean(self.cpu_usage)
            peak_cpu = max(self.cpu_usage)
        else:
            avg_cpu = 0
            peak_cpu = 0

        report = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "total_predictions": len(self.prediction_times),
                "average_time_ms": avg_prediction_time * 1000,
                "p95_time_ms": p95_prediction_time * 1000,
                "p99_time_ms": p99_prediction_time * 1000,
                "min_time_ms": min(self.prediction_times) * 1000,
                "max_time_ms": max(self.prediction_times) * 1000
            },
            "memory": {
                "average_usage_percent": avg_memory,
                "peak_usage_percent": peak_memory,
                "data_points": len(self.memory_usage)
            },
            "cpu": {
                "average_usage_percent": avg_cpu,
                "peak_usage_percent": peak_cpu,
                "data_points": len(self.cpu_usage)
            }
        }

        # Add GPU statistics if available
        if self.gpu_usage:
            gpu_loads = [g['load'] for g in self.gpu_usage]
            gpu_memory = [g['memory_used'] for g in self.gpu_usage]

            report["gpu"] = {
                "average_load_percent": np.mean(gpu_loads),
                "peak_load_percent": max(gpu_loads),
                "average_memory_used_mb": np.mean(gpu_memory),
                "peak_memory_used_mb": max(gpu_memory)
            }

        return report

class ModelOptimizer:
    """Optimize model for better performance"""

    def __init__(self, model):
        self.model = model

    def optimize_for_inference(self):
        """Optimize model for inference"""
        optimization_results = {}

        try:
            # 1. Model quantization (for supported models)
            if hasattr(self.model, 'predict'):
                optimization_results["quantization"] = self._try_quantization()

            # 2. Model pruning
            optimization_results["pruning"] = self._try_pruning()

            # 3. Caching optimization
            optimization_results["caching"] = self._optimize_caching()

            # 4. Memory optimization
            optimization_results["memory"] = self._optimize_memory()

        except Exception as e:
            optimization_results["error"] = str(e)

        return optimization_results

    def _try_quantization(self) -> Dict:
        """Try to quantize model for smaller size and faster inference"""
        try:
            # For PyTorch models
            try:
                import torch
                if isinstance(self.model, torch.nn.Module):
                    # Convert to quantized model
                    quantized_model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )

                    return {
                        "status": "success",
                        "original_size_mb": self._get_model_size_mb(self.model),
                        "quantized_size_mb": self._get_model_size_mb(quantized_model),
                        "technique": "dynamic_quantization"
                    }
            except ImportError:
                pass

            # For TensorFlow models
            try:
                import tensorflow as tf
                if isinstance(self.model, tf.keras.Model):
                    converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()

                    return {
                        "status": "success",
                        "original_size_mb": self._get_model_size_mb(self.model),
                        "quantized_size_mb": len(tflite_model) / (1024 * 1024),
                        "technique": "tflite_quantization"
                    }
            except ImportError:
                pass

            return {"status": "not_applicable", "reason": "Model type not supported for quantization"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _try_pruning(self) -> Dict:
        """Try to prune model weights"""
        try:
            # This is a simplified example - real pruning requires more sophisticated approaches
            import numpy as np

            # Get model parameters
            params = []
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()

            original_size = sum(p.size for p in params if hasattr(p, 'size'))

            # Simple magnitude-based pruning (keep top 90% of weights)
            threshold = np.percentile([abs(p).max() for p in params if hasattr(p, 'max')], 10)

            for param in params:
                if hasattr(param, 'numpy'):
                    param.data[abs(param.data) < threshold] = 0

            pruned_size = sum((param.data != 0).sum() for param in params if hasattr(param, 'data'))

            return {
                "status": "success",
                "original_parameters": original_size,
                "pruned_parameters": pruned_size,
                "compression_ratio": original_size / pruned_size if pruned_size > 0 else float('inf')
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _optimize_caching(self) -> Dict:
        """Optimize model caching"""
        try:
            # Clear any existing cache
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()

            # Enable inference caching if supported
            if hasattr(self.model, 'cache_inference'):
                self.model.cache_inference(enabled=True)

            return {"status": "success", "caching_enabled": True}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _optimize_memory(self) -> Dict:
        """Optimize model memory usage"""
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()

            # Get memory usage before and after
            memory_before = psutil.virtual_memory().used / (1024 * 1024)  # MB
            gc.collect()
            memory_after = psutil.virtual_memory().used / (1024 * 1024)  # MB

            return {
                "status": "success",
                "garbage_collected": collected,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_freed_mb": memory_before - memory_after
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _get_model_size_mb(self, model) -> float:
        """Get model size in megabytes"""
        try:
            import tempfile
            import pickle

            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(model, f)
                size_bytes = f.tell()
                return size_bytes / (1024 * 1024)
        except:
            return 0.0

# Usage example
def optimize_model_performance():
    """Example of model performance optimization"""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Create sample model
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Initialize performance monitor
    monitor = PerformanceMonitor(model, "iris-classifier")

    # Start monitoring
    monitor.start_monitoring()

    # Run some predictions
    for _ in range(10):
        test_input = X[0:1]
        prediction = model.predict(test_input)

        # Record prediction time
        start_time = time.time()
        prediction = model.predict(test_input)
        prediction_time = time.time() - start_time
        monitor.prediction_times.append(prediction_time)

    # Stop monitoring
    monitor.stop_monitoring()

    # Generate performance report
    report = monitor.generate_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))

    # Optimize model
    optimizer = ModelOptimizer(model)
    optimization_results = optimizer.optimize_for_inference()
    print("Optimization Results:", json.dumps(optimization_results, indent=2))

    # Test batch processing optimization
    batch_inputs = X[:1000]
    batch_results = monitor.optimize_batch_processing(batch_inputs)
    print("Batch Optimization Results:", json.dumps(batch_results, indent=2))

if __name__ == "__main__":
    optimize_model_performance()
```

---

## Summary

This comprehensive guide covers all aspects of model deploy

ment and production systems for machine learning:

### Key Topics Covered:

1. **Deployment Strategies**: Blue-green, canary, and A/B testing approaches with complete code implementations
2. **Cloud Platforms**: Detailed guides for AWS SageMaker, GCP Vertex AI, and Azure ML with cost comparisons
3. **Containerization**: Docker best practices, multi-stage builds, and production-ready configurations
4. **Orchestration**: Kubernetes deployments, scaling, and service management
5. **MLOps**: CI/CD pipelines, model versioning, and automated workflows
6. **Monitoring**: Prometheus metrics, ELK logging, and Grafana dashboards
7. **Scaling**: Load balancing, caching strategies, and performance optimization
8. **API Development**: FastAPI implementation with security, authentication, and rate limiting
9. **Production Workflows**: Complete deployment pipelines and model registry management
10. **Security**: Authentication, authorization, input validation, and audit logging
11. **Privacy**: Differential privacy, federated learning, and encryption techniques
12. **Troubleshooting**: Performance monitoring, optimization strategies, and diagnostic tools

### Production Checklist:

- ✅ Docker containerization with multi-stage builds
- ✅ Kubernetes orchestration with scaling policies
- ✅ CI/CD pipeline with automated testing
- ✅ Model monitoring with Prometheus metrics
- ✅ Security implementation with authentication
- ✅ Rate limiting and input validation
- ✅ Audit logging for compliance
- ✅ Performance optimization and caching
- ✅ Disaster recovery and backup procedures
- ✅ Cost optimization and resource management

This guide provides production-ready code and configurations that can be directly implemented in enterprise ML deployments, ensuring scalable, secure, and maintainable machine learning systems.
