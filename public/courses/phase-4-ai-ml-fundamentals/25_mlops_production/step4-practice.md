# MLOps & Production ML - Practice

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Model Development Workflow](#model-development-workflow)
3. [Model Experimentation and Tracking](#model-experimentation-and-tracking)
4. [Model Deployment](#model-deployment)
5. [Model Monitoring](#model-monitoring)
6. [ML Pipeline Automation](#ml-pipeline-automation)
7. [CI/CD for ML](#cicd-for-ml)
8. [Model Registry](#model-registry)
9. [Production Monitoring](#production-monitoring)
10. [End-to-End MLOps Project](#end-to-end-mlops-project)

## Setup and Installation

### Environment Setup

```bash
# Create virtual environment
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas scikit-learn numpy matplotlib seaborn
pip install mlflow wandb great-expectations
pip install jupyter ipython pytest black flake8
pip install docker kubernetes
pip install fastapi uvicorn joblib pickle-mixin

# Install additional ML libraries
pip install xgboost lightgbm catboost
pip install tensorflow torch
```

### Project Structure Creation

```python
import os
import pathlib

def create_mlops_project_structure(project_name):
    """Create standard MLOps project structure"""

    base_path = pathlib.Path(project_name)

    # Create directories
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'src/deployment',
        'tests/unit',
        'tests/integration',
        'models/trained',
        'models/archived',
        'notebooks',
        'config',
        'scripts',
        'docs',
        'monitoring'
    ]

    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/features/__init__.py',
        'src/models/__init__.py',
        'src/visualization/__init__.py',
        'src/deployment/__init__.py',
        'tests/__init__.py'
    ]

    for init_file in init_files:
        (base_path / init_file).touch()

    # Create configuration files
    create_config_files(base_path)

    print(f"Created MLOps project structure at {base_path}")

def create_config_files(base_path):
    """Create configuration files"""

    # requirements.txt
    requirements = """
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
mlflow>=2.0.0
great-expectations>=0.15.0
pytest>=7.0.0
fastapi>=0.80.0
uvicorn>=0.18.0
joblib>=1.2.0
"""

    (base_path / 'requirements.txt').write_text(requirements)

    # .gitignore
    gitignore = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environment files
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Data files
data/raw/
data/processed/
data/external/
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Models
models/trained/
models/archived/
!models/.gitkeep

# MLflow
mlruns/
mlartifacts/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
"""

    (base_path / '.gitignore').write_text(gitignore)

    # Makefile
    makefile = """
.PHONY: install test lint format train deploy clean

install:
\tpip install -r requirements.txt

test:
\tpython -m pytest tests/

lint:
\tflake8 src/ tests/
\tblack --check src/ tests/

format:
\tblack src/ tests/
\tisort src/ tests/

train:
\tpython src/models/train_model.py

deploy:
\tpython src/deployment/deploy_model.py

clean:
\trm -rf data/raw/
\trm -rf data/processed/
\trm -rf models/trained/
\trm -rf mlruns/
"""

    (base_path / 'Makefile').write_text(makefile)

# Usage
create_mlops_project_structure("mlops_project")
```

## Model Development Workflow

### Exercise 1: Data Processing Pipeline

```python
# src/data/make_dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, file_path):
        """Load data from file"""
        logger.info(f"Loading data from {file_path}")

        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def clean_data(self, df):
        """Clean and preprocess data"""
        logger.info("Cleaning data")

        # Handle missing values
        df = df.dropna(subset=['target'])  # Drop rows with missing target

        # Fill missing values in features
        for column in df.columns:
            if df[column].isnull().any() and column != 'target':
                if df[column].dtype in ['object']:
                    df[column].fillna('unknown', inplace=True)
                else:
                    df[column].fillna(df[column].median(), inplace=True)

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'target']

        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers instead of removing
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"After cleaning: {len(df)} records")
        return df

    def engineer_features(self, df):
        """Create new features"""
        logger.info("Engineering features")

        # Example feature engineering
        if 'amount' in df.columns and 'quantity' in df.columns:
            df['price_per_unit'] = df['amount'] / df['quantity']
            df['price_per_unit'] = df['price_per_unit'].replace([np.inf, -np.inf], 0)

        # Date features
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for column in date_columns:
            df[f'{column}_year'] = df[column].dt.year
            df[f'{column}_month'] = df[column].dt.month
            df[f'{column}_day'] = df[column].dt.day
            df[f'{column}_dayofweek'] = df[column].dt.dayofweek

        return df

    def encode_categorical_features(self, df):
        """Encode categorical features"""
        logger.info("Encoding categorical features")

        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'target']

        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                # Handle new categories in test data
                le = self.label_encoders[column]
                # For unseen categories, assign a default value
                df[column] = df[column].astype(str)
                df[column] = df[column].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return df

    def split_data(self, df):
        """Split data into train, validation, and test sets"""
        logger.info("Splitting data")

        X = df.drop('target', axis=1)
        y = df['target']

        # Stratified split to maintain class distribution
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )  # 0.25 * 0.8 = 0.2 of total

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """Scale numerical features"""
        logger.info("Scaling features")

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def process_data(self, data_path, output_dir):
        """Complete data processing pipeline"""
        logger.info("Starting data processing pipeline")

        # Load data
        df = self.load_data(data_path)

        # Clean data
        df = self.clean_data(df)

        # Engineer features
        df = self.engineer_features(df)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        # Save processed data
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train_scaled.to_csv(output_dir / 'X_train.csv', index=False)
        X_val_scaled.to_csv(output_dir / 'X_val.csv', index=False)
        X_test_scaled.to_csv(output_dir / 'X_test.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
        y_val.to_csv(output_dir / 'y_val.csv', index=False)
        y_test.to_csv(output_dir / 'y_test.csv', index=False)

        # Save preprocessors
        import joblib
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoders, output_dir / 'label_encoders.pkl')

        logger.info("Data processing completed successfully")

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

# Usage example
if __name__ == "__main__":
    config = {
        'target_column': 'target',
        'test_size': 0.2,
        'random_state': 42
    }

    processor = DataProcessor(config)
    results = processor.process_data(
        data_path=Path('data/raw/dataset.csv'),
        output_dir=Path('data/processed')
    )
```

### Exercise 2: Model Training Pipeline

```python
# src/models/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name="model_training"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def load_data(self, data_dir):
        """Load processed data"""
        logger.info(f"Loading data from {data_dir}")

        data_dir = Path(data_dir)

        X_train = pd.read_csv(data_dir / 'X_train.csv')
        X_val = pd.read_csv(data_dir / 'X_val.csv')
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv')
        y_val = pd.read_csv(data_dir / 'y_val.csv')
        y_test = pd.read_csv(data_dir / 'y_test.csv')

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train, y_train):
        """Train multiple models"""
        logger.info("Training multiple models")

        model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1, 10],
                    'random_state': 42,
                    'max_iter': 1000
                }
            }
        }

        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}")

            # Manual hyperparameter tuning (simplified)
            best_score = 0
            best_params = None
            best_model = None

            for param_combination in self._generate_param_combinations(config['params']):
                model = config['model'](**param_combination)

                try:
                    model.fit(X_train, y_train)

                    # Use a simple validation strategy for this example
                    train_score = model.score(X_train, y_train)

                    if train_score > best_score:
                        best_score = train_score
                        best_params = param_combination
                        best_model = model

                except Exception as e:
                    logger.warning(f"Failed to train model with params {param_combination}: {e}")
                    continue

            self.models[model_name] = {
                'model': best_model,
                'params': best_params,
                'score': best_score
            }

            logger.info(f"{model_name} best score: {best_score:.4f}")

    def _generate_param_combinations(self, param_dict):
        """Generate parameter combinations for grid search"""
        import itertools

        keys, values = zip(*param_dict.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def evaluate_models(self, X_val, y_val):
        """Evaluate all trained models"""
        logger.info("Evaluating models on validation set")

        for model_name, model_info in self.models.items():
            model = model_info['model']

            try:
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1_score': f1_score(y_val, y_pred, average='weighted')
                }

                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)

                self.results[model_name] = metrics

                logger.info(f"{model_name} metrics: {metrics}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")

    def select_best_model(self):
        """Select best model based on validation metrics"""
        logger.info("Selecting best model")

        # Use F1 score as primary metric for model selection
        best_model_name = None
        best_score = 0

        for model_name, metrics in self.results.items():
            if 'f1_score' in metrics and metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model_name = model_name

        if best_model_name:
            best_model_info = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
            return best_model_name, best_model_info
        else:
            logger.error("No valid model found")
            return None, None

    def log_to_mlflow(self, model_name, model_info, metrics):
        """Log model and metrics to MLflow"""
        logger.info(f"Logging {model_name} to MLflow")

        with mlflow.start_run(run_name=model_name):
            # Log parameters
            params = model_info['params']
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(
                model_info['model'],
                "model",
                registered_model_name=f"{model_name}_model"
            )

            # Log artifacts
            import shutil
            model_path = Path('models/trained') / f"{model_name}_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_info['model'], model_path)
            mlflow.log_artifact(str(model_path))

    def train_and_evaluate(self, data_dir, model_dir):
        """Complete training and evaluation pipeline"""
        logger.info("Starting model training pipeline")

        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data(data_dir)

        # Train models
        self.train_models(X_train, y_train)

        # Evaluate models
        self.evaluate_models(X_val, y_val)

        # Select best model
        best_model_name, best_model_info = self.select_best_model()

        if best_model_name:
            # Log to MLflow
            best_metrics = self.results[best_model_name]
            self.log_to_mlflow(best_model_name, best_model_info, best_metrics)

            # Save final model
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = model_dir / 'production_model.pkl'
            joblib.dump(best_model_info['model'], final_model_path)

            logger.info(f"Training pipeline completed. Best model: {best_model_name}")
            return best_model_name, best_model_info, best_metrics
        else:
            logger.error("Training pipeline failed - no valid model")
            return None, None, None

# Usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model_name, best_model_info, metrics = trainer.train_and_evaluate(
        data_dir='data/processed',
        model_dir='models/trained'
    )
```

## Model Experimentation and Tracking

### Exercise 3: MLflow Integration

```python
# src/experiments/experiment_tracking.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ExperimentTracker:
    def __init__(self, experiment_name="default"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_data_summary(self, df, data_name="dataset"):
        """Log data summary to MLflow"""
        with mlflow.start_run(run_name=f"{data_name}_summary"):
            mlflow.log_param("total_rows", len(df))
            mlflow.log_param("total_columns", len(df.columns))

            # Log data types
            for dtype in df.dtypes.value_counts().items():
                mlflow.log_param(f"columns_{dtype[0]}", dtype[1])

            # Log missing values
            missing_counts = df.isnull().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    mlflow.log_param(f"missing_{col}", count)

    def run_experiment(self, run_name, model, X_train, X_test, y_train, y_test, params):
        """Run ML experiment with full logging"""

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("accuracy_diff", abs(train_accuracy - test_accuracy))

            # Log classification report
            report = classification_report(y_test, y_pred_test, output_dict=True)
            for class_label in report:
                if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    mlflow.log_metric(f"precision_{class_label}", report[class_label]['precision'])
                    mlflow.log_metric(f"recall_{class_label}", report[class_label]['recall'])
                    mlflow.log_metric(f"f1_{class_label}", report[class_label]['f1-score'])

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Create and log plots
            self._create_and_log_plots(X_test, y_test, y_pred_test, model)

            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'run_id': mlflow.active_run().info.run_id
            }

    def _create_and_log_plots(self, X_test, y_test, y_pred, model):
        """Create and log visualization plots"""

        # Confusion matrix
        from sklearn.metrics import confusion_matrix

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plot_path = Path('plots/confusion_matrix.png')
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(str(plot_path))

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))

            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()

            importance_path = Path('plots/feature_importance.png')
            plt.savefig(importance_path)
            plt.close()

            mlflow.log_artifact(str(importance_path))

    def compare_experiments(self, experiment_name=None):
        """Compare multiple experiments"""
        from mlflow.tracking import MlflowClient

        if experiment_name is None:
            experiment_name = self.experiment_name

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_accuracy DESC"]
        )

        print(f"Experiment: {experiment_name}")
        print(f"Total runs: {len(runs)}")
        print("\nTop 5 runs:")

        for i, run in enumerate(runs[:5]):
            print(f"\nRun {i+1}:")
            print(f"  Run ID: {run.info.run_id}")
            print(f"  Test Accuracy: {run.data.metrics.get('test_accuracy', 'N/A')}")
            print(f"  Train Accuracy: {run.data.metrics.get('train_accuracy', 'N/A')}")

            # Log parameters
            for param_name, param_value in run.data.params.items():
                print(f"  {param_name}: {param_value}")

        return runs

    def load_best_model(self, experiment_name=None, metric="test_accuracy"):
        """Load the best model from experiments"""
        if experiment_name is None:
            experiment_name = self.experiment_name

        runs = self.compare_experiments(experiment_name)

        if runs:
            best_run = runs[0]
            model_uri = f"runs:/{best_run.info.run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            return model, best_run.info.run_id

        return None, None

# Usage example
if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    # Initialize tracker
    tracker = ExperimentTracker("customer_churn_experiment")

    # Run multiple experiments
    experiments = [
        {
            'run_name': 'rf_depth_10',
            'model': RandomForestClassifier(n_estimators=100, max_depth=10),
            'params': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'run_name': 'rf_depth_20',
            'model': RandomForestClassifier(n_estimators=100, max_depth=20),
            'params': {'n_estimators': 100, 'max_depth': 20}
        }
    ]

    results = []
    for exp in experiments:
        result = tracker.run_experiment(
            exp['run_name'],
            exp['model'],
            X_train, X_test, y_train, y_test,
            exp['params']
        )
        results.append(result)

    # Compare experiments
    runs = tracker.compare_experiments()

    # Load best model
    best_model, run_id = tracker.load_best_model()
```

## Model Deployment

### Exercise 4: FastAPI Deployment

```python
# src/deployment/deploy_model.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model API",
    description="API for machine learning model predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_metadata = {}

class PredictionRequest(BaseModel):
    """Request model for prediction"""
    features: dict

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature1": 1.5,
                    "feature2": 2.0,
                    "feature3": "value"
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: float
    prediction_label: str
    confidence: float
    timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    features_list: list

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    model_version: str = "unknown"

def load_model():
    """Load the trained model"""
    global model, model_metadata

    model_path = Path('models/trained/production_model.pkl')

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_metadata = {
                'loaded_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'model_type': type(model).__name__
            }
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    else:
        logger.error(f"Model file not found: {model_path}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting ML Model API")
    load_model()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        model_version=model_metadata.get('model_type', 'unknown')
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(features_df)[0]

        # Get probability if available
        confidence = 1.0
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
            confidence = float(max(probabilities))

        # Convert prediction to label if needed
        if isinstance(prediction, (int, np.integer)):
            prediction_label = str(prediction)
        else:
            prediction_label = prediction

        response = PredictionResponse(
            prediction=float(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('model_type', 'unknown')
        )

        # Log prediction
        logger.info(f"Prediction made: {response.dict()}")

        return response

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail="Invalid input data")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        features_df = pd.DataFrame(request.features_list)

        # Make predictions
        predictions = model.predict(features_df)

        # Get probabilities if available
        confidences = []
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)
            confidences = [float(max(prob)) for prob in probabilities]
        else:
            confidences = [1.0] * len(predictions)

        # Prepare response
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                "prediction": float(pred),
                "prediction_label": str(pred),
                "confidence": conf,
                "input_index": i
            })

        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = {
        "model_type": type(model).__name__,
        "model_path": model_metadata.get('model_path', 'unknown'),
        "loaded_at": model_metadata.get('loaded_at', 'unknown'),
        "features": list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else "unknown",
        "classes": list(model.classes_) if hasattr(model, 'classes_') else "unknown"
    }

    return info

@app.post("/reload")
async def reload_model():
    """Reload the model"""
    success = load_model()

    if success:
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    uvicorn.run(
        "deploy_model:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### Exercise 5: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.deployment.deploy_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/trained/production_model.pkl
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ml-api
    restart: unless-stopped

volumes:
  logs:
```

## Model Monitoring

### Exercise 6: Model Performance Monitoring

```python
# monitoring/model_monitor.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, db_path="monitoring/model_monitor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction REAL NOT NULL,
                actual REAL,
                features TEXT NOT NULL,
                model_version TEXT NOT NULL,
                confidence REAL,
                is_correct BOOLEAN
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                model_version TEXT NOT NULL,
                window_size INTEGER DEFAULT 100
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def log_prediction(self, features, prediction, actual=None, model_version="1.0", confidence=1.0):
        """Log prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Determine if prediction is correct
        is_correct = None
        if actual is not None:
            is_correct = (prediction == actual)

        cursor.execute('''
            INSERT INTO predictions
            (timestamp, prediction, actual, features, model_version, confidence, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction,
            actual,
            json.dumps(features),
            model_version,
            confidence,
            is_correct
        ))

        conn.commit()
        conn.close()

    def calculate_metrics(self, window_size=100):
        """Calculate rolling metrics"""
        conn = sqlite3.connect(self.db_path)

        # Get recent predictions
        query = '''
            SELECT prediction, actual, confidence
            FROM predictions
            WHERE actual IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        '''

        df = pd.read_sql_query(query, conn, params=(window_size,))
        conn.close()

        if len(df) == 0:
            return None

        metrics = {}

        # Accuracy
        if len(df) > 0:
            metrics['accuracy'] = accuracy_score(df['actual'], df['prediction'])
            metrics['precision'] = precision_score(df['actual'], df['prediction'], average='weighted', zero_division=0)
            metrics['recall'] = recall_score(df['actual'], df['prediction'], average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(df['actual'], df['prediction'], average='weighted', zero_division=0)

        # Average confidence
        metrics['avg_confidence'] = df['confidence'].mean()

        # Prediction distribution
        metrics['prediction_0_count'] = (df['prediction'] == 0).sum()
        metrics['prediction_1_count'] = (df['prediction'] == 1).sum()

        return metrics

    def store_metrics(self, metrics, model_version="1.0"):
        """Store calculated metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for metric_name, metric_value in metrics.items():
            cursor.execute('''
                INSERT INTO metrics (timestamp, metric_name, metric_value, model_version)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metric_name,
                metric_value,
                model_version
            ))

        conn.commit()
        conn.close()

    def check_for_alerts(self, metrics):
        """Check metrics for alert conditions"""
        alerts = []

        # Low accuracy alert
        if 'accuracy' in metrics and metrics['accuracy'] < 0.8:
            alerts.append({
                'type': 'low_accuracy',
                'message': f"Model accuracy dropped to {metrics['accuracy']:.3f}",
                'severity': 'high'
            })

        # Low confidence alert
        if 'avg_confidence' in metrics and metrics['avg_confidence'] < 0.6:
            alerts.append({
                'type': 'low_confidence',
                'message': f"Average prediction confidence dropped to {metrics['avg_confidence']:.3f}",
                'severity': 'medium'
            })

        # Prediction imbalance alert
        if 'prediction_0_count' in metrics and 'prediction_1_count' in metrics:
            total_predictions = metrics['prediction_0_count'] + metrics['prediction_1_count']
            if total_predictions > 0:
                class_balance = abs(metrics['prediction_0_count'] - metrics['prediction_1_count']) / total_predictions
                if class_balance > 0.8:
                    alerts.append({
                        'type': 'prediction_imbalance',
                        'message': f"Prediction distribution is highly imbalanced: {class_balance:.3f}",
                        'severity': 'medium'
                    })

        return alerts

    def store_alerts(self, alerts):
        """Store alerts in database"""
        if not alerts:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for alert in alerts:
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, message, severity)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                alert['type'],
                alert['message'],
                alert['severity']
            ))

        conn.commit()
        conn.close()

        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")

    def get_recent_alerts(self, hours=24):
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)

        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        query = '''
            SELECT * FROM alerts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        '''

        df = pd.read_sql_query(query, conn, params=(since,))
        conn.close()

        return df

    def generate_report(self, window_size=100):
        """Generate monitoring report"""
        metrics = self.calculate_metrics(window_size)

        if metrics is None:
            return {"error": "No data available for metrics calculation"}

        alerts = self.check_for_alerts(metrics)

        # Store metrics and alerts
        self.store_metrics(metrics)
        self.store_alerts(alerts)

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alerts': alerts,
            'total_alerts': len(alerts),
            'window_size': window_size
        }

        return report

    def get_performance_trend(self, metric_name='accuracy', days=7):
        """Get performance trend over time"""
        conn = sqlite3.connect(self.db_path)

        since = (datetime.now() - timedelta(days=days)).isoformat()
        query = '''
            SELECT timestamp, metric_value
            FROM metrics
            WHERE metric_name = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''

        df = pd.read_sql_query(query, conn, params=(metric_name, since))
        conn.close()

        if len(df) == 0:
            return {"error": f"No data available for {metric_name}"}

        # Convert timestamp to datetime for plotting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        # Aggregate by date
        daily_metrics = df.groupby('date')[metric_name].agg(['mean', 'std', 'count']).reset_index()

        return {
            'metric_name': metric_name,
            'trend_data': daily_metrics.to_dict('records'),
            'period_days': days
        }

# Usage example
def monitor_loop():
    """Continuous monitoring loop"""
    monitor = ModelMonitor()

    # Simulate some predictions with actual values
    for i in range(100):
        # Simulate features (replace with actual feature extraction)
        features = {
            'feature1': np.random.normal(0, 1),
            'feature2': np.random.normal(0, 1),
            'feature3': np.random.choice([0, 1])
        }

        # Simulate prediction
        prediction = np.random.choice([0, 1])
        actual = np.random.choice([0, 1])
        confidence = np.random.uniform(0.5, 1.0)

        # Log prediction
        monitor.log_prediction(
            features=features,
            prediction=prediction,
            actual=actual,
            model_version="1.0",
            confidence=confidence
        )

        time.sleep(0.1)

    # Generate report
    report = monitor.generate_report()
    print("Monitoring Report:", json.dumps(report, indent=2))

    # Get recent alerts
    alerts = monitor.get_recent_alerts()
    print(f"Recent alerts: {len(alerts)}")

    # Get performance trend
    trend = monitor.get_performance_trend()
    print("Performance trend:", trend)

if __name__ == "__main__":
    monitor_loop()
```

## ML Pipeline Automation

### Exercise 7: Airflow DAG for ML Pipeline

```python
# ml_pipeline_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import logging

# Default arguments
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

# Create DAG
dag = DAG(
    'ml_model_pipeline',
    default_args=default_args,
    description='Automated ML model training and deployment pipeline',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'pipeline']
)

def extract_data():
    """Extract data from source"""
    import pandas as pd
    import numpy as np

    # Simulate data extraction
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.choice([0, 1], 1000),
        'target': np.random.choice([0, 1], 1000)
    }

    df = pd.DataFrame(data)
    df.to_csv('/tmp/extracted_data.csv', index=False)
    logging.info(f"Extracted {len(df)} records")
    return '/tmp/extracted_data.csv'

def validate_data(file_path):
    """Validate extracted data"""
    import pandas as pd
    import logging

    df = pd.read_csv(file_path)

    # Basic validation
    assert len(df) > 0, "Data is empty"
    assert df.isnull().sum().sum() == 0, "Data contains null values"
    assert len(df.columns) == 4, "Unexpected number of columns"

    logging.info(f"Data validation passed: {len(df)} records, {len(df.columns)} columns")
    return file_path

def prepare_features(file_path):
    """Feature engineering and preparation"""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_path)

    # Simple feature engineering
    df['feature1_squared'] = df['feature1'] ** 2
    df['feature2_log'] = np.log(df['feature2'] + 2)  # Add 2 to avoid log(0)
    df['features_interaction'] = df['feature1'] * df['feature2']

    output_path = '/tmp/processed_data.csv'
    df.to_csv(output_path, index=False)

    logging.info(f"Feature engineering completed: {len(df.columns)} features")
    return output_path

def train_model(file_path):
    """Train machine learning model"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import logging

    # Load data
    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    model_path = '/tmp/production_model.pkl'
    joblib.dump(model, model_path)

    logging.info(f"Model trained with accuracy: {accuracy:.4f}")
    return model_path

def validate_model(model_path):
    """Validate trained model"""
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    import logging

    # Load model
    model = joblib.load(model_path)

    # Load test data
    df = pd.read_csv('/tmp/processed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)

    if accuracy < 0.7:
        raise ValueError(f"Model accuracy {accuracy:.4f} is below acceptable threshold")

    logging.info(f"Model validation passed with accuracy: {accuracy:.4f}")
    return accuracy

def deploy_model(model_path):
    """Deploy model to production"""
    import shutil
    import logging

    # Copy model to deployment directory
    deployment_path = '/opt/airflow/models/production_model.pkl'
    shutil.copy(model_path, deployment_path)

    logging.info(f"Model deployed to {deployment_path}")
    return deployment_path

# Define tasks
extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

prepare_features_task = PythonOperator(
    task_id='prepare_features',
    python_callable=prepare_features,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Set task dependencies
extract_data_task >> validate_data_task >> prepare_features_task
prepare_features_task >> train_model_task
train_model_task >> validate_model_task
validate_model_task >> deploy_model_task
```

## End-to-End MLOps Project

### Exercise 8: Complete MLOps Implementation

```python
# main_mlops_pipeline.py
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndMLOps:
    """Complete MLOps pipeline implementation"""

    def __init__(self, project_name="mlops_project", experiment_name="production_pipeline"):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.model = None
        self.metrics = {}

        # Create directories
        self._create_project_structure()

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def _create_project_structure(self):
        """Create project directory structure"""
        directories = [
            'data/raw',
            'data/processed',
            'data/external',
            'models/trained',
            'models/archived',
            'models/production',
            'logs',
            'reports',
            'monitoring'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def generate_sample_data(self):
        """Generate sample dataset for the project"""
        logger.info("Generating sample dataset")

        np.random.seed(42)
        n_samples = 1000

        # Generate features
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'education_years': np.random.randint(10, 20, n_samples),
            'experience_years': np.random.randint(0, 40, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'debt_to_income': np.random.uniform(0, 1, n_samples)
        }

        # Create target variable with some realistic relationships
        df = pd.DataFrame(data)

        # Simple target creation (e.g., loan approval)
        target = (
            (df['credit_score'] > 650) * 0.3 +
            (df['debt_to_income'] < 0.4) * 0.3 +
            (df['income'] > 40000) * 0.2 +
            (df['experience_years'] > 5) * 0.1 +
            np.random.normal(0, 0.2, n_samples)
        )
        df['loan_approved'] = (target > 0.5).astype(int)

        # Save raw data
        output_path = Path('data/raw/loan_data.csv')
        df.to_csv(output_path, index=False)

        logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")
        return output_path

    def load_and_validate_data(self, data_path):
        """Load and validate data"""
        logger.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)

        # Validation checks
        if len(df) == 0:
            raise ValueError("Dataset is empty")

        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains missing values")

        if 'loan_approved' not in df.columns:
            raise ValueError("Target column 'loan_approved' not found")

        logger.info(f"Data validation passed: {len(df)} records, {len(df.columns)} columns")

        # Log data summary to MLflow
        with mlflow.start_run(run_name="data_validation"):
            mlflow.log_param("total_rows", len(df))
            mlflow.log_param("total_columns", len(df.columns))
            mlflow.log_param("target_distribution", df['loan_approved'].value_counts().to_dict())

        return df

    def feature_engineering(self, df):
        """Perform feature engineering"""
        logger.info("Performing feature engineering")

        # Create new features
        df['income_per_year_education'] = df['income'] / df['education_years']
        df['debt_to_income_ratio'] = df['debt_to_income']
        df['credit_category'] = pd.cut(
            df['credit_score'],
            bins=[300, 600, 700, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )

        # Encode categorical features
        credit_category_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
        df['credit_category_encoded'] = df['credit_category'].map(credit_category_map)

        # Remove original categorical column
        df = df.drop('credit_category', axis=1)

        logger.info(f"Feature engineering completed: {len(df.columns)} total features")

        # Log feature information
        with mlflow.start_run(run_name="feature_engineering"):
            for feature in df.columns:
                if feature != 'loan_approved':
                    mlflow.log_param("feature", feature)

        return df

    def train_and_evaluate_models(self, df):
        """Train multiple models and select the best one"""
        logger.info("Training and evaluating models")

        # Prepare data
        X = df.drop('loan_approved', axis=1)
        y = df['loan_approved']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train different models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'random_forest_tuned': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
        }

        best_model = None
        best_score = 0
        best_model_name = None

        for model_name, model in models.items():
            logger.info(f"Training {model_name}")

            # Start MLflow run
            with mlflow.start_run(run_name=f"train_{model_name}"):
                # Train model
                model.fit(X_train, y_train)

                # Evaluate
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)

                # Log metrics
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("overfitting", train_accuracy - test_accuracy)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                # Select best model based on test accuracy
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_model = model
                    best_model_name = model_name

                logger.info(f"{model_name} - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

        logger.info(f"Best model: {best_model_name} with test accuracy: {best_score:.4f}")

        # Store final model and metrics
        self.model = best_model
        self.metrics = {
            'test_accuracy': best_score,
            'model_name': best_model_name,
            'features': list(X.columns)
        }

        return best_model, best_score

    def save_production_model(self, model, model_path):
        """Save model for production use"""
        logger.info(f"Saving production model to {model_path}")

        # Save model
        joblib.dump(model, model_path)

        # Save model metadata
        metadata_path = Path(model_path).parent / "model_metadata.json"
        metadata = {
            'model_type': type(model).__name__,
            'features': self.metrics.get('features', []),
            'test_accuracy': self.metrics.get('test_accuracy', 0),
            'training_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Production model saved successfully")

        # Log to MLflow
        with mlflow.start_run(run_name="save_production_model"):
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metadata_path))

    def deploy_api(self):
        """Deploy model as API"""
        logger.info("Deploying model API")

        # Create simple deployment script
        deployment_script = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Approval API", version="1.0.0")

class LoanRequest(BaseModel):
    age: int
    income: float
    education_years: int
    experience_years: int
    credit_score: int
    debt_to_income: float
    income_per_year_education: float
    debt_to_income_ratio: float
    credit_category_encoded: int

# Load model
try:
    model = joblib.load('models/production/production_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.post("/predict")
async def predict_loan_approval(request: LoanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    data = pd.DataFrame([request.dict()])

    # Make prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max() if hasattr(model, 'predict_proba') else 1.0

    return {
        "loan_approved": bool(prediction),
        "confidence": float(probability),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        api_script_path = Path('deployment/api_server.py')
        with open(api_script_path, 'w') as f:
            f.write(deployment_script)

        logger.info(f"API deployment script created at {api_script_path}")

    def run_complete_pipeline(self):
        """Run the complete MLOps pipeline"""
        logger.info("Starting complete MLOps pipeline")

        try:
            # Step 1: Generate data
            data_path = self.generate_sample_data()

            # Step 2: Load and validate data
            df = self.load_and_validate_data(data_path)

            # Step 3: Feature engineering
            df_engineered = self.feature_engineering(df)

            # Step 4: Train and evaluate models
            best_model, best_score = self.train_and_evaluate_models(df_engineered)

            # Step 5: Save production model
            model_path = Path('models/production/production_model.pkl')
            self.save_production_model(best_model, model_path)

            # Step 6: Deploy API
            self.deploy_api()

            logger.info("Complete MLOps pipeline finished successfully!")

            # Generate summary report
            summary = {
                'pipeline_status': 'success',
                'model_accuracy': best_score,
                'model_type': self.metrics.get('model_name'),
                'features_used': len(self.metrics.get('features', [])),
                'timestamp': datetime.now().isoformat()
            }

            summary_path = Path('reports/pipeline_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Pipeline summary saved to {summary_path}")
            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

            # Log failure
            with mlflow.start_run(run_name="pipeline_failure"):
                mlflow.log_param("error", str(e))
                mlflow.log_param("status", "failed")

            raise

# Usage
if __name__ == "__main__":
    mlops_pipeline = EndToEndMLOps()
    result = mlops_pipeline.run_complete_pipeline()
    print("Pipeline Result:", json.dumps(result, indent=2))
```

This comprehensive practice module provides hands-on experience with all aspects of MLOps and Production ML, from model development to deployment and monitoring. Each exercise builds upon the previous ones to create a complete understanding of modern ML operations.
