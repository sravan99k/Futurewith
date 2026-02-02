# Model Selection Cheat Sheet - Phase 13

## 📋 Table of Contents

1. [Data Science Pipeline Steps](#data-science-pipeline-steps)
2. [Model Selection Criteria](#model-selection-criteria)
3. [Cross-Validation Strategies](#cross-validation-strategies)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Feature Engineering Techniques](#feature-engineering-techniques)

---

## 🔄 Data Science Pipeline Steps

### 1. Business Understanding

```python
# Define problem, success metrics, constraints
problem_definition = {
    'objective': 'predict_target_variable',
    'success_metric': 'accuracy > 0.85',
    'business_impact': 'revenue_increase',
    'constraints': ['time', 'budget', 'regulatory']
}
```

### 2. Data Collection & Understanding

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
data = pd.read_csv('dataset.csv')
data.head()
data.info()
data.describe()
data.isnull().sum()
```

### 3. Data Preparation

```python
# Handle missing values
data['column'].fillna(data['column'].median(), inplace=True)
data.dropna(inplace=True)

# Handle outliers
Q1 = data['column'].quantile(0.25)
Q3 = data['column'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['column'] >= Q1 - 1.5*IQR) &
            (data['column'] <= Q3 + 1.5*IQR)]
```

### 4. Exploratory Data Analysis (EDA)

```python
# Visualizations
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.histplot(data['target'])
plt.subplot(2, 3, 2)
sns.boxplot(data=data, x='target', y='feature')
plt.subplot(2, 3, 3)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
```

### 5. Feature Engineering

```python
# Create new features
data['new_feature'] = data['feature1'] / data['feature2']
data['log_feature'] = np.log(data['feature'])

# Encode categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one_hot = pd.get_dummies(data['category'])
label_enc = LabelEncoder()
data['encoded'] = label_enc.fit_transform(data['category'])
```

### 6. Model Selection & Training

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. Model Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

### 8. Model Deployment

```python
# Save model
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load and predict
loaded_model = joblib.load('model.pkl')
predictions = loaded_model.predict(new_data)
```

---

## 🎯 Model Selection Criteria

### Problem Type Selection

```python
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```

### Model Selection Matrix

| Problem Type              | Dataset Size    | Performance Priority | Recommended Models                  |
| ------------------------- | --------------- | -------------------- | ----------------------------------- |
| Binary Classification     | Small (<10k)    | Interpretability     | Logistic Regression, Naive Bayes    |
| Binary Classification     | Medium (10k-1M) | Balance              | Random Forest, XGBoost              |
| Binary Classification     | Large (>1M)     | Performance          | XGBoost, LightGBM, Neural Networks  |
| Multiclass Classification | Small           | Interpretability     | Logistic Regression, Decision Trees |
| Multiclass Classification | Large           | Performance          | XGBoost, CatBoost, Neural Networks  |
| Regression                | Small           | Interpretability     | Linear Regression, Ridge            |
| Regression                | Medium          | Balance              | Random Forest, Gradient Boosting    |
| Regression                | Large           | Performance          | XGBoost, Neural Networks            |

### Selection Criteria Checklist

- [ ] **Accuracy**: Does the model achieve acceptable performance?
- [ ] **Interpretability**: Can stakeholders understand the model?
- [ ] **Training Time**: Is training time reasonable for the use case?
- [ ] **Prediction Speed**: Is inference time acceptable?
- [ ] **Memory Usage**: Does the model fit in available memory?
- [ ] **Robustness**: Is the model stable across different data?
- [ ] **Scalability**: Can the model handle larger datasets?

---

## 🔀 Cross-Validation Strategies

### Basic Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Multiple metrics
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)
```

### Stratified Cross-Validation (Classification)

```python
from sklearn.model_selection import StratifiedKFold

# For imbalanced datasets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

# Standard approach for regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# For time-dependent data
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
```

### Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

# For very small datasets
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
```

### Group Cross-Validation

```python
from sklearn.model_selection import GroupKFold

# When groups are important (e.g., patient IDs)
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf)
```

### Custom CV Strategies

```python
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# Random sampling
rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=rs)

# Stratified random sampling
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=sss)
```

---

## ⚙️ Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3)
}

# Randomized search
random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)
```

### Bayesian Optimization (Hyperopt)

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1001)),
    'max_depth': hp.choice('max_depth', range(3, 20)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.8, 1.0),
    'min_samples_split': hp.choice('min_samples_split', range(2, 20))
}

def objective(params):
    model = GradientBoostingClassifier(random_state=42, **params)
    score = cross_val_score(model, X_train, y_train, cv=3,
                           scoring='accuracy', n_jobs=-1).mean()
    return {'loss': -score, 'status': STATUS_OK}

# Optimize
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest,
           max_evals=100, trials=trials)
```

### Optuna (Advanced Bayesian Optimization)

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.8, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=3,
                           scoring='accuracy', n_jobs=-1).mean()
    return score

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")
```

---

## 📊 Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, matthews_corrcoef
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Predictions
y_pred = model.predict(X_test)

# Core metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
```

### Custom Evaluation Functions

```python
def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """Comprehensive classification evaluation"""
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    if y_pred_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    return results

def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""
    results = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    return results
```

### Visualization Functions

```python
def plot_confusion_matrix(y_true, y_pred, classes=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_learning_curve(estimator, X, y, cv=None):
    """Plot learning curve"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r',
             label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='r')

    plt.plot(train_sizes, test_mean, 'o-', color='g',
             label='Validation score')
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color='g')

    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

---

## 🔧 Feature Engineering Techniques

### 1. Numerical Feature Engineering

```python
# Scaling and Normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust Scaling (outlier-resistant)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Log Transformation (for skewed distributions)
X_log = np.log(X + 1)  # +1 to handle zeros

# Power Transformation (Box-Cox)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

### 2. Categorical Feature Engineering

```python
# One-Hot Encoding
pd.get_dummies(data['category'], prefix='cat')

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['encoded'] = le.fit_transform(data['category'])

# Target Encoding (use with caution)
from category_encoders import TargetEncoder
encoder = TargetEncoder()
data['encoded'] = encoder.fit_transform(data['category'], data['target'])

# Binary Encoding
from category_encoders import BinaryEncoder
encoder = BinaryEncoder()
data_encoded = encoder.fit_transform(data['category'])
```

### 3. Time Feature Engineering

```python
# Extract time components
data['year'] = pd.to_datetime(data['date']).dt.year
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day
data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
data['hour'] = pd.to_datetime(data['date']).dt.hour

# Cyclical encoding for time features
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
```

### 4. Text Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(text_data)

# Count Vectorizer
count_vec = CountVectorizer(max_features=1000, stop_words='english')
X_count = count_vec.fit_transform(text_data)

# LSA (Latent Semantic Analysis)
svd = TruncatedSVD(n_components=50, random_state=42)
X_lsa = svd.fit_transform(X_tfidf)

# Text statistics
data['text_length'] = data['text'].str.len()
data['word_count'] = data['text'].str.split().str.len()
data['avg_word_length'] = data['text'].apply(lambda x:
    np.mean([len(word) for word in x.split()]) if x else 0)
```

### 5. Advanced Feature Engineering

```python
# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature Interactions
data['feature_interaction'] = data['feature1'] * data['feature2']

# Binning
data['age_bin'] = pd.cut(data['age'], bins=[0, 18, 35, 50, 100],
                        labels=['child', 'young', 'middle', 'senior'])

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Feature Importance from tree-based models
model = RandomForestClassifier()
model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### 6. Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X)

# t-SNE (for visualization)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP (alternative to t-SNE)
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
```

### 7. Feature Selection Methods

```python
# Univariate selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# LASSO regularization (L1)
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

# Feature importance
model = RandomForestClassifier()
model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 🚀 Quick Reference Commands

### Essential Imports

```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *

# Feature Engineering
from sklearn.feature_selection import *
from sklearn.decomposition import PCA

# Advanced
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
```

### Quick Data Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Quick Model Training

```python
# Classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Quick Evaluation

```python
# Classification
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Regression
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
```

---

## ⚡ Pro Tips

### 1. **Start Simple**

- Begin with simple models (Linear Regression, Logistic Regression)
- Use them as baselines before moving to complex models

### 2. **Understand Your Data**

- Always visualize distributions and relationships
- Check for data leakage and class imbalance

### 3. **Feature Engineering First**

- Good features often beat complex models
- Domain knowledge is crucial

### 4. **Cross-Validation is Key**

- Never evaluate on training data only
- Use appropriate CV strategy for your problem

### 5. **Monitor for Overfitting**

- Plot learning curves
- Use regularization when needed

### 6. **Ensemble for Better Results**

- Combine multiple models (Random Forest, XGBoost, etc.)
- Use stacking and blending techniques

### 7. **Document Everything**

- Track hyperparameters and their effects
- Reproduceable experiments are crucial

---

## 📚 Common Pitfalls to Avoid

❌ **Don't:**

- Evaluate on training data only
- Use accuracy for imbalanced datasets
- Ignore feature scaling for distance-based algorithms
- Use leakage in time series data
- Skip proper data splitting

✅ **Do:**

- Use stratified sampling for classification
- Choose appropriate metrics for your problem
- Scale features for algorithms sensitive to scale
- Respect temporal order in time series
- Cross-validate properly

---

_This cheat sheet provides a comprehensive reference for model selection in data science projects. Keep it handy for quick lookups and best practices!_
