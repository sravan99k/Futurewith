# AI/ML Fundamentals Cheat Sheet

## 🤖 Machine Learning Types Overview

### 1. Supervised Learning

**Definition**: Learning with labeled training data

- **Input**: Features (X) + Labels (y)
- **Goal**: Learn mapping function f: X → y
- **Examples**: Classification, Regression

**Common Algorithms**:

- **Linear/Logistic Regression**: Continuous/Categorical outputs
- **Decision Trees**: Interpretable, handles non-linear patterns
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machines for complex boundaries
- **Neural Networks**: Deep learning for complex patterns

**Python Implementation**:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

# Regression example
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification example
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### 2. Unsupervised Learning

**Definition**: Finding patterns in data without labels

- **Input**: Features (X) only
- **Goal**: Discover hidden structures

**Common Algorithms**:

- **K-Means Clustering**: Group similar data points
- **Hierarchical Clustering**: Tree-like clustering structure
- **PCA**: Dimensionality reduction
- **Association Rules**: Find relationships between variables

**Python Implementation**:

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Clustering example
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA example
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
```

### 3. Reinforcement Learning

**Definition**: Learning through interaction with environment

- **Components**: Agent, Environment, Actions, Rewards
- **Goal**: Maximize cumulative reward

**Common Algorithms**:

- **Q-Learning**: Value-based method
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combines value and policy methods

**Python Implementation**:

```python
import gym
import numpy as np

# Simple Q-Learning example structure
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.epsilon = 0.1  # exploration rate

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        # Q-Learning update rule
        self.q_table[state][action] += self.learning_rate * (
            reward + 0.99 * np.max(self.q_table[next_state]) -
            self.q_table[state][action]
        )
```

## 🔄 Algorithm Selection Flowchart

```
START: What type of problem?
│
├── Prediction Problem?
│   ├── YES → Supervised Learning
│   │   ├── Continuous target? → Regression
│   │   │   ├── Small dataset? → Linear Regression
│   │   │   ├── Complex patterns? → Random Forest/Neural Networks
│   │   └── Categorical target? → Classification
│   │       ├── Binary? → Logistic Regression/SVM
│   │       ├── Multiple classes? → Random Forest/Neural Networks
│   │       └── Interpretability needed? → Decision Trees
│   └── NO → Unsupervised Learning
│       ├── How many clusters? Known? → K-Means/Hierarchical
│       └── Reduce dimensions? → PCA/t-SNE
│
└── Sequential Decision Problem?
    └── YES → Reinforcement Learning
        ├── Discrete actions? → Q-Learning/DQN
        └── Continuous actions? → Policy Gradient/Actor-Critic
```

## 📊 Data Preparation Steps

### 1. Data Collection & Understanding

```python
import pandas as pd
import numpy as np

# Load and explore data
df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

### 2. Data Cleaning

```python
# Handle missing values
df['column'].fillna(df['column'].median(), inplace=True)  # Numerical
df['column'].fillna(df['column'].mode()[0], inplace=True)  # Categorical

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle outliers
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['column'] < Q1 - 1.5 * IQR) | (df['column'] > Q3 + 1.5 * IQR))]
```

### 3. Feature Engineering

```python
# Create new features
df['new_feature'] = df['feature1'] / df['feature2']
df['category_encoded'] = pd.get_dummies(df['category'], prefix='cat')

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Or Min-Max scaling
minmax_scaler = MinMaxScaler()
X_scaled = minmax_scaler.fit_transform(X)
```

### 4. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Tree-based feature importance
forest = ExtraTreesClassifier(n_estimators=100)
forest.fit(X, y)
feature_importance = forest.feature_importances_
```

## 📏 Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ROC-AUC (for binary classification)
from sklearn.metrics import roc_auc_score, roc_curve
auc_score = roc_auc_score(y_true, y_prob)
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Stratified K-Fold (for classification)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=skf)
```

## 🐍 Python Integration Patterns

### 1. Complete ML Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class ML_Pipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier()

    def preprocess_data(self, df):
        # Handle missing values
        df = df.fillna(df.median())

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = self.label_encoder.fit_transform(df[col])

        # Scale features
        X_scaled = self.scaler.fit_transform(df)

        return X_scaled

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        return self.model

    def predict(self, X_new):
        return self.model.predict(X_new)

# Usage
pipeline = ML_Pipeline()
X_processed = pipeline.preprocess_data(df)
pipeline.train(X_processed, y)
predictions = pipeline.predict(X_new)
```

### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### 3. Model Persistence

```python
import joblib

# Save model
joblib.dump(model, 'trained_model.pkl')

# Load model
loaded_model = joblib.load('trained_model.pkl')

# Use loaded model
predictions = loaded_model.predict(X_new)
```

## 🎯 Quick Reference Guide

### When to use each algorithm:

| Problem Type                   | Best Algorithms                  | When to Use                                  |
| ------------------------------ | -------------------------------- | -------------------------------------------- |
| **Binary Classification**      | Logistic Regression, SVM         | Linear separable, interpretable              |
| **Multi-class Classification** | Random Forest, Neural Networks   | Complex patterns, mixed data types           |
| **Regression**                 | Linear Regression, Random Forest | Continuous target, feature importance needed |
| **Clustering**                 | K-Means, DBSCAN                  | Customer segmentation, pattern discovery     |
| **Dimensionality Reduction**   | PCA, t-SNE                       | Visualization, feature compression           |
| **Time Series**                | ARIMA, LSTM                      | Sequential data, temporal patterns           |

### Common Python Libraries:

- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow, pytorch
- **Visualization**: matplotlib, seaborn, plotly
- **Model Interpretation**: shap, lime

### Performance Optimization Tips:

1. **Feature Scaling**: Always scale features for distance-based algorithms
2. **Cross-Validation**: Use to get reliable performance estimates
3. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV
4. **Feature Selection**: Remove irrelevant features to improve performance
5. **Ensemble Methods**: Combine multiple models for better results

### Debugging Checklist:

- [ ] Check for data leakage
- [ ] Verify feature engineering steps
- [ ] Validate train/test split
- [ ] Monitor for overfitting
- [ ] Check class imbalance
- [ ] Validate assumptions of chosen algorithm

---

_Last Updated: November 2025_
_Phase 4: AI/ML Fundamentals_
