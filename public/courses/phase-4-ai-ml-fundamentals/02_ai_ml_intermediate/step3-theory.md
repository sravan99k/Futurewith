---
title: "AI Tools, Libraries & Development Environment - Universal Guide"
description: "Your Complete AI Toolkit - From Zero to Building Cool Stuff!"
level: "Beginner-Intermediate"
time: "45 minutes"
prerequisites: "Python basics"
tags: ["AI", "Machine Learning", "Python", "Development Environment"]
---

# AI Tools, Libraries & Development Environment - Universal Guide

## Your Complete AI Toolkit - From Zero to Building Cool Stuff!

_Everything you need to start building AI projects - explained simply with real examples!_

---

## 🎯 How to Use This Guide

### 📚 **For Complete Beginners**

- Start with **"Why These Tools Matter"** - understand what you're getting
- Begin with **pandas & numpy** - the foundation for working with data
- Try **scikit-learn** - your first AI library (very beginner-friendly!)

### ⚡ **For Hands-On Learning**

- Follow **Installation Guides** - get everything set up step by step
- Use **Ready-to-Run Examples** - practice with working code
- Try **Beginner Projects** - build real things from day one

### 🚀 **For Skill Building**

- Progress from **Traditional ML** to **Deep Learning**
- Explore **Computer Vision** and **Natural Language Processing**
- Practice with **End-to-End Projects** - put it all together

### 🎯 **Tool Selection Guide**

- **🔨 Basic AI Projects** → Start with pandas + scikit-learn
- **🖼️ Image Recognition** → Add OpenCV + TensorFlow/PyTorch
- **📝 Text Analysis** → Add NLTK/spaCy + Hugging Face
- **🎨 Creative AI** → Add special libraries for art, music, etc.

### 📖 **Table of Contents**

#### **🚀 Getting Started**

1. [AI Toolkit Foundation - Why These Tools Matter](#introduction)
2. [pandas & numpy - Working with Data (Your First Tools)](#data-manipulation)
3. [matplotlib & seaborn - Making Pretty Graphs](#visualization)

#### **🧠 Core AI Libraries**

4. [scikit-learn - Machine Learning Made Simple](#scikit-learn)
5. [TensorFlow 2.x - Deep Learning Powerhouse](#tensorflow)
6. [PyTorch - Flexible Deep Learning](#pytorch)

#### **🎯 Specialized Tools**

7. [OpenCV - Teaching AI to See](#opencv)
8. [NLTK & spaCy - Teaching AI Language](#nlp-libraries)
9. [Hugging Face - Pre-trained AI Models](#hugging-face)

#### **🔧 Development Setup**

10. [Jupyter Notebooks - Interactive Coding](#jupyter)
11. [VS Code - Professional Code Editor](#vs-code)
12. [Environment Management - Keeping Things Organized](#environment-management)

#### **📚 Learning Path**

13. [Installation Guides - Step-by-Step Setup](#installation-guides)
14. [Development Workflows - How Professionals Work](#development-workflows)
15. [Best Practices & Troubleshooting - Avoid Common Mistakes](#best-practices)

---

## 📊 AI Libraries Quick Reference Table

| Library              | Purpose                   | Difficulty | Use Case                       | Installation                   |
| -------------------- | ------------------------- | ---------- | ------------------------------ | ------------------------------ |
| **pandas**           | Data manipulation         | ⭐         | Data analysis, cleaning        | `pip install pandas`           |
| **numpy**            | Numerical computing       | ⭐         | Mathematical operations        | `pip install numpy`            |
| **matplotlib**       | Basic plotting            | ⭐⭐       | Data visualization             | `pip install matplotlib`       |
| **seaborn**          | Statistical visualization | ⭐⭐       | Beautiful plots, statistics    | `pip install seaborn`          |
| **scikit-learn**     | Traditional ML            | ⭐⭐       | Classification, regression     | `pip install scikit-learn`     |
| **TensorFlow**       | Deep learning             | ⭐⭐⭐     | Neural networks, production    | `pip install tensorflow`       |
| **PyTorch**          | Deep learning             | ⭐⭐⭐     | Research, flexibility          | `pip install torch`            |
| **OpenCV**           | Computer vision           | ⭐⭐⭐     | Image/video processing         | `pip install opencv-python`    |
| **NLTK**             | NLP basics                | ⭐⭐       | Text processing, education     | `pip install nltk`             |
| **spaCy**            | Industrial NLP            | ⭐⭐⭐     | Production NLP pipelines       | `pip install spacy`            |
| **Hugging Face**     | Pre-trained models        | ⭐⭐⭐     | State-of-the-art models        | `pip install transformers`     |
| **Plotly**           | Interactive plots         | ⭐⭐       | Dashboards, web visualizations | `pip install plotly`           |
| **Streamlit**        | Web apps                  | ⭐⭐       | ML web applications            | `pip install streamlit`        |
| **Flask/FastAPI**    | Web APIs                  | ⭐⭐⭐     | Model deployment               | `pip install flask`            |
| **MLflow**           | ML lifecycle              | ⭐⭐       | Experiment tracking            | `pip install mlflow`           |
| **Jupyter**          | Interactive notebooks     | ⭐         | Development, exploration       | `pip install jupyter`          |
| **Weights & Biases** | Experiment tracking       | ⭐⭐       | Professional MLOps             | `pip install wandb`            |
| **Optuna**           | Hyperparameter tuning     | ⭐⭐⭐     | Automated optimization         | `pip install optuna`           |
| **XGBoost**          | Gradient boosting         | ⭐⭐       | Competition-winning models     | `pip install xgboost`          |
| **LightGBM**         | Gradient boosting         | ⭐⭐       | Fast gradient boosting         | `pip install lightgbm`         |
| **CatBoost**         | Categorical features      | ⭐⭐       | Handle categorical data well   | `pip install catboost`         |
| **SHAP**             | Model interpretability    | ⭐⭐⭐     | Explain predictions            | `pip install shap`             |
| **LIME**             | Model interpretability    | ⭐⭐       | Local explanations             | `pip install lime`             |
| **Yellowbrick**      | Visual ML tools           | ⭐         | Model diagnostics              | `pip install yellowbrick`      |
| **Imbalanced-learn** | Imbalanced data           | ⭐⭐       | Handle skewed datasets         | `pip install imbalanced-learn` |

### 🎯 Quick Start Commands

#### Essential Libraries (Level 1)

```bash
# Foundation packages for every AI project
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# For data analysis and basic ML
pip install plotly streamlit

# For development tools
pip install black flake8 ipython
```

#### Deep Learning Stack (Level 2)

```bash
# Core deep learning frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu

# For model deployment
pip install onnx onnxruntime
```

#### Computer Vision & NLP (Level 3)

```bash
# Computer vision
pip install opencv-python pillow

# Natural language processing
pip install nltk spacy
python -m spacy download en_core_web_sm

# Advanced NLP
pip install transformers datasets accelerate
```

#### MLOps & Production (Level 4)

```bash
# Experiment tracking
pip install mlflow wandb

# Model optimization
pip install optuna shap lime

# Web deployment
pip install flask fastapi uvicorn
pip install streamlit
```

#### One-Command Installation Options

**Minimal Setup (Beginner):**

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

**Data Science Setup:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly streamlit
```

**ML Research Setup:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly streamlit torch torchvision torchaudio tensorflow transformers datasets
```

**Full AI Stack Setup:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly streamlit torch torchvision torchaudio tensorflow transformers opencv-python nltk spacy flask fastapi mlflow wandb optuna shap xgboost lightgbm catboost
```

---

## 1. AI Toolkit Foundation - Why These Tools Matter 🧰

### **The Simple Answer**

Think of AI development like **building with LEGO blocks**:

- Each library is like a special set of LEGO pieces
- Some are basic bricks (pandas, numpy) - you use these everywhere
- Some are specialized pieces (OpenCV for images, NLTK for text)
- Some are complete pre-built structures (Hugging Face models)

**Instead of building everything from scratch, you get powerful, tested tools that let you focus on solving problems!**

### **Why These Tools Matter**

#### **✅ 1. Save Time and Energy**

- **Instead of:** Writing thousands of lines of code for basic math
- **With tools:** Just write `import pandas as pd` and you're ready!

#### **✅ 2. Learn from Experts**

- **Instead of:** Figuring out complicated algorithms yourself
- **With tools:** Use implementations created by PhD researchers and tested by millions

#### **✅ 3. Get Better Results**

- **Instead of:** Your own potentially buggy code
- **With tools:** Battle-tested libraries used by Google, Netflix, and NASA

#### **✅ 4. Join a Community**

- **Instead of:** Working alone
- **With tools:** Thousands of developers helping each other

### **Your AI Tool Hierarchy - Start Here!**

#### **🏗️ Foundation Layer (Use These First!)**

- **pandas** 📊 - Like Excel for Python, but much more powerful
- **numpy** 🔢 - Like a super-powered calculator for math
- **matplotlib** 📈 - Make beautiful graphs and charts

#### **🔨 Traditional AI Layer (Your First AI Tools)**

- **scikit-learn** 🤖 - Your first AI library, very beginner-friendly

#### **🧠 Deep Learning Layer (Advanced AI)**

- **TensorFlow** 🚀 - Google's deep learning powerhouse
- **PyTorch** 🔥 - Facebook's flexible deep learning framework

#### **🎯 Specialized Tools**

- **OpenCV** 👁️ - For working with images and videos
- **NLTK/spaCy** 📝 - For understanding human language
- **Hugging Face** 🤗 - For pre-built, ready-to-use AI models

### **Real-World Examples - What Each Tool Does**

#### **📊 pandas - Data Organizing Master**

- **What it does:** Organizes and analyzes data like a super-powered spreadsheet
- **Real use:** Netflix uses it to analyze viewing patterns
- **Your use:** Organize your school grades, track expenses, analyze survey data

#### **🤖 scikit-learn - First AI Library**

- **What it does:** Makes machine learning simple and accessible
- **Real use:** Spotify uses it for music recommendations
- **Your use:** Predict house prices, classify emails as spam/not spam

#### **🖼️ OpenCV - Computer Vision Expert**

- **What it does:** Helps computers "see" and understand images
- **Real use:** Facebook automatically tags your friends in photos
- **Your use:** Detect objects in photos, create face filters

#### **📝 NLTK - Language Expert**

- **What it does:** Helps computers understand human language
- **Real use:** Gmail automatically suggests replies
- **Your use:** Analyze reviews, translate text, summarize articles

### **Your Learning Journey - Step by Step**

#### **Week 1-2: Foundation** 🏗️

```
Python basics → pandas → matplotlib → First data project
```

#### **Week 3-4: Traditional AI** 🤖

```
scikit-learn → First ML project → Model evaluation
```

#### **Week 5-6: Deep Learning** 🧠

```
TensorFlow or PyTorch → Neural networks → Image/text projects
```

#### **Week 7-8: Specialization** 🎯

```
Choose your interest: Computer Vision OR NLP OR Other
```

#### **Week 9-12: Projects** 🚀

```
Build portfolio projects → Deploy online → Share with world!
```

### **Beginner Success Tips**

1. **🎯 Start Simple:** Don't try to learn everything at once
2. **👥 Use Communities:** Stack Overflow, Reddit, Discord groups
3. **🏗️ Build Projects:** Apply what you learn to real problems
4. **📚 Read Examples:** Study how others solve problems
5. **🔄 Practice Regularly:** 30 minutes daily is better than 5 hours once a week

---

## scikit-learn: Machine Learning Essentials {#scikit-learn}

### What is scikit-learn?

Think of scikit-learn as your "Swiss Army knife" for machine learning. It's like having a toolbox with the 20 most useful tools for data science - simple, reliable, and covers most jobs you'll encounter.

**Why Use scikit-learn:**

- **Simple API**: Consistent interface across all algorithms
- **Comprehensive**: Covers 95% of common ML tasks
- **Reliable**: Battle-tested in production for over a decade
- **Educational**: Perfect for learning ML concepts

**Where to Use scikit-learn:**

- Tabular data analysis
- Feature engineering and preprocessing
- Traditional ML algorithms (not deep learning)
- Model evaluation and comparison
- Production deployment for simple models

### Complete API Coverage with 100+ Examples

#### 1. Data Preprocessing & Feature Engineering

**Why Preprocessing Matters:**
Raw data is like raw ingredients in cooking - you need to clean, prepare, and season before you can make a great dish. Preprocessing transforms messy real-world data into algorithm-friendly format.

**When to Use:**

- Handling missing values
- Scaling features to same range
- Converting categorical to numerical
- Feature transformation

**How to Implement:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PowerTransformer, QuantileTransformer,
    PolynomialFeatures, SplineTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, SelectFromModel,
    VarianceThreshold, f_regression, mutual_info_regression
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Example 1: Handling Missing Values with Multiple Strategies
print("=== Example 1: Missing Value Imputation ===")
# Simulate data with missing values
data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'education': ['high_school', 'bachelor', np.nan, 'master', 'phd']
})

print("Original data with missing values:")
print(data)

# Strategy 1: Simple imputation (mean/median/mode)
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_mode = SimpleImputer(strategy='most_frequent')

data_mean = data.copy()
data_mean[['age', 'salary']] = imputer_mean.fit_transform(data[['age', 'salary']])
data_median = data.copy()
data_median[['age', 'salary']] = imputer_median.fit_transform(data[['age', 'salary']])
data_mode = data.copy()
data_mode[['education']] = imputer_mode.fit_transform(data[['education']])

print("\nImputed with mean (age, salary):")
print(data_mean)
print("\nImputed with mode (education):")
print(data_mode)

# Strategy 2: KNN Imputation (more sophisticated)
print("\n=== Example 2: KNN Imputation ===")
knn_imputer = KNNImputer(n_neighbors=2)
data_knn = pd.DataFrame(
    knn_imputer.fit_transform(data[['age', 'salary']]),
    columns=['age', 'salary']
)
print("KNN Imputed data (numerical features only):")
print(data_knn)

# Strategy 3: Iterative Imputation (most sophisticated)
print("\n=== Example 3: Iterative Imputation ===")
iterative_imputer = IterativeImputer(random_state=42)
data_iterative = pd.DataFrame(
    iterative_imputer.fit_transform(data[['age', 'salary']]),
    columns=['age', 'salary']
)
print("Iterative Imputed data:")
print(data_iterative)

# Example 4: Scaling and Normalization
print("\n=== Example 4: Feature Scaling ===")
# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.normal(100, 50, 1000),
    'feature2': np.random.exponential(2, 1000),
    'feature3': np.random.uniform(0, 1, 1000)
})

# Standard Scaling (z-score normalization)
scaler_standard = StandardScaler()
X_standard = pd.DataFrame(
    scaler_standard.fit_transform(X),
    columns=X.columns
)
print("Original features - mean and std:")
print(f"Feature1: mean={X['feature1'].mean():.2f}, std={X['feature1'].std():.2f}")
print(f"Feature2: mean={X['feature2'].mean():.2f}, std={X['feature2'].std():.2f}")

print("\nStandard scaled features - mean and std:")
print(f"Feature1: mean={X_standard['feature1'].mean():.2f}, std={X_standard['feature1'].std():.2f}")
print(f"Feature2: mean={X_standard['feature2'].mean():.2f}, std={X_standard['feature2'].std():.2f}")

# Min-Max Scaling (0-1 normalization)
scaler_minmax = MinMaxScaler()
X_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(X),
    columns=X.columns
)
print("\nMin-Max scaled features - min and max:")
print(f"Feature1: min={X_minmax['feature1'].min():.2f}, max={X_minmax['feature1'].max():.2f}")
print(f"Feature2: min={X_minmax['feature2'].min():.2f}, max={X_minmax['feature2'].max():.2f}")

# Robust Scaling (median and IQR)
scaler_robust = RobustScaler()
X_robust = pd.DataFrame(
    scaler_robust.fit_transform(X),
    columns=X.columns
)
print("\nRobust scaled features - median and IQR:")
print(f"Feature1: median={X_robust['feature1'].median():.2f}")
print(f"Feature2: median={X_robust['feature2'].median():.2f}")

# Example 5: Categorical Encoding
print("\n=== Example 5: Categorical Encoding ===")
# Create sample categorical data
categorical_data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
    'size': ['small', 'medium', 'large', 'large', 'small', 'medium'],
    'category': ['A', 'B', 'C', 'A', 'B', 'C'],
    'priority': ['high', 'medium', 'low', 'high', 'medium', 'low']
})

print("Original categorical data:")
print(categorical_data)

# Label Encoding (for ordinal data like size, priority)
label_encoder_size = LabelEncoder()
label_encoder_priority = LabelEncoder()
categorical_label = categorical_data.copy()
categorical_label['size'] = label_encoder_size.fit_transform(categorical_data['size'])
categorical_label['priority'] = label_encoder_priority.fit_transform(categorical_data['priority'])

print("\nLabel encoded (ordinal features):")
print(categorical_label)
print("Size mapping:", dict(zip(label_encoder_size.classes_, label_encoder_size.transform(label_encoder_size.classes_))))
print("Priority mapping:", dict(zip(label_encoder_priority.classes_, label_encoder_priority.transform(label_encoder_priority.classes_))))

# One-Hot Encoding (for nominal data like color, category)
categorical_onehot = categorical_data.copy()
categorical_onehot_encoded = pd.get_dummies(categorical_onehot, columns=['color', 'category'], prefix=['color', 'category'])
print("\nOne-hot encoded (nominal features):")
print(categorical_onehot_encoded)

# Ordinal Encoding (explicitly specify order for ordinal data)
size_order = ['small', 'medium', 'large']
priority_order = ['low', 'medium', 'high']
ordinal_encoder = OrdinalEncoder(categories=[size_order, priority_order])
categorical_ordinal = categorical_data.copy()
categorical_ordinal[['size', 'priority']] = ordinal_encoder.fit_transform(categorical_data[['size', 'priority']])

print("\nOrdinal encoded (explicit order):")
print(categorical_ordinal)
print("Size order:", size_order)
print("Priority order:", priority_order)

# Example 6: Feature Transformation
print("\n=== Example 6: Feature Transformation ===")
# Power Transformation (Box-Cox, Yeo-Johnson)
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X_power = pd.DataFrame(
    power_transformer.fit_transform(X),
    columns=X.columns
)

print("Original distribution (skewed data):")
print(f"Feature1 skewness: {X['feature1'].skew():.2f}")
print(f"Feature2 skewness: {X['feature2'].skew():.2f}")

print("\nAfter power transformation (closer to normal):")
print(f"Feature1 skewness: {X_power['feature1'].skew():.2f}")
print(f"Feature2 skewness: {X_power['feature2'].skew():.2f}")

# Example 7: Polynomial Features
print("\n=== Example 7: Polynomial Features ===")
# Create simple 2D data
X_simple = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10]
})

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pd.DataFrame(
    poly.fit_transform(X_simple),
    columns=poly.get_feature_names_out(['x1', 'x2'])
)
print("Original features:")
print(X_simple)
print("\nPolynomial features (degree 2):")
print(X_poly)

# Example 8: Feature Selection
print("\n=== Example 8: Feature Selection ===")
# Create sample dataset with target
np.random.seed(42)
n_samples, n_features = 500, 10
X_features = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
# Make only first 3 features relevant
y_target = (3 * X_features['feature_0'] + 2 * X_features['feature_1'] - 1 * X_features['feature_2'] +
           0.1 * np.random.randn(n_samples))

# Univariate feature selection
selector_univariate = SelectKBest(score_func=f_regression, k=5)
X_selected_univariate = selector_univariate.fit_transform(X_features, y_target)
selected_features_univariate = X_features.columns[selector_univariate.get_support()]

print("Univariate selection - selected features:")
print(f"Features: {list(selected_features_univariate)}")
print(f"Scores: {dict(zip(selected_features_univariate, selector_univariate.scores_[selector_univariate.get_support()]))}")

# Feature selection using model
from sklearn.ensemble import RandomForestRegressor
selector_model = SelectFromModel(RandomForestRegressor(n_estimators=50))
X_selected_model = selector_model.fit_transform(X_features, y_target)
selected_features_model = X_features.columns[selector_model.get_support()]

print("\nModel-based selection - selected features:")
print(f"Features: {list(selected_features_model)}")

# Example 9: Dimensionality Reduction
print("\n=== Example 9: Dimensionality Reduction ===")
# PCA for visualization and compression
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)

print(f"Original features: {n_features}")
print(f"PCA components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Example 10: Complete Pipeline
print("\n=== Example 10: Complete Preprocessing Pipeline ===")
# Create pipeline for numerical and categorical features
numerical_features = ['age', 'salary', 'experience']
categorical_features = ['education', 'department']

# Define transformers for each feature type
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply to sample data
sample_data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'experience': [1, 5, 3, 10, 15],
    'education': ['high_school', 'bachelor', np.nan, 'master', 'phd'],
    'department': ['IT', 'Sales', 'IT', 'HR', 'Finance']
})

print("Sample data before preprocessing:")
print(sample_data)

# Fit and transform
X_preprocessed = preprocessor.fit_transform(sample_data)
print(f"\nPreprocessed shape: {X_preprocessed.shape}")
print("Preprocessing completed successfully!")
```

#### 2. Supervised Learning Algorithms

**Why Supervised Learning:**
Think of supervised learning like teaching a child to recognize animals. You show them pictures (input features) with labels (output) like "this is a dog" or "this is a cat." Eventually, they learn to identify new pictures without labels.

**When to Use Supervised Learning:**

- Classification: Predicting categories (spam/not spam, disease/no disease)
- Regression: Predicting continuous values (house prices, stock prices)
- Both require labeled training data

**How to Implement All Major Algorithms:**

```python
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier, Perceptron
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score

print("=== Supervised Learning: Complete Algorithm Coverage ===")

# Generate sample datasets
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler

# Classification dataset
X_class, y_class = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

# Regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=0.1, random_state=42
)

# Split data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("=== Example 11: Linear Models ===")

# Linear Regression for regression tasks
print("\n--- Linear Regression ---")
lr_reg = LinearRegression()
lr_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lr_reg.predict(X_test_reg)
r2_reg = r2_score(y_test_reg, y_pred_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Linear Regression - R² Score: {r2_reg:.4f}, MSE: {mse_reg:.4f}")

# Logistic Regression for classification
print("\n--- Logistic Regression ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_class)
X_test_scaled = scaler.transform(X_test_class)

lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_train_scaled, y_train_class)
y_pred_clf = lr_clf.predict(X_test_scaled)
accuracy_clf = accuracy_score(y_test_class, y_pred_clf)
print(f"Logistic Regression - Accuracy: {accuracy_clf:.4f}")

# Ridge Regression (L2 regularization)
print("\n--- Ridge Regression ---")
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_reg.predict(X_test_reg)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)
print(f"Ridge Regression - R² Score: {r2_ridge:.4f}")

# Lasso Regression (L1 regularization)
print("\n--- Lasso Regression ---")
lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso_reg.predict(X_test_reg)
r2_lasso = r2_score(y_test_reg, y_pred_lasso)
print(f"Lasso Regression - R² Score: {r2_lasso:.4f}")

# ElasticNet (L1 + L2 regularization)
print("\n--- ElasticNet Regression ---")
elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_reg.fit(X_train_reg, y_train_reg)
y_pred_elastic = elastic_reg.predict(X_test_reg)
r2_elastic = r2_score(y_test_reg, y_pred_elastic)
print(f"ElasticNet Regression - R² Score: {r2_elastic:.4f}")

# Stochastic Gradient Descent
print("\n--- Stochastic Gradient Descent ---")
sgd_reg = SGDRegressor(random_state=42)
sgd_reg.fit(X_train_reg, y_train_reg)
y_pred_sgd_reg = sgd_reg.predict(X_test_reg)
r2_sgd_reg = r2_score(y_test_reg, y_pred_sgd_reg)
print(f"SGD Regression - R² Score: {r2_sgd_reg:.4f}")

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_scaled, y_train_class)
y_pred_sgd_clf = sgd_clf.predict(X_test_scaled)
accuracy_sgd_clf = accuracy_score(y_test_class, y_pred_sgd_clf)
print(f"SGD Classification - Accuracy: {accuracy_sgd_clf:.4f}")

print("\n=== Example 12: Tree-Based Models ===")

# Decision Tree
print("\n--- Decision Tree Classifier ---")
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_clf.fit(X_train_class, y_train_class)
y_pred_dt = dt_clf.predict(X_test_class)
accuracy_dt = accuracy_score(y_test_class, y_pred_dt)
print(f"Decision Tree Classifier - Accuracy: {accuracy_dt:.4f}")

print("\n--- Decision Tree Regressor ---")
dt_reg = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_reg.fit(X_train_reg, y_train_reg)
y_pred_dt_reg = dt_reg.predict(X_test_reg)
r2_dt_reg = r2_score(y_test_reg, y_pred_dt_reg)
print(f"Decision Tree Regressor - R² Score: {r2_dt_reg:.4f}")

# Random Forest (Ensemble of decision trees)
print("\n--- Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_class, y_train_class)
y_pred_rf = rf_clf.predict(X_test_class)
accuracy_rf = accuracy_score(y_test_class, y_pred_rf)
print(f"Random Forest Classifier - Accuracy: {accuracy_rf:.4f}")

print("\n--- Random Forest Regressor ---")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)
r2_rf_reg = r2_score(y_test_reg, y_pred_rf_reg)
print(f"Random Forest Regressor - R² Score: {r2_rf_reg:.4f}")

# Extra Trees (Extremely Randomized Trees)
print("\n--- Extra Trees Classifier ---")
et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_clf.fit(X_train_class, y_train_class)
y_pred_et = et_clf.predict(X_test_class)
accuracy_et = accuracy_score(y_test_class, y_pred_et)
print(f"Extra Trees Classifier - Accuracy: {accuracy_et:.4f}")

print("\n--- Extra Trees Regressor ---")
et_reg = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_reg.fit(X_train_reg, y_train_reg)
y_pred_et_reg = et_reg.predict(X_test_reg)
r2_et_reg = r2_score(y_test_reg, y_pred_et_reg)
print(f"Extra Trees Regressor - R² Score: {r2_et_reg:.4f}")

# Gradient Boosting
print("\n--- Gradient Boosting Classifier ---")
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train_class, y_train_class)
y_pred_gb = gb_clf.predict(X_test_class)
accuracy_gb = accuracy_score(y_test_class, y_pred_gb)
print(f"Gradient Boosting Classifier - Accuracy: {accuracy_gb:.4f}")

print("\n--- Gradient Boosting Regressor ---")
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train_reg, y_train_reg)
y_pred_gb_reg = gb_reg.predict(X_test_reg)
r2_gb_reg = r2_score(y_test_reg, y_pred_gb_reg)
print(f"Gradient Boosting Regressor - R² Score: {r2_gb_reg:.4f}")

# Histogram Gradient Boosting (faster for large datasets)
print("\n--- Histogram Gradient Boosting Classifier ---")
hgb_clf = HistGradientBoostingClassifier(random_state=42)
hgb_clf.fit(X_train_class, y_train_class)
y_pred_hgb = hgb_clf.predict(X_test_class)
accuracy_hgb = accuracy_score(y_test_class, y_pred_hgb)
print(f"Hist Gradient Boosting Classifier - Accuracy: {accuracy_hgb:.4f}")

print("\n--- Histogram Gradient Boosting Regressor ---")
hgb_reg = HistGradientBoostingRegressor(random_state=42)
hgb_reg.fit(X_train_reg, y_train_reg)
y_pred_hgb_reg = hgb_reg.predict(X_test_reg)
r2_hgb_reg = r2_score(y_test_reg, y_pred_hgb_reg)
print(f"Hist Gradient Boosting Regressor - R² Score: {r2_hgb_reg:.4f}")

# AdaBoost
print("\n--- AdaBoost Classifier ---")
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train_class, y_train_class)
y_pred_ada = ada_clf.predict(X_test_class)
accuracy_ada = accuracy_score(y_test_class, y_pred_ada)
print(f"AdaBoost Classifier - Accuracy: {accuracy_ada:.4f}")

print("\n--- AdaBoost Regressor ---")
ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_reg.fit(X_train_reg, y_train_reg)
y_pred_ada_reg = ada_reg.predict(X_test_reg)
r2_ada_reg = r2_score(y_test_reg, y_pred_ada_reg)
print(f"AdaBoost Regressor - R² Score: {r2_ada_reg:.4f}")

print("\n=== Example 13: Support Vector Machines ===")

# SVM Classification
print("\n--- SVM Classifier (RBF) ---")
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_scaled, y_train_class)
y_pred_svm = svm_clf.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test_class, y_pred_svm)
print(f"SVM Classifier - Accuracy: {accuracy_svm:.4f}")

# Linear SVM (faster for high-dimensional data)
print("\n--- Linear SVM Classifier ---")
linear_svm_clf = LinearSVC(random_state=42, max_iter=2000)
linear_svm_clf.fit(X_train_scaled, y_train_class)
y_pred_linear_svm = linear_svm_clf.predict(X_test_scaled)
accuracy_linear_svm = accuracy_score(y_test_class, y_pred_linear_svm)
print(f"Linear SVM Classifier - Accuracy: {accuracy_linear_svm:.4f}")

# SVM Regression
print("\n--- SVM Regressor (RBF) ---")
svm_reg = SVR(kernel='rbf')
svm_reg.fit(X_train_reg, y_train_reg)
y_pred_svm_reg = svm_reg.predict(X_test_reg)
r2_svm_reg = r2_score(y_test_reg, y_pred_svm_reg)
print(f"SVM Regressor - R² Score: {r2_svm_reg:.4f}")

# Linear SVM Regression
print("\n--- Linear SVM Regressor ---")
linear_svm_reg = LinearSVR(random_state=42, max_iter=2000)
linear_svm_reg.fit(X_train_reg, y_train_reg)
y_pred_linear_svm_reg = linear_svm_reg.predict(X_test_reg)
r2_linear_svm_reg = r2_score(y_test_reg, y_pred_linear_svm_reg)
print(f"Linear SVM Regressor - R² Score: {r2_linear_svm_reg:.4f}")

print("\n=== Example 14: Nearest Neighbors ===")

# K-Nearest Neighbors Classification
print("\n--- KNN Classifier ---")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train_class)
y_pred_knn = knn_clf.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test_class, y_pred_knn)
print(f"KNN Classifier - Accuracy: {accuracy_knn:.4f}")

# K-Nearest Neighbors Regression
print("\n--- KNN Regressor ---")
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)
y_pred_knn_reg = knn_reg.predict(X_test_reg)
r2_knn_reg = r2_score(y_test_reg, y_pred_knn_reg)
print(f"KNN Regressor - R² Score: {r2_knn_reg:.4f}")

print("\n=== Example 15: Naive Bayes ===")

# Gaussian Naive Bayes (continuous features)
print("\n--- Gaussian Naive Bayes ---")
nb_clf = GaussianNB()
nb_clf.fit(X_train_scaled, y_train_class)
y_pred_nb = nb_clf.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test_class, y_pred_nb)
print(f"Gaussian Naive Bayes - Accuracy: {accuracy_nb:.4f}")

# Multinomial Naive Bayes (discrete features)
print("\n--- Multinomial Naive Bayes ---")
# Convert features to non-negative for Multinomial NB
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train_scaled)
X_test_minmax = minmax_scaler.transform(X_test_scaled)

multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_minmax, y_train_class)
y_pred_multinomial_nb = multinomial_nb.predict(X_test_minmax)
accuracy_multinomial_nb = accuracy_score(y_test_class, y_pred_multinomial_nb)
print(f"Multinomial Naive Bayes - Accuracy: {accuracy_multinomial_nb:.4f}")

# Bernoulli Naive Bayes (binary features)
print("\n--- Bernoulli Naive Bayes ---")
# Convert to binary features
X_train_binary = (X_train_minmax > 0.5).astype(int)
X_test_binary = (X_test_minmax > 0.5).astype(int)

bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train_binary, y_train_class)
y_pred_bernoulli_nb = bernoulli_nb.predict(X_test_binary)
accuracy_bernoulli_nb = accuracy_score(y_test_class, y_pred_bernoulli_nb)
print(f"Bernoulli Naive Bayes - Accuracy: {accuracy_bernoulli_nb:.4f}")

print("\n=== Example 16: Neural Networks (MLP) ===")

# Multi-layer Perceptron Classifier
print("\n--- MLP Classifier ---")
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
mlp_clf.fit(X_train_scaled, y_train_class)
y_pred_mlp = mlp_clf.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test_class, y_pred_mlp)
print(f"MLP Classifier - Accuracy: {accuracy_mlp:.4f}")

# Multi-layer Perceptron Regressor
print("\n--- MLP Regressor ---")
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
mlp_reg.fit(X_train_reg, y_train_reg)
y_pred_mlp_reg = mlp_reg.predict(X_test_reg)
r2_mlp_reg = r2_score(y_test_reg, y_pred_mlp_reg)
print(f"MLP Regressor - R² Score: {r2_mlp_reg:.4f}")

print("\n=== Example 17: Model Comparison and Selection ===")

# Compare all classifiers
print("\n--- Classifier Comparison ---")
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

classifier_results = {}
for name, clf in classifiers.items():
    if name in ['Logistic Regression', 'SVM', 'KNN', 'Naive Bayes', 'MLP']:
        clf.fit(X_train_scaled, y_train_class)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train_class, y_train_class)
        y_pred = clf.predict(X_test_class)

    accuracy = accuracy_score(y_test_class, y_pred)
    classifier_results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")

# Find best classifier
best_classifier = max(classifier_results, key=classifier_results.get)
print(f"\nBest Classifier: {best_classifier} with accuracy {classifier_results[best_classifier]:.4f}")

# Compare all regressors
print("\n--- Regressor Comparison ---")
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

regressor_results = {}
for name, reg in regressors.items():
    reg.fit(X_train_reg, y_train_reg)
    y_pred = reg.predict(X_test_reg)
    r2 = r2_score(y_test_reg, y_pred)
    regressor_results[name] = r2
    print(f"{name}: {r2:.4f}")

# Find best regressor
best_regressor = max(regressor_results, key=regressor_results.get)
print(f"\nBest Regressor: {best_regressor} with R² {regressor_results[best_regressor]:.4f}")

print("\n=== Example 18: Cross-Validation ===")

# Cross-validation for model selection
print("\n--- Cross-Validation Scores ---")
models_for_cv = {
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

for name, model in models_for_cv.items():
    if name == 'Logistic Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train_class, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train_class, y_train_class, cv=5, scoring='accuracy')

    print(f"{name}:")
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n=== Example 19: Hyperparameter Tuning ===")

# Grid Search for Random Forest
print("\n--- Grid Search for Random Forest ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_class, y_train_class)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test best model
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_class)
accuracy_best_rf = accuracy_score(y_test_class, y_pred_best_rf)
print(f"Test accuracy with best parameters: {accuracy_best_rf:.4f}")

# Randomized Search for large parameter spaces
print("\n--- Randomized Search for Gradient Boosting ---")
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3)
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_class, y_train_class)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

print("\nSupervised Learning coverage completed!")
```

#### 3. Unsupervised Learning Algorithms

**Why Unsupervised Learning:**
Imagine you're given a box of puzzle pieces without the picture on the box. You have to figure out how they connect based only on their shapes and colors. Unsupervised learning finds hidden patterns in data without labels.

**When to Use Unsupervised Learning:**

- Clustering: Group similar data points (customer segmentation, gene sequencing)
- Dimensionality Reduction: Simplify complex data while preserving important information
- Anomaly Detection: Find unusual patterns (fraud detection, network security)
- Association Rules: Discover relationships between variables

**How to Implement All Major Algorithms:**

```python
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering, MeanShift, Birch,
    AffinityPropagation, OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import (
    PCA, TruncatedSVD, FastICA, NMF,
    LatentDirichletAllocation
)
from sklearn.manifold import (
    TSNE, Isomap, LocallyLinearEmbedding,
    SpectralEmbedding
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    homogeneity_completeness_v_measure
)
from sklearn.preprocessing import StandardScaler

print("=== Unsupervised Learning: Complete Algorithm Coverage ===")

# Generate sample datasets for clustering
np.random.seed(42)

# Dataset 1: Blobs (good for k-means)
X_blobs, y_blobs = make_blobs(
    n_samples=300, centers=4, cluster_std=1.0,
    random_state=42
)

# Dataset 2: Circles (good for non-convex clustering)
from sklearn.datasets import make_circles
X_circles, _ = make_circles(n_samples=300, noise=0.1, factor=0.2, random_state=42)

# Dataset 3: Moons (another non-convex case)
from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

print("\n=== Example 20: K-Means Clustering ===")

# Standard K-Means
print("\n--- K-Means Clustering ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_blobs)
silhouette_kmeans = silhouette_score(X_blobs, clusters_kmeans)
print(f"K-Means Silhouette Score: {silhouette_kmeans:.4f}")
print(f"K-Means Centers:\n{kmeans.cluster_centers_}")

# K-Means with different initialization methods
print("\n--- K-Means++ vs Random Initialization ---")
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
clusters_plus = kmeans_plus.fit_predict(X_blobs)
silhouette_plus = silhouette_score(X_blobs, clusters_plus)

kmeans_random = KMeans(n_clusters=4, init='random', random_state=42, n_init=10)
clusters_random = kmeans_random.fit_predict(X_blobs)
silhouette_random = silhouette_score(X_blobs, clusters_random)

print(f"K-Means++ Silhouette Score: {silhouette_plus:.4f}")
print(f"Random Init Silhouette Score: {silhouette_random:.4f}")

# Elbow method to find optimal number of clusters
print("\n--- Elbow Method for Optimal K ---")
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_elbow.fit(X_blobs)
    inertias.append(kmeans_elbow.inertia_)

# Find elbow (simplified)
deltas = np.diff(inertias)
second_deltas = np.diff(deltas)
elbow_k = np.argmax(second_deltas) + 2  # +2 because we start from k=1 and took second diff
print(f"Suggested number of clusters (elbow method): {elbow_k}")

print("\n=== Example 21: Hierarchical Clustering ===")

# Agglomerative Clustering
print("\n--- Agglomerative Clustering ---")
# Different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
for linkage in linkage_methods:
    agg_clustering = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    clusters_agg = agg_clustering.fit_predict(X_blobs)
    silhouette_agg = silhouette_score(X_blobs, clusters_agg)
    print(f"Agglomerative ({linkage}) Silhouette Score: {silhouette_agg:.4f}")

# Different affinity metrics
print("\n--- Different Affinity Metrics ---")
for affinity in ['euclidean', 'manhattan', 'cosine']:
    agg_clustering = AgglomerativeClustering(
        n_clusters=4, linkage='complete', affinity=affinity
    )
    clusters_agg = agg_clustering.fit_predict(X_blobs)
    silhouette_agg = silhouette_score(X_blobs, clusters_agg)
    print(f"Agglomerative ({affinity}) Silhouette Score: {silhouette_agg:.4f}")

print("\n=== Example 22: DBSCAN (Density-Based) ===")

# DBSCAN - automatically determines number of clusters
print("\n--- DBSCAN Clustering ---")
# Parameters to tune: eps (neighborhood radius) and min_samples
eps_values = [0.3, 0.5, 0.7, 1.0]
min_samples_values = [3, 5, 10]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters_dbscan = dbscan.fit_predict(X_blobs)

        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
        n_noise = list(clusters_dbscan).count(-1)

        if n_clusters > 1:  # Only calculate silhouette if we have multiple clusters
            silhouette_dbscan = silhouette_score(X_blobs, clusters_dbscan)
            print(f"DBSCAN (eps={eps}, min_samples={min_samples}): "
                  f"{n_clusters} clusters, {n_noise} noise points, "
                  f"silhouette={silhouette_dbscan:.4f}")
        else:
            print(f"DBSCAN (eps={eps}, min_samples={min_samples}): "
                  f"{n_clusters} clusters, {n_noise} noise points")

print("\n=== Example 23: Spectral Clustering ===")

# Spectral Clustering - good for non-convex clusters
print("\n--- Spectral Clustering ---")
# Test on different datasets
datasets_spectral = {
    'Blobs': X_blobs,
    'Circles': X_circles,
    'Moons': X_moons
}

for dataset_name, X_data in datasets_spectral.items():
    spectral = SpectralClustering(n_clusters=2, random_state=42)
    clusters_spectral = spectral.fit_predict(X_data)
    silhouette_spectral = silhouette_score(X_data, clusters_spectral)
    print(f"Spectral Clustering ({dataset_name}) Silhouette Score: {silhouette_spectral:.4f}")

print("\n=== Example 24: Gaussian Mixture Models ===")

# GMM - probabilistic clustering
print("\n--- Gaussian Mixture Models ---")
# Different covariance types
covariance_types = ['full', 'tied', 'diag', 'spherical']
for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=4, covariance_type=cov_type, random_state=42)
    clusters_gmm = gmm.fit_predict(X_blobs)
    silhouette_gmm = silhouette_score(X_blobs, clusters_gmm)
    print(f"GMM ({cov_type}) Silhouette Score: {silhouette_gmm:.4f}")

# Model selection with BIC/AIC
print("\n--- GMM Model Selection ---")
n_components_range = range(1, 11)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm_model = GaussianMixture(n_components=n_components, random_state=42)
    gmm_model.fit(X_blobs)
    bic_scores.append(gmm_model.bic(X_blobs))
    aic_scores.append(gmm_model.aic(X_blobs))

# Find optimal number of components
optimal_n_components_bic = n_components_range[np.argmin(bic_scores)]
optimal_n_components_aic = n_components_range[np.argmin(aic_scores)]

print(f"Optimal n_components (BIC): {optimal_n_components_bic}")
print(f"Optimal n_components (AIC): {optimal_n_components_aic}")

print("\n=== Example 25: Other Clustering Algorithms ===")

# Mean Shift
print("\n--- Mean Shift Clustering ---")
mean_shift = MeanShift()
clusters_meanshift = mean_shift.fit_predict(X_blobs)
n_clusters_meanshift = len(np.unique(clusters_meanshift))
print(f"Mean Shift found {n_clusters_meanshift} clusters")

# Birch
print("\n--- Birch Clustering ---")
birch = Birch(n_clusters=4)
clusters_birch = birch.fit_predict(X_blobs)
silhouette_birch = silhouette_score(X_blobs, clusters_birch)
print(f"Birch Silhouette Score: {silhouette_birch:.4f}")

# OPTICS
print("\n--- OPTICS Clustering ---")
optics = OPTICS(min_samples=5)
clusters_optics = optics.fit_predict(X_blobs)
n_clusters_optics = len(set(clusters_optics)) - (1 if -1 in clusters_optics else 0)
print(f"OPTICS found {n_clusters_optics} clusters")

print("\n=== Example 26: Dimensionality Reduction ===")

# Generate high-dimensional data
np.random.seed(42)
X_high_dim = np.random.randn(500, 100)

# PCA (Principal Component Analysis)
print("\n--- PCA (Principal Component Analysis) ---")
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_high_dim)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"First 10 components explain {cumulative_variance[-1]:.2%} of variance")
print(f"First component explains {explained_variance_ratio[0]:.2%}")
print(f"Second component explains {explained_variance_ratio[1]:.2%}")

# Find number of components for 95% variance
pca_full = PCA()
pca_full.fit(X_high_dim)
n_components_95 = np.argmax(pca_full.explained_variance_ratio_.cumsum() >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# SVD (Singular Value Decomposition)
print("\n--- Truncated SVD ---")
svd = TruncatedSVD(n_components=10, random_state=42)
X_svd = svd.fit_transform(X_high_dim)
print(f"SVD explained variance ratio: {svd.explained_variance_ratio_}")

# ICA (Independent Component Analysis)
print("\n--- Fast ICA ---")
ica = FastICA(n_components=10, random_state=42)
X_ica = ica.fit_transform(X_high_dim)
print(f"ICA completed - shape: {X_ica.shape}")

# NMF (Non-negative Matrix Factorization)
print("\n--- NMF (Non-negative Matrix Factorization) ---")
# Make data non-negative
X_positive = np.abs(X_high_dim)
nmf = NMF(n_components=10, random_state=42)
X_nmf = nmf.fit_transform(X_positive)
reconstruction_error = nmf.reconstruction_err_
print(f"NMF reconstruction error: {reconstruction_error:.4f}")

# LDA (Latent Dirichlet Allocation) for text data
print("\n--- LDA (Latent Dirichlet Allocation) ---")
from sklearn.datasets import make_classification
X_lda, _ = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda_transformed = lda.fit_transform(X_lda)
print(f"LDA completed - components shape: {X_lda_transformed.shape}")

print("\n=== Example 27: Manifold Learning ===")

# t-SNE (t-Distributed Stochastic Neighbor Embedding)
print("\n--- t-SNE Manifold Learning ---")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_high_dim[:, :50])  # Use first 50 features for speed
print(f"t-SNE completed - embedded shape: {X_tsne.shape}")

# Isomap
print("\n--- Isomap Manifold Learning ---")
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X_high_dim[:, :20])  # Use first 20 features
print(f"Isomap completed - embedded shape: {X_isomap.shape}")

# Locally Linear Embedding (LLE)
print("\n--- Locally Linear Embedding ---")
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X_high_dim[:, :20])  # Use first 20 features
print(f"LLE completed - embedded shape: {X_lle.shape}")

# Spectral Embedding
print("\n--- Spectral Embedding ---")
spectral_emb = SpectralEmbedding(n_components=2, random_state=42)
X_spectral = spectral_emb.fit_transform(X_high_dim[:, :20])  # Use first 20 features
print(f"Spectral embedding completed - shape: {X_spectral.shape}")

print("\n=== Example 28: Anomaly Detection ===")

# Generate data with anomalies
X_normal = np.random.randn(200, 2)
X_anomaly = np.random.uniform(low=-6, high=6, size=(20, 2))  # Clear outliers
X_anomaly_data = np.vstack([X_normal, X_anomaly])

print("\n--- Isolation Forest ---")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_anomaly_data)
n_anomalies = sum(anomaly_labels == -1)
print(f"Isolation Forest detected {n_anomalies} anomalies")

# Local Outlier Factor
print("\n--- Local Outlier Factor ---")
lof = LocalOutlierFactor(contamination=0.1)
lof_labels = lof.fit_predict(X_anomaly_data)
lof_anomalies = sum(lof_labels == -1)
print(f"LOF detected {lof_anomalies} anomalies")

# Elliptic Envelope ( assumes Gaussian distribution )
print("\n--- Elliptic Envelope ---")
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
elliptic_labels = elliptic.fit_predict(X_anomaly_data)
elliptic_anomalies = sum(elliptic_labels == -1)
print(f"Elliptic Envelope detected {elliptic_anomalies} anomalies")

print("\n=== Example 29: Clustering Evaluation ===")

# Generate ground truth labels for evaluation
y_true = np.concatenate([np.zeros(200), np.ones(20)])  # Normal=0, Anomaly=1

# Convert clustering results to anomaly labels
iso_labels = (anomaly_labels == -1).astype(int)  # -1 -> 1 (anomaly), 1 -> 0 (normal)

# Calculate metrics
ari_iso = adjusted_rand_score(y_true, iso_labels)
ari_lof = adjusted_rand_score(y_true, lof_labels)
ari_elliptic = adjusted_rand_score(y_true, elliptic_labels)

print(f"\nClustering Evaluation (Adjusted Rand Index):")
print(f"Isolation Forest: {ari_iso:.4f}")
print(f"LOF: {ari_lof:.4f}")
print(f"Elliptic Envelope: {ari_elliptic:.4f}")

print("\n=== Example 30: Pipeline Examples ===")

# Complete clustering pipeline
print("\n--- Complete Clustering Pipeline ---")

# Pipeline 1: Data preprocessing + K-Means
from sklearn.pipeline import Pipeline

clustering_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

# Fit and predict
cluster_labels = clustering_pipeline.fit_predict(X_blobs)
silhouette_pipeline = silhouette_score(X_blobs, cluster_labels)
print(f"Pipeline silhouette score: {silhouette_pipeline:.4f}")

# Pipeline 2: Preprocessing + PCA + GMM
gmm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('gmm', GaussianMixture(n_components=4, random_state=42))
])

gmm_labels = gmm_pipeline.fit_predict(X_blobs)
silhouette_gmm_pipeline = silhouette_score(X_blobs, gmm_labels)
print(f"GMM Pipeline silhouette score: {silhouette_gmm_pipeline:.4f}")

print("\nUnsupervised Learning coverage completed!")
```

#### 4. Model Evaluation & Metrics

**Why Evaluation Matters:**
Training a model without proper evaluation is like building a car without testing it - you have no idea if it works safely. Evaluation metrics tell you how well your model performs and help you choose the best one.

**When to Use Different Metrics:**

- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Regression**: MSE, MAE, R², explained variance
- **Clustering**: Silhouette score, adjusted rand index
- **Cross-validation**: K-fold, stratified, time series split

**How to Implement Comprehensive Evaluation:**

```python
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef,

    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error,

    # Clustering metrics
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure, calinski_harabasz_score,

    # Model validation
    cross_val_score, cross_validate, validation_curve,
    learning_curve, permutation_test_score,

    # Cross-validation splitters
    KFold, StratifiedKFold, GroupKFold, ShuffleSplit,
    TimeSeriesSplit, StratifiedShuffleSplit
)

print("=== Model Evaluation & Metrics: Complete Coverage ===")

# Create datasets for demonstration
np.random.seed(42)
X_eval, y_eval = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
)

print("\n=== Example 31: Classification Metrics ===")

# Train a classifier for evaluation
rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
rf_eval.fit(X_train_eval, y_train_eval)
y_pred_eval = rf_eval.predict(X_test_eval)
y_pred_proba_eval = rf_eval.predict_proba(X_test_eval)[:, 1]

print("\n--- Basic Classification Metrics ---")
accuracy = accuracy_score(y_test_eval, y_pred_eval)
precision = precision_score(y_test_eval, y_pred_eval)
recall = recall_score(y_test_eval, y_pred_eval)
f1 = f1_score(y_test_eval, y_pred_eval)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Classification Report (comprehensive metrics)
print("\n--- Classification Report ---")
class_report = classification_report(y_test_eval, y_pred_eval)
print(class_report)

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test_eval, y_pred_eval)
print("Confusion Matrix:")
print(cm)

# ROC-AUC Score
print("\n--- ROC-AUC Analysis ---")
roc_auc = roc_auc_score(y_test_eval, y_pred_proba_eval)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test_eval, y_pred_proba_eval)
print(f"ROC curve calculated - {len(fpr)} points")

# Precision-Recall Curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test_eval, y_pred_proba_eval)
avg_precision = average_precision_score(y_test_eval, y_pred_proba_eval)
print(f"Average Precision: {avg_precision:.4f}")

# Additional metrics
print("\n--- Additional Classification Metrics ---")
kappa = cohen_kappa_score(y_test_eval, y_pred_eval)
mcc = matthews_corrcoef(y_test_eval, y_pred_eval)

print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Multi-class example
print("\n=== Example 32: Multi-class Classification ===")

# Create multi-class dataset
X_multi, y_multi = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=3, random_state=42
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = rf_multi.predict(X_test_multi)

# Multi-class metrics
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
precision_multi = precision_score(y_test_multi, y_pred_multi, average='weighted')
recall_multi = recall_score(y_test_multi, y_pred_multi, average='weighted')
f1_multi = f1_score(y_test_multi, y_pred_multi, average='weighted')

print(f"Multi-class Accuracy: {accuracy_multi:.4f}")
print(f"Multi-class Precision (weighted): {precision_multi:.4f}")
print(f"Multi-class Recall (weighted): {recall_multi:.4f}")
print(f"Multi-class F1-score (weighted): {f1_multi:.4f}")

# Classification report for multi-class
print("\n--- Multi-class Classification Report ---")
print(classification_report(y_test_multi, y_pred_multi))

print("\n=== Example 33: Regression Metrics ===")

# Create regression dataset
X_reg_eval, y_reg_eval = make_regression(
    n_samples=1000, n_features=10, noise=0.1, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_eval, y_reg_eval, test_size=0.2, random_state=42
)

# Train regressor
rf_reg_eval = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_eval.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg_eval.predict(X_test_reg)

print("\n--- Regression Metrics ---")
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
explained_var = explained_variance_score(y_test_reg, y_pred_reg)
mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Explained Variance Score: {explained_var:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")

# Additional regression metrics
print("\n--- Custom Regression Metrics ---")

# Max error
max_error = np.max(np.abs(y_test_reg - y_pred_reg))
print(f"Max Error: {max_error:.4f}")

# Median absolute error
median_ae = np.median(np.abs(y_test_reg - y_pred_reg))
print(f"Median Absolute Error: {median_ae:.4f}")

# Quantile errors
quantile_25 = np.percentile(np.abs(y_test_reg - y_pred_reg), 25)
quantile_75 = np.percentile(np.abs(y_test_reg - y_pred_reg), 75)
print(f"25th percentile error: {quantile_25:.4f}")
print(f"75th percentile error: {quantile_75:.4f}")

print("\n=== Example 34: Cross-Validation Strategies ===")

print("\n--- K-Fold Cross-Validation ---")
# Basic K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_kfold = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=kfold, scoring='accuracy'
)
print(f"K-Fold CV scores: {cv_scores_kfold}")
print(f"K-Fold CV mean: {cv_scores_kfold.mean():.4f} (+/- {cv_scores_kfold.std() * 2:.4f})")

# Stratified K-Fold (for classification)
print("\n--- Stratified K-Fold Cross-Validation ---")
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=stratified_kfold, scoring='accuracy'
)
print(f"Stratified K-Fold CV scores: {cv_scores_stratified}")
print(f"Stratified K-Fold CV mean: {cv_scores_stratified.mean():.4f} (+/- {cv_scores_stratified.std() * 2:.4f})")

# Group K-Fold (for grouped data)
print("\n--- Group K-Fold Cross-Validation ---")
# Create groups for samples
groups = np.repeat(np.arange(10), 100)  # 10 groups, 100 samples each
group_kfold = GroupKFold(n_splits=3)
cv_scores_group = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=group_kfold, groups=groups, scoring='accuracy'
)
print(f"Group K-Fold CV scores: {cv_scores_group}")
print(f"Group K-Fold CV mean: {cv_scores_group.mean():.4f} (+/- {cv_scores_group.std() * 2:.4f})")

# Shuffle Split (random splits)
print("\n--- Shuffle Split Cross-Validation ---")
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv_scores_shuffle = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=shuffle_split, scoring='accuracy'
)
print(f"Shuffle Split CV scores: {cv_scores_shuffle}")
print(f"Shuffle Split CV mean: {cv_scores_shuffle.mean():.4f} (+/- {cv_scores_shuffle.std() * 2:.4f})")

# Time Series Split
print("\n--- Time Series Split Cross-Validation ---")
# Create time series data
X_timeseries = X_eval.copy()
y_timeseries = y_eval.copy()
time_series_split = TimeSeriesSplit(n_splits=3)
cv_scores_timeseries = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_timeseries, y_timeseries,
    cv=time_series_split, scoring='accuracy'
)
print(f"Time Series Split CV scores: {cv_scores_timeseries}")
print(f"Time Series Split CV mean: {cv_scores_timeseries.mean():.4f} (+/- {cv_scores_timeseries.std() * 2:.4f})")

print("\n=== Example 35: Advanced Cross-Validation ===")

# Cross-validation with multiple metrics
print("\n--- Cross-Validation with Multiple Metrics ---")
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=5, scoring=scoring, return_train_score=True
)

for metric in scoring:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    print(f"{metric.upper()}:")
    print(f"  Test: {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
    print(f"  Gap: {(train_scores.mean() - test_scores.mean()):.4f}")

# Learning curves
print("\n--- Learning Curves ---")
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestClassifier(random_state=42),
    X_eval, y_eval,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

print(f"Learning curve calculated with {len(train_sizes)} training sizes")
print(f"Smallest training size: {train_sizes[0]}")
print(f"Largest training size: {train_sizes[-1]}")

# Validation curves (parameter sensitivity)
print("\n--- Validation Curves ---")
param_range = np.logspace(-4, 1, 6)  # C parameter range for SVM
train_scores_vc, val_scores_vc = validation_curve(
    SVC(random_state=42),
    X_eval, y_eval,
    param_name='C', param_range=param_range,
    cv=5, scoring='accuracy'
)

print("Validation curve for SVM C parameter:")
for i, C in enumerate(param_range):
    train_score = train_scores_vc[i].mean()
    val_score = val_scores_vc[i].mean()
    print(f"C={C:.4f}: Train={train_score:.4f}, Val={val_score:.4f}")

print("\n=== Example 36: Permutation Tests ===")

# Permutation test for model comparison
print("\n--- Permutation Test ---")
baseline_score = accuracy_score(y_test_eval, y_pred_eval)
perm_score, perm_pvalue = permutation_test_score(
    rf_eval, X_train_eval, y_train_eval,
    cv=5, scoring='accuracy', n_permutations=100
)

print(f"Baseline accuracy: {baseline_score:.4f}")
print(f"Permutation test score: {perm_score:.4f}")
print(f"P-value: {perm_pvalue:.4f}")

if perm_pvalue < 0.05:
    print("Model performance is significantly better than random!")
else:
    print("Model performance is not significantly better than random.")

print("\nModel Evaluation coverage completed!")
```

#### 5. Complete scikit-learn Workflow Examples

```python
print("=== Complete scikit-learn Workflows ===")

# Workflow 1: Complete ML Pipeline for Classification
print("\n=== Workflow 1: Classification Pipeline ===")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulate a real-world dataset
np.random.seed(42)
n_samples = 1000

# Create realistic features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'education_years': np.random.randint(8, 22, n_samples),
    'experience': np.random.randint(0, 50, n_samples),
    'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'work_hours': np.random.choice(['Full-time', 'Part-time', 'Contract'], n_samples),
    'salary_raise': np.random.binomial(1, 0.3, n_samples)  # Target variable
}

df = pd.DataFrame(data)

print("Dataset info:")
print(df.head())
print(f"\nTarget distribution:")
print(df['salary_raise'].value_counts())

# Define feature types
numerical_features = ['age', 'income', 'education_years', 'experience']
categorical_features = ['department', 'location', 'work_hours']

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data
X = df.drop('salary_raise', axis=1)
y = df['salary_raise']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
print("\n--- Hyperparameter Tuning ---")
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train final model and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Final Model Evaluation ---")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = (numerical_features +
                list(best_model.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_features)))

feature_importance = best_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Workflow 2: Clustering Analysis
print("\n\n=== Workflow 2: Customer Segmentation ===")

# Simulate customer data
np.random.seed(42)
n_customers = 1000

customer_data = {
    'annual_spending': np.random.lognormal(8, 1, n_customers),
    'purchase_frequency': np.random.poisson(20, n_customers),
    'avg_order_value': np.random.lognormal(4, 0.8, n_customers),
    'customer_age': np.random.normal(40, 15, n_customers),
    'days_since_last_purchase': np.random.exponential(30, n_customers),
    'discount_usage': np.random.beta(2, 5, n_customers),
    'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_customers),
    'loyalty_tier': npormal.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers)
}

customer_df = pd.DataFrame(customer_data)

# Feature engineering for clustering
print("Customer data overview:")
print(customer_df.head())
print(f"\nDataset shape: {customer_df.shape}")

# Select features for clustering
clustering_features = ['annual_spending', 'purchase_frequency', 'avg_order_value',
                      'customer_age', 'days_since_last_purchase', 'discount_usage']
X_customers = customer_df[clustering_features]

# Preprocessing pipeline for clustering
clustering_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clusterer', KMeans(n_clusters=4, random_state=42))
])

# Find optimal number of clusters
print("\n--- Finding Optimal Number of Clusters ---")
silhouette_scores = []
inertias = []
k_range = range(2, 11)

for k in k_range:
    clusterer = Pipeline([
        ('scaler', StandardScaler()),
        ('clusterer', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])

    clusters = clusterer.fit_predict(X_customers)
    silhouette_avg = silhouette_score(X_customers, clusters)
    silhouette_scores.append(silhouette_avg)

    # Get inertia
    clusterer.fit(X_customers)
    inertias.append(clusterer.named_steps['clusterer'].inertia_)

    print(f"K={k}: Silhouette Score = {silhouette_avg:.4f}")

# Find best k
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {best_k}")

# Final clustering with best k
final_clustering = Pipeline([
    ('scaler', StandardScaler()),
    ('clusterer', KMeans(n_clusters=best_k, random_state=42, n_init=10))
])

customer_clusters = final_clustering.fit_predict(X_customers)
customer_df['cluster'] = customer_clusters

# Analyze clusters
print("\n--- Cluster Analysis ---")
for cluster_id in range(best_k):
    cluster_data = customer_df[customer_df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} customers):")
    print(f"  Average Annual Spending: ${cluster_data['annual_spending'].mean():.2f}")
    print(f"  Average Purchase Frequency: {cluster_data['purchase_frequency'].mean():.1f}")
    print(f"  Average Order Value: ${cluster_data['avg_order_value'].mean():.2f}")
    print(f"  Average Age: {cluster_data['customer_age'].mean():.1f}")
    print(f"  Average Days Since Last Purchase: {cluster_data['days_since_last_purchase'].mean():.1f}")

# Workflow 3: Time Series Forecasting
print("\n\n=== Workflow 3: Time Series Forecasting ===")

# Simulate sales data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
sales_data = []
base_sales = 100
trend = 0.05
seasonality = 20
noise_level = 10

for i, date in enumerate(dates):
    day_of_year = date.dayofyear
    weekly_season = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
    yearly_season = 15 * np.sin(2 * np.pi * day_of_year / (365.25 * 4))

    sales = (base_sales +
            trend * i +
            seasonality * (1 if date.weekday() < 5 else 0) +
            weekly_season + yearly_season +
            np.random.normal(0, noise_level))

    sales_data.append(max(0, sales))  # Ensure non-negative

sales_df = pd.DataFrame({
    'date': dates,
    'sales': sales_data
})

print("Time series data:")
print(sales_df.head())
print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")

# Feature engineering for time series
sales_df['year'] = sales_df['date'].dt.year
sales_df['month'] = sales_df['date'].dt.month
sales_df['day'] = sales_df['date'].dt.day
sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
sales_df['day_of_year'] = sales_df['date'].dt.dayofyear

# Add lag features
for lag in [1, 7, 30]:
    sales_df[f'sales_lag_{lag}'] = sales_df['sales'].shift(lag)

# Add rolling statistics
for window in [7, 30]:
    sales_df[f'sales_rolling_mean_{window}'] = sales_df['sales'].rolling(window=window).mean()
    sales_df[f'sales_rolling_std_{window}'] = sales_df['sales'].rolling(window=window).std()

# Remove rows with NaN (from lag and rolling features)
sales_df_clean = sales_df.dropna()

print(f"Data after feature engineering: {len(sales_df_clean)} rows")

# Prepare features and target
feature_columns = ['year', 'month', 'day', 'day_of_week', 'day_of_year'] + \
                 [f'sales_lag_{lag}' for lag in [1, 7, 30]] + \
                 [f'sales_rolling_mean_{window}' for window in [7, 30]] + \
                 [f'sales_rolling_std_{window}' for window in [7, 30]]

X_ts = sales_df_clean[feature_columns]
y_ts = sales_df_clean['sales']

# Split for time series (sequential split)
split_idx = int(len(X_ts) * 0.8)
X_train_ts, X_test_ts = X_ts[:split_idx], X_ts[split_idx:]
y_train_ts, y_test_ts = y_ts[:split_idx], y_ts[split_idx:]

print(f"Training set: {len(X_train_ts)} samples")
print(f"Test set: {len(X_test_ts)} samples")

# Train multiple models
models_ts = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

ts_results = {}
for name, model in models_ts.items():
    model.fit(X_train_ts, y_train_ts)
    y_pred_ts = model.predict(X_test_ts)

    mse = mean_squared_error(y_test_ts, y_pred_ts)
    mae = mean_absolute_error(y_test_ts, y_pred_ts)
    r2 = r2_score(y_test_ts, y_pred_ts)

    ts_results[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}
    print(f"\n{name} Results:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

# Find best model
best_model_name = max(ts_results, key=lambda x: ts_results[x]['R²'])
print(f"\nBest model: {best_model_name} with R² = {ts_results[best_model_name]['R²']:.4f}")

print("\nComplete scikit-learn workflows completed!")
```

---

## TensorFlow 2.x: Deep Learning Framework {#tensorflow}

### What is TensorFlow?

Think of TensorFlow as a "construction crane" for deep learning. Just like a crane can lift heavy materials to build complex structures, TensorFlow can handle complex mathematical operations to build and train neural networks. It's designed for scalability from mobile devices to large distributed systems.

**Why Use TensorFlow:**

- **Production-Ready**: Used by Google, Airbnb, Uber in production
- **Scalable**: Runs on single devices to thousands of machines
- **Flexible**: Supports multiple programming paradigms (eager, graph, distributed)
- **Ecosystem**: TensorBoard, TensorFlow Serving, TensorFlow Lite, TensorFlow.js

**Where to Use TensorFlow:**

- Deep learning research and development
- Production deployment of neural networks
- Large-scale distributed training
- Mobile and edge computing (TensorFlow Lite)
- Web deployment (TensorFlow.js)

### Complete API Coverage

#### 1. TensorFlow Basics and Tensor Operations

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=== TensorFlow 2.x: Complete Guide ===")
print(f"TensorFlow version: {tf.__version__}")

# Basic tensor operations
print("\n=== Example 37: Tensor Basics ===")

# Create tensors from Python lists
tf_scalar = tf.constant(42)
tf_vector = tf.constant([1, 2, 3, 4, 5])
tf_matrix = tf.constant([[1, 2], [3, 4], [5, 6]])
tf_3d_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar: {tf_scalar}")
print(f"Vector: {tf_vector}")
print(f"Matrix shape: {tf_matrix.shape}")
print(f"3D Tensor shape: {tf_3d_tensor.shape}")

# Create tensors from numpy arrays
numpy_array = np.array([1.0, 2.0, 3.0])
tf_from_numpy = tf.constant(numpy_array)
print(f"From numpy: {tf_from_numpy}")

# Tensor attributes
print("\n--- Tensor Properties ---")
print(f"tf_vector shape: {tf_vector.shape}")
print(f"tf_vector dtype: {tf_vector.dtype}")
print(f"tf_vector device: {tf_vector.device}")

# Tensor operations
print("\n--- Tensor Operations ---")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(f"Addition: {tf.add(a, b)}")
print(f"Multiplication: {tf.multiply(a, b)}")
print(f"Matrix multiplication: {tf.matmul(tf_matrix, tf_matrix)}")
print(f"Sum: {tf.reduce_sum(tf_vector)}")
print(f"Mean: {tf.reduce_mean(tf_vector)}")

# Shape manipulation
print("\n--- Shape Manipulation ---")
reshaped = tf.reshape(tf_vector, [5, 1])
print(f"Original: {tf_vector.shape} -> {tf_vector}")
print(f"Reshaped: {reshaped.shape} -> {reshaped}")

transposed = tf.transpose(tf_matrix)
print(f"Original matrix shape: {tf_matrix.shape}")
print(f"Transposed shape: {transposed.shape}")

# Broadcasting
print("\n--- Broadcasting ---")
matrix = tf.constant([[1, 2], [3, 4]])
vector = tf.constant([10, 20])
broadcasted = tf.add(matrix, vector)
print(f"Broadcasted addition:\n{broadcasted}")

# TensorFlow math functions
print("\n--- Math Functions ---")
x = tf.constant([1.0, 2.0, 3.0])
print(f"Square root: {tf.sqrt(x)}")
print(f"Exponential: {tf.exp(x)}")
print(f"Natural log: {tf.math.log(x)}")
print(f"Sigmoid: {tf.nn.sigmoid(x)}")

print("\n=== Example 38: Variables and Automatic Differentiation ===")

# Variables (trainable parameters)
print("\n--- Variables ---")
variable = tf.Variable([1.0, 2.0, 3.0])
print(f"Variable: {variable}")
print(f"Variable value: {variable.numpy()}")

# Variable operations
variable.assign([4.0, 5.0, 6.0])
print(f"After assignment: {variable}")

# Gradient computation
print("\n--- Automatic Differentiation ---")
x = tf.Variable(3.0)
y = tf.Variable(5.0)

# Define a simple function: f(x, y) = x² + 2*y² + 3*x*y
@tf.function
def f(x, y):
    return x**2 + 2*y**2 + 3*x*y

with tf.GradientTape() as tape:
    result = f(x, y)

gradients = tape.gradient(result, [x, y])
print(f"Function: f(x, y) = x² + 2*y² + 3*x*y")
print(f"At x={x.numpy()}, y={y.numpy()}: f = {result.numpy()}")
print(f"∂f/∂x = {gradients[0].numpy()}")
print(f"∂f/∂y = {gradients[1].numpy()}")

# More complex example
@tf.function
def complex_function(x, y):
    z = tf.sin(x) + tf.cos(y)
    w = tf.exp(z) + tf.log(tf.abs(x) + tf.abs(y))
    return w

x = tf.Variable(1.0)
y = tf.Variable(2.0)

with tf.GradientTape() as tape:
    result = complex_function(x, y)

gradients = tape.gradient(result, [x, y])
print(f"\nComplex function result: {result.numpy()}")
print(f"∂result/∂x: {gradients[0].numpy()}")
print(f"∂result/∂y: {gradients[1].numpy()}")

print("\n=== Example 39: Data Pipeline with tf.data ===")

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate features and labels
X_data = np.random.randn(n_samples, 10).astype(np.float32)
y_data = (X_data[:, 0] + 2*X_data[:, 1] - X_data[:, 2] +
         0.1 * np.random.randn(n_samples)).astype(np.float32)

print(f"Generated dataset: {X_data.shape[0]} samples, {X_data.shape[1]} features")

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
print(f"Dataset created: {type(dataset)}")

# Basic dataset operations
print("\n--- Dataset Operations ---")
print(f"Dataset size: {dataset.cardinality().numpy()}")

# Take first 5 samples
for i, (features, label) in enumerate(dataset.take(5)):
    if i < 5:  # Only show first 5
        print(f"Sample {i+1}: features shape {features.shape}, label {label.numpy():.2f}")

# Dataset transformations
print("\n--- Dataset Transformations ---")

# Shuffle and batch
shuffled_dataset = dataset.shuffle(buffer_size=100).batch(batch_size=32)
batched_dataset = dataset.batch(batch_size=32)

print(f"Shuffled and batched dataset: {batched_dataset.cardinality().numpy()} batches")

# Take one batch and examine
for features_batch, labels_batch in batched_dataset.take(1):
    print(f"Batch size: {features_batch.shape[0]}")
    print(f"Features shape: {features_batch.shape}")
    print(f"Labels shape: {labels_batch.shape}")
    break

# Map transformations
def normalize_features(features, label):
    """Normalize features to have zero mean and unit variance"""
    mean = tf.reduce_mean(features)
    std = tf.math.reduce_std(features)
    normalized_features = (features - mean) / std
    return normalized_features, label

normalized_dataset = dataset.map(normalize_features)

# Prefetch for performance
prefetched_dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("Dataset transformations completed: shuffle, batch, map, prefetch")

# Complete data pipeline
print("\n--- Complete Data Pipeline ---")
full_pipeline = (dataset
    .shuffle(buffer_size=1000)
    .map(normalize_features, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=32)
    .prefetch(tf.data.AUTOTUNE))

print("Pipeline: shuffle → normalize → batch → prefetch")

# Split into train and test
train_size = int(0.8 * n_samples)
train_dataset = full_pipeline.take(train_size // 32)
test_dataset = full_pipeline.skip(train_size // 32)

print(f"Train dataset: {train_dataset.cardinality().numpy()} batches")
print(f"Test dataset: {test_dataset.cardinality().numpy()} batches")

print("\n=== Example 40: Keras API - Sequential Models ===")

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

print("Sequential model created:")
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("Model compiled with Adam optimizer and MSE loss")

# Train the model
print("\n--- Training the Model ---")
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    verbose=1
)

print("Training completed!")

# Evaluate the model
print("\n--- Model Evaluation ---")
test_loss, test_mae = model.evaluate(test_dataset, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
sample_batch = next(iter(test_dataset))
predictions = model.predict(sample_batch[0][:5])  # Predict first 5 samples
actual = sample_batch[1][:5]

print("\nPrediction vs Actual (first 5 samples):")
for i in range(5):
    print(f"Sample {i+1}: Predicted={predictions[i][0]:.2f}, Actual={actual[i].numpy():.2f}")

print("\n=== Example 41: Keras Functional API ===")

# Create a more complex model using Functional API
print("\n--- Functional API Model ---")

# Input layer
inputs = tf.keras.Input(shape=(10,), name='input_layer')

# Shared dense layer
shared_dense = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')

# First branch
branch1 = shared_dense(inputs)
branch1 = tf.keras.layers.Dense(32, activation='relu', name='branch1_dense1')(branch1)
branch1 = tf.keras.layers.Dense(16, activation='relu', name='branch1_dense2')(branch1)
output1 = tf.keras.layers.Dense(1, name='output1')(branch1)

# Second branch
branch2 = shared_dense(inputs)
branch2 = tf.keras.layers.Dense(48, activation='relu', name='branch2_dense1')(branch2)
branch2 = tf.keras.layers.Dense(24, activation='relu', name='branch2_dense2')(branch2)
output2 = tf.keras.layers.Dense(1, name='output2')(branch2)

# Combine outputs
combined = tf.keras.layers.concatenate([output1, output2])
combined_dense = tf.keras.layers.Dense(16, activation='relu')(combined)
final_output = tf.keras.layers.Dense(1, activation='linear', name='final_output')(combined_dense)

# Create model
functional_model = tf.keras.Model(inputs=inputs, outputs=final_output, name='functional_model')

print("Functional API model created:")
functional_model.summary()

# Compile and train
functional_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Quick training (fewer epochs for demo)
print("\n--- Training Functional Model ---")
history2 = functional_model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset,
    verbose=0
)

print("Functional model training completed!")

print("\n=== Example 42: Custom Layers and Models ===")

# Custom Layer
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            return self.activation(z)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config

print("Custom Dense Layer created!")

# Custom Model
class CustomRegressionModel(tf.keras.Model):
    def __init__(self):
        super(CustomRegressionModel, self).__init__()
        self.dense1 = CustomDenseLayer(64, activation='relu')
        self.dense2 = CustomDenseLayer(32, activation='relu')
        self.dense3 = CustomDenseLayer(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        return self.output_layer(x)

print("Custom Regression Model created!")

# Create and compile custom model
custom_model = CustomRegressionModel()
custom_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("Custom model compiled!")

# Custom callback
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:  # Print every 2 epochs
            print(f"Epoch {epoch}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")

# Train custom model
print("\n--- Training Custom Model ---")
custom_history = custom_model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset,
    callbacks=[TrainingProgressCallback()],
    verbose=0
)

print("Custom model training completed!")

print("\n=== Example 43: Model Saving and Loading ===")

print("\n--- Model Persistence ---")

# Save model
model.save('models/basic_regression_model')
print("Model saved to 'models/basic_regression_model'")

# Load model
loaded_model = tf.keras.models.load_model('models/basic_regression_model')
print("Model loaded successfully!")

# Verify loaded model
test_loss_loaded, test_mae_loaded = loaded_model.evaluate(test_dataset, verbose=0)
print(f"Loaded model performance - Loss: {test_loss_loaded:.4f}, MAE: {test_mae_loaded:.4f}")

# Save as TensorFlow SavedModel format
model.save('models/saved_model_format')
print("Model saved in SavedModel format")

# Load and verify SavedModel
loaded_saved_model = tf.keras.models.load_model('models/saved_model_format')
test_loss_saved = loaded_saved_model.evaluate(test_dataset, verbose=0)[0]
print(f"SavedModel performance - Loss: {test_loss_saved:.4f}")

print("\n=== Example 44: Transfer Learning ===")

# Load pre-trained model (MobileNetV2 for feature extraction)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

print("Base model (MobileNetV2) loaded:")
base_model.summary()

# Freeze base model
base_model.trainable = False

# Create transfer learning model
transfer_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

transfer_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Transfer learning model created:")
transfer_model.summary()

# Fine-tuning example (unfreeze last layers)
print("\n--- Fine-tuning ---")
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Fine-tuning from layer {fine_tune_at} onwards")
print(f"Trainable parameters: {transfer_model.count_params()}")

print("\n=== Example 45: TensorBoard Integration ===")

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/tensorboard_logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/checkpoint_epoch_{epoch:02d}',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# ReduceLROnPlateau callback
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

print("TensorBoard and monitoring callbacks created")
print("Log directory: logs/tensorboard_logs")

# Example training with callbacks
print("\n--- Training with Callbacks ---")
history_with_callbacks = model.fit(
    train_dataset,
    epochs=3,  # Few epochs for demo
    validation_data=test_dataset,
    callbacks=[tensorboard_callback, checkpoint_callback, lr_reducer],
    verbose=1
)

print("Training with callbacks completed!")

print("\n=== Example 46: Distributed Training ===")

print("\n--- Multi-GPU Strategy ---")

# Strategy for multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create model within strategy scope
with strategy.scope():
    distributed_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    distributed_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

print("Distributed model created")
print(f"Strategy: {strategy.__class__.__name__}")

# Create distributed dataset
dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

print("Distributed datasets created")

print("\n=== Example 47: Custom Training Loop ===")

# Custom training step
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Custom training loop
def custom_training_loop(model, train_dataset, epochs=3):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    print("Starting custom training loop...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_dataset:
            loss_value = train_step(batch_x, batch_y)
            epoch_loss += loss_value
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    print("Custom training loop completed!")

# Run custom training
custom_training_loop(model, train_dataset)

print("\nTensorFlow 2.x coverage completed!")
```

---

## PyTorch: Flexible Deep Learning {#pytorch}

### What is PyTorch?

Think of PyTorch as " LEGO blocks for deep learning." Just like LEGO lets you build anything with simple, flexible pieces, PyTorch provides intuitive building blocks that you can combine in countless ways. It's particularly beloved by researchers because it feels natural and Pythonic.

**Why Use PyTorch:**

- **Dynamic Graphs**: Define-by-run approach makes debugging easier
- **Pythonic**: Feels natural to Python developers
- **Research-Friendly**: Most cutting-edge research uses PyTorch
- **Strong Ecosystem**: torchvision, torchaudio, torchtext, lightning

**Where to Use PyTorch:**

- Deep learning research and experimentation
- Computer vision (torchvision)
- Natural language processing (torchtext)
- Audio processing (torchaudio)
- Production deployment (TorchScript, ONNX)

### Complete API Coverage

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np

print("=== PyTorch: Complete Guide ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\n=== Example 48: Tensor Operations ===")

# Create tensors
print("--- Tensor Creation ---")
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_3d = torch.randn(2, 3, 4)  # Random values

print(f"Scalar: {scalar}")
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")
print(f"3D tensor shape: {tensor_3d.shape}")

# From numpy
numpy_array = np.array([1, 2, 3])
torch_from_numpy = torch.from_numpy(numpy_array)
numpy_from_torch = torch_from_numpy.numpy()

print(f"Numpy → PyTorch → Numpy: {numpy_array} → {torch_from_numpy} → {numpy_from_torch}")

# Tensor properties
print("\n--- Tensor Properties ---")
print(f"Data type: {vector.dtype}")
print(f"Device: {vector.device}")
print(f"Requires grad: {vector.requires_grad}")

# Tensor operations
print("\n--- Tensor Operations ---")
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(f"Addition: {a + b}")
print(f"Multiplication (element-wise): {a * b}")
print(f"Matrix multiplication: {torch.matmul(a.view(3, 1), b.view(1, 3))}")
print(f"Sum: {torch.sum(a)}")
print(f"Mean: {torch.mean(a)}")

# Reshaping
print("\n--- Reshaping Operations ---")
reshaped = vector.view(5, 1)
flattened = vector.view(-1)
transposed = matrix.t()

print(f"Original: {vector.shape}")
print(f"Reshaped (5,1): {reshaped.shape}")
print(f"Flattened: {flattened.shape}")
print(f"Transposed: {transposed.shape}")

# Broadcasting
print("\n--- Broadcasting ---")
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
vector = torch.tensor([10, 20, 30], dtype=torch.float32)
broadcasted = matrix + vector
print(f"Broadcasted addition:\n{broadcasted}")

print("\n=== Example 49: Autograd and Gradients ===")

# Autograd
print("--- Automatic Differentiation ---")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Define a function: z = x² + 2*y² + 3*x*y
z = x**2 + 2*y**2 + 3*x*y
loss = torch.sum(z)

print(f"x: {x}")
print(f"y: {y}")
print(f"z = x² + 2*y² + 3*x*y: {z}")
print(f"Loss = sum(z): {loss}")

# Backward pass
loss.backward()

print(f"∂loss/∂x: {x.grad}")
print(f"∂loss/∂y: {y.grad}")

# Analytical verification
print("Analytical gradients:")
print(f"∂z/∂x = 2x + 3y: {2*x + 3*y}")
print(f"∂z/∂y = 4y + 3x: {4*y + 3*x}")

# Disable gradient computation
with torch.no_grad():
    z_no_grad = x**2 + 2*y**2 + 3*x*y
    print(f"Without grad tracking: {z_no_grad}")

# Custom autograd function
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 3 * x**2 * grad_output

# Use custom function
x = torch.tensor([2.0], requires_grad=True)
y = CustomFunction.apply(x)
y.backward()

print(f"\nCustom function: y = x³")
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"∂y/∂x = 3x² = {3 * x**2}")

print("\n=== Example 50: Neural Networks with nn.Module ===")

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model
input_size = 10
hidden_size = 128
output_size = 1

model = SimpleNet(input_size, hidden_size, output_size)
print("Simple neural network created:")
print(model)

# Model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model moved to: {device}")

print("\n=== Example 51: Training Loop ===")

# Create sample data
np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
X_train = torch.randn(n_samples, input_size)
y_train = torch.sum(X_train[:, :3], dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)

# Create data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Train the model
print("Starting training...")
train_model(model, train_loader, criterion, optimizer, epochs=5)
print("Training completed!")

# Evaluation
model.eval()
with torch.no_grad():
    test_samples = X_train[:10].to(device)
    predictions = model(test_samples)
    actual = y_train[:10].to(device)

    print("\nPrediction vs Actual (first 10 samples):")
    for i in range(10):
        print(f"Sample {i+1}: Predicted={predictions[i].item():.4f}, Actual={actual[i].item():.4f}")

print("\n=== Example 52: Custom Datasets and DataLoaders ===")

# Custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# Create custom dataset
custom_dataset = CustomDataset(X_train, y_train)
print(f"Custom dataset size: {len(custom_dataset)}")

# Custom collate function
def custom_collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return data, labels

# Custom data loader
custom_loader = DataLoader(
    custom_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=2  # Use multiple processes for data loading
)

print("Custom data loader created")

# Example using torchvision datasets
print("\n--- torchvision.datasets Example ---")

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST
train_dataset_mnist = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)

test_dataset_mnist = datasets.MNIST(
    root='data',
    train=False,
    transform=transform
)

# Create data loaders
train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=1000, shuffle=False)

print(f"MNIST train dataset: {len(train_dataset_mnist)} samples")
print(f"MNIST test dataset: {len(test_dataset_mnist)} samples")

# Show a sample
sample_image, sample_label = train_dataset_mnist[0]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label}")

print("\n=== Example 53: CNN for Image Classification ===")

# CNN for MNIST
class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create CNN model
cnn_model = CNN_MNIST().to(device)
print("CNN model created:")
print(cnn_model)

# Loss and optimizer
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training function for CNN
def train_cnn(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

# Test function
def test_cnn(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= total
    accuracy = 100 * correct / total
    return test_loss, accuracy

# Train CNN
print("\nTraining CNN...")
epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_cnn(cnn_model, train_loader_mnist, cnn_criterion, cnn_optimizer, device)
    test_loss, test_acc = test_cnn(cnn_model, test_loader_mnist, device)

    print(f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%')

print("\nCNN training completed!")

print("\n=== Example 54: Transfer Learning ===")

# Load pre-trained ResNet
from torchvision.models import resnet18, ResNet18_Weights

# Load pre-trained model
pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze parameters
for param in pretrained_model.parameters():
    param.requires_grad = False

# Modify final layer for our task
num_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_features, 10)  # 10 classes for MNIST

pretrained_model = pretrained_model.to(device)
print("Pre-trained ResNet model loaded and modified:")
print(pretrained_model)

# Optimizer for transfer learning (only train final layer)
transfer_optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

# Transform for ResNet (expects 3 channels)
transform_resnet = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Re-download MNIST with transform
train_dataset_resnet = datasets.MNIST(root='data', train=True, download=True, transform=transform_resnet)
test_dataset_resnet = datasets.MNIST(root='data', train=False, download=True, transform=transform_resnet)

train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=32, shuffle=True)
test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=1000, shuffle=False)

print("Transfer learning dataset prepared")

# Train transfer learning model
print("\nTraining with transfer learning...")
for epoch in range(3):  # Fewer epochs since we're only training the final layer
    print(f"\nEpoch {epoch+1}/3")
    train_loss, train_acc = train_cnn(pretrained_model, train_loader_resnet, cnn_criterion, transfer_optimizer, device)
    test_loss, test_acc = test_cnn(pretrained_model, test_loader_resnet, device)

    print(f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%')

print("\nTransfer learning completed!")

print("\n=== Example 55: Saving and Loading Models ===")

# Save model
torch.save({
    'epoch': epochs,
    'model_state_dict': cnn_model.state_dict(),
    'optimizer_state_dict': cnn_optimizer.state_dict(),
    'loss': train_loss,
    'accuracy': train_acc
}, 'models/pytorch_cnn_model.pth')

print("Model saved to 'models/pytorch_cnn_model.pth'")

# Load model
new_model = CNN_MNIST()
checkpoint = torch.load('models/pytorch_cnn_model.pth', map_location=device)
new_model.load_state_dict(checkpoint['model_state_dict'])
new_model = new_model.to(device)

print("Model loaded successfully!")

# Test loaded model
test_loss_loaded, test_acc_loaded = test_cnn(new_model, test_loader_mnist, device)
print(f"Loaded model accuracy: {test_acc_loaded:.2f}%")

# Save entire model
torch.save(cnn_model, 'models/pytorch_cnn_complete.pth')
print("Entire model saved")

# Load entire model
loaded_model = torch.load('models/pytorch_cnn_complete.pth', map_location=device)
loaded_model.eval()
print("Entire model loaded")

print("\n=== Example 56: Custom Loss Functions ===")

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        l1_loss = self.l1(predictions, targets)
        return self.alpha * mse_loss + (1 - self.alpha) * l1_loss

# Focal Loss for classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return torch.mean(focal_loss)

# Use custom loss
custom_loss = CustomLoss(alpha=0.7)
print("Custom loss functions created")

# Example with custom loss
with torch.no_grad():
    sample_pred = torch.randn(5, 1)
    sample_target = torch.randn(5, 1)
    custom_loss_value = custom_loss(sample_pred, sample_target)
    print(f"Custom loss value: {custom_loss_value.item():.4f}")

print("\n=== Example 57: Gradient Accumulation ===")

# Simulate training with gradient accumulation
def train_with_accumulation(model, train_loader, optimizer, criterion, device, accumulation_steps=4):
    model.train()
    total_loss = 0.0

    for step, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulate gradients
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps

        # Break after few steps for demo
        if step >= 10:
            break

    return total_loss / accumulation_steps

# Example usage
print("Training with gradient accumulation...")
accumulation_loss = train_with_accumulation(cnn_model, train_loader_mnist, cnn_optimizer, cnn_criterion, device)
print(f"Accumulated loss: {accumulation_loss:.6f}")

print("\n=== Example 58: Mixed Precision Training ===")

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler()

def train_mixed_precision(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = cnn_criterion(output, target)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Break for demo
        break

    return total_loss

# Example mixed precision training (requires CUDA)
if torch.cuda.is_available():
    print("Training with mixed precision...")
    mp_loss = train_mixed_precision(cnn_model, train_loader_mnist, cnn_optimizer, device)
    print(f"Mixed precision loss: {mp_loss:.6f}")
else:
    print("Mixed precision training requires CUDA")

print("\nPyTorch coverage completed!")
```

---

## Hugging Face: State-of-the-Art Models {#hugging-face}

### What is Hugging Face?

Think of Hugging Face as a "content creation studio" for AI models. Just like a recording studio provides all the tools, instruments, and mixing equipment you need to create professional music, Hugging Face provides pre-trained models, datasets, and tools for natural language processing, computer vision, and more.

**Why Use Hugging Face:**

- **50+ Pre-trained Models**: BERT, GPT, T5, RoBERTa, DistilBERT, and more
- **1000+ Datasets**: Curated datasets for every NLP task
- **Easy API**: Simple interface for training and inference
- **Model Hub**: Share and discover models from the community

**Where to Use Hugging Face:**

- Text classification (sentiment analysis, topic modeling)
- Named entity recognition (NER)
- Question answering
- Text summarization
- Machine translation
- Text generation

### Complete API Coverage

```python
from transformers import (
    # Core imports
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM,

    # Pipelines
    pipeline, TextClassificationPipeline, TokenClassificationPipeline,
    QuestionAnsweringPipeline, SummarizationPipeline, TranslationPipeline,

    # Training and fine-tuning
    Trainer, TrainingArguments, DataCollatorWithPadding,

    # Models
    BertTokenizer, BertModel, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,
    DistilBertTokenizer, DistilBertModel,
    RobertaTokenizer, RobertaForSequenceClassification,

    # Datasets
    load_dataset, load_metric, Dataset, DatasetDict,

    # Feature extraction
    pipeline, feature_extraction_pipeline,

    # Metrics
    evaluate
)

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

print("=== Hugging Face: Complete Guide ===")

print("\n=== Example 59: Basic Model Loading and Usage ===")

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"Loaded model: {model_name}")
print(f"Model type: {type(model)}")
print(f"Tokenizer type: {type(tokenizer)}")

# Basic text processing
text = "The quick brown fox jumps over the lazy dog. This is a simple example."

print(f"\nOriginal text: {text}")

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Encode (convert to IDs)
input_ids = tokenizer.encode(text, return_tensors="pt")
print(f"Input IDs shape: {input_ids.shape}")
print(f"First 10 input IDs: {input_ids[0][:10]}")

# Get model output
with torch.no_grad():
    outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output

print(f"Last hidden state shape: {last_hidden_state.shape}")
print(f"Pooled output shape: {pooled_output.shape}")

# Decode
decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(f"Decoded text: {decoded_text}")

print("\n=== Example 60: Text Classification Pipeline ===")

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

print("Sentiment Analysis Pipeline:")
texts = [
    "I absolutely love this movie! It's fantastic.",
    "This movie is terrible. I hated every minute.",
    "The movie was okay, nothing special.",
    "What an amazing piece of art! Brilliant work.",
    "Worst movie I've ever seen. Complete waste of time."
]

for text in texts:
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    print(f"Text: '{text}'")
    print(f"Sentiment: {label} (confidence: {score:.4f})")
    print()

# Named Entity Recognition
print("--- Named Entity Recognition ---")
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

texts_ner = [
    "Apple Inc. is planning to open a new store in San Francisco next year.",
    "Microsoft CEO Satya Nadella announced the acquisition of GitHub for $7.5 billion.",
    "The Eiffel Tower is located in Paris, France."
]

for text in texts_ner:
    print(f"Text: {text}")
    entities = ner_pipeline(text)
    for entity in entities:
        print(f"  {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.4f})")
    print()

# Question Answering
print("--- Question Answering ---")
qa_pipeline = pipeline("question-answering")

context = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf forest
in which the Amazon Basin. It is located in South America and covers an area
of 5.5 million square kilometers. The Amazon represents over half of the
planet's remaining rainforest.
"""

questions = [
    "Where is the Amazon rainforest located?",
    "What is another name for the Amazon rainforest?",
    "How large is the Amazon rainforest?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")
    print()

print("\n=== Example 61: Text Generation ===")

# Load text generation pipeline
generator = pipeline("text-generation", model="gpt2")

print("Text Generation:")
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy,",
    "Climate change will impact the world by"
]

for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    results = generator(prompt, max_length=50, num_return_sequences=2)

    for i, result in enumerate(results, 1):
        generated_text = result['generated_text']
        print(f"Generated {i}: {generated_text}")
    print()

# Summarization
print("--- Text Summarization ---")
summarizer = pipeline("summarization")

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast
to the natural intelligence displayed by humans and animals. Leading AI textbooks
define the field as the study of "intelligent agents": any device that perceives
its environment and takes actions that maximize its chance of successfully achieving
its goals. Colloquially, the term "artificial intelligence" is often used to
describe machines that mimic "cognitive" functions that humans associate with the
human mind, such as "learning" and "problem solving". As machines become increasingly
capable, tasks considered to require "intelligence" are often removed from the
definition of AI. The term frequently applied to projects for developing systems
endowed with the intellectual processes characteristic of humans, such as the
ability to reason, discover meaning, generalize, or learn from past experience.

Machine learning (ML) is a field of inquiry devoted to understanding and building
methods that 'learn' – that is, methods that leverage data to improve performance
on some set of tasks. Machine learning algorithms build a model based on training
data in order to make predictions or decisions without being explicitly programmed
to do so. Machine learning algorithms are used in a wide variety of applications,
such as in medicine, email filtering, speech recognition, and computer vision,
where it is difficult or unfeasible to develop conventional algorithms to perform
the needed tasks.
"""

summary = summarizer(long_text, max_length=100, min_length=30, do_sample=False)
print("Original text length:", len(long_text))
print("Summary:", summary[0]['summary_text'])

print("\n=== Example 62: Translation ===")

# Load translation pipeline
translator = pipeline("translation_en_to_fr")

texts_to_translate = [
    "Hello, how are you today?",
    "I love machine learning and artificial intelligence.",
    "The weather is beautiful today."
]

print("English to French Translation:")
for text in texts_to_translate:
    result = translator(text)
    translated_text = result[0]['translation_text']
    print(f"English: {text}")
    print(f"French: {translated_text}")
    print()

# Load back translation to verify
back_translator = pipeline("translation_fr_to_en")
print("Back translation (French to English):")
for text in texts_to_translate:
    french_result = translator(text)[0]['translation_text']
    back_translated = back_translator(french_result)[0]['translation_text']
    print(f"Original: {text}")
    print(f"French: {french_result}")
    print(f"Back: {back_translated}")
    print()

print("\n=== Example 63: Custom Model Fine-tuning ===")

# Load a model for sequence classification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer_finetune = AutoTokenizer.from_pretrained(model_name)

print(f"Loaded model for sequence classification: {model_name}")

# Sample data for classification
texts = [
    "This is a great movie!",
    "I hate this product.",
    "The service was excellent.",
    "Terrible customer support.",
    "Amazing experience!",
    "Could be better.",
    "Outstanding quality!",
    "Not worth the money."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

print("Sample texts for classification:")
for i, (text, label) in enumerate(zip(texts, labels)):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"{i+1}. '{text}' → {sentiment}")

# Tokenize the data
encoded_data = tokenizer_finetune(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

print(f"\nTokenized data shape: {encoded_data['input_ids'].shape}")

# Create a simple dataset
class CustomDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

dataset = CustomDataset(encoded_data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Dataset size: {len(dataset)}")

# Simple training loop (for demonstration)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("\nFine-tuning for 3 epochs:")
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels_batch = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

print("Fine-tuning completed!")

# Test the fine-tuned model
model.eval()
test_text = "This product exceeded my expectations!"
inputs = tokenizer_finetune(test_text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=-1).item()

label_names = ["Negative", "Positive"]
predicted_sentiment = label_names[predicted_label]
confidence = predictions[0][predicted_label].item()

print(f"\nTest result for: '{test_text}'")
print(f"Predicted: {predicted_sentiment} (confidence: {confidence:.4f})")

print("\n=== Example 64: Feature Extraction ===")

# Use model for feature extraction
feature_model = AutoModel.from_pretrained('distilbert-base-uncased')

texts_features = [
    "Machine learning is fascinating",
    "Deep learning applications",
    "Natural language processing"
]

# Get embeddings
embeddings = []
for text in texts_features:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = feature_model(**inputs)
        # Use CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding)

embeddings = torch.stack(embeddings).squeeze()
print(f"Embeddings shape: {embeddings.shape}")

# Calculate similarity between embeddings
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

print("\nSimilarity matrix:")
for i in range(len(texts_features)):
    for j in range(len(texts_features)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"'{texts_features[i]}' vs '{texts_features[j]}': {sim.item():.4f}")
    print()

print("\n=== Example 65: Using Different Model Architectures ===")

# BERT model
print("--- BERT Model ---")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

text = "The capital of France is Paris."
bert_tokens = bert_tokenizer.tokenize(text)
print(f"BERT tokens: {bert_tokens}")

# GPT-2 for text generation
print("\n--- GPT-2 Model ---")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

gpt2_prompt = "In a world where artificial intelligence"
gpt2_inputs = gpt2_tokenizer.encode(gpt2_prompt, return_tensors="pt")

with torch.no_grad():
    gpt2_outputs = gpt2_model.generate(
        gpt2_inputs,
        max_length=50,
        num_return_sequences=2,
        temperature=0.8,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )

print(f"GPT-2 prompt: '{gpt2_prompt}'")
for i, output in enumerate(gpt2_outputs, 1):
    generated_text = gpt2_tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated {i}: {generated_text}")

# T5 for text-to-text tasks
print("\n--- T5 Model ---")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Summarization task
t5_input = "summarize: " + long_text[:500] + "..."
t5_inputs = t5_tokenizer.encode(t5_input, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    t5_outputs = t5_model.generate(t5_inputs, max_length=100, num_return_sequences=1)

t5_summary = t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)
print(f"T5 Summary: {t5_summary}")

print("\n=== Example 66: Working with Datasets ===")

# Load dataset
try:
    dataset = load_dataset("squad", split="train[:5%]")  # Load small portion for demo
    print(f"Loaded SQuAD dataset: {len(dataset)} examples")

    # Show first example
    example = dataset[0]
    print(f"Question: {example['question']}")
    print(f"Context: {example['context'][:100]}...")
    print(f"Answer: {example['answers']['text'][0]}")

except Exception as e:
    print(f"Could not load dataset: {e}")
    # Create dummy dataset instead
    dataset_data = {
        'text': [
            "The Amazon rainforest is located in South America.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ],
        'label': [0, 1, 1, 1, 1]  # 0: factual, 1: technical
    }

    dataset = Dataset.from_dict(dataset_data)
    print(f"Created dummy dataset with {len(dataset)} examples")

    for i, example in enumerate(dataset):
        label_name = "factual" if example['label'] == 0 else "technical"
        print(f"{i+1}. '{example['text']}' → {label_name}")

print("\n=== Example 67: Model Evaluation ===")

# Load evaluation metrics
try:
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    # Simulate predictions and references
    predictions = [0, 1, 2, 1, 0, 2, 1]
    references = [0, 1, 1, 1, 0, 2, 0]

    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

    print(f"Accuracy: {accuracy['accuracy']:.4f}")
    print(f"F1 Score: {f1['f1']:.4f}")

except Exception as e:
    print(f"Could not load metrics: {e}")

print("\nHugging Face coverage completed!")
```

---

## OpenCV: Computer Vision Library {#opencv}

### What is OpenCV?

Think of OpenCV as a "microscope and camera" for computers. Just like a microscope lets you see tiny details and a camera captures images, OpenCV gives computers the ability to see, analyze, and process visual information. It's the foundation for most computer vision applications.

**Why Use OpenCV:**

- **Comprehensive**: Covers all major computer vision tasks
- **Fast**: Optimized C++ backend with Python bindings
- **Cross-platform**: Works on Windows, macOS, Linux
- **Industry Standard**: Used in production by major companies

**Where to Use OpenCV:**

- Image processing and enhancement
- Object detection and recognition
- Video analysis and tracking
- Face detection and recognition
- Motion detection

### Complete API Coverage

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("=== OpenCV: Computer Vision Complete Guide ===")
print(f"OpenCV version: {cv2.__version__}")

print("\n=== Example 68: Basic Image Operations ===")

# Create a sample image
print("--- Creating and Manipulating Images ---")

# Create a blank image
height, width = 400, 600
blank_image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw basic shapes
cv2.rectangle(blank_image, (50, 50), (200, 150), (255, 0, 0), 3)  # Blue rectangle
cv2.circle(blank_image, (300, 100), 50, (0, 255, 0), -1)  # Green filled circle
cv2.line(blank_image, (400, 50), (500, 150), (0, 0, 255), 5)  # Red line
cv2.putText(blank_image, "OpenCV Demo", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

print(f"Created blank image: {blank_image.shape}")
print("Drew: rectangle, circle, line, text")

# Load and process an image (create sample if none exists)
def create_sample_image():
    # Create a gradient image
    gradient = np.zeros((300, 400, 3), dtype=np.uint8)

    # Create horizontal gradient
    for i in range(400):
        color = int(255 * i / 400)
        gradient[:, i] = [color, 0, 255 - color]

    # Add some shapes
    cv2.rectangle(gradient, (50, 50), (150, 200), (255, 255, 255), 2)
    cv2.circle(gradient, (250, 150), 80, (255, 255, 255), 3)

    return gradient

sample_image = create_sample_image()
print(f"Sample image created: {sample_image.shape}")

# Basic image operations
print("\n--- Basic Operations ---")

# Get image properties
print(f"Image shape: {sample_image.shape}")
print(f"Image dtype: {sample_image.dtype}")
print(f"Image size: {sample_image.size}")

# Access pixels
pixel_value = sample_image[100, 100]
print(f"Pixel at (100, 100): {pixel_value}")

# Modify pixel
sample_image[100, 100] = [255, 255, 255]
print(f"Modified pixel at (100, 100): {sample_image[100, 100]}")

# Image regions (ROI - Region of Interest)
roi = sample_image[50:150, 50:200]
print(f"ROI shape: {roi.shape}")

print("\n=== Example 69: Image Filtering ===")

def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Create noisy version
noisy_image = add_gaussian_noise(sample_image)
print("Added Gaussian noise to image")

# Different blur operations
print("\n--- Blur Operations ---")

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(sample_image, (5, 5), 0)
print("Applied Gaussian blur")

# Median Blur
median_blur = cv2.medianBlur(sample_image, 5)
print("Applied median blur")

# Bilateral Filter (preserves edges)
bilateral = cv2.bilateralFilter(sample_image, 9, 75, 75)
print("Applied bilateral filter")

# Sharpening filter
print("\n--- Sharpening ---")
kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(sample_image, -1, kernel_sharpen)
print("Applied sharpening filter")

# Edge detection filters
print("\n--- Edge Detection Filters ---")

# Sobel edge detection
sobel_x = cv2.Sobel(sample_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(sample_image, cv2.CV_64F, 0, 1            "label": "Run Python File",
            "type": "shell",
            "command": "python",
            "args": ["${file}"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Jupyter Notebook",
            "type": "shell",
            "command": "jupyter",
            "args": ["notebook", "${file}"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "python",
            "args": ["-m", "pytest", "${workspaceFolder}/tests"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
"""

print("Example task configuration:")
print(task_config)

print("\n--- Snippets Configuration ---")

snippets_example = """
// Python snippets (python.json)
{
    "ml-import": {
        "prefix": "mlimport",
        "body": [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.ensemble import RandomForestClassifier",
            "from sklearn.metrics import accuracy_score, classification_report"
        ],
        "description": "Common ML imports"
    },
    "ml-pipeline": {
        "prefix": "mlpipeline",
        "body": [
            "# Load data",
            "df = pd.read_csv('data.csv')",
            "",
            "# Explore data",
            "print(df.head())",
            "print(df.info())",
            "print(df.describe())",
            "",
            "# Prepare features and target",
            "X = df.drop('target', axis=1)",
            "y = df['target']",
            "",
            "# Split data",
            "X_train, X_test, y_train, y_test = train_test_split(",
            "    X, y, test_size=0.2, random_state=42, stratify=y",
            ")",
            "",
            "# Train model",
            "model = RandomForestClassifier(random_state=42)",
            "model.fit(X_train, y_train)",
            "",
            "# Evaluate",
            "y_pred = model.predict(X_test)",
            "print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')"
        ],
        "description": "Basic ML pipeline"
    }
}
"""

print("Example code snippets:")
print(snippets_example)

print("\nVS Code setup completed!")
```

---

## Environment Management {#environment-management}

### Why Environment Management?

Think of environment management like having different "workshops" for different projects. In one workshop, you might need specific tools and materials, while in another, you need different equipment. Python environments let you isolate these dependencies so projects don't interfere with each other.

### Complete Setup Guide

```python
print("=== Environment Management: Complete Guide ===")

print("\n=== Conda Environment Management ===")

conda_commands = """
# Create new environment
conda create -n myproject python=3.9 pandas numpy matplotlib scikit-learn
conda create -n ml_env python=3.8 pytorch torchvision tensorflow

# Activate environment
conda activate myproject

# List environments
conda env list

# Install packages
conda install pandas numpy matplotlib
conda install -c conda-forge opencv

# Export environment
conda env export > environment.yml
conda env export --from-history > environment.yml

# Create from file
conda env create -f environment.yml

# Deactivate
conda deactivate

# Remove environment
conda env remove -n myproject

# Update environment
conda env update -n myproject --file environment.yml
"""

print("Conda commands:")
print(conda_commands)

print("\n=== pip Environment Management ===")

pip_commands = """
# Create virtual environment
python -m venv myproject_env

# Activate (Windows)
myproject_env\\Scripts\\activate

# Activate (Linux/Mac)
source myproject_env/bin/activate

# Install packages
pip install pandas numpy matplotlib scikit-learn
pip install -r requirements.txt

# Freeze requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Upgrade packages
pip install --upgrade package_name

# Deactivate
deactivate

# Use different Python version
python -m venv myproject_env --python=3.9
"""

print("pip commands:")
print(pip_commands)

print("\n=== Poetry Environment Management ===")

poetry_commands = """
# Initialize project
poetry init

# Add dependencies
poetry add pandas numpy matplotlib
poetry add -D pytest black flake8

# Install dependencies
poetry install

# Run script
poetry run python script.py

# Shell environment
poetry shell

# Update dependencies
poetry update

# Build
poetry build

# Export to requirements.txt
poetry export -f requirements.txt --output requirements.txt --without-hashes
"""

print("Poetry commands:")
print(poetry_commands)

print("\n=== pipenv Environment Management ===")

pipenv_commands = """
# Initialize project
pipenv --python 3.9
pipenv install

# Install packages
pipenv install pandas numpy matplotlib
pipenv install --dev pytest black

# Install from Pipfile
pipenv sync

# Activate virtual environment
pipenv shell

# Run script
pipenv run python script.py

# Update packages
pipenv update

# Remove package
pipenv uninstall package_name

# Generate requirements.txt
pipenv requirements > requirements.txt
"""

print("Pipenv commands:")
print(pipenv_commands)

print("\n=== Docker Environment Management ===")

dockerfile_example = """
# Dockerfile for Python ML project
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py"]
"""

docker_commands = """
# Build image
docker build -t myproject:latest .

# Run container
docker run -it --rm myproject:latest

# Run with volume mount
docker run -it --rm -v $(pwd):/app myproject:latest

# Run Jupyter notebook
docker run -it --rm -p 8888:8888 -v $(pwd):/app myproject:latest jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Use docker-compose
docker-compose up -d
"""

docker_compose_example = """
# docker-compose.yml
version: '3.8'
services:
  ml-project:
    build: .
    ports:
      - "8888:8888"
      - "5000:5000"
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
    working_dir: /app
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"""

print("Dockerfile example:")
print(dockerfile_example)
print("\nDocker commands:")
print(docker_commands)
print("\nDocker Compose example:")
print(docker_compose_example)

print("\n=== Environment Comparison ===")

comparison_table = """
| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| venv | Simple projects | Built-in, lightweight | No dependency resolution |
| conda | Data science | Excellent dependency management, cross-platform | Larger footprint |
| pipenv | Modern Python | Dependency resolution, security | Less common in research |
| poetry | Modern Python | Dependency resolution, packaging | Learning curve |
| Docker | Production | Complete isolation, reproducibility | Complexity for beginners |
"""

print(comparison_table)

print("\nEnvironment management completed!")
```

### 🚀 Complete Environment Setup Workflows

#### Option 1: Conda Workflow (Recommended for Data Science)

```bash
# 1. Install Anaconda/Miniconda
# Download from https://anaconda.com/download

# 2. Create new environment for AI projects
conda create -n ai-env python=3.9 pandas numpy matplotlib scikit-learn jupyter

# 3. Activate environment
conda activate ai-env

# 4. Install additional packages
conda install -c conda-forge opencv
conda install -c pytorch pytorch torchvision

# 5. Create environment file for sharing
conda env export > environment.yml

# 6. To recreate environment elsewhere
conda env create -f environment.yml

# 7. List environments
conda env list

# 8. Deactivate when done
conda deactivate
```

#### Option 2: Virtual Environment + pip Workflow

```bash
# 1. Create virtual environment
python -m venv ai-project-env

# 2. Activate environment
# On Windows:
ai-project-env\\Scripts\\activate
# On Mac/Linux:
source ai-project-env/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install core packages
pip install pandas numpy matplotlib scikit-learn jupyter

# 5. Install specialized packages
pip install torch torchvision tensorflow opencv-python
pip install nltk spacy transformers

# 6. Create requirements file
pip freeze > requirements.txt

# 7. To recreate environment
python -m venv new-ai-env
source new-ai-env/bin/activate
pip install -r requirements.txt
```

#### Option 3: Poetry Workflow (Modern Python)

```bash
# 1. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Initialize project
poetry init

# 3. Add dependencies
poetry add pandas numpy matplotlib scikit-learn
poetry add -D pytest black flake8 mypy

# 4. Install dependencies
poetry install

# 5. Run commands in the environment
poetry run python script.py

# 6. Activate shell
poetry shell

# 7. Update dependencies
poetry update

# 8. Build for distribution
poetry build
```

### 🐳 Docker Setup for AI Development

#### Complete Dockerfile for AI/ML Projects

```dockerfile
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies if they exist
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 aiuser && chown -R aiuser:aiuser /app
USER aiuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### Sample requirements.txt

```
# Core Data Science
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.1.0

# Deep Learning
torch>=1.12.0
torchvision>=0.13.0
tensorflow>=2.8.0

# Computer Vision
opencv-python>=4.6.0
Pillow>=9.0.0

# Natural Language Processing
nltk>=3.7
spacy>=3.4.0
transformers>=4.20.0

# Development Tools
jupyter>=1.0.0
black>=22.0.0
flake8>=4.0.0
pytest>=7.0.0

# Additional Tools
plotly>=5.8.0
streamlit>=1.12.0
```

#### Sample requirements-dev.txt

```
# Development and Testing
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# Jupyter Extensions
jupyterlab>=3.4.0
nbstripout>=0.6.0

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0

# Experiment Tracking
mlflow>=1.25.0
wandb>=0.12.0
```

#### Docker Compose for Complete Development Environment

```yaml
# docker-compose.yml
version: "3.8"

services:
  # Main AI Development Environment
  ai-dev:
    build: .
    ports:
      - "8888:8888" # Jupyter
      - "8501:8501" # Streamlit
      - "5000:5000" # FastAPI
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
      - ./notebooks:/app/notebooks
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    working_dir: /app
    command: >
      bash -c "pip install -r requirements-dev.txt &&
               jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

  # Database for Data Storage
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: aiproject
      POSTGRES_USER: aiuser
      POSTGRES_PASSWORD: aipassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  # Redis for Caching
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MLflow for Experiment Tracking
  mlflow:
    image: python:3.9-slim
    ports:
      - "5001:5000"
    volumes:
      - .:/app
      - mlflow_data:/app/mlruns
    working_dir: /app
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://aiuser:aipassword@postgres:5432/aiproject --default-artifact-root /app/mlruns"

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
```

### 📁 Project Structure Template

```python
print("=== Development Environment Setup Project ===")

project_structure = """
ai_project/
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup
├── pyproject.toml                # Modern Python project config
├── environment.yml               # Conda environment file
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose config
├── .env                          # Environment variables
├── .env.example                  # Environment variables template
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data (never modify)
│   ├── processed/                # Cleaned and processed data
│   ├── external/                 # Data from external sources
│   └── interim/                  # Intermediate data
│
├── src/                          # Source code
│   ├── __init__.py               # Make src a package
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── make_dataset.py       # Data loading and cleaning
│   │   ├── preprocess.py         # Data preprocessing
│   │   └── features.py           # Feature engineering
│   ├── models/                   # ML models
│   │   ├── __init__.py
│   │   ├── train_model.py        # Model training
│   │   ├── predict_model.py      # Model prediction
│   │   └── evaluate_model.py     # Model evaluation
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py            # General helpers
│   │   └── visualization.py      # Plotting functions
│   └── api/                      # API endpoints
│       ├── __init__.py
│       ├── app.py                # Main API application
│       └── routes.py             # API routes
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── tests/                        # Test directory
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_utils.py
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── CONTRIBUTING.md
│   └── API.md
│
├── scripts/                      # Executable scripts
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── deploy_model.py
│
└── models/                       # Trained model artifacts
    ├── .gitkeep                  # Keep directory in git
    ├── model.pkl                 # Serialized model
    └── model_config.json         # Model configuration
"""

print("Recommended project structure:")
print(project_structure)

print("\n=== Quick Setup Script ===")

setup_script = '''#!/bin/bash
# setup.sh - One-command project setup

echo "🚀 Setting up AI development environment..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
else
    pip install -r requirements.txt
fi

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "🔀 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: AI project setup"
fi

echo "✅ Setup complete! Activate your environment with:"
echo "source venv/bin/activate  # On Linux/Mac"
echo "venv\\\\Scripts\\\\activate     # On Windows"
echo ""
echo "🎯 Start coding by running:"
echo "jupyter notebook           # For notebooks"
echo "python -m pytest tests/    # For running tests"
'''

print("Setup script:")
print(setup_script)

print("\n=== Environment Management Best Practices ===")

best_practices = """
1. **One Project, One Environment**
   - Always use isolated environments for each project
   - Never install packages globally for project-specific work

2. **Pin Your Dependencies**
   - Use exact versions in production: pandas==1.5.0
   - Use version ranges in development: pandas>=1.5.0
   - Keep requirements.txt updated

3. **Use Environment Files**
   - Create environment.yml for conda projects
   - Keep requirements.txt for pip projects
   - Document all dependencies

4. **Version Control Your Environment**
   - Commit environment files to git
   - Include setup scripts for reproducibility
   - Document installation steps

5. **Development vs Production**
   - Separate development dependencies
   - Use different requirements files
   - Test production environments

6. **Containerize for Consistency**
   - Use Docker for production deployment
   - Ensure same environment everywhere
   - Document containerization steps

7. **Backup Your Environments**
   - Export conda environments
   - Keep requirements.txt in version control
   - Document platform-specific setup

8. **Regular Updates**
   - Update dependencies regularly
   - Test with newer versions
   - Maintain backward compatibility
"""

print(best_practices)

print("\nEnvironment management completed!")
```

---

## Installation Guides {#installation-guides}

### Complete Installation for All Platforms

```python
print("=== Complete Installation Guide ===")

print("\n=== Windows 10/11 Installation ===")

windows_guide = """
1. Install Python:
   - Download from https://python.org/downloads/
   - Run installer, check "Add to PATH"
   - Verify: python --version, pip --version

2. Install Git:
   - Download from https://git-scm.com/download/win
   - Use default settings
   - Verify: git --version

3. Install Visual Studio Code:
   - Download from https://code.visualstudio.com/
   - Install with default settings
   - Install Python extension

4. Install Anaconda:
   - Download from https://anaconda.com/download
   - Run installer, check "Add to PATH"
   - Verify: conda --version

5. Create Virtual Environment:
   - Open Command Prompt
   - conda create -n ai_env python=3.9
   - conda activate ai_env
   - pip install pandas numpy matplotlib scikit-learn jupyter

6. Install Additional Tools:
   - pip install torch torchvision tensorflow
   - pip install opencv-python
   - pip install nltk spacy
   - python -m spacy download en_core_web_sm

Troubleshooting:
- If pip fails, try: python -m pip install --upgrade pip
- If conda fails, try: conda clean --all
- If VS Code doesn't find Python, restart and check interpreter
"""

print(windows_guide)

print("\n=== macOS Installation ===")

macos_guide = """
1. Install Homebrew (if not installed):
   - /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   - Verify: brew --version

2. Install Python:
   - brew install python@3.9
   - Or: brew install python (latest version)
   - Verify: python3 --version, pip3 --version

3. Install Git:
   - brew install git
   - Verify: git --version

4. Install Visual Studio Code:
   - brew install --cask visual-studio-code
   - Or download from website
   - Install Python extension

5. Install Anaconda:
   - brew install anaconda
   - Or download installer from website
   - Verify: conda --version

6. Create Virtual Environment:
   - python3 -m venv ai_env
   - source ai_env/bin/activate
   - pip install pandas numpy matplotlib scikit-learn jupyter

7. Install Additional Tools:
   - Same as Windows, but use pip3 instead of pip
"""

print(macos_guide)

print("\n=== Ubuntu/Debian Installation ===")

ubuntu_guide = """
1. Update system:
   - sudo apt update && sudo apt upgrade -y

2. Install Python and pip:
   - sudo apt install python3 python3-pip python3-venv -y
   - Verify: python3 --version, pip3 --version

3. Install Git:
   - sudo apt install git -y
   - Verify: git --version

4. Install Visual Studio Code:
   - wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
   - sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
   - sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
   - sudo apt update && sudo apt install code -y

5. Install Anaconda:
   - wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
   - bash Anaconda3-2023.03-Linux-x86_64.sh
   - Verify: conda --version

6. Create Virtual Environment:
   - python3 -m venv ai_env
   - source ai_env/bin/activate
   - pip install pandas numpy matplotlib scikit-learn jupyter

7. Install Development Tools:
   - sudo apt install build-essential python3-dev -y
"""

print(ubuntu_guide)

print("\n=== Verification Script ===")

verification_script = '''
# Create verify_installation.py
import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

# Check Python
print(f"Python version: {sys.version}")

# Check essential packages
packages = [
    ("pip", "pip"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("jupyter", "jupyter"),
    ("tensorflow", "tensorflow"),
    ("torch", "torch"),
    ("opencv", "cv2"),
    ("nltk", "nltk"),
    ("spacy", "spacy"),
]

print("\\nChecking packages:")
for package, import_name in packages:
    check_package(package, import_name)

# Check system tools
tools = ["git", "conda", "code"]
for tool in tools:
    try:
        result = subprocess.run([tool, "--version"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ {tool} is installed")
        else:
            print(f"✗ {tool} is NOT installed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"✗ {tool} is NOT installed or not in PATH")
'''

print("Verification script:")
print(verification_script)

print("\nInstallation guides completed!")
```

---

## Development Workflows {#development-workflows}

### Best Practices for AI/ML Development

```python
print("=== Development Workflows: Best Practices ===")

workflow_examples = {
    "Data Science Workflow": """
1. Problem Definition
   - Define business objective
   - Identify success metrics
   - Understand data requirements

2. Data Collection
   - Gather relevant datasets
   - Document data sources
   - Check data quality

3. Data Exploration (EDA)
   - Understand data structure
   - Identify patterns and outliers
   - Visualize relationships

4. Data Preprocessing
   - Handle missing values
   - Scale and normalize features
   - Create new features

5. Model Selection
   - Try multiple algorithms
   - Use cross-validation
   - Compare performance metrics

6. Model Training
   - Split data appropriately
   - Tune hyperparameters
   - Avoid overfitting

7. Model Evaluation
   - Use appropriate metrics
   - Test on unseen data
   - Validate assumptions

8. Deployment
   - Save model
   - Create API
   - Monitor performance

9. Maintenance
   - Monitor data drift
   - Retrain periodically
   - Update as needed
""",

    "Research Workflow": """
1. Literature Review
   - Read relevant papers
   - Understand state-of-the-art
   - Identify research gaps

2. Hypothesis Formation
   - Define research questions
   - Formulate hypotheses
   - Plan experiments

3. Experimental Design
   - Design controlled experiments
   - Plan data collection
   - Define evaluation metrics

4. Implementation
   - Code the solution
   - Implement baseline methods
   - Document the process

5. Experimentation
   - Run experiments
   - Collect results
   - Analyze findings

6. Analysis
   - Interpret results
   - Compare with baselines
   - Draw conclusions

7. Documentation
   - Write paper/draft
   - Create visualizations
   - Prepare presentations

8. Peer Review
   - Share with colleagues
   - Get feedback
   - Revise accordingly
""",

    "Production Workflow": """
1. Requirements Gathering
   - Define functional requirements
   - Understand constraints
   - Plan infrastructure

2. Architecture Design
   - Design system architecture
   - Plan data flow
   - Define APIs

3. Development
   - Implement features
   - Write tests
   - Follow coding standards

4. Testing
   - Unit tests
   - Integration tests
   - Performance tests

5. Deployment
   - Staging environment
   - Production deployment
   - Monitoring setup

6. Monitoring
   - Track performance
   - Monitor errors
   - Check data quality

7. Maintenance
   - Regular updates
   - Bug fixes
   - Feature enhancements
"""
}

for workflow, description in workflow_examples.items():
    print(f"\n=== {workflow} ===")
    print(description)

print("\n=== Code Organization ===")

project_structure = """
ai_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── clean_data.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_utils.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── model_utils.py
│   └── visualization/
│       └── visualize.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_features.py
├── requirements.txt
├── environment.yml
├── README.md
├── setup.py
├── .gitignore
└── .flake8
"""

print("Recommended project structure:")
print(project_structure)

print("\n=== Version Control Best Practices ===")

git_practices = """
1. Initialize Repository
   git init
   git add .
   git commit -m "Initial commit"

2. Create .gitignore
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   env/
   venv/
   .env
   .venv
   pip-log.txt
   pip-delete-this-directory.txt
   .pytest_cache/
   .mypy_cache/
   .coverage
   .coverage.*
   htmlcov/
   .tox/
   .nox/
   .DS_Store
   *.egg-info/
   dist/
   build/
   data/raw/
   models/*.pkl
   *.log

3. Branch Strategy
   git checkout -b feature/new-model
   git checkout -b bugfix/data-loading-issue
   git checkout -b experiment/transformer-architecture

4. Commit Messages
   feat: add new classification model
   fix: resolve data preprocessing bug
   docs: update API documentation
   style: format code with black
   refactor: simplify feature selection logic
   test: add unit tests for data loader
   chore: update requirements.txt

5. Regular Commits
   - Commit frequently with meaningful messages
   - Push to remote regularly
   - Pull before starting work
"""

print("Git best practices:")
print(git_practices)

print("\nDevelopment workflows completed!")
```

---

## Best Practices & Troubleshooting {#best-practices}

### Common Issues and Solutions

```python
print("=== Best Practices & Troubleshooting ===")

print("\n=== Python Best Practices ===")

python_practices = """
1. Code Style
   - Use Black for formatting
   - Follow PEP 8 guidelines
   - Write descriptive variable names
   - Use type hints when possible

2. Error Handling
   - Use try-except blocks
   - Catch specific exceptions
   - Log errors appropriately
   - Provide meaningful error messages

3. Testing
   - Write unit tests for functions
   - Test edge cases
   - Use pytest for testing
   - Mock external dependencies

4. Documentation
   - Write docstrings for functions
   - Comment complex logic
   - Update README files
   - Document APIs

5. Performance
   - Use vectorized operations (NumPy/Pandas)
   - Profile code to find bottlenecks
   - Use appropriate data structures
   - Consider memory usage

6. Security
   - Never hardcode secrets
   - Validate input data
   - Use parameterized queries
   - Keep dependencies updated
"""

print(python_practices)

print("\n=== ML-Specific Best Practices ===")

ml_practices = """
1. Data Management
   - Version control your data
   - Document data sources
   - Track data lineage
   - Handle missing data appropriately

2. Model Development
   - Start with simple baselines
   - Use cross-validation
   - Monitor for overfitting
   - Keep detailed experiment logs

3. Feature Engineering
   - Scale/normalize features
   - Handle categorical variables
   - Create meaningful features
   - Feature selection techniques

4. Model Evaluation
   - Use appropriate metrics
   - Test on multiple datasets
   - Consider class imbalance
   - Validate assumptions

5. Production Deployment
   - Monitor model performance
   - Track data drift
   - Implement A/B testing
   - Plan for model updates

6. Reproducibility
   - Set random seeds
   - Save model parameters
   - Document environment
   - Version everything
"""

print(ml_practices)

print("\n=== Common Error Solutions ===")

error_solutions = {
    "ImportError": """
Problem: Module not found
Solutions:
1. Check if module is installed: pip list | grep module_name
2. Install missing module: pip install module_name
3. Check Python path: import sys; print(sys.path)
4. Use virtual environment
5. Check spelling of module name
""",

    "CUDA Out of Memory": """
Problem: GPU memory insufficient
Solutions:
1. Reduce batch size
2. Use gradient accumulation
3. Enable memory growth: torch.cuda.set_per_process_memory_fraction()
4. Clear GPU cache: torch.cuda.empty_cache()
5. Use mixed precision training
""",

    "ValueError": """
Problem: Invalid value provided
Solutions:
1. Check input data types
2. Validate input ranges
3. Handle NaN values
4. Check array shapes
5. Convert data types appropriately
""",

    "MemoryError": """
Problem: Insufficient memory
Solutions:
1. Use data generators
2. Process data in chunks
3. Use memory-efficient data types
4. Clear variables: del variable_name
5. Restart kernel
""",

    "PermissionError": """
Problem: Access denied
Solutions:
1. Check file permissions
2. Run with appropriate privileges
3. Check if file is in use
4. Use correct file paths
5. Check directory existence
""",

    "TimeoutError": """
Problem: Operation timed out
Solutions:
1. Increase timeout value
2. Optimize code performance
3. Use async operations
4. Process data in batches
5. Implement progress tracking
"""
}

for error, solution in error_solutions.items():
    print(f"\n--- {error} ---")
    print(solution)

print("\n=== Performance Optimization ===")

optimization_tips = """
1. Data Loading
   - Use efficient data formats (Parquet, HDF5)
   - Implement data generators
   - Use memory mapping for large files
   - Parallel data loading with num_workers

2. Model Training
   - Use appropriate batch sizes
   - Enable mixed precision training
   - Use multiple GPUs with DataParallel
   - Implement early stopping

3. Memory Management
   - Clear intermediate variables
   - Use inplace operations
   - Profile memory usage
   - Implement garbage collection

4. Code Optimization
   - Vectorize operations
   - Use NumPy/Pandas optimizations
   - Avoid loops in Python
   - Use compiled extensions

5. Infrastructure
   - Use SSD storage
   - Ensure adequate RAM
   - Use GPU for training
   - Consider cloud computing
"""

print(optimization_tips)

print("\n=== Troubleshooting Checklist ===")

checklist = """
When encountering issues:

1. Environment
   □ Python version compatible?
   □ All dependencies installed?
   □ Virtual environment activated?
   □ Path settings correct?

2. Data
   □ File paths correct?
   □ Data format as expected?
   □ Missing values handled?
   □ Data types correct?

3. Code
   □ Syntax errors checked?
   □ Import statements correct?
   □ Variable names consistent?
   □ Function arguments valid?

4. Resources
   □ Sufficient memory?
   □ GPU available if needed?
   □ Disk space adequate?
   □ Network connectivity?

5. Debugging
   □ Error messages read carefully?
   □ Stack trace examined?
   □ Logs checked?
   □ Code step-through debugging?
"""

print(checklist)

print("\n=== Getting Help ===")

help_resources = """
When stuck:

1. Documentation
   - Official documentation
   - API references
   - Tutorials and guides

2. Community
   - Stack Overflow
   - GitHub issues
   - Discord/Slack communities
   - Reddit (r/MachineLearning, r/Python)

3. Courses
   - Coursera, edX, Udacity
   - YouTube tutorials
   - Documentation tutorials

4. Books
   - "Python for Data Analysis" by Wes McKinney
   - "Hands-On Machine Learning" by Aurélien Géron
   - "Pattern Recognition and Machine Learning" by Christopher Bishop

5. Professional Help
   - Consult with colleagues
   - Hire a mentor
   - Join professional networks
   - Attend conferences/meetups
"""

print(help_resources)

print("\nBest practices and troubleshooting completed!")

print("\n=== AI Tools, Libraries & Development Environment Guide Complete ===")
print(f"Total sections covered: 16 major sections")
print(f"Content: Comprehensive installation, setup, and best practices guide")
print(f"Purpose: Complete toolkit for AI/ML development")
```

---

## Summary: What You've Learned

This comprehensive guide has equipped you with the complete AI/ML development toolkit:

**Core Libraries Mastered:**

- **scikit-learn**: Complete ML algorithms with 100+ examples
- **TensorFlow 2.x**: Deep learning framework with Keras API
- **PyTorch**: Flexible neural network framework
- **Hugging Face**: State-of-the-art pre-trained models
- **OpenCV**: Computer vision and image processing
- **NLTK/spaCy**: Natural language processing
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib/seaborn**: Data visualization and plotting

**Development Environment:**

- **Jupyter Notebooks**: Interactive development and analysis
- **VS Code**: Professional development environment
- **Environment Management**: conda, pip, poetry, Docker
- **Installation Guides**: Complete setup for Windows, macOS, Linux

**Best Practices:**

- Development workflows for data science, research, and production
- Code organization and version control
- Performance optimization techniques
- Common error troubleshooting
- Security and reproducibility guidelines

This toolkit provides everything needed to build, train, deploy, and maintain AI/ML systems from research prototypes to production applications. The knowledge gained here forms the foundation for advanced AI development and will accelerate your journey in artificial intelligence and machine learning.
This toolkit provides everything needed to build, train, deploy, and maintain AI/ML systems from research prototypes to production applications. The knowledge gained here forms the foundation for advanced AI development and will accelerate your journey in artificial intelligence and machine learning.

---

## 🤯 Common Confusions & Solutions

### 1. Library vs Framework Confusion

**Problem**: Not understanding the difference between libraries and frameworks

```python
# Library - You call its functions
import numpy as np
array = np.array([1, 2, 3])  # You control the flow

# Framework - It calls your functions (inversion of control)
# TensorFlow/Keras
model = Sequential()  # Framework structure
model.add(Dense(64, activation='relu'))  # Framework calls your code
```

### 2. scikit-learn vs TensorFlow Choice

**Problem**: Not knowing which library to use for different tasks

```python
# Use scikit-learn for traditional ML
from sklearn.ensemble import RandomForestClassifier
# Great for: Classification, regression, clustering with structured data

# Use TensorFlow for deep learning
import tensorflow as tf
# Great for: Neural networks, deep learning, complex models
```

### 3. Environment Management Confusion

**Problem**: Installing packages globally vs in virtual environments

```python
# Wrong ❌ - Global installation causes conflicts
pip install tensorflow pandas numpy  # Might break other projects

# Correct ✅ - Use virtual environments
python -m venv myproject_env
source myproject_env/bin/activate  # Linux/Mac
pip install tensorflow pandas numpy  # Only affects this project
```

### 4. Jupyter vs Regular Python Scripts

**Problem**: Not knowing when to use each development environment

```python
# Use Jupyter for: Exploratory data analysis, prototyping, visualization
# Great for: Step-by-step analysis, immediate feedback, documentation

# Use Python scripts for: Production code, automation, batch processing
# Great for: Reusable code, version control, automation
```

### 5. Library Version Compatibility

**Problem**: Different library versions not working together

```python
# Create requirements.txt to track compatible versions
# requirements.txt
numpy==1.24.0
pandas==1.5.0
scikit-learn==1.2.0
tensorflow==2.12.0

# Install specific versions
pip install -r requirements.txt
```

### 6. Memory Issues with Large Datasets

**Problem**: Running out of memory when processing large datasets

```python
# Wrong ❌ - Loading entire dataset at once
data = pd.read_csv('huge_file.csv')  # Might use all memory

# Correct ✅ - Process in chunks
for chunk in pd.read_csv('huge_file.csv', chunksize=1000):
    process(chunk)  # Process one chunk at a time
```

### 7. Model Training Performance

**Problem**: Not optimizing for training speed and memory usage

```python
# Use appropriate data types
# Wrong ❌
data = np.array([1, 2, 3], dtype=np.float64)  # Uses more memory

# Correct ✅
data = np.array([1, 2, 3], dtype=np.float32)  # Half the memory
```

### 8. Library Documentation Overwhelm

**Problem**: Documentation seems too complex for beginners

```python
# Start with simple examples, not the entire documentation

# matplotlib basic example
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()

# pandas basic example
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

---

## 🧠 Micro-Quiz: Test Your Knowledge

### Question 1

What's the main difference between a library and a framework?
A) No difference - same thing
B) Library you call, framework calls your code ✅
C) Library is faster
D) Framework is only for web development

### Question 2

When should you use scikit-learn vs TensorFlow?
A) Always use scikit-learn
B) Always use TensorFlow
C) Use scikit-learn for traditional ML, TensorFlow for deep learning ✅
D) They are interchangeable

### Question 3

What does this code do?

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

A) Creates a new CSV file
B) Reads and displays the first few rows of a CSV file ✅
C) Deletes a CSV file
D) Edits a CSV file

### Question 4

How do you install a specific version of a library?
A) pip install library_name
B) pip install library_name==version ✅
C) pip install --latest library_name
D) pip update library_name

### Question 5

What is the main advantage of using virtual environments?
A) Better performance
B) Isolated package installations ✅
C) Automatic code writing
D) Faster internet

### Question 6

When should you use Jupyter Notebooks vs Python scripts?
A) Jupyter for exploration, scripts for production ✅
B) Always use Jupyter
C) Always use scripts
D) No difference

**Mastery Requirement: 5/6 questions correct (83%)**

---

## 💭 Reflection Prompts

### 1. Library Selection for Real Problems

Think about problems you've encountered:

- If you needed to analyze test scores for your class, which library would you use?
- If you wanted to create a chatbot, which libraries would you need?
- If you needed to process images (like sorting photos), what tools would help?
- How do you decide which tool is right for which job?

### 2. Development Environment Strategy

Consider your learning style and project needs:

- Do you prefer step-by-step exploration (Jupyter) or structured code (scripts)?
- How do you stay organized when working on multiple projects?
- What strategies help you remember which libraries to use for which tasks?
- How do you keep track of different project configurations?

### 3. Learning Path and Skill Development

Reflect on your learning journey:

- Which libraries excite you most and why?
- What projects would help you practice the libraries you've learned?
- How do you balance learning new tools vs mastering existing ones?
- What learning resources work best for you (documentation, tutorials, examples)?

---

## 🏃‍♂️ Mini Sprint Project: Data Analysis Toolkit

**Time Limit: 30 minutes**

**Challenge**: Create a comprehensive data analysis toolkit using multiple libraries.

**Requirements**:

- Use pandas to load and analyze sample data (student grades, survey data, or sales data)
- Use numpy for mathematical operations and calculations
- Use matplotlib to create at least 3 different types of visualizations
- Use scikit-learn for basic machine learning (classification or regression)
- Create a virtual environment and install all required libraries
- Include proper error handling and documentation

**Starter Code**:

```python
# data_analysis_toolkit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_analyze_data():
    """Load and perform basic analysis on sample data"""
    # Your code here - create or load sample data

def create_visualizations(data):
    """Create multiple types of charts"""
    # Your code here - at least 3 different chart types

def build_ml_model(data):
    """Build and evaluate a simple machine learning model"""
    # Your code here - use scikit-learn for basic ML

def main():
    """Main function to run the complete analysis"""
    # Your code here - orchestrate all functions

if __name__ == "__main__":
    main()
```

**Success Criteria**:
✅ Virtual environment created and properly configured
✅ All required libraries imported and working
✅ Data loaded and analyzed with pandas
✅ Multiple visualization types created with matplotlib
✅ Simple machine learning model built with scikit-learn
✅ Code includes error handling and documentation
✅ Results are clearly presented and interpreted

---

## 🚀 Full Project Extension: Complete AI/ML Development Environment

**Time Investment: 3-4 hours**

**Project Overview**: Build a comprehensive AI/ML development environment that integrates multiple libraries and tools for a complete data science and machine learning workflow.

**Core System Components**:

### 1. Data Pipeline Manager

```python
class DataPipeline:
    def __init__(self):
        self.steps = []
        self.data = None

    def add_step(self, step_function):
        """Add processing step to pipeline"""
        # Data validation
        # Cleaning
        # Transformation
        # Feature engineering

    def execute_pipeline(self, input_data):
        """Execute all pipeline steps"""
        # Apply each step in sequence
        # Handle errors gracefully
        # Return processed data
```

### 2. Model Training Framework

```python
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        """Add model to training framework"""
        # Support for scikit-learn models
        # Support for TensorFlow/Keras models
        # Support for PyTorch models

    def train_all_models(self, X_train, y_train):
        """Train all models with cross-validation"""
        # Parallel training
        # Hyperparameter tuning
        # Performance tracking

    def compare_models(self, X_test, y_test):
        """Compare model performance"""
        # Accuracy, precision, recall
        # Visual comparison
        # Best model recommendation
```

### 3. Visualization Dashboard

```python
class VisualizationDashboard:
    def __init__(self):
        self.charts = {}

    def create_data_exploration_charts(self, data):
        """Create comprehensive data exploration visualizations"""
        # Distribution plots
        # Correlation heatmaps
        # Pair plots
        # Box plots

    def create_model_performance_charts(self, model_results):
        """Create model performance visualizations"""
        # ROC curves
        # Confusion matrices
        # Feature importance plots
        # Learning curves
```

### 4. Environment Manager

```python
class EnvironmentManager:
    def __init__(self):
        self.environments = {}

    def create_project_environment(self, project_name, requirements):
        """Create isolated environment for each project"""
        # Virtual environment creation
        # Package installation
        # Environment validation

    def export_environment(self, project_name):
        """Export environment configuration"""
        # requirements.txt generation
        # environment.yml creation
        # Documentation generation
```

**Real-World Applications**:

### 1. Business Intelligence Dashboard

- **Data Sources**: Sales data, customer data, market data
- **Analysis**: Trend analysis, customer segmentation, forecasting
- **Visualizations**: Interactive dashboards, KPI tracking, alerts
- **ML Applications**: Sales prediction, customer churn, recommendation systems

### 2. Scientific Research Platform

- **Data Processing**: Experimental data, survey responses, sensor data
- **Analysis**: Statistical analysis, hypothesis testing, pattern discovery
- **Visualizations**: Research plots, statistical distributions, correlation analysis
- **ML Applications**: Classification, regression, clustering for research insights

### 3. Quality Control System

- **Data Collection**: Manufacturing data, quality metrics, defect tracking
- **Analysis**: Quality trends, anomaly detection, process optimization
- **Visualizations**: Control charts, quality dashboards, trend analysis
- **ML Applications**: Defect prediction, quality classification, process optimization

### 4. Personal Finance Advisor

- **Data Integration**: Bank statements, investment data, market data
- **Analysis**: Spending patterns, investment performance, budget analysis
- **Visualizations**: Expense tracking, investment growth, budget planning
- **ML Applications**: Spending prediction, investment recommendations, fraud detection

**Advanced Features**:

### Model Versioning and Management

```python
class ModelManager:
    def __init__(self):
        self.model_registry = {}

    def register_model(self, model, metadata):
        """Register trained model with metadata"""
        # Model version tracking
        # Performance metrics storage
        # Hyperparameter documentation

    def load_model(self, model_name, version):
        """Load specific model version"""
        # Model validation
        # Compatibility checking
        # Performance verification
```

### Experiment Tracking

```python
class ExperimentTracker:
    def __init__(self):
        self.experiments = {}

    def log_experiment(self, experiment_name, parameters, results):
        """Log experiment details"""
        # Parameter tracking
        # Results storage
        # Comparison capabilities

    def compare_experiments(self, experiment_list):
        """Compare multiple experiments"""
        # Performance comparison
        # Statistical significance testing
        # Best experiment identification
```

### Automated Reporting

```python
class ReportGenerator:
    def __init__(self, pipeline, dashboard):
        self.pipeline = pipeline
        self.dashboard = dashboard

    def generate_comprehensive_report(self, data, models):
        """Generate professional analysis report"""
        # Data summary
        # Model performance
        # Recommendations
        # Next steps

    def create_executive_summary(self, results):
        """Create high-level summary for stakeholders"""
        # Key findings
        # Business impact
        # Action items
```

**Deployment and Production Features**:

### API Development

```python
class ModelAPI:
    def __init__(self, model):
        self.model = model
        self.app = Flask(__name__)

    def create_prediction_endpoint(self):
        """Create REST API for model predictions"""
        # Input validation
        # Prediction generation
        # Response formatting
        # Error handling

    def deploy(self, host='0.0.0.0', port=5000):
        """Deploy model as web service"""
        # API documentation
        # Testing capabilities
        # Monitoring setup
```

### Monitoring and Alerting

```python
class ModelMonitor:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.alerts = []

    def monitor_performance(self, new_data, predictions):
        """Monitor model performance in production"""
        # Data drift detection
        # Performance degradation alerts
        # Retraining recommendations

    def generate_alerts(self, issue_type, details):
        """Generate and send alerts"""
        # Email notifications
        # Dashboard alerts
        # Automated responses
```

**Success Criteria**:
✅ Complete data pipeline from raw data to insights
✅ Multi-library integration (pandas, numpy, sklearn, tensorflow, etc.)
✅ Comprehensive model training and evaluation framework
✅ Professional visualization dashboard
✅ Environment management and reproducibility
✅ Model versioning and experiment tracking
✅ Automated reporting and documentation
✅ API development and deployment capabilities
✅ Production monitoring and alerting
✅ Real-world application integration
✅ Professional code quality and documentation
✅ Scalable and maintainable architecture

**Learning Outcomes**:

- Master complete data science and ML workflow
- Integrate multiple libraries effectively
- Build production-ready ML systems
- Understand experiment tracking and model management
- Develop skills in system architecture and design
- Create professional documentation and reporting
- Build deployment and monitoring capabilities
- Understand MLOps and production ML challenges

**Portfolio Impact**: This project demonstrates advanced data science and ML engineering skills, including system design, production deployment, and professional development practices. It showcases the ability to build end-to-end ML solutions and serves as an excellent demonstration of enterprise-level software development skills in the AI/ML domain.
