# AI/ML Intermediate Tools & Libraries Cheat Sheet

## Table of Contents

- [Installation Commands](#installation-commands)
- [NumPy](#numpy)
- [Pandas](#pandas)
- [Matplotlib](#matplotlib)
- [Scikit-learn](#scikit-learn)
- [TensorFlow](#tensorflow)
- [PyTorch](#pytorch)
- [Code Templates](#code-templates)
- [Performance Tips](#performance-tips)

---

## Installation Commands

### Core Libraries

```bash
# NumPy
pip install numpy

# Pandas
pip install pandas

# Matplotlib
pip install matplotlib

# Scikit-learn
pip install scikit-learn

# TensorFlow
pip install tensorflow

# PyTorch
pip install torch torchvision torchaudio

# With CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Additional Dependencies

```bash
# Data manipulation and visualization
pip install seaborn plotly jupyter

# Image processing
pip install opencv-python pillow

# Scientific computing
pip install scipy statsmodels

# Model deployment
pip install flask fastapi uvicorn
```

---

## NumPy

### Basic Operations

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)
random = np.random.rand(3, 3)
random_int = np.random.randint(0, 10, (2, 3))

# Array properties
arr.shape       # (5,)
arr.dtype       # dtype('int64')
arr.ndim        # 1
arr.size        # 5

# Reshaping
arr.reshape(5, 1)    # Column vector
arr.reshape(1, 5)    # Row vector
arr.reshape(-1, 1)   # Auto-reshape to column

# Indexing and slicing
arr[0]              # First element
arr[-1]             # Last element
arr[1:4]            # Elements 1 to 3
arr[::2]            # Every other element
arr[arr > 3]        # Boolean indexing

# Mathematical operations
np.sum(arr)                    # Sum all elements
np.mean(arr)                   # Mean
np.std(arr)                    # Standard deviation
np.min(arr), np.max(arr)       # Min/Max
np.argmin(arr), np.argmax(arr) # Indices of min/max

# Matrix operations
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)

np.dot(A, B)           # Matrix multiplication
A @ B                  # Matrix multiplication (alternative)
A.T                    # Transpose
np.linalg.inv(A)       # Inverse
np.linalg.eig(A)       # Eigenvalues and vectors
```

### Advanced Features

```python
# Broadcasting
arr + 5                # Add 5 to each element
A + np.ones((3, 1))    # Add column vector to matrix

# Concatenation
np.concatenate([arr1, arr2])      # Along axis 0
np.vstack([arr1, arr2])           # Vertical stack
np.hstack([arr1, arr2])           # Horizontal stack

# Stack, split
np.stack([arr1, arr2], axis=1)    # Stack along axis 1
np.split(arr, 3)                  # Split into 3 parts

# Unique values
np.unique(arr)                    # Unique elements and counts
np.unique(arr, return_counts=True)

# Sorting
np.sort(arr)                      # Sorted array
np.argsort(arr)                   # Indices of sorted elements
```

---

## Pandas

### DataFrame Creation and Basic Operations

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
    'C': [1.5, 2.5, 3.5]
})

# From files
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_json('file.json')

# Display and info
df.head()         # First 5 rows
df.info()         # Data types and null counts
df.describe()     # Statistical summary
df.shape          # (rows, columns)
df.columns        # Column names
df.dtypes         # Data types

# Indexing and selection
df['A']                       # Single column (Series)
df[['A', 'B']]                # Multiple columns
df.loc[0]                     # Row by label
df.iloc[0]                    # Row by position
df.loc[0, 'A']                # Specific value
df.iloc[0:2, 0:2]             # Subset by position

# Boolean indexing
df[df['A'] > 1]                      # Filter rows
df[(df['A'] > 1) & (df['B'] == 'a')] # Multiple conditions
df.query('A > 1 and B == "a"')       # Query method
```

### Data Manipulation

```python
# Missing values
df.isnull()                # Boolean mask of nulls
df.dropna()                # Remove rows with nulls
df.fillna(0)               # Fill nulls with value
df.fillna(method='ffill')  # Forward fill

# Duplicates
df.duplicated()            # Boolean mask of duplicates
df.drop_duplicates()       # Remove duplicates

# Sorting
df.sort_values('A')                # Sort by column
df.sort_values(['A', 'B'], ascending=[True, False])

# Grouping
df.groupby('B').mean()             # Group and aggregate
df.groupby(['B', 'C']).agg({'A': 'mean', 'B': 'sum'})

# Merging
pd.merge(df1, df2, on='key')       # Inner join
df1.merge(df2, on='key', how='left')  # Left join
pd.concat([df1, df2])              # Concatenate
```

### Data Analysis

```python
# Value counts
df['B'].value_counts()             # Frequency counts

# Correlation
df.corr()                          # Correlation matrix

# Pivot tables
df.pivot_table(values='A', index='B', columns='C', aggfunc='mean')

# Time series
df['date'] = pd.to_datetime(df['date'])
df.set_index('date').resample('M').mean()

# Apply functions
df['A'].apply(lambda x: x * 2)     # Apply function
df.apply(np.mean)                  # Apply to columns
```

---

## Matplotlib

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, label='sin(x)', linewidth=2, color='blue')

# Scatter plot
ax.scatter(x, y, c='red', s=50, alpha=0.6)

# Bar plot
categories = ['A', 'B', 'C']
values = [1, 2, 3]
ax.bar(categories, values, color=['red', 'green', 'blue'])

# Histogram
ax.hist(data, bins=30, alpha=0.7, color='skyblue')

# Multiple plots
ax.plot(x, np.sin(x), label='sin')
ax.plot(x, np.cos(x), label='cos')
ax.legend()
ax.grid(True)
ax.set_title('Trigonometric Functions')
ax.set_xlabel('x')
ax.set_ylabel('y')
```

### Advanced Plotting

```python
# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data, bins=20)

# Heatmap
from sklearn.datasets import load_iris
iris = load_iris()
correlation_matrix = np.corrcoef(iris.data.T)
im = ax.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar(im)

# Box plot
ax.boxplot([data1, data2, data3], labels=['X', 'Y', 'Z'])

# Violin plot
parts = ax.violinplot([data1, data2, data3], positions=[1, 2, 3])

# Customize appearance
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Save plots
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Seaborn (Statistical Visualization)

```python
import seaborn as sns

# Load dataset
tips = sns.load_dataset('tips')

# Styling
sns.set_style("whitegrid")
sns.set_palette("husl")

# Plots
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day')
sns.lineplot(data=tips, x='day', y='total_bill', estimator='mean')
sns.barplot(data=tips, x='day', y='total_bill')
sns.histplot(data=tips, x='total_bill', kde=True)
sns.boxplot(data=tips, x='day', y='total_bill')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
sns.pairplot(tips)  # Pairwise relationships

# Regression plots
sns.regplot(data=tips, x='total_bill', y='tip')
sns.lmplot(data=tips, x='total_bill', y='tip', col='day')
```

---

## Scikit-learn

### Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max scaling
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)

# Encoding categorical variables
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(categorical_data)
```

### Model Selection and Evaluation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
```

### Classification Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Support Vector Machine
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)
```

### Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Support Vector Regressor
svr = SVR(kernel='rbf', C=1.0)
svr.fit(X_train, y_train)
```

### Clustering

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
silhouette_avg = silhouette_score(X, clusters)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

---

## TensorFlow

### Basic Setup and Operations

```python
import tensorflow as tf
from tensorflow import keras

# Check version
print(f"TensorFlow version: {tf.__version__}")

# Create tensors
tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 3])
random = tf.random.normal([3, 3])

# Tensor operations
result = tf.matmul(tensor, tensor)
result = tf.nn.softmax(tensor)
result = tf.reduce_mean(tensor, axis=0)

# GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {len(gpus)} devices")
else:
    print("No GPU detected")
```

### Model Building

```python
# Sequential API
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Model summary
model.summary()
```

### Training and Evaluation

```python
# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    factor=0.2, patience=5
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Predict
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

### Custom Training Loop

```python
# Training function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Custom training loop
for epoch in range(epochs):
    for batch_x, batch_y in train_dataset:
        loss = train_step(batch_x, batch_y)
    print(f'Epoch {epoch+1}: Loss = {loss:.4f}')
```

### Model Saving and Loading

```python
# Save model
model.save('model.h5')
model.save_weights('model_weights')

# Load model
loaded_model = keras.models.load_model('model.h5')
loaded_model.load_weights('model_weights')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

---

## PyTorch

### Basic Operations

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create tensors
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
random = torch.randn(3, 3)

# Tensor operations
result = torch.matmul(tensor, tensor)
result = torch.softmax(tensor, dim=1)
result = torch.mean(tensor, dim=0)

# Move to device
tensor = tensor.to(device)
```

### Model Definition

```python
# Using nn.Module
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create model
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params}')
```

### Training Loop

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
from torch.utils.data import DataLoader, TensorDataset

# Convert to tensors and create dataset
X_tensor = torch.FloatTensor(X_train).to(device)
y_tensor = torch.LongTensor(y_train).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Test accuracy: {accuracy:.4f}')
```

### Advanced Features

```python
# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, alpha=0.5):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        return alpha * mse_loss + (1 - alpha) * l1_loss

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler.step()

# Save and load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

---

## Code Templates

### Classification Template

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def classification_template(X, y, test_size=0.2, random_state=42):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f'CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    return model, scaler

# Usage
# model, scaler = classification_template(X, y)
```

### Regression Template

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def regression_template(X, y, test_size=0.2, random_state=42):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'R² Score: {r2:.4f}')

    return model, scaler

# Usage
# model, scaler = regression_template(X, y)
```

### Neural Network Template (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_nn(X, y, epochs=100, batch_size=32, lr=0.001, device='cpu'):
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    hidden_sizes = [128, 64]  # Can be tuned
    model = NeuralNetwork(input_size, hidden_sizes, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model

# Usage
# model = train_nn(X_train, y_train, epochs=100)
```

### Grid Search Template

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_template(X, y):
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize model
    rf = RandomForestClassifier(random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X, y)

    # Results
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_

# Usage
# best_model = grid_search_template(X_train, y_train)
```

---

## Performance Tips

### General Best Practices

1. **Vectorization**: Use NumPy/Pandas vectorized operations instead of loops
2. **Data Types**: Use appropriate data types (e.g., `float32` instead of `float64` for neural networks)
3. **Memory Management**: Use generators and batch processing for large datasets
4. **Feature Scaling**: Always scale features for algorithms sensitive to scale
5. **Cross-Validation**: Use stratified CV for classification, regular CV for regression

### NumPy Performance

```python
# Bad - using Python loops
for i in range(len(arr)):
    result[i] = arr[i] * 2

# Good - vectorized operation
result = arr * 2

# Use np.dot for matrix operations instead of nested loops
# Pre-allocate arrays when possible
result = np.zeros((m, n))  # Instead of appending in loop
```

### Pandas Performance

```python
# Bad - using apply with custom functions
df['new_col'] = df['col'].apply(lambda x: complex_function(x))

# Good - vectorized operations when possible
df['new_col'] = df['col'] * 2

# For complex operations, use numba or cython
# Use categorical data types for repeated strings
df['category_col'] = df['category_col'].astype('category')
```

### Scikit-learn Performance

```python
# Use n_jobs=-1 for parallel processing
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# For large datasets, use partial_fit when available
# Scale features before training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use appropriate cross-validation strategy
# StratifiedKFold for classification, KFold for regression
```

### TensorFlow/PyTorch Performance

```python
# Use mixed precision training
# TensorFlow
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# PyTorch
model = model.half()

# Use DataLoader with num_workers > 0
train_loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Use model.eval() and torch.no_grad() for inference
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Use @tf.function for TensorFlow function optimization
# Use torch.jit.script for PyTorch JIT compilation
```

### Memory Optimization

```python
# Use generators for large datasets
def data_generator(file_path):
    for chunk in pd.read_csv(file_path, chunksize=1000):
        yield chunk

# Use appropriate data types
# float16 for storage, float32 for computation
# int8 for categorical data

# Clear intermediate variables
import gc
del intermediate_variable
gc.collect()
```

---

## Quick Reference Links

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/)

---

_This cheat sheet covers the essential tools and libraries for AI/ML intermediate practitioners. Use this as a reference guide and remember to always check the latest documentation for updates and best practices._
