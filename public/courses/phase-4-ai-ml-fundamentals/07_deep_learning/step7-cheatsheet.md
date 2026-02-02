# Deep Learning Basic Cheatsheet: Quick Reference Guide

## Table of Contents

1. [Neural Network Basics](#basics)
2. [Activation Functions Reference](#activation-functions)
3. [Architecture Patterns](#architecture)
4. [Quick Code Templates](#code-templates)
5. [Troubleshooting Guide](#troubleshooting)
6. [Performance Optimization](#optimization)
7. [Common Formulas](#formulas)
8. [Best Practices](#best-practices)

---

## Neural Network Basics {#basics}

### Core Concepts

**Neuron Structure:**

```python
# Single neuron
output = activation(np.dot(weights, inputs) + bias)
```

**Network Types:**

- **Perceptron**: Single layer, binary classification
- **MLP (Multi-Layer Perceptron)**: 1+ hidden layers
- **Deep Neural Network**: 3+ hidden layers

**Key Parameters:**

- **Weights**: Connection strengths between neurons
- **Biases**: Neuron activation thresholds
- **Learning Rate**: Step size for weight updates
- **Epochs**: Complete passes through training data
- **Batch Size**: Samples processed simultaneously

### Forward Propagation

```python
# Mathematical flow
z[l] = W[l] · a[l-1] + b[l]     # Linear transformation
a[l] = f(z[l])                   # Activation function
```

**Step-by-step process:**

1. Input layer receives data
2. Each layer computes: z = W·x + b
3. Apply activation: a = f(z)
4. Pass to next layer
5. Output layer produces final prediction

### Backpropagation

**Chain Rule Application:**

```python
∂L/∂W[l] = ∂L/∂a[L] · ∂a[L]/∂z[L] · ∂z[L]/∂W[L]
```

**Algorithm:**

1. Forward pass (calculate loss)
2. Backward pass (compute gradients)
3. Update weights: W = W - α·∂L/∂W

---

## Activation Functions Reference {#activation-functions}

### Quick Comparison Table

| Function       | Formula                   | Range        | Derivative             | Use Case           | Pros                     | Cons                |
| -------------- | ------------------------- | ------------ | ---------------------- | ------------------ | ------------------------ | ------------------- |
| **Sigmoid**    | 1/(1+e^(-x))              | (0,1)        | σ(x)(1-σ(x))           | Binary output      | Probabilistic            | Vanishing gradient  |
| **Tanh**       | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1)       | 1-tanh²(x)             | Hidden layers      | Zero-centered            | Vanishing gradient  |
| **ReLU**       | max(0,x)                  | [0,∞)        | 1 if x>0 else 0        | Hidden layers      | Fast, no vanishing       | Dead neurons        |
| **Leaky ReLU** | max(αx,x)                 | (-∞,∞)       | 1 if x>0 else α        | Hidden layers      | No dead neurons          | Not zero-centered   |
| **ELU**        | x if x>0 else α(e^x-1)    | (-α,∞)       | 1 if x>0 else α(e^x-1) | Hidden layers      | Smooth, negative outputs | Slower computation  |
| **Softmax**    | e^x_i/Σe^x_j              | (0,1), sum=1 | For classification     | Multi-class output | Probabilistic            | Requires all inputs |

### Implementation Guide

**Sigmoid Function:**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

**ReLU Function:**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Softmax Function (for multi-class):**

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    # Jacobian of softmax
    s = softmax(x)
    return s * (np.eye(s.shape[-1]) - s.T)
```

### When to Use Each

**Output Layer:**

- **Sigmoid**: Binary classification
- **Softmax**: Multi-class classification
- **Linear**: Regression

**Hidden Layer:**

- **ReLU**: Default choice, most common
- **Leaky ReLU**: If you encounter dead neurons
- **ELU**: If you need smooth activations
- **Tanh**: If zero-centered output is important

---

## Architecture Patterns {#architecture}

### Common Network Architectures

**Binary Classifier (2 classes):**

```python
# Input: n_features
# Hidden: 2-4 layers, decreasing size
# Output: 1 neuron, sigmoid activation

layer_sizes = [n_features, 64, 32, 16, 1]
activation = 'relu'  # hidden layers
output_activation = 'sigmoid'
```

**Multi-class Classifier (k classes):**

```python
# Input: n_features
# Hidden: 2-4 layers
# Output: k neurons, softmax activation

layer_sizes = [n_features, 128, 64, 32, n_classes]
activation = 'relu'
output_activation = 'softmax'
```

**Regression Network:**

```python
# Input: n_features
# Hidden: 1-3 layers
# Output: 1 neuron, linear activation

layer_sizes = [n_features, 64, 32, 1]
activation = 'relu'  # or 'tanh'
output_activation = 'linear'
```

### Architecture Guidelines

**Layer Sizing Rules:**

- **Input layer**: Match feature dimension
- **Hidden layers**: Start wide, gradually narrow
- **Output layer**: Match number of targets
- **Rule of thumb**: Hidden size = (input + output) / 2

**Depth Guidelines:**

- **Shallow**: 1-2 hidden layers (simple problems)
- **Medium**: 3-5 hidden layers (moderate complexity)
- **Deep**: 6+ hidden layers (complex problems)

**Parameter Estimation:**

```python
def estimate_parameters(n_features, n_samples, n_classes):
    """Estimate reasonable network size"""

    # Hidden layer estimation
    hidden_factor = min(2 * n_features, 100)
    hidden_layers = [hidden_factor, hidden_factor // 2, hidden_factor // 4]

    # Total parameters estimation
    total_params = 0
    layer_sizes = [n_features] + hidden_layers + [n_classes]

    for i in range(len(layer_sizes) - 1):
        total_params += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]

    return layer_sizes, total_params
```

---

## Quick Code Templates {#code-templates}

### Simple MLP Template

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class SimpleMLP:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights (Xavier/Glorot)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        current = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.z_values.append(z)
            self.activations.append(a)
            current = a

        # Output layer
        z_out = np.dot(current, self.weights[-1]) + self.biases[-1]
        a_out = 1 / (1 + np.exp(-z_out))  # sigmoid
        self.z_values.append(z_out)
        self.activations.append(a_out)

        return a_out

    def backward(self, X, y, output):
        m = X.shape[0]

        # Output layer delta
        dA = output - y
        dZ = dA * (output * (1 - output))  # sigmoid derivative
        dW = np.dot(self.activations[-2].T, dZ) / m
        dB = np.sum(dZ, axis=0, keepdims=True) / m

        d_weights = [dW]
        d_biases = [dB]

        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            dA_prev = np.dot(dZ, self.weights[i+1].T)
            dZ_prev = dA_prev * self.relu_derivative(self.z_values[i])
            dW_prev = np.dot(self.activations[i].T, dZ_prev) / m
            dB_prev = np.sum(dZ_prev, axis=0, keepdims=True) / m

            d_weights.insert(0, dW_prev)
            d_biases.insert(0, dB_prev)
            dZ = dZ_prev

        return d_weights, d_biases

    def fit(self, X, y, epochs=100):
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_scaled)

            # Calculate loss
            loss = np.mean((output - y) ** 2)

            # Backward pass
            dW, dB = self.backward(X_scaled, y, output)

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * dB[i]

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return self.forward(X_scaled)
```

### Data Preprocessing Template

```python
def preprocess_data(X_train, X_test, y_train, y_test):
    """Standard data preprocessing pipeline"""

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle target variable
    if len(np.unique(y_train)) == 2:  # Binary classification
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
```

### Training Loop Template

```python
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, early_stopping=True):
    """Complete training loop with validation and early stopping"""

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.fit(X_train, y_train, epochs=1)  # Train one epoch

        # Validation
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val) ** 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    return train_losses, val_losses
```

### Evaluation Template

```python
def evaluate_model(model, X_test, y_test, task_type='classification'):
    """Comprehensive model evaluation"""

    predictions = model.predict(X_test)

    if task_type == 'classification':
        # Accuracy
        if predictions.shape[1] == 1:  # Binary
            pred_classes = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_classes == y_test)
        else:  # Multi-class
            pred_classes = np.argmax(predictions, axis=1)
            y_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            accuracy = np.mean(pred_classes == y_classes)

        # Precision, Recall, F1 (simplified)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    else:  # Regression
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))

        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return {'mse': mse, 'rmse': rmse, 'mae': mae}
```

---

## Troubleshooting Guide {#troubleshooting}

### Common Problems and Solutions

| Problem                 | Symptoms                                     | Causes                                  | Solutions                                                                             |
| ----------------------- | -------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| **Vanishing Gradients** | Training very slow, early layers don't learn | Sigmoid/tanh in deep networks           | • Use ReLU activation<br>• Use residual connections<br>• Use proper initialization    |
| **Exploding Gradients** | Loss becomes NaN, weights very large         | Large learning rate, deep networks      | • Reduce learning rate<br>• Use gradient clipping<br>• Proper weight initialization   |
| **Overfitting**         | High train acc, low test acc                 | Too many parameters, too few samples    | • Add regularization<br>• Use dropout<br>• Reduce model complexity<br>• Get more data |
| **Underfitting**        | Low train and test accuracy                  | Too few parameters, too simple model    | • Increase model capacity<br>• Train longer<br>• Reduce regularization                |
| **Slow Convergence**    | Loss decreases very slowly                   | Poor learning rate, bad initialization  | • Increase learning rate<br>• Use adaptive optimizers<br>• Check data scaling         |
| **NaN Loss**            | Loss becomes NaN                             | Gradient explosion, poor initialization | • Reduce learning rate<br>• Check data for outliers<br>• Use gradient clipping        |
| **Dead Neurons**        | ReLU neurons stuck at 0                      | ReLU with poor initialization           | • Use Leaky ReLU<br>• Better weight initialization<br>• Lower learning rate           |

### Debugging Checklist

**Data Issues:**

- [ ] Check data shapes and types
- [ ] Verify data is properly scaled
- [ ] Look for missing values or outliers
- [ ] Ensure balanced classes (for classification)
- [ ] Check for data leakage

**Model Issues:**

- [ ] Verify network architecture matches problem
- [ ] Check activation function choices
- [ ] Ensure proper weight initialization
- [ ] Validate loss function choice
- [ ] Check learning rate settings

**Training Issues:**

- [ ] Monitor both training and validation metrics
- [ ] Check for over/underfitting
- [ ] Verify gradient flow
- [ ] Ensure proper batch size
- [ ] Check for gradient explosion/vanishing

### Quick Diagnostics

**Loss Analysis:**

```python
def analyze_training_loss(train_losses, val_losses):
    """Analyze training progress"""

    # Check for convergence
    recent_train = np.mean(train_losses[-10:])
    recent_val = np.mean(val_losses[-10:])

    if recent_train > train_losses[0] * 0.9:
        print("⚠️ Training loss not decreasing significantly")

    # Check for overfitting
    gap = recent_train - recent_val
    if gap > 0.1:
        print("⚠️ Large train-val gap - possible overfitting")

    # Check for underfitting
    if abs(recent_train - recent_val) < 0.01 and recent_train > 0.1:
        print("⚠️ Both losses high - possible underfitting")

    return {
        'recent_train': recent_train,
        'recent_val': recent_val,
        'gap': gap
    }
```

**Gradient Analysis:**

```python
def check_gradients(model, X_sample, y_sample):
    """Check for gradient problems"""

    # Forward pass
    output = model.forward(X_sample)

    # Backward pass to get gradients
    dW, dB = model.backward(X_sample, y_sample, output)

    # Analyze gradient statistics
    print("Gradient Analysis:")
    for i, (dw, db) in enumerate(zip(dW, dB)):
        dw_mean, dw_std = np.mean(dw), np.std(dw)
        db_mean, db_std = np.mean(db), np.std(db)

        print(f"Layer {i}:")
        print(f"  Weight grads: mean={dw_mean:.6f}, std={dw_std:.6f}")
        print(f"  Bias grads: mean={db_mean:.6f}, std={db_std:.6f}")

        if dw_std < 1e-6:
            print(f"  ⚠️ Very small gradients - possible vanishing")
        elif dw_std > 1:
            print(f"  ⚠️ Very large gradients - possible exploding")
```

---

## Performance Optimization {#optimization}

### Learning Rate Selection

**Automatic Learning Rate Finding:**

```python
def find_good_learning_rate(model, X_train, y_train,
                           start_lr=1e-6, end_lr=1e-1, num_epochs=10):
    """Find a good learning rate range"""

    learning_rates = np.logspace(np.log10(start_lr), np.log10(end_lr), num_epochs)
    losses = []

    for lr in learning_rates:
        model.learning_rate = lr
        model.fit(X_train, y_train, epochs=1)

        # Get loss (you'll need to implement loss tracking)
        loss = get_current_loss(model)  # Implement this
        losses.append(loss)

    # Plot learning rate vs loss
    plt.semilogx(learning_rates, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    # Find good learning rate (steepest descent)
    best_idx = np.argmin(losses[1:]) + 1  # Skip first point
    return learning_rates[best_idx]
```

**Learning Rate Schedules:**

```python
def step_decay_schedule(initial_lr, drop_factor=0.5, epochs_drop=20):
    """Step decay learning rate schedule"""
    def schedule(epoch):
        return initial_lr * (drop_factor ** (epoch // epochs_drop))
    return schedule

def exponential_decay(initial_lr, decay_rate=0.95):
    """Exponential decay learning rate schedule"""
    def schedule(epoch):
        return initial_lr * (decay_rate ** epoch)
    return schedule

# Usage:
# lr_scheduler = step_decay_schedule(initial_lr=0.01, drop_factor=0.5, epochs_drop=20)
# for epoch in range(epochs):
#     lr = lr_scheduler(epoch)
#     model.learning_rate = lr
```

### Batch Size Guidelines

**Batch Size Impact:**

- **Small batch (1-32)**: More noise, better generalization, slower training
- **Medium batch (32-256)**: Good balance of speed and stability
- **Large batch (256+)**: Faster training, may need careful tuning

**Memory-Efficient Training:**

```python
def create_batch_generator(X, y, batch_size):
    """Create batches for large datasets"""
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
```

### Regularization Techniques

**L2 Regularization (Weight Decay):**

```python
def add_l2_regularization(weights, lambda_reg=0.01):
    """Add L2 regularization to loss"""
    l2_loss = 0
    for w in weights:
        l2_loss += np.sum(w ** 2)
    return lambda_reg * l2_loss
```

**Dropout Implementation:**

```python
def dropout(X, dropout_rate=0.5):
    """Apply dropout to input"""
    mask = np.random.binomial(1, 1 - dropout_rate, X.shape) / (1 - dropout_rate)
    return X * mask
```

**Early Stopping:**

```python
def early_stopping(val_losses, patience=10, min_delta=0.001):
    """Check if training should stop early"""
    if len(val_losses) < patience:
        return False

    recent_losses = val_losses[-patience:]
    best_loss = min(val_losses[:-patience]) if len(val_losses) > patience else val_losses[0]
    best_recent = min(recent_losses)

    return (best_recent - best_loss) < min_delta
```

---

## Common Formulas {#formulas}

### Activation Functions

**Sigmoid:**

```python
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x) * (1 - σ(x))
```

**ReLU:**

```python
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0 else 0
```

**Tanh:**

```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

**Softmax:**

```python
softmax(x_i) = e^x_i / Σⱼ e^x_j
```

### Loss Functions

**Mean Squared Error (MSE):**

```python
MSE = (1/n) * Σᵢ (yᵢ - ŷᵢ)²
```

**Cross-Entropy (Binary):**

```python
CE = -(1/n) * Σᵢ [yᵢ * log(ŷᵢ) + (1-yᵢ) * log(1-ŷᵢ)]
```

**Cross-Entropy (Multi-class):**

```python
CE = -(1/n) * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ)
```

### Backpropagation

**Chain Rule:**

```python
∂L/∂Wₗ = ∂L/∂aₗ * ∂aₗ/∂zₗ * ∂zₗ/∂Wₗ
```

**Weight Update:**

```python
Wₗ = Wₗ - α * ∂L/∂Wₗ
```

### Gradient Descent Variants

**Stochastic Gradient Descent (SGD):**

```python
θ = θ - α * ∇J(θ; xⁱ, yⁱ)
```

**Momentum:**

```python
vₜ = β * vₜ₋₁ + α * ∇J(θ)
θ = θ - vₜ
```

**Adam:**

```python
mₜ = β₁ * mₜ₋₁ + (1 - β₁) * ∇J(θ)
vₜ = β₂ * vₜ₋₁ + (1 - β₂) * (∇J(θ))²
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)
θ = θ - α * m̂ₜ / (√(v̂ₜ) + ε)
```

---

## Best Practices {#best-practices}

### Data Preprocessing

**Feature Scaling:**

```python
# For neural networks, always scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Target Variable:**

- **Classification**: Convert to one-hot or normalized
- **Regression**: Consider scaling if values are large

### Model Architecture

**Initialization:**

- Use Xavier/Glorot or He initialization
- Initialize biases to small positive values

**Activation Selection:**

- **Hidden layers**: ReLU (default)
- **Output layer**: Match problem type (sigmoid, softmax, linear)

**Layer Sizing:**

- Start with simpler models
- Gradually increase complexity if needed
- Use validation set to guide architecture choice

### Training Strategy

**Learning Rate:**

- Start with 0.01 (rule of thumb)
- Use learning rate scheduling
- Monitor both training and validation metrics

**Regularization:**

- Use dropout (0.2-0.5) for hidden layers
- L2 regularization for weight decay
- Early stopping to prevent overfitting

**Validation:**

- Always use train/validation/test split
- Use early stopping based on validation loss
- Monitor multiple metrics (loss, accuracy)

### Common Pitfalls to Avoid

1. **Data Leakage**: Including future information in training
2. **Wrong Loss Function**: Using MSE for classification
3. **Activation Mismatch**: Using sigmoid for hidden layers in deep networks
4. **Ignoring Data Preprocessing**: Not scaling features
5. **Overfitting**: Not using validation or regularization
6. **Wrong Problem Type**: Treating regression as classification or vice versa
7. **Poor Initialization**: Using random values without proper scaling
8. **Learning Rate Issues**: Too high (divergence) or too low (slow)

### Quick Decision Tree

**Choosing Architecture:**

```
Is this a classification problem?
├─ Yes:
│  ├─ Binary? → Output: 1 neuron, sigmoid
│  └─ Multi-class? → Output: n_classes neurons, softmax
└─ No: → Output: 1 neuron, linear

How complex is the data?
├─ Simple patterns → 1-2 hidden layers
├─ Medium complexity → 3-5 hidden layers
└─ Very complex → 6+ hidden layers

How much data do you have?
├─ Little data (< 1000 samples) → Simple model
├─ Medium data (1000-100k samples) → Medium model
└─ Large data (> 100k samples) → Complex model possible

What's your computational budget?
├─ Limited → Smaller, simpler network
└─ Ample → Can experiment with larger networks
```

### Performance Monitoring

**Key Metrics to Track:**

- Training loss and validation loss
- Training accuracy and validation accuracy
- Learning rate
- Gradient norms
- Parameter update magnitudes

**When to Stop Training:**

- Validation loss stops improving
- Training loss is much lower than validation loss (overfitting)
- Loss becomes NaN
- Accuracy plateaus

**Post-Training Analysis:**

- Check for over/underfitting
- Analyze misclassified examples
- Visualize learned features
- Test on completely new data
- Compare with baseline models

---

This cheatsheet provides essential quick-reference information for deep learning basics. Keep it handy during implementation and experimentation to ensure you're following best practices and avoiding common pitfalls.

**Remember**: Deep learning is both an art and a science. While these guidelines provide a solid foundation, always experiment and adapt based on your specific problem and data characteristics.
