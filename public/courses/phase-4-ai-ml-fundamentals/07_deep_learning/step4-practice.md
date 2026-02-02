# Deep Learning Basic Practice - Hands-On Exercises

_Build Your First Neural Networks with Step-by-Step Code_

## Table of Contents

1. [Setting Up Your Environment](#setup)
2. [Practice 1: Simple Perceptron](#perceptron)
3. [Practice 2: Multi-Layer Perceptron](#mlp)
4. [Practice 3: Working with Activation Functions](#activation)
5. [Practice 4: Building Your First Neural Network](#first-network)
6. [Practice 5: Training and Testing](#training)
7. [Common Mistakes and Debugging](#debugging)

---

## 🛠️ Setting Up Your Environment {#setup}

### Install Required Libraries

```bash
pip install numpy matplotlib scikit-learn
```

### Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")
print("📦 NumPy version:", np.__version__)
```

---

## 🎯 Practice 1: Build a Simple Perceptron {#perceptron}

### What You'll Learn

- How neurons work
- Basic forward propagation
- Simple decision boundaries

### Step 1: Create the Perceptron Class

```python
class SimplePerceptron:
    def __init__(self, learning_rate=0.1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for i in range(self.max_iterations):
            for idx, x_i in enumerate(X):
                # Calculate output
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        # Step function
        return 1 if x >= 0 else 0

    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            y_predicted = self.activation_function(linear_output)
            predictions.append(y_predicted)
        return np.array(predictions)
```

### Step 2: Test the Perceptron

```python
# Create simple binary classification data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, random_state=1, n_clusters_per_class=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train perceptron
perceptron = SimplePerceptron(learning_rate=0.1, max_iterations=100)
perceptron.fit(X_train, y_train)

# Make predictions
y_pred = perceptron.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"🎯 Perceptron Accuracy: {accuracy:.2%}")

# Visualize the decision boundary
def plot_decision_boundary(X, y, perceptron, title="Decision Boundary"):
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Visualize results
plot_decision_boundary(X_test, y_test, perceptron, "Simple Perceptron Decision Boundary")
```

### 🎓 Practice Task 1

**Challenge**: Modify the perceptron to work with the XOR problem. What do you observe?

```python
# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR truth table

# Try training
perceptron_xor = SimplePerceptron()
perceptron_xor.fit(X_xor, y_xor)
predictions_xor = perceptron_xor.predict(X_xor)

print("XOR Predictions:", predictions_xor)
print("XOR Actual:", y_xor)
print("Can a single perceptron solve XOR? Why or why not?")
```

---

## 🧠 Practice 2: Multi-Layer Perceptron (MLP) {#mlp}

### What You'll Learn

- How hidden layers work
- Non-linear decision boundaries
- Network architecture design

### Step 1: MLP Implementation

```python
class SimpleMLP:
    def __init__(self, hidden_size=4, learning_rate=0.1, max_iterations=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        # Initialize weights randomly
        np.random.seed(42)
        self.W1 = np.random.randn(2, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        y = y.reshape(-1, 1)  # Reshape for matrix operations

        for i in range(self.max_iterations):
            # Forward propagation
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)

            # Backward propagation
            dz2 = a2 - y
            dW2 = np.dot(a1.T, dz2) / len(X)
            db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.sigmoid_derivative(a1)
            dW1 = np.dot(X.T, dz1) / len(X)
            db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return (a2 > 0.5).astype(int).flatten()
```

### Step 2: Test MLP on XOR

```python
# Test MLP on XOR problem
mlp = SimpleMLP(hidden_size=4, learning_rate=0.5, max_iterations=1000)
mlp.fit(X_xor, y_xor)
predictions_mlp = mlp.predict(X_xor)

print("MLP XOR Predictions:", predictions_mlp)
print("MLP XOR Actual:", y_xor)
print("✅ Can MLP solve XOR? Yes! How many hidden neurons did you use?")
```

### 🎓 Practice Task 2

**Challenge**: Experiment with different hidden layer sizes and learning rates. What happens to the accuracy?

```python
# Experiment with different configurations
configurations = [
    {"hidden_size": 2, "learning_rate": 0.1},
    {"hidden_size": 4, "learning_rate": 0.5},
    {"hidden_size": 8, "learning_rate": 1.0}
]

for config in configurations:
    mlp_test = SimpleMLP(**config)
    mlp_test.fit(X_xor, y_xor)
    pred_test = mlp_test.predict(X_xor)
    accuracy_test = np.mean(pred_test == y_xor)
    print(f"Hidden Size: {config['hidden_size']}, LR: {config['learning_rate']}, Accuracy: {accuracy_test:.2%}")
```

---

## ⚡ Practice 3: Working with Activation Functions {#activation}

### What You'll Learn

- Different activation functions
- When to use each one
- Gradient problems and solutions

### Step 1: Compare Activation Functions

```python
def compare_activation_functions():
    x = np.linspace(-5, 5, 100)

    # Different activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)

    # Plot all functions
    plt.figure(figsize=(15, 10))

    functions = [
        (sigmoid, "Sigmoid", "blue"),
        (tanh, "Tanh", "red"),
        (relu, "ReLU", "green"),
        (leaky_relu, "Leaky ReLU", "orange")
    ]

    for i, (func, name, color) in enumerate(functions, 1):
        plt.subplot(2, 2, i)
        plt.plot(x, func, color=color, linewidth=2)
        plt.title(f'{name} Activation Function')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.grid(True, alpha=0.3)
        plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.show()

compare_activation_functions()
```

### Step 2: Build MLP with Different Activations

```python
class FlexibleMLP:
    def __init__(self, hidden_size=4, learning_rate=0.1, activation='sigmoid'):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.activation = activation

        np.random.seed(42)
        self.W1 = np.random.randn(2, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)

    def activate_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - x**2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)

    def fit(self, X, y):
        y = y.reshape(-1, 1)

        for i in range(1000):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.activate(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.activate(z2)

            # Backward pass
            dz2 = a2 - y
            dW2 = np.dot(a1.T, dz2) / len(X)
            db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.activate_derivative(a1)
            dW1 = np.dot(X.T, dz1) / len(X)
            db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.activate(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.activate(z2)
        return (a2 > 0.5).astype(int).flatten()
```

### 🎓 Practice Task 3

**Challenge**: Test all activation functions on the XOR problem. Which one works best?

```python
activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
results = {}

for act in activations:
    mlp_act = FlexibleMLP(activation=act, learning_rate=0.5)
    mlp_act.fit(X_xor, y_xor)
    pred_act = mlp_act.predict(X_xor)
    acc_act = np.mean(pred_act == y_xor)
    results[act] = acc_act
    print(f"{act.capitalize()}: {acc_act:.2%} accuracy")

print("\n🏆 Best activation function:", max(results, key=results.get))
```

---

## 🏗️ Practice 4: Building Your First Neural Network {#first-network}

### What You'll Learn

- Complete neural network architecture
- Data preprocessing
- Training and validation

### Step 1: Create a Complete Neural Network

```python
class CompleteNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # Track training progress
        self.losses = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        # Calculate loss
        loss = -np.mean(y * np.log(output + 1e-15) + (1 - y) * np.log(1 - output + 1e-15))
        self.losses.append(loss)

        # Backward propagation
        dz2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if verbose and epoch % 100 == 0:
                loss = self.losses[-1]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int).flatten()

    def plot_training_progress(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.show()
```

### Step 2: Train and Test the Network

```python
# Create more complex data
from sklearn.datasets import make_moons

# Generate moon-shaped data
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42)

# Create and train network
nn = CompleteNeuralNetwork(hidden_size=8, learning_rate=0.5)
nn.fit(X_train_moons, y_train_moons, epochs=1000)

# Make predictions
y_pred_moons = nn.predict(X_test_moons)
accuracy_moons = np.mean(y_pred_moons == y_test_moons)

print(f"🎯 Neural Network Accuracy on Moons Dataset: {accuracy_moons:.2%}")

# Plot training progress
nn.plot_training_progress()

# Visualize decision boundary
plot_decision_boundary(X_test_moons, y_test_moons, nn, "Neural Network Decision Boundary")
```

---

## 🎯 Practice 5: Training and Testing {#training}

### What You'll Learn

- Proper train/validation/test splits
- Overfitting detection
- Model evaluation metrics

### Step 1: Comprehensive Training Script

```python
def train_and_evaluate_model(X, y, hidden_sizes=[2, 4, 8, 16]):
    """Train models with different hidden layer sizes and compare"""

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    results = {}

    for hidden_size in hidden_sizes:
        print(f"\n🧠 Training with {hidden_size} hidden neurons...")

        # Train model
        model = CompleteNeuralNetwork(hidden_size=hidden_size, learning_rate=0.5)
        model.fit(X_train, y_train, epochs=1000, verbose=False)

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        train_acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)
        test_acc = np.mean(test_pred == y_test)

        results[hidden_size] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'model': model
        }

        print(f"   Train: {train_acc:.2%} | Val: {val_acc:.2%} | Test: {test_acc:.2%}")

    return results

# Test on different datasets
print("🔍 Testing on XOR Problem:")
xor_results = train_and_evaluate_model(X_xor, y_xor, [2, 4, 8])

print("\n🔍 Testing on Moons Dataset:")
moons_results = train_and_evaluate_model(X_moons, y_moons, [4, 8, 16])
```

### Step 2: Analyze Results

```python
def analyze_results(results, dataset_name):
    """Analyze training results and identify best model"""
    print(f"\n📊 Analysis for {dataset_name}:")
    print("Hidden Size | Train Acc | Val Acc | Test Acc | Overfitting?")
    print("-" * 60)

    best_model = None
    best_score = 0

    for hidden_size, metrics in results.items():
        train_acc = metrics['train_accuracy']
        val_acc = metrics['val_accuracy']
        test_acc = metrics['test_accuracy']

        # Check for overfitting
        overfitting = "Yes" if (train_acc - val_acc) > 0.1 else "No"

        print(f"{hidden_size:^11} | {train_acc:^9.2%} | {val_acc:^7.2%} | {test_acc:^8.2%} | {overfitting:^11}")

        # Select best model based on validation accuracy
        if val_acc > best_score:
            best_score = val_acc
            best_model = metrics['model']

    return best_model

# Analyze results
best_xor_model = analyze_results(xor_results, "XOR Problem")
best_moons_model = analyze_results(moons_results, "Moons Dataset")
```

---

## 🐛 Common Mistakes and Debugging {#debugging}

### Mistake 1: Vanishing Gradients

```python
# Problem: Very deep networks with sigmoid activation
# Solution: Use ReLU or proper initialization

def debug_vanishing_gradients():
    print("🔍 Debugging Vanishing Gradients...")

    # Create a deep network with sigmoid
    deep_sigmoid = CompleteNeuralNetwork(hidden_size=100)

    # Generate data
    X_deep, y_deep = make_classification(n_samples=1000, n_features=10,
                                        n_informative=5, random_state=42)

    # Train and check loss progression
    deep_sigmoid.fit(X_deep, y_deep, epochs=100, verbose=False)

    print(f"Final loss: {deep_sigmoid.losses[-1]:.4f}")
    print(f"Loss reduction: {deep_sigmoid.losses[0] - deep_sigmoid.losses[-1]:.4f}")

    # Check if loss is decreasing slowly
    if len(deep_sigmoid.losses) > 10:
        recent_change = deep_sigmoid.losses[-10] - deep_sigmoid.losses[-1]
        print(f"Recent loss change: {recent_change:.6f}")
        if recent_change < 0.001:
            print("⚠️  Possible vanishing gradient problem detected!")
            print("💡 Try: ReLU activation, better initialization, or fewer layers")

debug_vanishing_gradients()
```

### Mistake 2: Overfitting

```python
# Problem: Model memorizes training data
# Solution: Regularization, more data, simpler model

def detect_overfitting(results):
    print("🔍 Overfitting Detection:")

    for hidden_size, metrics in results.items():
        train_acc = metrics['train_accuracy']
        val_acc = metrics['val_accuracy']
        gap = train_acc - val_acc

        if gap > 0.1:
            print(f"⚠️  Model with {hidden_size} hidden neurons: Overfitting detected!")
            print(f"   Train: {train_acc:.2%} | Val: {val_acc:.2%} | Gap: {gap:.2%}")
            print("💡 Solutions: Add dropout, reduce model complexity, get more data")

detect_overfitting(moons_results)
```

### Mistake 3: Wrong Learning Rate

```python
# Problem: Learning too slow or oscillating
# Solution: Tune learning rate

def test_learning_rates():
    print("🔍 Testing Different Learning Rates...")

    X_test, y_test = make_classification(n_samples=200, random_state=42)

    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

    for lr in learning_rates:
        model = CompleteNeuralNetwork(hidden_size=4, learning_rate=lr)
        model.fit(X_test, y_test, epochs=100, verbose=False)

        final_loss = model.losses[-1]
        loss_reduction = model.losses[0] - model.losses[-1]

        print(f"LR: {lr:^5} | Final Loss: {final_loss:^7.4f} | Reduction: {loss_reduction:^7.4f}")

        if lr > 0.5 and loss_reduction < 0.1:
            print("   ⚠️  Learning rate might be too high!")

test_learning_rates()
```

---

## 🎯 Final Practice Challenge

### Build a Complete Image Classifier

```python
def final_challenge():
    """
    🎯 CHALLENGE: Build a neural network to classify handwritten digits (simplified)
    Use what you've learned to create the best possible model!
    """
    print("🎯 FINAL CHALLENGE: Build an Image Classifier")
    print("=" * 50)

    # Generate synthetic "image" data (28x28 = 784 features)
    from sklearn.datasets import fetch_openml
    from sklearn.utils import resample

    try:
        # This might take a while, so we'll use synthetic data for now
        X_images, y_images = make_classification(
            n_samples=1000, n_features=784, n_classes=10,
            n_informative=100, random_state=42
        )

        # Your task: Build the best classifier you can!
        print("📋 Your Tasks:")
        print("1. Split data into train/validation/test")
        print("2. Try different architectures (hidden sizes)")
        print("3. Experiment with learning rates")
        print("4. Use proper activation functions")
        print("5. Detect and prevent overfitting")
        print("6. Report your best accuracy")
        print("\n🚀 Start building! Remember all the techniques you've learned.")

        # Starter code:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_images, y_images, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        print(f"Error loading data: {e}")
        print("💡 Try with simpler synthetic data for now!")

# Run the challenge
X_train, X_val, X_test, y_train, y_val, y_test = final_challenge()
```

---

## 🏆 Congratulations!

You've completed all the deep learning basic practices! You now know how to:

✅ **Build neural networks from scratch**  
✅ **Implement forward and backward propagation**  
✅ **Work with different activation functions**  
✅ **Debug common problems**  
✅ **Evaluate and improve your models**

### Next Steps:

1. **Experiment** with the code examples
2. **Try** different datasets and architectures
3. **Build** your own neural network projects
4. **Share** your creations with others

Remember: **Every expert was once a beginner!** Keep practicing and experimenting! 🚀
