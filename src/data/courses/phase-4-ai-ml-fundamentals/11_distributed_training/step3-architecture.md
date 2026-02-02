# Neural Network Architecture: A Comprehensive Guide

## Table of Contents

1. [Introduction to Neural Network Architecture](#introduction)
2. [Fundamental Building Blocks](#building-blocks)
3. [Activation Functions](#activation-functions)
4. [Network Topology and Design](#topology)
5. [Advanced Architectural Patterns](#architectural-patterns)
6. [Regularization Techniques](#regularization)
7. [Optimization Algorithms](#optimization)
8. [Practical Implementation Examples](#implementation)
9. [Architecture Selection Guidelines](#selection)
10. [Visual Network Topologies](#visualizations)
11. [Advanced Topics](#advanced-topics)
12. [Case Studies and Real-World Applications](#case-studies)
13. [Performance Optimization](#performance)
14. [Future Directions](#future-directions)

---

## 1. Introduction to Neural Network Architecture {#introduction}

Neural network architecture is the blueprint that defines how neurons are organized, connected, and how information flows through the network. The architecture fundamentally determines the network's capacity to learn, its computational efficiency, and its ability to solve specific types of problems.

### 1.1 Historical Perspective

The journey of neural network architecture has been marked by several breakthrough innovations:

- **1943**: McCulloch-Pitts neuron model
- **1957**: Frank Rosenblatt's perceptron
- **1960s**: ADALINE and MADALINE networks
- **1986**: Backpropagation algorithm
- **1990s**: Multi-layer perceptrons with hidden layers
- **2006**: Deep learning emergence with deep belief networks
- **2012**: AlexNet revolutionizing computer vision
- **2015**: ResNet introducing residual connections
- **2017**: Transformer architecture
- **2020s**: Large language models and massive architectures

### 1.2 Why Architecture Matters

The architecture of a neural network is crucial because it determines:

1. **Representation Power**: How complex patterns the network can represent
2. **Computational Complexity**: Training and inference time requirements
3. **Memory Requirements**: Storage needs for parameters and activations
4. **Generalization**: Ability to perform well on unseen data
5. **Interpretability**: Understanding of learned representations
6. **Scalability**: Ability to handle larger datasets and more complex problems

### 1.3 Architectural Taxonomy

Neural network architectures can be categorized based on:

- **Connectivity Pattern**: Feedforward, recurrent, convolutional, hybrid
- **Depth**: Shallow, deep, very deep networks
- **Width**: Narrow, wide, or hybrid architectures
- **Topology**: Feedforward, recurrent, graph-based, attention-based
- **Specialization**: General-purpose, vision-specific, language-specific, etc.

---

## 2. Fundamental Building Blocks {#building-blocks}

### 2.1 The Neuron (Perceptron)

The fundamental unit of neural networks is the artificial neuron, also known as a perceptron. Understanding this building block is essential before exploring complex architectures.

#### 2.1.1 Mathematical Formulation

A single neuron receives inputs \(x_1, x_2, ..., x_n\) and produces an output \(y\) according to:

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

Where:

- \(w_i\) are the weights
- \(b\) is the bias
- \(f\) is the activation function

#### 2.1.2 Python Implementation

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.learning_rate = learning_rate

    def forward(self, inputs):
        """Forward pass through the perceptron"""
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        # Apply step function (perceptron activation)
        return 1 if weighted_sum >= 0 else 0

    def predict(self, X):
        """Make predictions on multiple inputs"""
        predictions = []
        for x in X:
            predictions.append(self.forward(x))
        return np.array(predictions)
```

### 2.2 Multi-Layer Perceptrons (MLP)

Multi-layer perceptrons extend the basic perceptron by adding hidden layers between input and output layers.

#### 2.2.1 Architecture Components

An MLP consists of:

- **Input Layer**: Receives raw data
- **Hidden Layers**: Process and transform features
- **Output Layer**: Produces final predictions
- **Activation Functions**: Applied at each layer
- **Loss Function**: Measures prediction error
- **Optimization Algorithm**: Updates weights and biases

#### 2.2.2 Forward Propagation

For a 3-layer MLP (input-hidden-output):

```python
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        # Initialize weights and biases randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.activation_name = activation

    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)

    def tanh(self, z):
        """Tanh activation function"""
        return np.tanh(z)

    def get_activation(self, z):
        """Get activation based on selected function"""
        if self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:  # ReLU (default)
            return self.relu(z)

    def forward(self, X):
        """Forward propagation through the network"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.get_activation(self.z1)

        # Output layer (usually linear for regression, sigmoid for classification)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Using sigmoid for binary classification

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 2.3 Feedforward Networks

Feedforward networks are the simplest form of neural networks where connections flow in one direction from input to output.

#### 2.3.1 General Feedforward Architecture

```python
class FeedforwardNetwork:
    def __init__(self, layer_sizes, activations=None):
        """
        Initialize a feedforward network
        Args:
            layer_sizes: List of integers representing layer sizes
            activations: List of activation function names for each layer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activations = activations if activations else ['relu'] * (self.num_layers - 1)

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def activation_function(self, x, name):
        """Apply activation function"""
        if name == 'relu':
            return np.maximum(0, x)
        elif name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif name == 'tanh':
            return np.tanh(x)
        elif name == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif name == 'elu':
            return np.where(x >= 0, x, np.exp(x) - 1)
        else:  # linear
            return x

    def forward(self, X):
        """Forward propagation through the network"""
        self.activations = [X]  # Store all activations

        for i in range(self.num_layers - 1):
            # Compute linear transformation
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            # Apply activation
            a = self.activation_function(z, self.activations[i])
            self.activations.append(a)

        return self.activations[-1]
```

---

## 3. Activation Functions {#activation-functions}

Activation functions are crucial components that introduce non-linearity into neural networks, enabling them to learn complex patterns.

### 3.1 Linear Activation

The simplest activation function that doesn't introduce non-linearity:

$$f(x) = x$$

```python
def linear(x):
    """Linear activation function"""
    return x
```

**Pros:**

- No saturation
- Derivative is constant (1)
- Good for regression problems

**Cons:**

- Cannot introduce non-linearity
- Networks collapse to single-layer with multiple linear activations

### 3.2 Sigmoid Activation

$$f(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x):
    """Sigmoid activation function"""
    # Clip input to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)
```

**Pros:**

- Smooth, differentiable everywhere
- Output in range (0, 1)
- Good for binary classification

**Cons:**

- Vanishing gradient problem
- Non-zero centered output
- Computational overhead

### 3.3 Hyperbolic Tangent (Tanh)

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh function"""
    t = np.tanh(x)
    return 1 - t**2
```

**Pros:**

- Smooth, differentiable
- Zero-centered output
- Stronger gradients than sigmoid

**Cons:**

- Vanishing gradient problem
- Slower computation than ReLU

### 3.4 Rectified Linear Unit (ReLU)

$$f(x) = \max(0, x)$$

```python
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)
```

**Pros:**

- Simple and fast computation
- Helps mitigate vanishing gradient problem
- Promotes sparsity in activations

**Cons:**

- Dying ReLU problem (neurons can get stuck)
- Unbounded output
- Not zero-centered

### 3.5 Leaky ReLU

$$f(x) = \max(\alpha x, x) \text{ where } \alpha > 0$$

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU function"""
    return np.where(x > 0, 1, alpha)
```

**Pros:**

- Addresses dying ReLU problem
- Computationally efficient
- Allows small negative values

**Cons:**

- Still not zero-centered
- Parameter α needs tuning

### 3.6 Exponential Linear Unit (ELU)

$$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

```python
def elu(x, alpha=1.0):
    """ELU activation function"""
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivative of ELU function"""
    return np.where(x >= 0, 1, alpha * np.exp(x))
```

**Pros:**

- Smooth activation function
- Negative outputs help push mean activations toward zero
- Faster convergence than ReLU

**Cons:**

- More computationally expensive
- Saturates for large negative values

### 3.7 Swish

$$f(x) = x \cdot \sigma(x)$$

Where $\sigma(x)$ is the sigmoid function.

```python
def swish(x):
    """Swish activation function"""
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivative of Swish function"""
    s = sigmoid(x)
    return s + x * s * (1 - s)
```

**Pros:**

- Smooth and non-monotonic
- Often outperforms ReLU
- Self-gated (like sigmoid)

**Cons:**

- Computationally more expensive than ReLU

### 3.8 Mish

$$f(x) = x \cdot \tanh(\ln(1 + e^x))$$

```python
def mish(x):
    """Mish activation function"""
    return x * np.tanh(np.log(1 + np.exp(np.clip(x, -20, 20))))

def mish_derivative(x):
    """Derivative of Mish function"""
    # Complex derivative implementation
    sp = np.log(1 + np.exp(np.clip(x, -20, 20)))
    t = np.tanh(sp)
    s = sigmoid(x)
    return t + x * (1 - t**2) * s
```

**Pros:**

- Often achieves better performance than ReLU and Swish
- Smooth, non-monotonic
- Better gradient flow

**Cons:**

- More computationally expensive
- Complex derivative calculation

### 3.9 Activation Function Comparison

```python
import matplotlib.pyplot as plt

def plot_activation_functions():
    """Plot various activation functions and their derivatives"""
    x = np.linspace(-5, 5, 1000)

    # Define activation functions
    activations = {
        'Linear': linear(x),
        'Sigmoid': sigmoid(x),
        'Tanh': tanh(x),
        'ReLU': relu(x),
        'Leaky ReLU': leaky_relu(x),
        'ELU': elu(x),
        'Swish': swish(x),
        'Mish': mish(x)
    }

    # Plot activations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (name, y) in enumerate(activations.items()):
        axes[i].plot(x, y, linewidth=2, label=name)
        axes[i].set_title(f'{name} Activation')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Input')
        axes[i].set_ylabel('Output')
        axes[i].set_ylim(-1.5, 4)

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Advanced activation function combinations
class AdaptiveActivation:
    def __init__(self):
        self.learnable_params = {}

    def gated_linear_unit(self, x, num_gates=2):
        """Gated Linear Unit activation"""
        gates = []
        for i in range(num_gates):
            gate = np.random.randn(x.shape[-1], x.shape[-1]) * 0.1
            gates.append(sigmoid(np.dot(x, gate)))

        output = x * gates[0]
        for i in range(1, num_gates):
            output = output + gates[i] * x

        return output / num_gates

    def maxout(self, x, k=2):
        """Maxout activation function"""
        # Split channels into k groups
        chunk_size = x.shape[-1] // k
        chunks = [x[..., i*chunk_size:(i+1)*chunk_size] for i in range(k)]
        return np.maximum.reduce(chunks)

    def learnable_activation(self, x, temperature=1.0):
        """Soft gating with learnable parameters"""
        if 'weights' not in self.learnable_params:
            self.learnable_params['weights'] = np.ones(x.shape[-1]) / x.shape[-1]

        # Apply softmax with learnable weights
        logits = x * self.learnable_params['weights'] / temperature
        return sigmoid(logits) * x
```

---

## 4. Network Topology and Design {#topology}

Network topology refers to the arrangement and interconnection patterns of neurons within a neural network. It encompasses decisions about layer organization, neuron counts, connectivity patterns, and overall structure.

### 4.1 Layer Types and Organization

#### 4.1.1 Dense (Fully Connected) Layers

```python
class DenseLayer:
    def __init__(self, input_units, output_units, activation='relu', use_bias=True):
        """
        Initialize a dense layer
        Args:
            input_units: Number of input features
            output_units: Number of output units
            activation: Activation function name
            use_bias: Whether to use bias term
        """
        self.input_units = input_units
        self.output_units = output_units
        self.activation = activation
        self.use_bias = use_bias
        self.use_batchnorm = False
        self.use_dropout = False
        self.dropout_rate = 0.0

        # Initialize weights using Xavier/Glorot initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.bias = np.zeros((1, output_units)) if use_bias else None

        # Storage for forward pass
        self.input = None
        self.output = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, input_data):
        """Forward pass through dense layer"""
        self.input = input_data

        # Compute linear transformation
        linear_output = np.dot(input_data, self.weights)
        if self.use_bias:
            linear_output += self.bias

        # Apply batch normalization if enabled
        if self.use_batchnorm:
            linear_output = self.batch_normalize(linear_output, training=True)

        # Apply activation function
        self.output = self.activation_function(linear_output)

        # Apply dropout if enabled
        if self.use_dropout:
            self.output = self.apply_dropout(self.output)

        return self.output

    def activation_function(self, x):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == 'elu':
            return np.where(x >= 0, x, np.exp(x) - 1)
        elif self.activation == 'gelu':
            # Gaussian Error Linear Unit
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
        else:  # linear
            return x

    def batch_normalize(self, x, training=True):
        """Simple batch normalization"""
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            variance = np.var(x, axis=0, keepdims=True)
            x_normalized = (x - mean) / np.sqrt(variance + 1e-8)
            return x_normalized
        else:
            return x

    def apply_dropout(self, x):
        """Apply dropout regularization"""
        if self.training:  # Assume training attribute exists
            mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * mask / (1.0 - self.dropout_rate)
        else:
            return x
```

#### 4.1.2 Architectural Design Patterns

```python
class NetworkBuilder:
    def __init__(self):
        self.layers = []
        self.layer_configs = []

    def add_dense_layer(self, units, activation='relu', **kwargs):
        """Add a dense layer to the network"""
        if len(self.layers) == 0:
            # First layer needs input dimension
            input_dim = kwargs.get('input_dim', None)
            if input_dim is None:
                raise ValueError("First layer must specify input_dim")
        else:
            # Infer input dimension from previous layer
            input_dim = self.layers[-1].output_units

        layer = DenseLayer(input_dim, units, activation, **kwargs)
        self.layers.append(layer)
        self.layer_configs.append({
            'type': 'dense',
            'units': units,
            'activation': activation,
            'kwargs': kwargs
        })
        return self

    def add_batch_normalization(self):
        """Add batch normalization after last layer"""
        if self.layers:
            self.layers[-1].use_batchnorm = True
        return self

    def add_dropout(self, rate):
        """Add dropout after last layer"""
        if self.layers:
            self.layers[-1].use_dropout = True
            self.layers[-1].dropout_rate = rate
        return self

    def build(self):
        """Build and return the network"""
        return Network(self.layers)

    def compile(self, loss='mse', optimizer='adam', metrics=None):
        """Compile the network with loss and optimizer"""
        network = self.build()
        network.compile(loss, optimizer, metrics)
        return network

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optimizer = None
        self.metrics = []

    def forward(self, X):
        """Forward pass through entire network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

    def compile(self, loss='mse', optimizer='adam', metrics=None):
        """Compile the network"""
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics or []
```

### 4.2 Width vs Depth Trade-offs

#### 4.2.1 Universal Approximation Theory

The Universal Approximation Theorem states that a feedforward network with a single hidden layer can approximate any continuous function on compact sets, given sufficient width.

```python
def universal_approximation_analysis():
    """Analysis of width vs depth trade-offs"""
    print("Universal Approximation Theorem:")
    print("- Single hidden layer with finite width can approximate any continuous function")
    print("- Depth enables exponential efficiency in representation")
    print("- Deeper networks can learn hierarchical features")
    print()

    print("Width vs Depth Trade-offs:")
    print("Width-focused:")
    print("  + Can approximate any function")
    print("  + Easier to optimize")
    print("  - May require exponentially many neurons")
    print("  - Less efficient representation")
    print()
    print("Depth-focused:")
    print("  + Exponential efficiency")
    print("  + Hierarchical feature learning")
    print("  + Better generalization")
    print("  - More complex optimization")
    print("  - Vanishing gradient problems")
```

#### 4.2.2 Practical Width/Depth Guidelines

```python
def calculate_optimal_dimensions(input_dim, output_dim, problem_complexity='medium'):
    """Calculate optimal network dimensions"""
    complexity_multipliers = {
        'simple': 1.5,
        'medium': 3.0,
        'complex': 6.0,
        'very_complex': 10.0
    }

    multiplier = complexity_multipliers.get(problem_complexity, 3.0)

    # Rule of thumb calculations
    # Hidden layer sizes (should generally decrease)
    hidden_layers = []
    current_size = int(input_dim * multiplier)

    # First hidden layer: typically larger than input
    hidden_layers.append(current_size)

    # Subsequent layers: gradually reduce
    for i in range(2, 5):  # 2-4 hidden layers
        prev_size = hidden_layers[-1]
        new_size = max(int(prev_size * 0.7), output_dim * 2)
        if new_size != prev_size:
            hidden_layers.append(new_size)

    # Ensure final layer has enough capacity
    if hidden_layers[-1] < output_dim * 2:
        hidden_layers[-1] = output_dim * 2

    return {
        'input_dim': input_dim,
        'hidden_layers': hidden_layers,
        'output_dim': output_dim,
        'total_params': calculate_total_params(input_dim, hidden_layers, output_dim)
    }

def calculate_total_params(input_dim, hidden_layers, output_dim):
    """Calculate total number of parameters"""
    total = 0
    prev_size = input_dim

    for hidden_size in hidden_layers:
        total += prev_size * hidden_size  # weights
        total += hidden_size  # biases
        prev_size = hidden_size

    # Output layer
    total += prev_size * output_dim
    total += output_size

    return total
```

### 4.3 Connectivity Patterns

#### 4.3.1 Standard Feedforward

The simplest pattern where each layer connects only to the next layer.

#### 4.3.2 Skip Connections

Skip connections (also called residual connections) allow information to bypass layers.

```python
class ResidualBlock:
    def __init__(self, input_dim, hidden_dim, activation='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Two paths: identity and transformation
        self.dense1 = DenseLayer(input_dim, hidden_dim, activation)
        self.dense2 = DenseLayer(hidden_dim, hidden_dim, 'linear')  # No activation

        # Projection layer if dimensions don't match
        if input_dim != hidden_dim:
            self.projection = DenseLayer(input_dim, hidden_dim, 'linear')
        else:
            self.projection = None

    def forward(self, x):
        # Main path
        main_path = self.dense1.forward(x)
        main_path = self.dense2.forward(main_path)

        # Skip connection
        if self.projection is not None:
            shortcut = self.projection.forward(x)
        else:
            shortcut = x

        # Residual connection
        return self.activation_function(main_path + shortcut)

    def activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
        else:
            return x
```

#### 4.3.3 Highway Networks

Highway networks use learned gating mechanisms to control information flow.

```python
class HighwayBlock:
    def __init__(self, input_dim, hidden_dim, activation='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Main transformation
        self.transform_gate = DenseLayer(input_dim, hidden_dim, 'sigmoid')
        self.transform_path = DenseLayer(input_dim, hidden_dim, activation)

        # Carry gate (1 - transform_gate)
        self.carry_gate = None  # Computed as 1 - transform_gate

    def forward(self, x):
        # Calculate gates
        transform_gate = self.transform_gate.forward(x)
        carry_gate = 1 - transform_gate  # Highway networks ensure carry_gate + transform_gate = 1

        # Apply transformations
        transformed = self.transform_path.forward(x)
        carried = x  # Identity mapping

        # Highway connection
        output = transform_gate * transformed + carry_gate * carried

        return output
```

### 4.4 Architecture Search and AutoML

#### 4.4.1 Random Architecture Search

```python
import random
from itertools import combinations

class ArchitectureSearch:
    def __init__(self, input_dim, output_dim, max_layers=10, min_layers=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.min_layers = min_layers

    def random_architecture(self):
        """Generate a random architecture"""
        num_layers = random.randint(self.min_layers, self.max_layers)
        layers = []

        # Start with input dimension
        prev_dim = self.input_dim

        for i in range(num_layers):
            # Random layer size (powers of 2 typically work well)
            layer_sizes = [32, 64, 128, 256, 512, 1024]
            if i == num_layers - 1:  # Output layer
                layer_size = self.output_dim
            else:
                layer_size = random.choice(layer_sizes)

            # Random activation function
            activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish']
            activation = random.choice(activations)

            # Add dense layer
            layer_config = {
                'type': 'dense',
                'units': layer_size,
                'activation': activation
            }
            layers.append(layer_config)

            # Randomly add regularization
            if random.random() < 0.5 and i < num_layers - 1:
                if random.random() < 0.3:  # Batch norm
                    layers.append({'type': 'batchnorm'})
                elif random.random() < 0.5:  # Dropout
                    rate = random.uniform(0.1, 0.5)
                    layers.append({'type': 'dropout', 'rate': rate})

            prev_dim = layer_size

        return layers

    def evolutionary_architecture_search(self, population_size=20, generations=50):
        """Evolutionary algorithm for architecture search"""
        # Initialize population
        population = [self.random_architecture() for _ in range(population_size)]
        best_architectures = []

        for generation in range(generations):
            # Evaluate all architectures (this would require training)
            # For demonstration, we'll use a mock fitness function
            fitness_scores = [self.mock_fitness(arch) for arch in population]

            # Sort by fitness
            population_with_fitness = list(zip(population, fitness_scores))
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            # Keep best architectures
            elite_count = population_size // 4
            elite = [arch for arch, fitness in population_with_fitness[:elite_count]]
            best_architectures.append(elite[0])

            # Generate next generation
            next_generation = elite.copy()

            # Add mutations
            while len(next_generation) < population_size:
                parent = random.choice(elite)
                mutated = self.mutate_architecture(parent)
                next_generation.append(mutated)

            population = next_generation

        return best_architectures

    def mutate_architecture(self, architecture):
        """Mutate an architecture"""
        mutated = architecture.copy()

        # Random mutation
        mutation_type = random.choice(['add_layer', 'remove_layer', 'change_activation', 'change_size'])

        if mutation_type == 'add_layer' and len(mutated) < self.max_layers:
            # Add a new layer
            position = random.randint(0, len(mutated))
            new_layer = self.random_layer()
            mutated.insert(position, new_layer)

        elif mutation_type == 'remove_layer' and len(mutated) > self.min_layers:
            # Remove a random layer (not input/output)
            non_output_indices = [i for i, layer in enumerate(mutated)
                                if layer.get('type') == 'dense' and layer.get('units') != self.output_dim]
            if non_output_indices:
                idx = random.choice(non_output_indices)
                mutated.pop(idx)

        elif mutation_type == 'change_activation':
            # Change activation of a dense layer
            dense_indices = [i for i, layer in enumerate(mutated) if layer.get('type') == 'dense']
            if dense_indices:
                idx = random.choice(dense_indices)
                activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish']
                current_activation = mutated[idx]['activation']
                available = [a for a in activations if a != current_activation]
                if available:
                    mutated[idx]['activation'] = random.choice(available)

        elif mutation_type == 'change_size':
            # Change size of a dense layer
            dense_indices = [i for i, layer in enumerate(mutated) if layer.get('type') == 'dense']
            if dense_indices:
                idx = random.choice(dense_indices)
                if mutated[idx]['units'] != self.output_dim:  # Don't change output layer
                    layer_sizes = [32, 64, 128, 256, 512, 1024]
                    mutated[idx]['units'] = random.choice(layer_sizes)

        return mutated

    def random_layer(self):
        """Generate a random layer configuration"""
        layer_types = ['dense', 'batchnorm', 'dropout']
        layer_type = random.choice(layer_types)

        if layer_type == 'dense':
            return {
                'type': 'dense',
                'units': random.choice([32, 64, 128, 256]),
                'activation': random.choice(['relu', 'leaky_relu', 'elu', 'gelu'])
            }
        elif layer_type == 'dropout':
            return {
                'type': 'dropout',
                'rate': random.uniform(0.1, 0.5)
            }
        else:  # batchnorm
            return {'type': 'batchnorm'}

    def mock_fitness(self, architecture):
        """Mock fitness function for demonstration"""
        # This would typically involve training the architecture and measuring performance
        # For demonstration, we'll use a simple heuristic based on architecture complexity
        dense_layers = sum(1 for layer in architecture if layer.get('type') == 'dense')
        total_units = sum(layer.get('units', 0) for layer in architecture if layer.get('type') == 'dense')

        # Simple fitness heuristic: more layers and units generally increase capacity
        fitness = dense_layers * 10 + total_units / 1000

        # Add some random noise to make it more realistic
        fitness += random.uniform(-5, 5)

        return fitness
```

### 4.5 Specialized Topologies

#### 4.5.1 Inverted Pyramid

```python
class InvertedPyramidNetwork:
    def __init__(self, input_dim, layer_sizes, activation='relu'):
        """
        Network with decreasing layer sizes (inverted pyramid)
        Args:
            input_dim: Input dimension
            layer_sizes: List of layer sizes (should be decreasing)
            activation: Activation function
        """
        self.layers = []
        prev_size = input_dim

        for size in layer_sizes:
            self.layers.append(DenseLayer(prev_size, size, activation))
            prev_size = size

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
```

#### 4.5.2 Bottleneck Architecture

```python
class BottleneckNetwork:
    def __init__(self, input_dim, bottleneck_dim, expansion_factors=[4, 4, 4]):
        """
        Bottleneck architecture with expanding-compressing pattern
        Args:
            input_dim: Input dimension
            bottleneck_dim: Size of bottleneck layer
            expansion_factors: Factors by which to expand before bottleneck
        """
        self.layers = []
        prev_size = input_dim

        for i, factor in enumerate(expansion_factors):
            # Expand
            expanded_size = bottleneck_dim * factor
            self.layers.append(DenseLayer(prev_size, expanded_size, 'relu'))

            # Compress to bottleneck
            self.layers.append(DenseLayer(expanded_size, bottleneck_dim, 'relu'))

            prev_size = bottleneck_dim

        # Final output layer
        self.output_layer = DenseLayer(bottleneck_dim, input_dim, 'linear')

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        output = self.output_layer.forward(output)
        return output
```

---

## 5. Advanced Architectural Patterns {#architectural-patterns}

Modern neural network architectures have evolved beyond simple feedforward networks to incorporate sophisticated design patterns that address specific challenges like vanishing gradients, feature representation, and computational efficiency.

### 5.1 Residual Networks (ResNet)

ResNet introduced the revolutionary concept of skip connections, allowing gradients to flow directly through the network.

#### 5.1.1 Basic Residual Block

```python
class BasicResidualBlock:
    def __init__(self, input_dim, hidden_dim, use_projection=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_projection = use_projection

        # First convolution layer
        self.conv1 = DenseLayer(input_dim, hidden_dim, 'relu')
        # Second convolution layer
        self.conv2 = DenseLayer(hidden_dim, hidden_dim, 'linear')

        # Projection shortcut if dimensions don't match
        if use_projection or input_dim != hidden_dim:
            self.shortcut = DenseLayer(input_dim, hidden_dim, 'linear')
        else:
            self.shortcut = None

    def forward(self, x):
        # Forward through main path
        identity = x

        out = self.conv1.forward(x)
        out = self.conv2.forward(out)

        # Add shortcut connection
        if self.shortcut is not None:
            identity = self.shortcut.forward(x)

        out = out + identity
        out = np.maximum(0, out)  # ReLU activation

        return out

    def backward(self, grad_output):
        # Backward pass with gradient flow through skip connections
        # Implementation depends on automatic differentiation framework
        pass
```

#### 5.1.2 Bottleneck Residual Block

```python
class BottleneckResidualBlock:
    def __init__(self, input_dim, bottleneck_dim=64, expansion=4):
        self.bottleneck_dim = bottleneck_dim
        self.expansion = expansion
        self.output_dim = bottleneck_dim * expansion

        # 1x1 bottleneck (reducing dimension)
        self.conv1 = DenseLayer(input_dim, bottleneck_dim, 'relu')

        # 3x3 layer (middle dimension)
        self.conv2 = DenseLayer(bottleneck_dim, bottleneck_dim, 'relu')

        # 1x1 expanding back
        self.conv3 = DenseLayer(bottleneck_dim, self.output_dim, 'linear')

        # Projection shortcut
        self.shortcut = None
        if input_dim != self.output_dim:
            self.shortcut = DenseLayer(input_dim, self.output_dim, 'linear')

    def forward(self, x):
        identity = x

        # Forward through bottleneck
        out = self.conv1.forward(x)  # Reduce
        out = self.conv2.forward(out)  # Process
        out = self.conv3.forward(out)  # Expand

        # Add shortcut
        if self.shortcut is not None:
            identity = self.shortcut.forward(x)

        out = out + identity
        out = np.maximum(0, out)  # ReLU

        return out
```

#### 5.1.3 Complete ResNet Architecture

```python
class ResNet:
    def __init__(self, input_dim, num_classes, block_config, blocks_per_stage):
        """
        ResNet architecture
        Args:
            input_dim: Input dimension
            num_classes: Number of output classes
            block_config: List of block types for each stage
            blocks_per_stage: Number of blocks per stage
        """
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = DenseLayer(input_dim, 64, 'relu')

        # ResNet stages with different block types
        self.stages = []
        prev_dim = 64

        for i, (block_type, num_blocks) in enumerate(zip(block_config, blocks_per_stage)):
            stage = ResNetStage(prev_dim, block_type, num_blocks)
            self.stages.append(stage)
            prev_dim = stage.output_dim

        # Final classification layer
        self.fc = DenseLayer(prev_dim, num_classes, 'linear')

    def forward(self, x):
        # Initial convolution
        out = self.conv1.forward(x)

        # Forward through stages
        for stage in self.stages:
            out = stage.forward(out)

        # Global average pooling would go here (for image data)
        # For our MLP version, we'll use flatten if needed

        # Final classification
        out = self.fc.forward(out)

        return out

class ResNetStage:
    def __init__(self, input_dim, block_type, num_blocks):
        self.input_dim = input_dim
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks = []

        # Determine block configuration
        if block_type == 'basic':
            hidden_dim = input_dim
        elif block_type == 'bottleneck':
            hidden_dim = 64
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # First block might need projection
        first_block = self.create_block(input_dim, hidden_dim, use_projection=True)
        self.blocks.append(first_block)

        # Remaining blocks
        for _ in range(1, num_blocks):
            block = self.create_block(hidden_dim, hidden_dim, use_projection=False)
            self.blocks.append(block)

        self.output_dim = hidden_dim if block_type == 'basic' else hidden_dim * 4

    def create_block(self, input_dim, hidden_dim, use_projection=False):
        if self.block_type == 'basic':
            return BasicResidualBlock(input_dim, hidden_dim, use_projection)
        elif self.block_type == 'bottleneck':
            return BottleneckResidualBlock(input_dim, hidden_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block.forward(out)
        return out
```

### 5.2 Dense Networks (DenseNet)

DenseNet connects each layer to every other layer in a feed-forward fashion.

```python
class DenseLayer:
    def __init__(self, input_dim, growth_rate, bn_size=4):
        """
        Dense layer that concatenates inputs from all previous layers
        Args:
            input_dim: Total input dimension from all previous layers
            growth_rate: Number of new features added by this layer
            bn_size: Bottleneck layer size multiplier
        """
        self.input_dim = input_dim
        self.growth_rate = growth_rate
        self.bn_size = bn_size

        # Batch normalization
        self.bn1 = BatchNormLayer(input_dim)

        # ReLU + 1x1 Convolution (bottleneck)
        bottleneck_dim = bn_size * growth_rate
        self.conv1 = DenseLayer(input_dim, bottleneck_dim, 'relu')

        # Batch normalization
        self.bn2 = BatchNormLayer(bottleneck_dim)

        # 1x1 Convolution (expansion)
        self.conv2 = DenseLayer(bottleneck_dim, growth_rate, 'linear')

    def forward(self, x):
        # Concatenate all previous features
        if hasattr(x, 'shape'):
            # Flatten all dimensions except batch
            if len(x.shape) > 2:
                x = x.reshape(x.shape[0], -1)

        out = self.bn1.forward(x)
        out = self.conv1.forward(out)
        out = self.bn2.forward(out)
        out = self.conv2.forward(out)

        # Concatenate input and new features
        return np.concatenate([x, out], axis=-1)

class TransitionLayer:
    def __init__(self, input_dim, compression_factor=0.5):
        """
        Transition layer between dense blocks
        Args:
            input_dim: Input dimension
            compression_factor: Factor by which to reduce features (0-1)
        """
        self.input_dim = input_dim
        self.compression_factor = compression_factor
        output_dim = int(input_dim * compression_factor)

        self.bn = BatchNormLayer(input_dim)
        self.conv = DenseLayer(input_dim, output_dim, 'relu')
        self.pool = AveragePooling1D()  # Would need to implement for actual use

    def forward(self, x):
        out = self.bn.forward(x)
        out = self.conv.forward(out)
        out = self.pool.forward(out)
        return out

class DenseNet:
    def __init__(self, input_dim, num_classes, growth_rate=32, num_blocks=[6, 12, 24, 16]):
        """
        DenseNet architecture
        Args:
            input_dim: Input dimension
            num_classes: Number of output classes
            growth_rate: Growth rate for dense layers
            num_blocks: Number of dense layers per block
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.growth_rate = growth_rate

        # Initial convolution
        self.conv1 = DenseLayer(input_dim, 2 * growth_rate, 'relu')

        # Dense blocks
        self.dense_blocks = []
        num_features = 2 * growth_rate

        for i, num_layers in enumerate(num_blocks):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            # Add transition layer except for last block
            if i != len(num_blocks) - 1:
                transition = TransitionLayer(num_features)
                self.dense_blocks.append(transition)
                num_features = int(num_features * 0.5)

        # Final classification layer
        self.fc = DenseLayer(num_features, num_classes, 'linear')

    def forward(self, x):
        out = self.conv1.forward(x)

        for block in self.dense_blocks:
            out = block.forward(out)

        out = self.fc.forward(out)
        return out

class BatchNormLayer:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Moving averages
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Training mode: use batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Update running averages
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean, var = batch_mean, batch_var
        else:
            # Inference mode: use running statistics
            mean, var = self.running_mean, self.running_var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out
```

### 5.3 Attention Mechanisms

Attention mechanisms allow networks to focus on relevant parts of the input.

```python
class SelfAttention:
    def __init__(self, embed_dim, num_heads=8):
        """
        Multi-head self-attention mechanism
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.W_q = DenseLayer(embed_dim, embed_dim, 'linear')
        self.W_k = DenseLayer(embed_dim, embed_dim, 'linear')
        self.W_v = DenseLayer(embed_dim, embed_dim, 'linear')

        # Output projection
        self.W_o = DenseLayer(embed_dim, embed_dim, 'linear')

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V matrices
        Q = self.W_q.forward(x)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # Attention scores: (Q * K^T) / sqrt(d_k)
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)

        # Softmax
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        attention_output = np.matmul(attention_weights, V)

        # Reshape back
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.W_o.forward(attention_output)

        return output, attention_weights

    def softmax(self, x):
        # Stable softmax implementation
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        self.attention = SelfAttention(embed_dim, num_heads)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(embed_dim, embed_dim * 4, embed_dim)

        # Layer normalization (applied before attention and ffn)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        self.dropout_rate = dropout_rate

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention.forward(x, mask)

        # Apply dropout
        if hasattr(self, 'training') and self.training:
            attn_output = self.dropout(attn_output)

        # Residual connection + layer norm
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn.forward(x)

        if hasattr(self, 'training') and self.training:
            ffn_output = self.dropout(ffn_output)

        x = self.norm2(x + ffn_output)

        return x, attention_weights

    def dropout(self, x):
        if hasattr(self, 'training') and self.training:
            mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * mask / (1.0 - self.dropout_rate)
        return x

class LayerNorm:
    def __init__(self, embed_dim, epsilon=1e-6):
        self.embed_dim = embed_dim
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)

    def forward(self, x):
        # Compute mean and variance per feature
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        return self.gamma * x_norm + self.beta

class FeedForwardNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        self.layers = [
            DenseLayer(input_dim, hidden_dim, activation),
            DenseLayer(hidden_dim, output_dim, 'linear')
        ]

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
```

### 5.4 Convolutional Neural Networks (CNN) Patterns

While primarily used for images, CNN concepts can be applied to 1D sequences.

```python
class Conv1D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weights = np.random.randn(kernel_size, in_channels, out_channels) * np.sqrt(2.0 / (kernel_size * in_channels))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, seq_len, in_channels = x.shape

        # Pad input if needed
        if self.padding > 0:
            x = self.pad_input(x)
            seq_len += 2 * self.padding

        # Compute output length
        output_len = (seq_len - self.kernel_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, output_len, self.out_channels))

        # Perform convolution
        for i in range(output_len):
            start = i * self.stride
            end = start + self.kernel_size
            window = x[:, start:end, :]  # (batch_size, kernel_size, in_channels)

            # Convolve across channels
            for oc in range(self.out_channels):
                conv_result = np.sum(window * self.weights[:, :, oc], axis=(1, 2))
                output[:, i, oc] = conv_result + self.bias[oc]

        return output

    def pad_input(self, x):
        batch_size, seq_len, channels = x.shape
        padded = np.zeros((batch_size, seq_len + 2 * self.padding, channels))
        padded[:, self.padding:seq_len + self.padding, :] = x
        return padded

class ResNet1D:
    def __init__(self, input_channels, num_classes, layers=[2, 2, 2, 2]):
        """
        1D ResNet for sequence classification
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            layers: Number of layers per stage
        """
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = Conv1D(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = BatchNormLayer(64)
        self.relu = lambda x: np.maximum(0, x)
        self.maxpool = lambda x: self.max_pool1d(x, 3, stride=2)

        # ResNet stages
        self.stage1 = self._make_stage(64, layers[0], stride=1)
        self.stage2 = self._make_stage(128, layers[1], stride=2)
        self.stage3 = self._make_stage(256, layers[2], stride=2)
        self.stage4 = self._make_stage(512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = lambda x: np.mean(x, axis=1)  # Adaptive average pooling
        self.fc = DenseLayer(512, num_classes, 'linear')

    def _make_stage(self, channels, num_blocks, stride=1):
        """Create a ResNet stage with multiple blocks"""
        blocks = []

        # First block might need projection
        first_block = ResNet1DBlock(64 if len(blocks) == 0 else channels, channels, stride=stride)
        blocks.append(first_block)

        # Remaining blocks
        for _ in range(1, num_blocks):
            block = ResNet1DBlock(channels, channels, stride=1)
            blocks.append(block)

        return blocks

    def max_pool1d(self, x, kernel_size, stride):
        """Max pooling for 1D sequences"""
        batch_size, seq_len, channels = x.shape
        output_len = (seq_len - kernel_size) // stride + 1

        output = np.zeros((batch_size, output_len, channels))

        for i in range(output_len):
            start = i * stride
            end = start + kernel_size
            output[:, i, :] = np.max(x[:, start:end, :], axis=1)

        return output

    def forward(self, x):
        # Initial convolution
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Forward through stages
        for block in self.stage1:
            x = block.forward(x)
        for block in self.stage2:
            x = block.forward(x)
        for block in self.stage3:
            x = block.forward(x)
        for block in self.stage4:
            x = block.forward(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Classification
        x = self.fc.forward(x)

        return x

class ResNet1DBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # First convolution
        self.conv1 = Conv1D(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNormLayer(out_channels)
        self.relu = lambda x: np.maximum(0, x)

        # Second convolution
        self.conv2 = Conv1D(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNormLayer(out_channels)

        # Projection shortcut
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv1D(in_channels, out_channels, 1, stride=stride),
                BatchNormLayer(out_channels)
            )

    def forward(self, x):
        shortcut = x

        # Forward through main path
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # Add shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut.forward(x)

        out = out + shortcut
        out = self.relu(out)

        return out
```

### 5.5 Autoencoder Architectures

Autoencoders learn efficient representations by compressing input to a latent space and reconstructing it.

```python
class Autoencoder:
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        """
        Basic autoencoder architecture
        Args:
            input_dim: Input dimension
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions for encoder
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Encoder
        self.encoder = self.build_encoder(hidden_dims)

        # Decoder (mirror of encoder)
        decoder_dims = hidden_dims[::-1] + [input_dim]
        self.decoder = self.build_decoder(decoder_dims)

    def build_encoder(self, hidden_dims):
        """Build encoder network"""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Latent space
        layers.append(DenseLayer(prev_dim, self.latent_dim, 'linear'))

        return layers

    def build_decoder(self, decoder_dims):
        """Build decoder network"""
        layers = []

        for i, dim in enumerate(decoder_dims):
            activation = 'sigmoid' if i == len(decoder_dims) - 1 else 'relu'  # Sigmoid for reconstruction
            layers.append(DenseLayer(prev_dim, dim, activation))
            prev_dim = dim

        return layers

    def encode(self, x):
        """Encode input to latent space"""
        output = x
        for layer in self.encoder:
            output = layer.forward(output)
        return output

    def decode(self, z):
        """Decode from latent space to reconstruction"""
        output = z
        for layer in self.decoder:
            output = layer.forward(output)
        return output

    def forward(self, x):
        """Full forward pass: encode then decode"""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z

    def reconstruct(self, x):
        """Reconstruct input"""
        reconstructed, _ = self.forward(x)
        return reconstructed

class VariationalAutoencoder(Autoencoder):
    def __init__(self, input_dim, latent_dim, hidden_dims=None, beta=1.0):
        """
        Variational Autoencoder with KL divergence loss
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            beta: KL divergence weighting factor
        """
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta

        # Separate encoders for mean and log variance
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.encoder_mean = self.build_encoder(hidden_dims + [latent_dim])
        self.encoder_logvar = self.build_encoder(hidden_dims + [latent_dim])

    def encode(self, x):
        """Encode to parameters of variational distribution"""
        # Compute mean
        z_mean = x
        for layer in self.encoder_mean:
            z_mean = layer.forward(z_mean)

        # Compute log variance
        z_logvar = x
        for layer in self.encoder_logvar:
            z_logvar = layer.forward(z_logvar)

        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        """Reparameterization trick for backpropagation"""
        # Sample from standard normal
        eps = np.random.randn(*z_mean.shape)

        # Reparameterize: z = μ + σ ⊙ ε
        z = z_mean + np.sqrt(np.exp(z_logvar)) * eps
        return z

    def forward(self, x):
        """Forward pass with VAE sampling"""
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        reconstructed = self.decode(z)

        return reconstructed, z_mean, z_logvar

    def compute_loss(self, x, reconstructed, z_mean, z_logvar):
        """Compute VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss (binary cross-entropy)
        recon_loss = -np.sum(x * np.log(reconstructed + 1e-8) +
                           (1 - x) * np.log(1 - reconstructed + 1e-8), axis=1)
        recon_loss = np.mean(recon_loss)

        # KL divergence loss
        kl_loss = -0.5 * np.sum(1 + z_logvar - np.square(z_mean) - np.exp(z_logvar), axis=1)
        kl_loss = np.mean(kl_loss)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

class ConvolutionalAutoencoder:
    def __init__(self, input_shape, latent_dim):
        """
        Convolutional autoencoder for image-like data
        Args:
            input_shape: Shape of input (height, width, channels)
            latent_dim: Dimension of latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Calculate flattened dimension after convolutions
        # This is a simplified version - actual implementation would need specific architecture

        # Encoder
        self.encoder_conv_layers = [
            Conv1D(input_shape[2], 32, 3, padding=1),  # First conv layer
            lambda x: np.maximum(0, x),  # ReLU
            Conv1D(32, 64, 3, padding=1),  # Second conv layer
            lambda x: np.maximum(0, x),  # ReLU
        ]

        # Flatten and dense layers for latent space
        flattened_dim = input_shape[0] * input_shape[1] * 64  # Simplified calculation
        self.encoder_dense_layers = [
            DenseLayer(flattened_dim, 128, 'relu'),
            DenseLayer(128, latent_dim, 'linear')
        ]

        # Decoder (mirror)
        self.decoder_dense_layers = [
            DenseLayer(latent_dim, 128, 'relu'),
            DenseLayer(128, flattened_dim, 'relu'),
            lambda x: x.reshape(-1, input_shape[0], input_shape[1], 64)  # Reshape
        ]

        self.decoder_conv_layers = [
            Conv1D(64, 32, 3, padding=1),  # First conv layer
            lambda x: np.maximum(0, x),  # ReLU
            Conv1D(32, input_shape[2], 3, padding=1),  # Output conv layer
            lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Sigmoid
        ]

    def forward(self, x):
        # Encoder
        encoded = x
        for layer in self.encoder_conv_layers:
            encoded = layer(encoded)

        # Flatten and encode to latent space
        batch_size = encoded.shape[0]
        encoded = encoded.reshape(batch_size, -1)

        for layer in self.encoder_dense_layers:
            encoded = layer.forward(encoded)

        # Decoder
        decoded = encoded
        for layer in self.decoder_dense_layers:
            if hasattr(layer, 'forward'):
                decoded = layer.forward(decoded)
            else:
                decoded = layer(decoded)

        for layer in self.decoder_conv_layers:
            if hasattr(layer, 'forward'):
                decoded = layer.forward(decoded)
            else:
                decoded = layer(decoded)

        return decoded, encoded

    def encode(self, x):
        """Encode to latent space"""
        encoded = x
        for layer in self.encoder_conv_layers:
            encoded = layer(encoded)

        batch_size = encoded.shape[0]
        encoded = encoded.reshape(batch_size, -1)

        for layer in self.encoder_dense_layers:
            encoded = layer.forward(encoded)

        return encoded

    def decode(self, z):
        """Decode from latent space"""
        decoded = z
        for layer in self.decoder_dense_layers:
            if hasattr(layer, 'forward'):
                decoded = layer.forward(decoded)
            else:
                decoded = layer(decoded)

        for layer in self.decoder_conv_layers:
            if hasattr(layer, 'forward'):
                decoded = layer.forward(decoded)
            else:
                decoded = layer(decoded)

        return decoded
```

---

## 6. Regularization Techniques {#regularization}

Regularization techniques prevent overfitting and improve generalization by adding constraints or penalties to the learning process.

### 6.1 L1 and L2 Regularization

L1 (Lasso) and L2 (Ridge) regularization add penalty terms to the loss function.

```python
class RegularizedDenseLayer(DenseLayer):
    def __init__(self, input_units, output_units, activation='relu',
                 l1_reg=0.0, l2_reg=0.0, use_bias=True):
        super().__init__(input_units, output_units, activation, use_bias)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def compute_regularization_loss(self):
        """Compute regularization loss"""
        l1_loss = 0
        l2_loss = 0

        if self.l1_reg > 0:
            l1_loss = self.l1_reg * np.sum(np.abs(self.weights))

        if self.l2_reg > 0:
            l2_loss = 0.5 * self.l2_reg * np.sum(self.weights ** 2)

        return l1_loss + l2_loss

    def compute_gradients(self, d_output, learning_rate, include_reg=True):
        """Compute gradients with regularization"""
        # Standard gradient computation
        d_weights = np.dot(self.input.T, d_output)
        d_weights /= self.input.shape[0]  # Average over batch

        if self.use_bias:
            d_bias = np.sum(d_output, axis=0, keepdims=True) / self.input.shape[0]

        # Add regularization gradients
        if include_reg and self.l2_reg > 0:
            d_weights += self.l2_reg * self.weights

        if include_reg and self.l1_reg > 0:
            d_weights += self.l1_reg * np.sign(self.weights)

        # Update weights
        self.weights -= learning_rate * d_weights
        if self.use_bias:
            self.bias -= learning_rate * d_bias

        return d_weights, d_bias if self.use_bias else None
```

### 6.2 Dropout Regularization

Dropout randomly sets neurons to zero during training to prevent co-adaptation.

```python
class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x, training=True):
        if training:
            # Create dropout mask
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate)
            # Scale activations to maintain expected value
            return x * self.mask / (1.0 - self.dropout_rate)
        else:
            return x

    def backward(self, d_output):
        if self.mask is not None:
            return d_output * self.mask / (1.0 - self.dropout_rate)
        return d_output

class AdvancedDropoutLayer:
    def __init__(self, dropout_rate=0.5, dropout_type='standard'):
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self.mask = None
        self.noise = None
        self.training = True

    def forward(self, x, training=True):
        if training:
            if self.dropout_type == 'standard':
                return self.standard_dropout(x)
            elif self.dropout_type == 'dropconnect':
                return self.dropconnect(x)
            elif self.dropout_type == 'scheduled':
                return self.scheduled_dropout(x)
            elif self.dropout_type == 'alpha':
                return self.alpha_dropout(x)
        else:
            return x

    def standard_dropout(self, x):
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate)
        return x * self.mask / (1.0 - self.dropout_rate)

    def dropconnect(self, x):
        """Dropout weights instead of activations"""
        self.mask = (np.random.rand(*x.shape) < (1.0 - self.dropout_rate))
        return x * self.mask

    def alpha_dropout(self, x):
        """Alpha dropout preserves mean and variance"""
        # Implementation would use SELU activation properties
        mask = np.random.rand(*x.shape) > self.dropout_rate
        masked = x * mask / (1.0 - self.dropout_rate)
        # Apply scaling and shifting to maintain SELU statistics
        return self.alpha_dropout_transform(masked)

    def alpha_dropout_transform(self, x):
        # Simplified alpha dropout transform
        alpha = -1.7580993408473766  # SELU alpha value
        scale = 1.0507009873554802   # SELU scale value
        return scale * x
```

### 6.3 Batch Normalization

Batch normalization normalizes inputs to each layer, reducing internal covariate shift.

```python
class BatchNormalizationLayer:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backward pass
        self.input = None
        self.normalized = None
        self.variance = None

    def forward(self, x, training=True):
        self.input = x

        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        self.normalized = x_normalized
        self.variance = var

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        return output

    def backward(self, d_output):
        # Simplified backward pass
        d_normalized = d_output * self.gamma

        # Gradient with respect to gamma and beta
        d_gamma = np.sum(d_output * self.normalized, axis=0)
        d_beta = np.sum(d_output, axis=0)

        # Gradient with respect to input (simplified)
        batch_size = self.input.shape[0]
        d_input = d_normalized / batch_size

        return d_input, d_gamma, d_beta

class LayerNormalization:
    def __init__(self, num_features, epsilon=1e-6):
        self.num_features = num_features
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, x):
        # Compute statistics along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        return output
```

### 6.4 Advanced Regularization Methods

```python
class WeightNormalization:
    def __init__(self, input_units, output_units, activation='relu'):
        self.input_units = input_units
        self.output_units = output_units
        self.activation = activation

        # Separated scale and direction parameters
        self.v = np.random.normal(0, 1, (input_units, output_units))
        self.g = np.ones(output_units)
        self.b = np.zeros(output_units)

    def forward(self, x):
        # Normalize weights
        self.weights = self.g * (self.v / np.sqrt(np.sum(self.v ** 2, axis=0, keepdims=True)))

        # Forward pass
        output = np.dot(x, self.weights) + self.b
        return self.activation_function(output)

    def activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

class SpectralNormalization:
    def __init__(self, layer, n_power_iterations=1):
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        self.u = None
        self.sigma = None

    def normalize_weights(self):
        """Normalize weights using spectral norm"""
        weights = self.layer.weights
        w_shape = weights.shape

        # Reshape weights for power iteration
        weights_flat = weights.reshape(-1, w_shape[-1])

        if self.u is None:
            self.u = np.random.normal(0, 1, (1, weights_flat.shape[1]))

        # Power iteration
        for _ in range(self.n_power_iterations):
            v = np.dot(self.u, weights_flat.T)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            u = np.dot(v, weights_flat)
            u = u / np.linalg.norm(u, axis=1, keepdims=True)

        self.u = u
        self.sigma = np.dot(np.dot(v, weights_flat), u.T)

        # Normalize
        weights_normalized = weights / self.sigma
        return weights_normalized

    def forward(self, x):
        self.layer.weights = self.normalize_weights()
        return self.layer.forward(x)

class ElasticWeightConsolidation:
    def __init__(self, model, lambda_=1000):
        self.model = model
        self.lambda_ = lambda_
        self.prev_params = {}
        self.fisher_information = {}

    def store_params(self):
        """Store current parameters as old parameters"""
        for name, param in self.model.parameters.items():
            self.prev_params[name] = param.copy()
            self.fisher_information[name] = np.zeros_like(param)

    def compute_fisher_information(self, data_loader):
        """Compute Fisher Information Matrix for each parameter"""
        for name, param in self.model.parameters.items():
            fisher = np.zeros_like(param)

            for batch in data_loader:
                for x, y in batch:
                    # Compute probability for each class (simplified)
                    probs = self.model.forward(x)

                    for i in range(len(y)):
                        # One-hot encoding
                        y_onehot = np.zeros_like(probs[i])
                        y_onehot[y[i]] = 1

                        # Gradient of log probability
                        grad = probs[i] - y_onehot
                        fisher += np.outer(grad, grad)

            self.fisher_information[name] = fisher / len(data_loader)

    def compute_ewc_loss(self):
        """Compute EWC loss term"""
        ewc_loss = 0

        for name, param in self.model.parameters.items():
            if name in self.fisher_information:
                diff = param - self.prev_params[name]
                ewc_loss += np.sum(self.fisher_information[name] * diff ** 2)

        return self.lambda_ * ewc_loss

class ProgressiveNeuralNetwork:
    def __init__(self, input_dim, task_configs):
        """
        Progressive Neural Network for continual learning
        Args:
            task_configs: List of dictionaries with layer configurations
        """
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.task_networks = []
        self.activations_history = []

    def add_task(self, task_config):
        """Add a new task network"""
        prev_layers = []
        for network in self.task_networks:
            prev_layers.extend(network.layers)

        new_network = ProgressiveTaskNetwork(
            self.input_dim,
            task_config,
            prev_layers,
            len(self.task_networks)
        )

        self.task_networks.append(new_network)
        return new_network

    def forward(self, x, task_id):
        """Forward pass for specific task"""
        if task_id >= len(self.task_networks):
            raise ValueError(f"Task {task_id} not found")

        return self.task_networks[task_id].forward(x)

class ProgressiveTaskNetwork:
    def __init__(self, input_dim, task_config, prev_layers, task_id):
        self.input_dim = input_dim
        self.task_id = task_id
        self.layers = []

        prev_dim = input_dim
        for layer_config in task_config:
            layer_type = layer_config['type']
            layer_params = layer_config.get('params', {})

            if layer_type == 'dense':
                layer = DenseLayer(prev_dim, **layer_params)
            elif layer_type == 'conv1d':
                layer = Conv1D(**layer_params)
            # Add other layer types

            self.layers.append(layer)
            prev_dim = layer.output_units if hasattr(layer, 'output_units') else layer_params.get('units', 64)

        # Add lateral connections to previous tasks
        self.lateral_connections = []
        for i, prev_layer in enumerate(prev_layers):
            if hasattr(prev_layer, 'output_units'):
                self.lateral_connections.append({
                    'source': prev_layer,
                    'target': self.layers[0],
                    'weight': np.random.normal(0, 0.01, (prev_layer.output_units, self.layers[0].output_units))
                })

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer.forward(output)

        return output
```

---

## 7. Optimization Algorithms {#optimization}

Optimization algorithms determine how neural network parameters are updated during training.

### 7.1 Gradient Descent Variants

```python
class SGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def update(self, model):
        """Update model parameters using SGD"""
        for name, param in model.parameters.items():
            if param.grad is not None:
                if self.momentum > 0:
                    if name not in self.velocities:
                        self.velocities[name] = np.zeros_like(param)

                    velocity = self.momentum * self.velocities[name] - self.learning_rate * param.grad

                    if self.nesterov:
                        velocity = self.momentum * velocity - self.learning_rate * param.grad

                    self.velocities[name] = velocity
                else:
                    velocity = -self.learning_rate * param.grad

                param += velocity

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, model):
        """Update model parameters using Adam"""
        self.t += 1

        for name, param in model.parameters.items():
            if param.grad is not None:
                # Initialize first and second moment estimates
                if name not in self.m:
                    self.m[name] = np.zeros_like(param)
                    self.v[name] = np.zeros_like(param)

                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * param.grad

                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * param.grad ** 2

                # Compute bias-corrected first moment estimate
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                # Update parameters
                param += -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdaGradOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-10):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G = {}

    def update(self, model):
        """Update parameters using AdaGrad"""
        for name, param in model.parameters.items():
            if param.grad is not None:
                if name not in self.G:
                    self.G[name] = np.zeros_like(param)

                # Accumulate squared gradients
                self.G[name] += param.grad ** 2

                # Update parameters
                param += -self.learning_rate * param.grad / (np.sqrt(self.G[name]) + self.epsilon)

class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, alpha=0.99, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.v = {}

    def update(self, model):
        """Update parameters using RMSprop"""
        for name, param in model.parameters.items():
            if param.grad is not None:
                if name not in self.v:
                    self.v[name] = np.zeros_like(param)

                # Update running average of squared gradients
                self.v[name] = self.alpha * self.v[name] + (1 - self.alpha) * param.grad ** 2

                # Update parameters
                param += -self.learning_rate * param.grad / (np.sqrt(self.v[name]) + self.epsilon)

class AdamWOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, model):
        """Update parameters using AdamW"""
        self.t += 1

        for name, param in model.parameters.items():
            if param.grad is not None:
                if name not in self.m:
                    self.m[name] = np.zeros_like(param)
                    self.v[name] = np.zeros_like(param)

                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * param.grad

                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * param.grad ** 2

                # Compute bias-corrected estimates
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                # Apply weight decay (decoupled from gradient)
                param += -self.weight_decay * param

                # Update parameters
                param += -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class LookaheadOptimizer:
    def __init__(self, base_optimizer, lookahead_steps=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.lookahead_steps = lookahead_steps
        self.alpha = alpha
        self.slow_weights = {}
        self.step_count = 0

    def update(self, model):
        """Update using Lookahead"""
        self.base_optimizer.update(model)
        self.step_count += 1

        # Update slow weights every lookahead_steps
        if self.step_count % self.lookahead_steps == 0:
            for name, param in model.parameters.items():
                if name not in self.slow_weights:
                    self.slow_weights[name] = param.copy()

                # Interpolate between slow and fast weights
                self.slow_weights[name] = self.alpha * param + (1 - self.alpha) * self.slow_weights[name]
                param[:] = self.slow_weights[name]

class SWAOptimizer:
    def __init__(self, base_optimizer, start_epoch=10, annealing_factor=0.5):
        self.base_optimizer = base_optimizer
        self.start_epoch = start_epoch
        self.annealing_factor = annealing_factor
        self.swa_weights = {}
        self.swa_count = 0

    def update(self, model, epoch):
        """Update using Stochastic Weight Averaging"""
        self.base_optimizer.update(model)

        if epoch >= self.start_epoch:
            for name, param in model.parameters.items():
                if name not in self.swa_weights:
                    self.swa_weights[name] = param.copy()
                else:
                    # Running average of weights
                    self.swa_weights[name] = (self.swa_weights[name] * self.swa_count + param) / (self.swa_count + 1)

                # Optionally use SWA weights for final predictions
                if hasattr(model, 'use_swa') and model.use_swa:
                    param[:] = self.swa_weights[name]

            self.swa_count += 1
```

### 7.2 Learning Rate Schedules

```python
class LearningRateScheduler:
    def __init__(self, schedule_type='constant', **kwargs):
        self.schedule_type = schedule_type
        self.schedule_params = kwargs
        self.current_epoch = 0

    def get_learning_rate(self, base_lr):
        """Get learning rate for current epoch"""
        if self.schedule_type == 'constant':
            return base_lr
        elif self.schedule_type == 'step':
            step_size = self.schedule_params.get('step_size', 30)
            gamma = self.schedule_params.get('gamma', 0.1)
            return base_lr * (gamma ** (self.current_epoch // step_size))
        elif self.schedule_type == 'exponential':
            gamma = self.schedule_params.get('gamma', 0.95)
            return base_lr * (gamma ** self.current_epoch)
        elif self.schedule_type == 'cosine':
            eta_min = self.schedule_params.get('eta_min', 0)
            eta_max = self.schedule_params.get('eta_max', base_lr)
            T_max = self.schedule_params.get('T_max', 100)
            return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * self.current_epoch / T_max))
        elif self.schedule_type == 'warmup':
            warmup_steps = self.schedule_params.get('warmup_steps', 100)
            if self.current_epoch < warmup_steps:
                return base_lr * self.current_epoch / warmup_steps
            else:
                return base_lr
        else:
            return base_lr

    def step(self):
        """Increment epoch counter"""
        self.current_epoch += 1

class OneCycleScheduler:
    def __init__(self, max_lr, epochs, pct_start=0.3, pct_end=0.7):
        self.max_lr = max_lr
        self.epochs = epochs
        self.pct_start = pct_start
        self.pct_end = pct_end
        self.current_epoch = 0

    def get_learning_rate(self):
        """Get learning rate for current epoch using one-cycle policy"""
        if self.current_epoch < self.pct_start * self.epochs:
            # Increasing phase
            lr = self.max_lr * self.current_epoch / (self.pct_start * self.epochs)
        elif self.current_epoch < self.pct_end * self.epochs:
            # High learning rate phase
            lr = self.max_lr
        else:
            # Decreasing phase
            pct = (self.current_epoch - self.pct_end * self.epochs) / ((1 - self.pct_end) * self.epochs)
            lr = self.max_lr * (0.1 ** (5 * pct))

        return lr

    def step(self):
        self.current_epoch += 1
```

### 7.3 Gradient Clipping and Normalization

```python
class GradientClipping:
    def __init__(self, clip_value=None, clip_norm=None, clip_type='value'):
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.clip_type = clip_type

    def clip_gradients(self, model):
        """Clip gradients to prevent explosion"""
        if self.clip_type == 'value' and self.clip_value is not None:
            for param in model.parameters.values():
                if param.grad is not None:
                    np.clip(param.grad, -self.clip_value, self.clip_value, out=param.grad)

        elif self.clip_type == 'norm' and self.clip_norm is not None:
            total_norm = 0
            for param in model.parameters.values():
                if param.grad is not None:
                    total_norm += np.sum(param.grad ** 2)

            total_norm = np.sqrt(total_norm)

            if total_norm > self.clip_norm:
                clip_factor = self.clip_norm / total_norm
                for param in model.parameters.values():
                    if param.grad is not None:
                        param.grad *= clip_factor

class GradientNormalization:
    def __init__(self, norm_order=2, epsilon=1e-8):
        self.norm_order = norm_order
        self.epsilon = epsilon

    def normalize_gradients(self, model):
        """Normalize gradients to unit norm"""
        for param in model.parameters.values():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.flatten(), self.norm_order)
                if grad_norm > 0:
                    param.grad = param.grad / (grad_norm + self.epsilon)
```

---

## 8. Modern Architecture Patterns {#modern-architectures}

### 8.1 Vision Transformers (ViT)

```python
class VisionTransformer:
    def __init__(self, patch_size=16, num_classes=1000, dim=768, depth=12,
                 heads=12, mlp_dim=3072, dropout=0.1):
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, dim)

        # Positional embedding
        self.pos_embed = PositionalEncoding(dim)

        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(depth):
            self.transformer_blocks.append(
                TransformerBlock(dim, heads, mlp_dim, dropout)
            )

        # Classification head
        self.classifier = DenseLayer(dim, num_classes, 'linear')

        # Class token
        self.class_token = ClassToken(dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Reshape to patches
        patches = self.patch_embed.forward(x)

        # Add class token
        tokens = self.class_token.forward(patches)

        # Add positional encoding
        tokens = self.pos_embed.forward(tokens)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens, _ = block.forward(tokens)

        # Use class token for classification
        cls_token = tokens[:, 0]  # First token is class token

        # Classification
        output = self.classifier.forward(cls_token)

        return output

class PatchEmbedding:
    def __init__(self, patch_size, dim):
        self.patch_size = patch_size
        self.dim = dim
        self.projection = Conv1D(3, dim, patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Reshape for patch embedding
        # (batch, channels, height, width) -> (batch, num_patches, channels*patch_size*patch_size)
        patches = self.extract_patches(x)

        # Project to embedding dimension
        patches = patches.reshape(batch_size, -1, self.dim)

        return patches

    def extract_patches(self, x):
        batch_size, channels, height, width = x.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Extract patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = j * self.patch_size
                end_w = start_w + self.patch_size

                patch = x[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch.reshape(batch_size, -1))

        return np.stack(patches, axis=1)

class PositionalEncoding:
    def __init__(self, dim, max_length=5000):
        self.dim = dim
        self.max_length = max_length
        self.pos_enc = self.create_positional_encoding()

    def create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pos_enc = np.zeros((self.max_length, self.dim))

        for pos in range(self.max_length):
            for i in range(0, self.dim, 2):
                # Even indices
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.dim)))
                # Odd indices
                if i + 1 < self.dim:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.dim)))

        return pos_enc

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        return tokens + self.pos_enc[:seq_len]

class ClassToken:
    def __init__(self, dim):
        self.dim = dim
        self.cls_token = np.random.normal(0, 0.02, (1, dim))

    def forward(self, patches):
        # Add class token to beginning of sequence
        batch_size = patches.shape[0]
        cls_tokens = np.tile(self.cls_token, (batch_size, 1, 1))
        return np.concatenate([cls_tokens, patches], axis=1)

class HybridViT(VisionTransformer):
    def __init__(self, conv_layers, patch_size=16, **kwargs):
        super().__init__(patch_size, **kwargs)

        # Add convolutional stem
        self.conv_stem = []
        for conv_config in conv_layers:
            layer_type = conv_config['type']
            params = conv_config.get('params', {})

            if layer_type == 'conv1d':
                self.conv_stem.append(Conv1D(**params))
            elif layer_type == 'batchnorm':
                self.conv_stem.append(BatchNormalizationLayer(**params))
            elif layer_type == 'relu':
                self.conv_stem.append(lambda x: np.maximum(0, x))
            elif layer_type == 'avgpool':
                self.conv_stem.append(lambda x: np.mean(x, axis=-1, keepdims=True))

    def forward(self, x):
        # Apply convolutional stem
        for layer in self.conv_stem:
            if hasattr(layer, 'forward'):
                x = layer.forward(x)
            else:
                x = layer(x)

        # Reshape for transformer
        batch_size = x.shape[0]
        if len(x.shape) == 4:  # (batch, channels, height, width)
            x = x.reshape(batch_size, x.shape[1], -1).transpose(0, 2, 1)
        elif len(x.shape) == 3:  # (batch, channels, length)
            x = x.transpose(0, 2, 1)

        return super().forward(x)
```

### 8.2 Modern ConvNets Architectures

```python
class EfficientNet:
    def __init__(self, num_classes=1000, compound_scaling=True):
        self.num_classes = num_classes
        self.compound_scaling = compound_scaling

        # Base model parameters
        self.base_params = {
            'width_coefficient': 1.0,
            'depth_coefficient': 1.0,
            'resolution': 224,
            'dropout_rate': 0.2
        }

        if compound_scaling:
            self._apply_compound_scaling()

        # Build the model
        self._build_model()

    def _apply_compound_scaling(self):
        """Apply compound scaling to model parameters"""
        # Simplified scaling rules
        depth_coef, width_coef, res_coef = 1.1, 1.2, 1.15

        self.base_params['depth_coefficient'] = depth_coef
        self.base_params['width_coefficient'] = width_coef
        self.base_params['resolution'] = int(224 * res_coef)

    def _build_model(self):
        """Build EfficientNet architecture"""
        # ImageNet EfficientNet-B0 configuration
        config = [
            # [kernel_size, channels, num_layers, stride, expansion_factor]
            [3, 16, 1, 1, 1],
            [3, 24, 2, 2, 6],
            [3, 40, 2, 2, 6],
            [3, 80, 3, 2, 6],
            [3, 112, 3, 1, 6],
            [3, 192, 4, 2, 6],
            [3, 320, 1, 1, 6],
        ]

        self.stages = []
        prev_channels = 3

        for i, (kernel_size, channels, num_layers, stride, expansion) in enumerate(config):
            if i == 0:
                # Initial stem
                stage = EfficientNetStage(
                    prev_channels, channels, kernel_size, stride,
                    num_layers, expansion, use_se=True
                )
            else:
                stage = EfficientNetStage(
                    prev_channels, channels, kernel_size, stride,
                    num_layers, expansion, use_se=True
                )

            self.stages.append(stage)
            prev_channels = channels

        # Final layers
        final_channels = int(1280 * self.base_params['width_coefficient'])
        self.final_conv = Conv1D(prev_channels, final_channels, 1)
        self.classifier = DenseLayer(final_channels, self.num_classes, 'linear')

    def forward(self, x):
        # Initial convolution
        x = self.stages[0].forward(x)

        # Apply middle stages
        for stage in self.stages[1:]:
            x = stage.forward(x)

        # Final convolution and global average pooling
        x = self.final_conv.forward(x)
        x = np.mean(x, axis=1)  # Global average pooling

        # Classification
        x = self.classifier.forward(x)

        return x

class EfficientNetStage:
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 num_layers, expansion_factor, use_se=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.use_se = use_se

        # Expansion layer
        expanded_channels = in_channels * expansion_factor
        if expansion_factor != 1:
            self.expand_conv = Conv1D(in_channels, expanded_channels, 1, activation='relu')

        # Depthwise separable convolution blocks
        self.depthwise_blocks = []
        for i in range(num_layers):
            block = SqueezeExcitationBlock(
                expanded_channels if expansion_factor != 1 else in_channels,
                out_channels,
                kernel_size,
                stride if i == 0 else 1,  # Only first layer has stride
                use_se=use_se
            )
            self.depthwise_blocks.append(block)

        # Projection layer
        if in_channels != out_channels or stride != 1:
            self.projection = Conv1D(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        # Skip connection for first block if needed
        identity = x

        # Apply expansion
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv.forward(x)

        # Apply depthwise blocks
        for block in self.depthwise_blocks:
            x = block.forward(x)

        # Add skip connection
        if self.projection is not None:
            identity = self.projection.forward(identity)

        x = x + identity

        return x

class SqueezeExcitationBlock:
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_se=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se

        # Depthwise convolution
        self.depthwise = DepthwiseConv1D(in_channels, kernel_size, stride)

        # Squeeze and Excitation
        if use_se:
            self.se = SqueezeExcitation(in_channels)

        # Pointwise convolution
        self.pointwise = Conv1D(in_channels, out_channels, 1)

        # Projection if needed
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = Conv1D(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x

        # Apply depthwise convolution
        x = self.depthwise.forward(x)

        # Apply Squeeze and Excitation
        if self.use_se:
            x = self.se.forward(x)

        # Apply pointwise convolution
        x = self.pointwise.forward(x)

        # Add skip connection
        if self.projection is not None:
            identity = self.projection.forward(identity)

        x = x + identity

        return x

class SqueezeExcitation:
    def __init__(self, channels, reduction=16):
        self.channels = channels
        self.reduction = reduction
        squeezed_channels = max(1, channels // reduction)

        # Squeeze
        self.squeeze = Conv1D(channels, squeezed_channels, 1, activation='relu')
        # Excite
        self.excite = Conv1D(squeezed_channels, channels, 1, activation='sigmoid')

    def forward(self, x):
        # Global average pooling
        squeezed = np.mean(x, axis=1, keepdims=True)

        # Squeeze and excite
        squeezed = self.squeeze.forward(squeezed)
        attention = self.excite.forward(squeezed)

        # Apply attention
        return x * attention

class DepthwiseConv1D:
    def __init__(self, in_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = in_channels

        # Depthwise separable convolution
        self.depthwise = Conv1D(in_channels, in_channels, kernel_size,
                               stride=stride, groups=in_channels, padding=kernel_size//2)

    def forward(self, x):
        return self.depthwise.forward(x)

class RegNet:
    def __init__(self, num_classes=1000, width_per_group=32,
                 groups=8, depth=13, q=8):
        """
        RegNet architecture
        Args:
            width_per_group: Width per group
            groups: Number of groups
            depth: Network depth
            q: Quantization parameter
        """
        self.num_classes = num_classes
        self.width_per_group = width_per_group
        self.groups = groups
        self.depth = depth
        self.q = q

        self._build_model()

    def _build_model(self):
        """Build RegNet architecture"""
        # Initial convolution
        self.stem = Conv1D(3, 32, 3, stride=2, padding=1)

        # RegNet stages
        stage_widths = self._get_stage_widths()
        stage_depths = [2, 3, 6, 4]  # Default depths for RegNet-Y

        self.stages = []
        prev_width = 32

        for i, (width, depth) in enumerate(zip(stage_widths, stage_depths)):
            stride = 2 if i > 0 else 1
            stage = RegNetStage(
                prev_width, width, depth,
                groups=self.groups, stride=stride
            )
            self.stages.append(stage)
            prev_width = width

        # Final layers
        self.final_conv = Conv1D(prev_width, 1280, 1)
        self.classifier = DenseLayer(1280, self.num_classes, 'linear')

    def _get_stage_widths(self):
        """Compute stage widths for RegNet"""
        widths = []
        current_width = self.width_per_group * self.groups

        for i in range(4):
            if i == 0:
                widths.append(32)  # First stage fixed width
            else:
                current_width = self._quantize_width(current_width * 1.6)
                widths.append(current_width)

        return widths

    def _quantize_width(self, width):
        """Quantize width to power of 2"""
        import math
        return int(2 ** (round(math.log(width) / math.log(2)) / self.q) * self.q * 2)

    def forward(self, x):
        # Apply stem
        x = self.stem.forward(x)

        # Apply stages
        for stage in self.stages:
            x = stage.forward(x)

        # Final convolution and pooling
        x = self.final_conv.forward(x)
        x = np.mean(x, axis=1)

        # Classification
        x = self.classifier.forward(x)

        return x

class RegNetStage:
    def __init__(self, in_channels, out_channels, depth, groups, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.groups = groups
        self.stride = stride

        # First block with stride
        first_block = RegNetBlock(in_channels, out_channels, groups, stride)
        self.blocks = [first_block]

        # Remaining blocks
        for _ in range(1, depth):
            block = RegNetBlock(out_channels, out_channels, groups, 1)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

class RegNetBlock:
    def __init__(self, in_channels, out_channels, groups, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride

        # Bottleneck block
        bottleneck_ratio = 1  # Different from ResNet's 0.25
        bottleneck_channels = out_channels // bottleneck_ratio

        # Project if needed
        if in_channels != out_channels or stride != 1:
            self.projection = Conv1D(in_channels, out_channels, 1, stride=stride)
        else:
            self.projection = None

        # Bottleneck convolution
        self.conv1 = Conv1D(in_channels, bottleneck_channels, 1, groups=groups, activation='relu')
        self.conv2 = Conv1D(bottleneck_channels, bottleneck_channels, 3,
                           stride=stride, groups=groups, activation='relu')
        self.conv3 = Conv1D(bottleneck_channels, out_channels, 1, groups=groups)

    def forward(self, x):
        identity = x

        # Apply projections
        if self.projection is not None:
            identity = self.projection.forward(identity)

        # Apply bottleneck
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)

        # Add skip connection
        x = x + identity
        x = np.maximum(0, x)  # ReLU activation

        return x
```

---

## 9. Performance Comparisons and Benchmarks {#performance-comparisons}

### 9.1 Architecture Performance Analysis

```python
class ArchitectureBenchmark:
    def __init__(self, datasets, metrics=['accuracy', 'f1_score', 'inference_time']):
        self.datasets = datasets
        self.metrics = metrics
        self.results = {}

    def benchmark_architecture(self, model_func, model_name, dataset_name):
        """Benchmark a model architecture on a dataset"""
        if model_name not in self.results:
            self.results[model_name] = {}

        # Load dataset
        X_train, y_train, X_test, y_test = self.datasets[dataset_name]

        # Train and evaluate model
        model = model_func()
        start_time = time.time()
        model.train(X_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time

        # Calculate metrics
        metrics_result = {}
        for metric in self.metrics:
            if metric == 'accuracy':
                metrics_result[metric] = self.calculate_accuracy(y_test, predictions)
            elif metric == 'f1_score':
                metrics_result[metric] = self.calculate_f1_score(y_test, predictions)
            elif metric == 'inference_time':
                metrics_result[metric] = inference_time / len(X_test)

        self.results[model_name][dataset_name] = {
            'training_time': training_time,
            'metrics': metrics_result,
            'model_size': self.get_model_size(model)
        }

    def generate_comparison_report(self):
        """Generate comparison report for all models"""
        report = []
        report.append("# Neural Network Architecture Performance Comparison\n")

        for model_name in self.results:
            report.append(f"## {model_name}\n")
            for dataset_name in self.results[model_name]:
                result = self.results[model_name][dataset_name]
                report.append(f"### {dataset_name} Dataset")
                report.append(f"- Training Time: {result['training_time']:.2f}s")
                report.append(f"- Model Size: {result['model_size']:.1f} MB")

                for metric, value in result['metrics'].items():
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")

                report.append("")

        return "\n".join(report)

    def calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy score"""
        return np.mean(y_true == y_pred)

    def calculate_f1_score(self, y_true, y_pred):
        """Calculate F1 score"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def get_model_size(self, model):
        """Calculate model size in MB"""
        total_params = sum(p.size for p in model.parameters.values())
        return total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)

def create_benchmark_datasets():
    """Create benchmark datasets for comparison"""
    # MNIST-like dataset
    mnist_data = generate_synthetic_data(n_samples=10000, input_dim=784, num_classes=10)

    # CIFAR-10-like dataset
    cifar_data = generate_synthetic_data(n_samples=10000, input_dim=3072, num_classes=10)

    # Large-scale dataset
    large_data = generate_synthetic_data(n_samples=50000, input_dim=1024, num_classes=100)

    return {
        'mnist': mnist_data,
        'cifar10': cifar_data,
        'large_scale': large_data
    }

def generate_synthetic_data(n_samples, input_dim, num_classes):
    """Generate synthetic data for benchmarking"""
    np.random.seed(42)

    X = np.random.randn(n_samples, input_dim)
    y = np.random.randint(0, num_classes, n_samples)

    # Split into train and test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test

# Benchmark different architectures
benchmark = ArchitectureBenchmark(create_benchmark_datasets())

# Define models to benchmark
models_to_benchmark = {
    'MLP_3layers': lambda: create_mlp_model(input_dim=784, hidden_dims=[256, 128, 64], num_classes=10),
    'MLP_5layers': lambda: create_mlp_model(input_dim=784, hidden_dims=[512, 256, 128, 64, 32], num_classes=10),
    'ResNet_10layers': lambda: create_resnet_model(input_dim=784, depth=10, num_classes=10),
    'Transformer_small': lambda: create_transformer_model(input_dim=784, d_model=64, num_layers=3, num_classes=10),
    'VisionTransformer_tiny': lambda: create_vit_model(patch_size=16, num_classes=10, dim=64, depth=3),
    'EfficientNet_tiny': lambda: create_efficientnet_model(num_classes=10, width_coefficient=0.5)
}

# Run benchmarks
for dataset_name in benchmark.datasets:
    for model_name, model_func in models_to_benchmark.items():
        try:
            print(f"Benchmarking {model_name} on {dataset_name}...")
            benchmark.benchmark_architecture(model_func, model_name, dataset_name)
        except Exception as e:
            print(f"Error benchmarking {model_name} on {dataset_name}: {str(e)}")

# Generate comparison report
comparison_report = benchmark.generate_comparison_report()
print(comparison_report)
```

### 9.2 Efficiency Analysis

```python
class EfficiencyAnalyzer:
    def __init__(self):
        self.efficiency_metrics = {}

    def analyze_memory_efficiency(self, model):
        """Analyze memory efficiency of a model"""
        memory_info = {
            'model_parameters': sum(p.size for p in model.parameters.values()),
            'total_flops': self.estimate_flops(model),
            'peak_memory_usage': self.estimate_peak_memory(model),
            'parameter_efficiency': self.calculate_parameter_efficiency(model)
        }
        return memory_info

    def analyze_computational_efficiency(self, model, input_size):
        """Analyze computational efficiency"""
        import time

        # Warmup run
        for _ in range(5):
            _ = model.forward(np.random.randn(*input_size))

        # Benchmark inference
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model.forward(np.random.randn(*input_size))
        avg_inference_time = (time.time() - start_time) / num_runs

        # Memory bandwidth analysis
        memory_bandwidth = self.estimate_memory_bandwidth(model, input_size)

        return {
            'avg_inference_time': avg_inference_time,
            'throughput': 1.0 / avg_inference_time,
            'memory_bandwidth': memory_bandwidth,
            'flops_efficiency': self.calculate_flops_efficiency(model, avg_inference_time)
        }

    def estimate_flops(self, model):
        """Estimate FLOPs for a model"""
        total_flops = 0

        for name, param in model.parameters.items():
            if 'weight' in name:
                # Estimate FLOPs as 2 * input_size * output_size for dense layers
                total_flops += param.size * 2

        return total_flops

    def estimate_peak_memory(self, model):
        """Estimate peak memory usage"""
        # Simplified estimation
        param_memory = sum(p.size for p in model.parameters.values())
        activation_memory = param_memory * 4  # Rough estimate

        return param_memory + activation_memory

    def calculate_parameter_efficiency(self, model):
        """Calculate parameter efficiency (FLOPs per parameter)"""
        flops = self.estimate_flops(model)
        params = sum(p.size for p in model.parameters.values())

        return flops / params if params > 0 else 0

    def estimate_memory_bandwidth(self, model, input_size):
        """Estimate memory bandwidth requirements"""
        # Simplified estimation based on model size and input size
        param_size = sum(p.size for p in model.parameters.values())
        input_size_bytes = np.prod(input_size) * 4  # Assuming float32

        return param_size + input_size_bytes

    def calculate_flops_efficiency(self, model, inference_time):
        """Calculate FLOPs efficiency"""
        flops = self.estimate_flops(model)
        flops_per_second = flops / inference_time
        peak_flops = 10**12  # Assume 1 TFLOPS peak performance

        return flops_per_second / peak_flops

def compare_architectures_efficiency():
    """Compare efficiency of different architectures"""
    architectures = {
        'MLP_small': create_mlp_model(input_dim=1000, hidden_dims=[512, 256]),
        'MLP_large': create_mlp_model(input_dim=1000, hidden_dims=[2048, 1024, 512, 256]),
        'ResNet_medium': create_resnet_model(input_dim=1000, depth=20),
        'Transformer_small': create_transformer_model(input_dim=1000, d_model=128, num_layers=4)
    }

    analyzer = EfficiencyAnalyzer()

    for name, model in architectures.items():
        print(f"\n=== {name} Efficiency Analysis ===")

        memory_info = analyzer.analyze_memory_efficiency(model)
        print(f"Parameters: {memory_info['model_parameters']:,}")
        print(f"FLOPs: {memory_info['total_flops']:,}")
        print(f"Peak Memory: {memory_info['peak_memory_usage']:,} bytes")
        print(f"Parameter Efficiency: {memory_info['parameter_efficiency']:.2f}")

        comp_info = analyzer.analyze_computational_efficiency(model, (1, 1000))
        print(f"Average Inference Time: {comp_info['avg_inference_time']*1000:.2f} ms")
        print(f"Throughput: {comp_info['throughput']:.2f} samples/sec")
        print(f"FLOPs Efficiency: {comp_info['flops_efficiency']:.4f}")
```

### 9.3 Scalability Analysis

```python
class ScalabilityAnalyzer:
    def __init__(self):
        self.scaling_results = {}

    def analyze_depth_scaling(self, model_func, depths, input_dim=1000, num_classes=10):
        """Analyze how performance scales with depth"""
        results = {}

        for depth in depths:
            print(f"Testing depth {depth}...")
            model = model_func(input_dim=input_dim, depth=depth, num_classes=num_classes)

            # Train and evaluate
            X_train, y_train, X_test, y_test = generate_synthetic_data(
                n_samples=5000, input_dim=input_dim, num_classes=num_classes
            )

            start_time = time.time()
            model.train(X_train, y_train, epochs=10)  # Limited epochs for speed
            training_time = time.time() - start_time

            accuracy = model.evaluate(X_test, y_test)
            param_count = sum(p.size for p in model.parameters.values())

            results[depth] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'param_count': param_count,
                'param_efficiency': accuracy / (param_count / 1000000)  # Accuracy per million params
            }

        return results

    def analyze_width_scaling(self, model_func, widths, input_dim=1000, num_classes=10):
        """Analyze how performance scales with width"""
        results = {}

        for width in widths:
            print(f"Testing width {width}...")
            model = model_func(input_dim=input_dim, width=width, num_classes=num_classes)

            # Train and evaluate
            X_train, y_train, X_test, y_test = generate_synthetic_data(
                n_samples=5000, input_dim=input_dim, num_classes=num_classes
            )

            start_time = time.time()
            model.train(X_train, y_train, epochs=10)
            training_time = time.time() - start_time

            accuracy = model.evaluate(X_test, y_test)
            param_count = sum(p.size for p in model.parameters.values())

            results[width] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'param_count': param_count,
                'param_efficiency': accuracy / (param_count / 1000000)
            }

        return results

    def analyze_dataset_scaling(self, model_func, dataset_sizes, input_dim=1000, num_classes=10):
        """Analyze how performance scales with dataset size"""
        results = {}

        for size in dataset_sizes:
            print(f"Testing dataset size {size}...")
            X_train, y_train, X_test, y_test = generate_synthetic_data(
                n_samples=size, input_dim=input_dim, num_classes=num_classes
            )

            model = model_func(input_dim=input_dim, num_classes=num_classes)

            start_time = time.time()
            model.train(X_train, y_train, epochs=5)
            training_time = time.time() - start_time

            accuracy = model.evaluate(X_test, y_test)

            results[size] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'samples_per_second': size / training_time
            }

        return results

    def generate_scaling_report(self, depth_results, width_results, dataset_results):
        """Generate comprehensive scaling report"""
        report = []
        report.append("# Neural Network Scaling Analysis\n")

        # Depth scaling
        report.append("## Depth Scaling Analysis\n")
        report.append("| Depth | Accuracy | Training Time (s) | Parameters (M) | Efficiency |\n")
        report.append("|-------|----------|-------------------|----------------|------------|\n")

        for depth in sorted(depth_results.keys()):
            result = depth_results[depth]
            efficiency = result['param_efficiency']
            report.append(f"| {depth} | {result['accuracy']:.4f} | {result['training_time']:.2f} | "
                         f"{result['param_count']/1000000:.1f} | {efficiency:.2f} |\n")

        # Width scaling
        report.append("\n## Width Scaling Analysis\n")
        report.append("| Width | Accuracy | Training Time (s) | Parameters (M) | Efficiency |\n")
        report.append("|-------|----------|-------------------|----------------|------------|\n")

        for width in sorted(width_results.keys()):
            result = width_results[width]
            efficiency = result['param_efficiency']
            report.append(f"| {width} | {result['accuracy']:.4f} | {result['training_time']:.2f} | "
                         f"{result['param_count']/1000000:.1f} | {efficiency:.2f} |\n")

        # Dataset scaling
        report.append("\n## Dataset Size Scaling Analysis\n")
        report.append("| Dataset Size | Accuracy | Training Time (s) | Samples/sec |\n")
        report.append("|--------------|----------|-------------------|-------------|\n")

        for size in sorted(dataset_results.keys()):
            result = dataset_results[size]
            report.append(f"| {size} | {result['accuracy']:.4f} | {result['training_time']:.2f} | "
                         f"{result['samples_per_second']:.1f} |\n")

        return "\n".join(report)

# Example usage
def run_scaling_analysis():
    """Run comprehensive scaling analysis"""
    analyzer = ScalabilityAnalyzer()

    # Test depth scaling
    print("Running depth scaling analysis...")
    depth_results = analyzer.analyze_depth_scaling(
        create_mlp_model, depths=[2, 4, 6, 8, 10, 12]
    )

    # Test width scaling
    print("Running width scaling analysis...")
    width_results = analyzer.analyze_width_scaling(
        create_mlp_model, widths=[128, 256, 512, 1024, 2048]
    )

    # Test dataset scaling
    print("Running dataset scaling analysis...")
    dataset_results = analyzer.analyze_dataset_scaling(
        create_mlp_model, dataset_sizes=[1000, 5000, 10000, 25000, 50000]
    )

    # Generate report
    scaling_report = analyzer.generate_scaling_report(depth_results, width_results, dataset_results)
    print(scaling_report)

    return scaling_report
```

This enhanced neural networks architecture document now provides comprehensive coverage of modern architectural patterns, optimization techniques, regularization methods, and performance analysis. It includes cutting-edge architectures like Vision Transformers, EfficientNets, and RegNets, along with practical implementation examples and benchmarking methodologies.
