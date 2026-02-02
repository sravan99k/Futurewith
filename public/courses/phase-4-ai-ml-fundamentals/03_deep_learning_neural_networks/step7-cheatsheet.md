# Deep Learning & Neural Networks Cheat Sheet

## Table of Contents

1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Optimizers](#optimizers)
5. [TensorFlow/Keras Quick Reference](#tensorflowkeras-quick-reference)
6. [PyTorch Patterns](#pytorch-patterns)
7. [Common Architectures](#common-architectures)
8. [Training Tips & Best Practices](#training-tips--best-practices)

---

## Neural Network Fundamentals

### Basic Components

- **Neuron**: Basic computational unit
- **Weights**: Learnable parameters
- **Bias**: Learnable offset parameter
- **Layers**: Groups of neurons
  - Input Layer
  - Hidden Layers
  - Output Layer

### Forward Propagation

```
z = W·x + b
a = f(z)  # activation function
```

### Backward Propagation

```
# Gradient calculation chain rule
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

### Common Layer Types

- **Dense/Fully Connected**: All neurons connected
- **Convolutional**: For spatial data
- **Recurrent**: For sequential data
- **Batch Normalization**: Normalize layer inputs
- **Dropout**: Regularization technique
- **Embedding**: Dense vector representation

---

## Activation Functions

### Linear

```python
f(x) = x
```

- **Use Case**: Regression output layer
- **Derivative**: 1

### Sigmoid

```python
f(x) = 1 / (1 + e^(-x))
```

- **Range**: (0, 1)
- **Use Case**: Binary classification
- **Derivative**: f(x) \* (1 - f(x))
- **Problem**: Vanishing gradients

### Tanh

```python
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- **Range**: (-1, 1)
- **Use Case**: Hidden layers in RNNs
- **Derivative**: 1 - f(x)²
- **Advantage**: Zero-centered output

### ReLU (Rectified Linear Unit)

```python
f(x) = max(0, x)
```

- **Range**: [0, ∞)
- **Use Case**: Hidden layers (most common)
- **Derivative**: 1 if x > 0, 0 otherwise
- **Advantage**: Computationally efficient, reduces vanishing gradient

### Leaky ReLU

```python
f(x) = max(αx, x) where α = 0.01
```

- **Range**: (-∞, ∞)
- **Use Case**: When ReLU has dead neuron problem
- **Advantage**: Allows small negative values

### Swish

```python
f(x) = x * sigmoid(x)
```

- **Use Case**: Modern替代 to ReLU
- **Advantage**: Smooth, non-monotonic

### GELU (Gaussian Error Linear Unit)

```python
f(x) = x * Φ(x)  # where Φ is standard normal CDF
```

- **Use Case**: Transformer models
- **Advantage**: Smooth approximation of ReLU

### Softmax

```python
f(x_i) = e^x_i / Σ(e^x_j)
```

- **Range**: (0, 1) for each, sums to 1
- **Use Case**: Multi-class classification output
- **Property**: Normalizes output to probability distribution

---

## Loss Functions

### Regression Losses

#### Mean Squared Error (MSE)

```python
L = (1/n) * Σ(y_true - y_pred)²
```

- **Use Case**: Continuous value prediction
- **Derivative**: 2(y_pred - y_true)

#### Mean Absolute Error (MAE)

```python
L = (1/n) * Σ|y_true - y_pred|
```

- **Use Case**: Robust to outliers
- **Derivative**: -1 if y_true > y_pred, +1 if y_pred > y_true

#### Huber Loss

```python
L = 0.5 * (y_true - y_pred)² if |δ| <= δ
L = δ * |y_true - y_pred| - 0.5 * δ² otherwise
```

- **Use Case**: Robust regression
- **Advantage**: Combines MSE and MAE benefits

### Classification Losses

#### Binary Cross-Entropy

```python
L = -[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
```

- **Use Case**: Binary classification
- **Sigmoid output activation**

#### Categorical Cross-Entropy

```python
L = -Σ(y_true * log(y_pred))
```

- **Use Case**: Multi-class classification
- **Softmax output activation**

#### Sparse Categorical Cross-Entropy

```python
L = -log(y_pred[class_index])
```

- **Use Case**: Multi-class with integer labels
- **Memory efficient when many classes**

### Advanced Losses

#### Focal Loss

```python
L = -α_t * (1 - p_t)^γ * log(p_t)
```

- **Use Case**: Class imbalance problems
- **γ**: Focusing parameter (usually 2)

#### Dice Loss

```python
L = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
```

- **Use Case**: Image segmentation
- **Range**: [0, 1]

#### Triplet Loss

```python
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

- **Use Case**: Metric learning, face recognition
- **d**: Distance function (usually Euclidean)

---

## Optimizers

### Gradient Descent

```python
W = W - learning_rate * gradient
```

- **Pros**: Simple, guarantees convergence to local minimum
- **Cons**: Slow for large datasets, can get stuck in saddle points

### Stochastic Gradient Descent (SGD)

```python
# Update for each sample
W = W - learning_rate * gradient(sample)
```

- **Pros**: Faster, can escape local minima
- **Cons**: Noisy, needs hyperparameter tuning

### Momentum

```python
v = momentum * v + gradient
W = W - learning_rate * v
```

- **momentum**: Usually 0.9
- **Pros**: Faster convergence, reduces oscillations

### Nesterov Accelerated Gradient (NAG)

```python
v = momentum * v + learning_rate * gradient(W - momentum * v)
W = W - v
```

- **Pros**: Better convergence than momentum
- **Look-ahead**: Uses future position

### Adagrad

```python
G = G + gradient²  # Accumulated squared gradients
W = W - learning_rate * gradient / (√G + ε)
```

- **ε**: Small constant (1e-8)
- **Pros**: Adaptive learning rates per parameter
- **Cons**: Learning rate decreases monotonically

### RMSprop

```python
G = decay_rate * G + (1 - decay_rate) * gradient²
W = W - learning_rate * gradient / (√G + ε)
```

- **decay_rate**: Usually 0.9-0.99
- **Pros**: Solves Adagrad's vanishing learning rate

### Adam (Adaptive Moment Estimation)

```python
m = β1 * m + (1 - β1) * gradient  # First moment
v = β2 * v + (1 - β2) * gradient²  # Second moment
m_hat = m / (1 - β1^t)  # Bias correction
v_hat = v / (1 - β2^t)
W = W - learning_rate * m_hat / (√v_hat + ε)
```

- **β1**: 0.9, **β2**: 0.999
- **Most popular optimizer**
- **Combines momentum and RMSprop benefits**

### AdamW

```python
# Adam with decoupled weight decay
L_total = L_original + weight_decay * ||W||²
```

- **Decouples weight decay from gradient update**
- **Better generalization**

### Learning Rate Schedules

#### Step Decay

```python
learning_rate = initial_lr * gamma^(floor(epoch / step_size))
```

#### Exponential Decay

```python
learning_rate = initial_lr * exp(-k * epoch)
```

#### Cosine Annealing

```python
learning_rate = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / T)) / 2
```

---

## TensorFlow/Keras Quick Reference

### Model Creation

#### Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

#### Functional API

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

#### Model Subclassing

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)
```

### Layer Patterns

#### Convolution

```python
# 2D Convolution
layers.Conv2D(32, (3, 3), activation='relu', padding='same')
layers.MaxPooling2D((2, 2))
layers.GlobalAveragePooling2D()

# 1D Convolution (for sequences)
layers.Conv1D(64, 3, activation='relu')
```

#### Recurrent

```python
# LSTM
layers.LSTM(128, return_sequences=True, return_state=False)
layers.LSTM(128, return_sequences=False, return_state=True)

# GRU
layers.GRU(128, return_sequences=True)

# Bidirectional
layers.Bidirectional(layers.LSTM(64))
```

#### Attention

```python
# Multi-Head Attention
layers.MultiHeadAttention(num_heads=8, key_dim=64)

# Attention (deprecated, use MultiHeadAttention)
# layers.Attention()
```

### Common Layers

```python
# Dense
layers.Dense(units, activation=None, use_bias=True)

# Dropout
layers.Dropout(rate, seed=None)

# Batch Normalization
layers.BatchNormalization(momentum=0.99, epsilon=0.001)

# Layer Normalization
layers.LayerNormalization(epsilon=1e-6)

# Embedding
layers.Embedding(input_dim, output_dim, mask_zero=True)

# Flatten
layers.Flatten()
layers.GlobalAveragePooling2D()
```

### Compiling and Training

```python
# Compile
model.compile(
    optimizer='adam',  # or keras.optimizers.Adam(learning_rate=0.001)
    loss='sparse_categorical_crossentropy',  # or 'mse', 'binary_crossentropy'
    metrics=['accuracy']  # or ['mae', 'mse']
)

# Training
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.h5'),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)
```

### Common Callbacks

```python
# Early Stopping
keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# Model Checkpoint
keras.callbacks.ModelCheckpoint(
    filepath='model_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_loss'
)

# Learning Rate Scheduler
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
)

# TensorBoard
keras.callbacks.TensorBoard(log_dir='./logs')
```

### Preprocessing

```python
# Text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=100, padding='post')

# Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

### Utilities

```python
# Load model
model = keras.models.load_model('model.h5')

# Model summary
model.summary()

# Get model configuration
config = model.get_config()
weights = model.get_weights()

# Custom layer
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units))
        self.b = self.add_weight(shape=(self.units,))

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

---

## PyTorch Patterns

### Model Definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Common Modules

```python
# Linear
nn.Linear(in_features, out_features, bias=True)

# Convolution
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
nn.Conv1d(in_channels, out_channels, kernel_size)

# Pooling
nn.MaxPool2d(kernel_size, stride=None)
nn.AvgPool2d(kernel_size)
nn.AdaptiveAvgPool2d(output_size)

# Recurrent
nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers=1)
nn.RNN(input_size, hidden_size, num_layers=1)

# Attention
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)

# Normalization
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Dropout
nn.Dropout(p=0.5, inplace=False)
nn.Dropout2d(p=0.5)

# Embedding
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
```

### Training Loop

```python
import torch.optim as optim

# Model, loss, optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Calculate loss
        loss.backward()        # Backward pass
        optimizer.step()       # Update weights

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
```

### Data Loading

```python
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]

# DataLoader
dataset = CustomDataset(X, y, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### CUDA and Device Management

```python
# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = model.to(device)

# Move data to device
data = data.to(device)
target = target.to(device)

# GPU memory management
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear cache
    torch.cuda.ipc_collect()  # Clear memory fragments
```

### Saved and Loading Models

```python
# Save entire model
torch.save(model, 'model.pth')

# Save state dict
torch.save(model.state_dict(), 'model_state.pth')

# Load model
model = torch.load('model.pth')
model.eval()

# Load state dict
model = Net()
model.load_state_dict(torch.load('model_state.pth'))
model.eval()
```

### Common Optimizers and Losses

```python
# Optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Loss functions
criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
criterion = nn.NLLLoss()  # Negative log likelihood
```

### Autograd and Custom Gradients

```python
# Enable/disable gradients
with torch.no_grad():
    # No gradients computed
    pass

# Custom autograd function
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2

# Use in module
class MyModule(nn.Module):
    def forward(self, x):
        return MyFunction.apply(x)
```

---

## Common Architectures

### Convolutional Neural Networks (CNN)

#### LeNet-5 (1998)

```python
# For MNIST
model = nn.Sequential(
    nn.Conv2d(1, 6, 5), nn.Sigmoid(), nn.AvgPool2d(2),
    nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
```

#### AlexNet (2012)

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```

#### VGGNet (2014)

```python
# VGG16 configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Where 'M' means max pooling

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

#### ResNet (2015)

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, 1)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### Recurrent Neural Networks (RNN)

#### Simple RNN

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, hidden = self.rnn(x)
        # Use the last hidden state
        out = self.fc(out[:, -1, :])
        return out
```

#### LSTM

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out
```

#### GRU

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### Transformer (2017)

#### Positional Encoding

```python
import math

def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
```

#### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(attention_output)
        return output
```

#### Full Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

#### BERT-like Architecture

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, num_classes):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(0.1)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        seq_length = x.size(1)

        # Create position indices
        position_indices = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)

        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(position_indices)
        x = self.dropout(token_emb + pos_emb)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Use [CLS] token (first token) for classification
        x = self.norm(x)
        cls_output = x[:, 0]  # First token
        output = self.classifier(cls_output)

        return output
```

---

## Training Tips & Best Practices

### General Guidelines

1. **Start Simple**: Begin with simple models before complex ones
2. **Data Quality**: Garbage in, garbage out - focus on data quality
3. **Validation**: Always use proper train/validation/test splits
4. **Early Stopping**: Prevent overfitting with patience-based stopping
5. **Learning Rate**: Most important hyperparameter to tune
6. **Batch Size**: Affects convergence and generalization
7. **Regularization**: Use dropout, weight decay, data augmentation

### Hyperparameter Tuning

```python
# Common hyperparameter ranges
learning_rate: [0.001, 0.0001, 0.00001]  # Log scale
batch_size: [16, 32, 64, 128]  # Powers of 2
dropout_rate: [0.1, 0.2, 0.3, 0.5]  # Not too high
weight_decay: [1e-5, 1e-4, 1e-3, 1e-2]  # Log scale
```

### Debugging Training

```python
# Check for NaN losses
if torch.isnan(loss):
    print("NaN loss detected!")
    break

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.6f}")
```

### Data Normalization

```python
# Image data (TensorFlow)
model.compile(...,
              normalization_layer=keras.layers.Rescaling(1./255))

# PyTorch normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Transfer Learning

```python
# TensorFlow
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base model

# Add custom head
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# PyTorch
model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze parameters

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

### Memory Optimization

```python
# Gradient accumulation for large batch sizes
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision training (PyTorch)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Evaluation

```python
# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=class_names))
```

### Save/Load Best Practices

```python
# TensorFlow
model.save('model.h5')  # Single file
model.save_weights('weights.h5')  # Just weights

# PyTorch
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Common Issues and Solutions

#### Vanishing Gradients

- **Problem**: Gradients become very small
- **Solutions**:
  - Use ReLU family activations
  - Proper initialization (Xavier, He)
  - Batch normalization
  - Residual connections

#### Exploding Gradients

- **Problem**: Gradients become very large
- **Solutions**:
  - Gradient clipping
  - Lower learning rate
  - Proper initialization
  - Regularization

#### Overfitting

- **Problem**: Model performs well on training, poor on validation
- **Solutions**:
  - More training data
  - Data augmentation
  - Regularization (dropout, weight decay)
  - Early stopping
  - Reduce model complexity

#### Underfitting

- **Problem**: Poor performance on both training and validation
- **Solutions**:
  - Increase model complexity
  - Train longer
  - Reduce regularization
  - Add features

---

## Quick Reference Tables

### Activation Function Comparison

| Function   | Formula                   | Range  | Derivative                               | Best Use            |
| ---------- | ------------------------- | ------ | ---------------------------------------- | ------------------- |
| Sigmoid    | 1/(1+e^(-x))              | (0,1)  | x(1-x)                                   | Binary output       |
| Tanh       | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1) | 1-x²                                     | Hidden layers (RNN) |
| ReLU       | max(0,x)                  | [0,∞)  | 1 if x>0 else 0                          | Hidden layers       |
| Leaky ReLU | max(αx,x) α=0.01          | (-∞,∞) | 1 if x>0 else α                          | Alternative to ReLU |
| Swish      | x·sigmoid(x)              | (-∞,∞) | sigmoid(x) + x·sigmoid(x)·(1-sigmoid(x)) | Modern alternative  |

### Loss Function Selection

| Problem Type                 | Loss Function                    | Output Activation |
| ---------------------------- | -------------------------------- | ----------------- |
| Binary Classification        | Binary Cross-Entropy             | Sigmoid           |
| Multi-class Classification   | Categorical Cross-Entropy        | Softmax           |
| Multi-class (integer labels) | Sparse Categorical Cross-Entropy | Softmax           |
| Regression                   | MSE/MAE/Huber                    | Linear            |
| Class Imbalance              | Focal Loss                       | Sigmoid/Softmax   |
| Segmentation                 | Dice Loss                        | Sigmoid           |

### Optimizer Comparison

| Optimizer | Key Parameter            | Learning Rate | Pros                  | Cons                            |
| --------- | ------------------------ | ------------- | --------------------- | ------------------------------- |
| SGD       | lr, momentum             | Manual tuning | Simple, stable        | Slow, local minima              |
| Momentum  | lr, momentum=0.9         | Manual tuning | Faster convergence    | Oscillations                    |
| Adam      | lr, β1=0.9, β2=0.999     | Auto          | Adaptive, fast        | Overfitting risk                |
| AdamW     | lr, β1, β2, weight_decay | Auto          | Better generalization | More hyperparameters            |
| RMSprop   | lr, α=0.99               | Auto          | Adapts to data        | Sensitive to initial conditions |

### Architecture Summary

| Architecture | Year  | Key Innovation              | Parameters (approx) | Use Case             |
| ------------ | ----- | --------------------------- | ------------------- | -------------------- |
| LeNet        | 1998  | CNN for digits              | 60K                 | Handwritten digits   |
| AlexNet      | 2012  | Deep CNN, ReLU              | 60M                 | Image classification |
| VGG          | 2014  | Deep stacks of 3x3          | 138M                | Transfer learning    |
| ResNet       | 2015  | Skip connections            | 25M-152M            | Very deep networks   |
| LSTM         | 1997  | Memory cells                | Varies              | Sequence modeling    |
| Transformer  | 2017  | Self-attention              | Varies              | NLP, computer vision |
| BERT         | 2018  | Bidirectional transformers  | 110M-340M           | NLP tasks            |
| GPT          | 2018+ | Autoregressive transformers | 117M-175B           | Text generation      |

---

**Last Updated**: November 2025
**Version**: 1.0
**Maintainer**: Deep Learning Reference Team
