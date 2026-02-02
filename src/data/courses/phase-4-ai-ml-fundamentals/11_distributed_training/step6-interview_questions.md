# Advanced AI Interview Questions

## Table of Contents

1. [Neural Network Architectures](#network-architectures)
2. [Attention and Transformer Mechanisms](#attention-transformers)
3. [Specialized Applications](#specialized-applications)
4. [Research-Level Questions](#research-level)
5. [Implementation and Optimization](#implementation-optimization)
6. [Multi-modal and Emerging Architectures](#multimodal-emerging)
7. [Theory and Mathematical Foundations](#theory-mathematical)

---

## 1. Neural Network Architectures {#network-architectures}

### Q1: Explain the fundamental differences between ResNet, DenseNet, and Highway Networks in terms of feature propagation.

**Answer:** These architectures address the vanishing gradient problem in different ways:

**ResNet (Residual Networks):**

- Uses skip connections: `y = F(x) + x`
- Preserves identity mapping: Direct addition of input to output
- Each layer learns residual function `F(x) = y - x`
- Gradient flows directly through skip connections
- Dimension matching via 1×1 convolutions when needed

**DenseNet (Dense Convolutional Networks):**

- Uses dense connections: `x_l = H_l([x_0, x_1, ..., x_{l-1}])`
- Concatenates all previous feature maps
- Each layer receives feature maps from all preceding layers
- Bottleneck layers (1×1 conv) reduce parameters
- Transition layers compress feature maps

**Highway Networks:**

- Uses gating mechanism: `y = T(x) × F(x) + (1-T(x)) × x`
- Learnable transform gate `T(x)` and carry gate `1-T(x)`
- Controlled information flow through learnable parameters
- Allows some information to bypass transformations

**Key Differences:**

- ResNet: Addition preserves information
- DenseNet: Concatenation increases feature diversity
- Highway: Learnable control of information flow

### Q2: How do you choose between using a 1×1, 3×3, or 5×5 convolution in a CNN?

**Answer:** The choice depends on the computational cost vs. receptive field requirements:

**1×1 Convolutions:**

- **Purpose:** Dimensionality reduction, channel mixing
- **Cost:** O(N) per pixel (N = number of channels)
- **Use when:** Reducing parameters, increasing non-linearity, creating bottlenecks
- **Example:** MobileNet uses 1×1 convs to reduce computation by 10×

**3×3 Convolutions:**

- **Purpose:** Capturing local patterns with good balance
- **Cost:** O(9N) per pixel
- **Use when:** Standard feature extraction, best cost-benefit ratio
- **Modern networks:** Most commonly used due to efficiency

**5×5 Convolutions:**

- **Purpose:** Larger receptive field
- **Cost:** O(25N) per pixel (2.8× cost of 3×3)
- **Use when:** Need larger spatial context
- **Example:** Some early Inception modules used 5×5 convs

**Best Practices:**

- Replace 5×5 with two 3×3 convs (same receptive field, 28% less computation)
- Use 1×1 convs for efficient dimensionality reduction
- Consider factorized convolutions (spatial + depthwise)

### Q3: Compare Batch Normalization, Layer Normalization, and Group Normalization. When would you use each?

**Answer:** These normalization techniques differ in computation and application:

**Batch Normalization:**

```python
# Normalize across batch dimension
bn = (x - mean_batch) / sqrt(var_batch + ε) * γ + β
```

- **Pros:** Accelerates training, reduces internal covariate shift
- **Cons:** Batch-dependent, unstable for small batches, conflicts with RNNs
- **Use when:** Large batch sizes, feedforward networks, CNNs with sufficient memory

**Layer Normalization:**

```python
# Normalize across feature dimension for each sample
ln = (x - mean_feature) / sqrt(var_feature + ε) * γ + β
```

- **Pros:** Batch-independent, works with small batches
- **Cons:** Computationally more expensive, may not leverage batch statistics
- **Use when:** RNNs, Transformers, small batch sizes

**Group Normalization:**

```python
# Normalize across feature groups
gn = (x - mean_group) / sqrt(var_group + ε) * γ + β
```

- **Pros:** Between BN and LN, works well with small batches
- **Cons:** More hyperparameters to tune (number of groups)
- **Use when:** Object detection, instance segmentation, video understanding

**Recommendation:**

- **Vision tasks with large batches:** BatchNorm
- **Language tasks, small batches:** LayerNorm (Transformers)
- **Computer vision with small batches:** GroupNorm

### Q4: Explain the concept of neural architecture search (NAS) and discuss the challenges.

**Answer:** NAS automates the design of neural network architectures:

**Key Components:**

- **Search Space:** Defines possible architectures
- **Search Strategy:** How to explore the space (evolution, RL, gradient-based)
- **Performance Estimation:** How to evaluate architecture performance

**Search Strategies:**

1. **Reinforcement Learning-based:**

   ```python
   # Example: RNN controller generates architecture
   action = controller_rnn()
   reward = evaluate_architecture(action)
   update_controller_rnn(reward)
   ```

   - Pro: Effective, widely used
   - Con: Computationally expensive, requires many architectures

2. **Evolutionary Algorithms:**
   - Initialize population of architectures
   - Evaluate fitness (performance)
   - Apply mutation and crossover
   - Select best architectures

3. **Gradient-based (DARTS):**
   - Treat architecture as continuous variable
   - Use gradient descent to optimize architecture weights
   - Much faster than discrete NAS

**Challenges:**

- **Computational Cost:** Evaluating thousands of architectures
- **Transferability:** Architectures optimized for one task may not transfer
- **Search Space Design:** Balancing expressiveness and tractability
- **Hardware-Aware NAS:** Accounting for deployment constraints
- **Fair Evaluation:** Ensuring consistent comparison

**Modern Approaches:**

- **ENAS:** Efficient Neural Architecture Search
- **ProxylessNAS:** Direct hardware search
- **FBNet:** Hardware-efficient mobile architectures
- **Once-for-All:** Single trained network, many architectures

### Q5: How do attention mechanisms in CNNs differ from those in Transformers?

**Answer:** These attention types serve different purposes and operate differently:

**CNN Attention (e.g., CBAM, SENet):**

```python
# Channel attention
channel_att = sigmoid(MLP(GlobalAvgPool(features)))
# Spatial attention
spatial_att = sigmoid(Conv(GlobalMaxPool(features)))
# Apply attention
attended = features * channel_att * spatial_att
```

**Purpose:** Enhance important features, suppress irrelevant ones
**Scope:** Local spatial regions within each feature map
**Application:** Channel-wise and spatial attention on intermediate features
**Computation:** Lightweight MLPs and convolutions
**Use case:** Improve CNN feature representations

**Transformer Attention:**

```python
# Self-attention
attn = softmax(QK^T / sqrt(d_k))V
# Multi-head
multi_head = Concat(head1, head2, ...)W
```

**Purpose:** Model long-range dependencies, capture sequence relationships
**Scope:** Global relationships across entire sequence
**Application:** Primary computation mechanism in Transformer blocks
**Computation:** Heavy matrix multiplications (quadratic in sequence length)
**Use case:** Sequence modeling, language understanding

**Key Differences:**

| Aspect                 | CNN Attention       | Transformer Attention        |
| ---------------------- | ------------------- | ---------------------------- |
| **Scope**              | Local spatial       | Global sequence              |
| **Purpose**            | Feature enhancement | Relationship modeling        |
| **Computation**        | Light               | Heavy                        |
| **Dependencies**       | Local features      | All sequence positions       |
| **Position awareness** | Implicit            | Requires positional encoding |
| **Memory**             | Low                 | High (quadratic)             |

**Hybrid Approaches:**

- **Swin Transformer:** Hierarchical attention with windowed computation
- **ConvNeXt:** Modern convolutions inspired by Transformers
- **MobileViT:** Combining CNN locality with Transformer global modeling

---

## 2. Attention and Transformer Mechanisms {#attention-transformers}

### Q6: Derive the self-attention mechanism mathematically and explain why scaling is necessary.

**Answer:** Self-attention computes relationships between all pairs of positions in a sequence:

**Mathematical Derivation:**

For input sequence `X ∈ R^(B×N×D)` where B=batch, N=sequence length, D=embedding dim:

1. **Linear Projections:**

```
Q = XW_Q, K = XW_K, V = XW_V
where W_Q, W_K, W_V ∈ R^(D×D), Q, K, V ∈ R^(B×N×D)
```

2. **Scaled Dot-Product:**

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Why Scaling is Necessary:**

Without scaling, the variance of attention scores grows with `d_k`:

```
E[QK^T] = E[XW_Q * XW_K^T] = E[XX^T] * E[W_Q W_K^T]
```

If `Q` and `K` are zero-centered with variance `σ²`:

```
Var(QK^T) = N * σ⁴ * d_k
```

**Problem:** As `d_k` increases:

- Attention scores become more extreme (large positive/negative)
- Softmax becomes saturated (very peaked distribution)
- Gradients become very small
- Training becomes unstable

**Solution:** Divide by `√d_k`:

```
Attention = softmax(QK^T / √d_k)V
```

With scaling:

```
Var(QK^T / √d_k) = σ⁴ * N
```

Variance remains constant, keeping softmax in stable region.

### Q7: Explain the different types of positional encodings in Transformers and their trade-offs.

**Answer:** Transformers lack inherent positional information, requiring explicit positional encodings:

**1. Sinusoidal Positional Encoding:**

```python
# Original Transformer approach
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Pros:**

- No learned parameters
- Can extrapolate to longer sequences
- Linear relationships allow relative attention
- Stable training

**Cons:**

- Fixed pattern may not be optimal
- Cannot handle variable-length sequences during training
- Limited expressiveness

**2. Learned Positional Embeddings:**

```python
# Add learned position embeddings
embedding = token_embedding + position_embedding
```

**Pros:**

- Flexible, can learn optimal encoding
- Better performance on many tasks
- Simpler implementation

**Cons:**

- Limited to maximum sequence length seen in training
- Cannot extrapolate to longer sequences
- Additional parameters to learn

**3. Relative Positional Encodings:**

```python
# Relative position between tokens matters
attention_score = (Q_i)K_j + relative_pos(i,j)
```

**Pros:**

- Better for tasks requiring relative positioning
- More efficient for variable lengths
- Captures relative relationships

**Cons:**

- More complex implementation
- Higher computational cost

**4. Rotary Positional Embeddings (RoPE):**

```python
# Apply rotation in embedding space
q_m = R_m q_m, k_n = R_n k_n
# where R_m is rotation matrix
```

**Pros:**

- Preserves relative position information
- Works for any sequence length
- Good extrapolation properties

**Cons:**

- More complex mathematics
- Slightly higher computational cost

**Modern Approaches:**

- **ALiBi:** Linear bias for attention
- **T5 Relative Position:** Relative position bias
- **Position Information Integration:** Various hybrid approaches

### Q8: How would you implement a multi-head attention mechanism from scratch, and what are the key optimization strategies?

**Answer:** Multi-head attention allows the model to attend to different representation subspaces:

**Implementation:**

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 1. Linear projections and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 4. Output linear projection
        output = self.W_o(attention_output)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
```

**Key Optimizations:**

1. **Memory Efficiency:**

```python
# Use torch.bmm for batch matrix multiplication
attention_output = torch.bmm(attention_weights, V)

# For training, use efficient attention implementations
# Flash Attention, Memory Efficient Attention
```

2. **Computation Reduction:**

```python
# Linear attention (approximation)
# Replaces O(n²) with O(n) computation
Q, K, V = query, key, value
φ = FeatureMap()  # e.g., ELU + 1

K_exp = φ(K)
V_exp = φ(V)

# Linear complexity attention
attention = softmax(qv @ k^T) @ V ≈ (φ(qv)^T @ φ(k)) @ V
```

3. **Hardware Optimization:**

```python
# Use mixed precision for GPU training
with torch.cuda.amp.autocast():
    output = mha(query, key, value)

# Gradient checkpointing for memory savings
output = torch.utils.checkpoint.checkpoint(mha, query, key, value)
```

4. **Batch Optimization:**

```python
# Pad sequences to power-of-2 lengths
seq_lens = [len(seq) for seq in sequences]
padded_len = max(2**i for i in range(int(math.log2(max(seq_lens))) + 2))

# Use packed sequences for variable length
packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lens, batch_first=True)
```

### Q9: Compare the computational complexity of different attention mechanisms and discuss their trade-offs.

**Answer:** Attention mechanism complexity varies significantly based on sequence length and model size:

**Standard Self-Attention:**

```
Time Complexity: O(n²d)
Space Complexity: O(n²)
```

Where n=sequence length, d=embedding dimension

**Breakdown:**

- QK^T computation: O(n²d)
- Softmax and matmul: O(n²d)
- Output projection: O(nd²)

**Efficient Attention Variants:**

1. **Linear Attention:**

```
Time Complexity: O(nd²)
Space Complexity: O(nd)
```

**Approximation Method:**

```python
# Replaces O(n²) with O(n) through factorization
def linear_attention(Q, K, V):
    # Feature maps
    φ = nn.ELU()
    K_proj = φ(K)
    V_proj = φ(V)

    # Linear complexity computation
    K_sum = torch.cumsum(K_proj, dim=1)
    V_sum = torch.cumsum(V_proj, dim=1)

    # Attention computation
    output = K_sum * V_sum  # Simplified
    return output
```

**Trade-offs:**

- ✅ O(n) instead of O(n²) for long sequences
- ✅ Better for very long sequences (>4096)
- ❌ Approximation may reduce accuracy
- ❌ Doesn't capture precise attention patterns

2. **Local Attention:**

```
Time Complexity: O(nwd)
Space Complexity: O(nw)
```

Where w=window size (typically 64-256)

**Implementation:**

```python
def local_attention(Q, K, V, window_size=64):
    batch_size, seq_len, d_model = Q.shape
    output = torch.zeros_like(Q)

    for i in range(seq_len):
        # Local window
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)

        # Compute attention only within window
        local_Q = Q[:, i:i+1]  # [batch, 1, d]
        local_K = K[:, start:end]  # [batch, window, d]
        local_V = V[:, start:end]  # [batch, window, d]

        attn_scores = torch.matmul(local_Q, local_K.transpose(-2, -1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output[:, i:i+1] = torch.matmul(attn_weights, local_V)

    return output
```

**Trade-offs:**

- ✅ Linear complexity in sequence length
- ✅ Good for tasks with local dependencies
- ❌ Cannot capture long-range dependencies
- ❌ May need multiple layers for global context

3. **Hierarchical Attention:**

```
Time Complexity: O(nd log n)
Space Complexity: O(nd)
```

**Multi-scale Processing:**

```python
def hierarchical_attention(X, levels=3):
    # Process at multiple scales
    outputs = []
    current = X

    for level in range(levels):
        # Attention at this level
        output, _ = self_attention(current)

        # Downsample for next level
        if level < levels - 1:
            current = F.avg_pool1d(
                current.transpose(1, 2),
                kernel_size=2,
                stride=2
            ).transpose(1, 2)

        outputs.append(output)

    # Combine multi-scale outputs
    return combine_multi_scale(outputs)
```

**Comparison Table:**

| Attention Type   | Time        | Space  | Best for            | Limitation         |
| ---------------- | ----------- | ------ | ------------------- | ------------------ |
| **Standard**     | O(n²d)      | O(n²)  | n < 1024            | Quadratic cost     |
| **Linear**       | O(nd²)      | O(nd)  | Very long sequences | Accuracy trade-off |
| **Local**        | O(nwd)      | O(nw)  | Local patterns      | No long-range      |
| **Hierarchical** | O(nd log n) | O(nd)  | Multi-scale         | Complexity         |
| **Sparse**       | O(n√n)      | O(n√n) | Structured data     | Implementation     |

**Modern Efficient Implementations:**

- **Flash Attention:** Memory-optimized attention
- **Performer:** Kernel-based linear attention
- **BigBird:** Combined local + random + global attention
- **Longformer:** Local + global attention pattern

### Q10: Explain how Vision Transformers (ViT) work and compare them to CNNs. When would you choose ViT over CNNs?

**Answer:** Vision Transformers apply attention mechanisms to computer vision tasks:

**ViT Architecture:**

```python
class VisionTransformer:
    def __init__(self, patch_size, d_model, num_heads, num_layers):
        # Convert image to patches
        self.patch_embed = PatchEmbedding(patch_size, d_model)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding
        self.pos_embed = PositionalEncoding(d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, N_patches, d_model]

        # Add class token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N_patches+1, d_model]

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Use class token for classification
        cls_output = x[:, 0]  # [B, d_model]
        logits = self.classifier(cls_output)

        return logits
```

**Key Components:**

1. **Patch Embedding:**
   - Split image into fixed-size patches (16×16 or 32×32)
   - Flatten and linearly project to embedding dimension
   - Preserves spatial information through positional encodings

2. **Class Token:**
   - Special token for classification
   - Aggregates information from all patches
   - Similar to BERT's [CLS] token

3. **Positional Encoding:**
   - Learns or computes position information
   - Preserves 2D spatial relationships
   - Essential for spatial understanding

**Comparison with CNNs:**

| Aspect                       | CNNs                              | ViT                         |
| ---------------------------- | --------------------------------- | --------------------------- |
| **Inductive Biases**         | Built-in translation equivariance | None (learns from data)     |
| **Data Requirements**        | Works well with limited data      | Requires large datasets     |
| **Computational Complexity** | Local operations, efficient       | Global attention, expensive |
| **Scalability**              | Better for small models           | Better for large models     |
| **Interpretability**         | More interpretable features       | Less interpretable          |
| **Long-range Dependencies**  | Through multiple layers           | Direct attention            |

**When to Choose ViT:**

✅ **Choose ViT when:**

- Large datasets (>1M images)
- Global context is important
- Model size is large (>100M parameters)
- Need state-of-the-art performance
- Computational resources are abundant

✅ **Choose CNNs when:**

- Limited training data
- Real-time inference required
- Mobile/edge deployment
- Strong inductive bias is beneficial
- Resource-constrained environments

**Hybrid Approaches:**

- **Swin Transformer:** Hierarchical attention with windowed computation
- **ConvNeXt:** CNNs inspired by Transformer designs
- **MobileViT:** Lightweight Vision Transformers
- **CoAtNet:** Combined convolution and attention

**Performance Considerations:**

- ViT scales better with model size and data
- CNNs are more efficient for small models
- ViT benefits more from pre-training
- CNNs have better hardware support

---

## 3. Specialized Applications {#specialized-applications}

### Q11: How would you design a neural network for time series forecasting with multiple variables and long-term dependencies?

**Answer:** Time series forecasting requires handling temporal dependencies, multiple variables, and often long sequences:

**Architecture Design:**

1. **Temporal Convolutional Network (TCN):**

```python
class TemporalConvNet:
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            self.layers.append(
                ConvBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=(kernel_size-1)//2 * dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out
```

2. **Transformer for Time Series:**

```python
class TimeSeriesTransformer:
    def __init__(self, input_dim, d_model, num_heads, num_layers, seq_len, forecast_horizon):
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Encoder for historical data
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )

        # Decoder for forecasting
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt=None):
        # Project input and add positional encoding
        src = self.input_proj(src)
        src = self.pos_encoding(src)

        # Encode historical data
        memory = self.encoder(src)

        if tgt is not None:  # Training
            tgt = self.input_proj(tgt)
            tgt = self.pos_encoding(tgt)
            output = self.decoder(tgt, memory)
        else:  # Inference - autoregressive
            output = self.autoregressive_decode(memory)

        return self.output_proj(output)

    def autoregressive_decode(self, memory):
        # Generate one step at a time
        batch_size = memory.size(0)
        tgt = torch.zeros(batch_size, 1, memory.size(-1), device=memory.device)

        for _ in range(self.forecast_horizon):
            tgt = self.pos_encoding(tgt)
            output = self.decoder(tgt, memory)
            next_token = self.output_proj(output[:, -1:])
            tgt = torch.cat([tgt, next_token], dim=1)

        return tgt[:, 1:]  # Remove initial zero step
```

3. **LSTM with Attention:**

```python
class LSTMForecaster:
    def __init__(self, input_dim, hidden_dim, num_layers, attention_dim):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = TemporalAttention(hidden_dim, attention_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)

        # Temporal attention
        attended_out, attention_weights = self.attention(lstm_out)

        # Final prediction
        prediction = self.decoder(attended_out[:, -1, :])  # Last time step

        return prediction, attention_weights

class TemporalAttention:
    def __init__(self, hidden_dim, attention_dim):
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, lstm_out):
        # Compute attention scores
        scores = self.attention_net(lstm_out)  # [batch, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum
        attended = torch.sum(lstm_out * attention_weights, dim=1)

        return attended, attention_weights
```

**Key Design Considerations:**

1. **Handling Long-term Dependencies:**
   - Use dilated convolutions (TCN)
   - Skip connections in deep networks
   - Transformer attention mechanism
   - LSTM/GRU with attention

2. **Multi-variable Handling:**
   - Shared encoder with variable-specific heads
   - Cross-variable attention
   - Graph neural networks for variable relationships

3. **Feature Engineering:**
   - Time-based features (hour, day, month)
   - Lagged features
   - Rolling statistics
   - Seasonal decomposition

**Training Strategies:**

- **Sliding window approach:** Train on past N steps to predict next M steps
- **Multi-step loss:** Combine losses for different forecast horizons
- **Regularization:** Dropout, weight decay, early stopping
- **Data augmentation:** Time warping, scaling, noise injection

### Q12: Design a neural network architecture for graph neural networks that can handle both node classification and link prediction simultaneously.

**Answer:** Graph Neural Networks extend neural networks to graph-structured data:

**Unified GNN Architecture:**

```python
class MultiTaskGNN:
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, num_edge_types):
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, input_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Node classification head
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        # Link prediction head
        self.link_predictor = LinkPredictionLayer(hidden_dim, num_edge_types)

        # Message passing functions
        self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_features, adjacency, edge_features=None, edge_types=None):
        # Initial node features
        h = self.node_embedding.weight if node_features is None else node_features

        # Message passing
        for layer in self.gnn_layers:
            h = self.message_passing(h, adjacency, edge_features, layer)

        # Multi-task outputs
        node_logits = self.node_classifier(h)
        edge_scores = self.link_predictor(h, adjacency, edge_types)

        return node_logits, edge_scores

    def message_passing(self, h, adjacency, edge_features, gnn_layer):
        # Compute messages
        messages = self.compute_messages(h, adjacency, edge_features)

        # Aggregate messages
        aggregated = self.aggregate_messages(messages, adjacency)

        # Update node representations
        h_new = gnn_layer(aggregated)

        return h_new + h  # Residual connection

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h, adjacency):
        # Simple GCN: A_hat^(-1/2) * A_hat * A_hat^(-1/2) * H * W
        # Simplified version
        h_transformed = self.linear(h)
        h_aggr = torch.mm(adjacency, h_transformed)
        h_new = F.relu(h_aggr)
        return self.dropout(h_new)

class LinkPredictionLayer(nn.Module):
    def __init__(self, hidden_dim, num_edge_types):
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # Link scoring functions for each edge type
        self.link_scorers = nn.ModuleList([
            LinkScoringFunction(hidden_dim) for _ in range(num_edge_types)
        )

    def forward(self, node_embeddings, adjacency, edge_types):
        batch_size = node_embeddings.size(0)
        edge_scores = {}

        # For each edge type, compute link scores
        for edge_type in range(self.num_edge_types):
            # Get nodes connected by this edge type
            mask = edge_types == edge_type
            if not mask.any():
                continue

            # Compute scores for this edge type
            scores = self.link_scorers[edge_type](node_embeddings, adjacency, mask)
            edge_scores[f"type_{edge_type}"] = scores

        return edge_scores

class LinkScoringFunction(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Different scoring methods
        self.dot_product = DotProduct()
        self.bilinear = Bilinear(hidden_dim)
        self.neural_net = NeuralNet(hidden_dim)

        # Attention over scoring methods
        self.attention = nn.Linear(hidden_dim * 3, 1)

    def forward(self, node_embeddings, adjacency, edge_mask):
        # Get connected node pairs
        row, col = edge_mask.nonzero(as_tuple=True)

        # Extract node embeddings for connected pairs
        node_i = node_embeddings[row]
        node_j = node_embeddings[col]

        # Different scoring functions
        dot_scores = self.dot_product(node_i, node_j)
        bilin_scores = self.bilinear(node_i, node_j)
        neural_scores = self.neural_net(node_i, node_j)

        # Combine scores with attention
        combined = torch.cat([dot_scores, bilin_scores, neural_scores], dim=-1)
        attention_weights = F.softmax(self.attention(combined), dim=0)
        final_scores = torch.sum(combined * attention_weights, dim=-1)

        return final_scores

class DotProduct(nn.Module):
    def forward(self, node_i, node_j):
        return torch.sum(node_i * node_j, dim=-1)

class Bilinear(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, node_i, node_j):
        return torch.sum(node_i @ self.weight * node_j, dim=-1)

class NeuralNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_i, node_j):
        combined = torch.cat([node_i, node_j], dim=-1)
        return self.net(combined).squeeze(-1)
```

**Key Design Features:**

1. **Message Passing Framework:**
   - Information flows between connected nodes
   - Handles arbitrary graph structures
   - Supports edge features and types

2. **Multi-task Learning:**
   - Shared encoder for common features
   - Task-specific heads for different objectives
   - Joint training improves both tasks

3. **Link Prediction Strategies:**
   - **Adamic-Adar:** Simple path-based method
   - **Jaccard:** Neighborhood similarity
   - **Preferential Attachment:** Based on node degrees
   - **Neural approaches:** Learned similarity functions

**Training Strategy:**

```python
def train_multi_task_gnn(model, graph_data, optimizer, criterion):
    node_logits, edge_scores = model(
        graph_data.node_features,
        graph_data.adjacency,
        graph_data.edge_features,
        graph_data.edge_types
    )

    # Node classification loss
    node_loss = criterion(node_logits, graph_data.node_labels)

    # Link prediction loss
    edge_loss = 0
    for edge_type, scores in edge_scores.items():
        edge_labels = graph_data.edge_labels[edge_type]
        edge_loss += criterion(scores, edge_labels)

    # Combined loss
    total_loss = node_loss + edge_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**Applications:**

- Social network analysis
- Molecular property prediction
- Knowledge graph completion
- Network topology prediction
- Recommender systems

### Q13: How would you implement a neural network for federated learning across heterogeneous devices?

**Answer:** Federated learning trains models across distributed devices while preserving privacy:

**Federated Neural Network Architecture:**

```python
class FederatedClient:
    def __init__(self, model_fn, device_id, data_loader):
        self.device_id = device_id
        self.data_loader = data_loader
        self.model = model_fn().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def local_train(self, global_model_params, local_epochs=5):
        # Update local model with global parameters
        self.update_model_parameters(global_model_params)

        # Local training
        self.model.train()
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

        # Return model updates
        return self.get_parameter_updates()

    def update_model_parameters(self, global_params):
        """Update local model with global parameters"""
        for local_param, global_param in zip(self.model.parameters(), global_params):
            local_param.data = global_param.data.clone()

    def get_parameter_updates(self):
        """Get parameter deltas since last sync"""
        return [param.data.clone() for param in self.model.parameters()]

class FederatedServer:
    def __init__(self, model_fn, num_clients, aggregation_method='fedavg'):
        self.model = model_fn()
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.client_weights = np.ones(num_clients) / num_clients

    def federated_averaging(self, client_updates, client_weights=None):
        """FedAvg aggregation algorithm"""
        if client_weights is None:
            client_weights = self.client_weights

        # Weighted average of parameters
        aggregated_params = []
        for param_idx in range(len(client_updates[0])):
            weighted_sum = 0
            total_weight = 0

            for client_idx, (client_params, weight) in enumerate(zip(client_updates, client_weights)):
                weighted_sum += client_params[param_idx] * weight
                total_weight += weight

            aggregated_params.append(weighted_sum / total_weight)

        return aggregated_params

    def fedprox_aggregation(self, client_updates, global_params, mu=0.01):
        """FedProx aggregation with proximal term"""
        aggregated_params = self.federated_averaging(client_updates)

        # Add proximal term for heterogeneity
        for param_idx, (client_params, global_param) in enumerate(zip(client_updates[0], global_params)):
            # Compute variance term
            variance_term = sum(
                (client_params[p] - global_params[p]) ** 2
                for p in range(len(client_params))
            )

            # Update with proximal term
            aggregated_params[param_idx] -= mu * variance_term

        return aggregated_params

    def personalized_aggregation(self, client_updates, client_params, personalization_rate=0.5):
        """Personalized FedAvg"""
        # Global model (average of all clients)
        global_params = self.federated_averaging(client_updates)

        # Personalized parameters (interpolate)
        personalized_params = []
        for client_idx, client_params in enumerate(client_params):
            personalized_client_params = []
            for global_param, client_param in zip(global_params, client_params):
                # Interpolate between global and local
                personalized_param = (1 - personalization_rate) * global_param + personalization_rate * client_param
                personalized_client_params.append(personalized_param)
            personalized_params.append(personalized_client_params)

        return global_params, personalized_params

class HeterogeneousFederatedModel(nn.Module):
    """Model that can adapt to heterogeneous device capabilities"""
    def __init__(self, base_model, device_capability):
        super().__init__()
        self.base_model = base_model
        self.device_capability = device_capability

        # Dynamically adjust model based on device capability
        self.adapt_model_capacity()

    def adapt_model_capacity(self):
        """Adapt model architecture based on device capability"""
        if self.device_capability == 'low':
            # Reduce model complexity
            self.reduce_parameters()
        elif self.device_capability == 'medium':
            # Keep standard configuration
            pass
        elif self.device_capability == 'high':
            # Use full model or ensemble
            self.increase_capacity()

    def reduce_parameters(self):
        """Reduce model parameters for low-capability devices"""
        # Implementation to reduce model size
        pass

    def increase_capacity(self):
        """Increase model capacity for high-capability devices"""
        # Implementation to increase model size
        pass

class FederatedTrainingOrchestrator:
    def __init__(self, model_fn, client_data_loaders, aggregation_method='fedavg'):
        self.model_fn = model_fn
        self.client_data_loaders = client_data_loaders
        self.aggregation_method = aggregation_method
        self.server = FederatedServer(model_fn, len(client_data_loaders))
        self.clients = [
            FederatedClient(model_fn, i, loader)
            for i, loader in enumerate(client_data_loaders)
        ]

    def train_round(self, num_local_epochs=5, client_fraction=1.0):
        """Execute one federated training round"""
        # Select random subset of clients
        num_selected = int(self.server.num_clients * client_fraction)
        selected_clients = np.random.choice(self.clients, num_selected, replace=False)

        # Get current global model parameters
        global_params = [param.data.clone() for param in self.server.model.parameters()]

        # Collect updates from selected clients
        client_updates = []
        client_weights = []

        for client in selected_clients:
            # Local training
            updates = client.local_train(global_params, num_local_epochs)
            client_updates.append(updates)

            # Weight by amount of data
            weight = len(client.data_loader.dataset) / sum(len(c.data_loader.dataset) for c in selected_clients)
            client_weights.append(weight)

        # Aggregate updates
        if self.aggregation_method == 'fedavg':
            aggregated_params = self.server.federated_averaging(client_updates, client_weights)
        elif self.aggregation_method == 'fedprox':
            aggregated_params = self.server.fedprox_aggregation(client_updates, global_params)
        elif self.aggregation_method == 'personalized':
            aggregated_params, personalized_params = self.server.personalized_aggregation(
                client_updates, client_updates  # Use updates as personalization base
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Update global model
        for param, new_param in zip(self.server.model.parameters(), aggregated_params):
            param.data = new_param.data.clone()

        return aggregated_params

# Privacy-preserving techniques
class SecureAggregation:
    """Secure aggregation for federated learning"""
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.generate_key_pairs()

    def generate_key_pairs(self):
        """Generate public/private key pairs for each client"""
        # Simplified key generation (use proper cryptography in practice)
        self.public_keys = [self.generate_key_pair() for _ in range(self.num_clients)]
        self.private_keys = [keys[1] for keys in self.public_keys]
        self.public_keys = [keys[0] for keys in self.public_keys]

    def generate_key_pair(self):
        """Generate a simple key pair (placeholder)"""
        # In practice, use proper cryptographic key generation
        return (np.random.rand(10), np.random.rand(10))

    def secure_aggregate(self, client_updates):
        """Aggregate encrypted updates"""
        # Encrypt each client's update
        encrypted_updates = []
        for i, update in enumerate(client_updates):
            encrypted = self.encrypt_update(update, self.public_keys[i])
            encrypted_updates.append(encrypted)

        # Aggregate encrypted updates
        aggregated = self.aggregate_encrypted_updates(encrypted_updates)

        # Decrypt final result
        decrypted = self.decrypt_aggregated_update(aggregated)

        return decrypted

    def encrypt_update(self, update, public_key):
        """Encrypt update with public key"""
        # Simplified encryption (use proper cryptography)
        return [param * public_key for param in update]

    def aggregate_encrypted_updates(self, encrypted_updates):
        """Aggregate encrypted updates"""
        aggregated = []
        for param_idx in range(len(encrypted_updates[0])):
            summed = sum(update[param_idx] for update in encrypted_updates)
            aggregated.append(summed)
        return aggregated

    def decrypt_aggregated_update(self, aggregated):
        """Decrypt aggregated update"""
        # Simplified decryption (proper implementation needed)
        return [param / self.num_clients for param in aggregated]
```

**Key Challenges and Solutions:**

1. **Statistical Heterogeneity:**
   - **Problem:** Different data distributions across clients
   - **Solution:** FedProx, personalized models, meta-learning

2. **System Heterogeneity:**
   - **Problem:** Different device capabilities
   - **Solution:** Dynamic model adaptation, adaptive learning rates

3. **Privacy:**
   - **Problem:** Need to protect client data
   - **Solution:** Secure aggregation, differential privacy, homomorphic encryption

4. **Communication Efficiency:**
   - **Problem:** Limited communication bandwidth
   - **Solution:** Model compression, selective parameter updates, quantization

5. **Fault Tolerance:**
   - **Problem:** Client dropouts, Byzantine clients
   - **Solution:** Robust aggregation, client sampling, outlier detection

This federated learning system provides a complete framework for training neural networks across heterogeneous devices while preserving privacy and handling various challenges in distributed machine learning.
