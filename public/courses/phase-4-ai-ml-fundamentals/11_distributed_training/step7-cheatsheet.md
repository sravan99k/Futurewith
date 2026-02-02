# Advanced Neural Networks Cheatsheet

## Table of Contents

1. [Specialized Architecture Patterns](#architecture-patterns)
2. [Attention Mechanism Quick Reference](#attention-reference)
3. [Transformer Architecture Guide](#transformer-guide)
4. [Multi-modal Network Patterns](#multimodal-patterns)
5. [Advanced Optimization Strategies](#optimization-strategies)
6. [Performance Metrics & Monitoring](#performance-monitoring)
7. [Common Architectures Overview](#architectures-overview)
8. [Troubleshooting Guide](#troubleshooting)

---

## 1. Specialized Architecture Patterns {#architecture-patterns}

### 1.1 Residual Connections

```python
# Basic residual block
class ResidualBlock:
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if identity.shape != out.shape:
            identity = self.projection(identity)
        return out + identity

# BottleNet residual block
class BottleNetResidual:
    def forward(self, x):
        identity = x
        out = self.conv1(x)  # 1x1 reduce
        out = self.conv2(out)  # 3x3 process
        out = self.conv3(out)  # 1x1 expand
        if identity.shape != out.shape:
            identity = self.projection(identity)
        return out + identity
```

### 1.2 Attention Patterns

```python
# Multi-head attention formula
Attention(Q, K, V) = softmax(QK^T/√d_k)V

# Self-attention (same sequence as Q, K, V)
output = MultiHeadAttention(x, x, x, mask=causal_mask)

# Cross-attention (query from decoder, key-value from encoder)
output = MultiHeadAttention(decoder_output, encoder_output, encoder_output)

# Cross-attention with padding mask
output = MultiHeadAttention(query, key, value, mask=padding_mask)
```

### 1.3 Normalization Patterns

```python
# Layer Norm (pre-norm transformer style)
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FeedForward(x))

# Batch Norm (post-activation style)
x = BatchNorm(Conv(x))

# Group Norm (3D data)
x = GroupNorm(x, num_groups=32)
```

### 1.4 Embedding Patterns

```python
# Token + Position + Segment embeddings
embeddings = token_emb + position_emb + segment_emb

# Scaled embeddings (ViT style)
embeddings = token_emb * sqrt(d_model) + position_emb

# Learned positional vs Sinusoidal
# Learned: trainable parameters
# Sinusoidal: fixed mathematical pattern
```

---

## 2. Attention Mechanism Quick Reference {#attention-reference}

### 2.1 Attention Types

| Type               | Use Case                 | Formula                     | Complexity             |
| ------------------ | ------------------------ | --------------------------- | ---------------------- |
| Scaled Dot-Product | General attention        | softmax(QK^T/√d)V           | O(n²d)                 |
| Multi-Head         | Multiple representations | Concat(head₁...headₕ)Wᵒ     | O(h·n²·d/h) = O(n²d)   |
| Self-Attention     | Sequence understanding   | Q=K=V=X                     | O(n²d)                 |
| Cross-Attention    | Encoder-decoder          | Q=X_dec, K=V=X_enc          | O(n_decoder·n_encoder) |
| Local Attention    | Long sequences           | Only attend to local window | O(n·w)                 |
| Linear Attention   | Efficient attention      | Linear approximation        | O(n·d)                 |

### 2.2 Attention Components

```python
# Core attention computation
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose() / sqrt(K.size(-1))  # (batch, seq_len, seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = softmax(scores, dim=-1)
    return weights @ V, weights

# Multi-head attention
def multi_head_attention(Q, K, V, num_heads):
    head_dim = Q.size(-1) // num_heads
    Q = Q.view(*Q.shape[:-1], num_heads, head_dim).transpose(-2, -3)
    K = K.view(*K.shape[:-1], num_heads, head_dim).transpose(-2, -3)
    V = V.view(*V.shape[:-1], num_heads, head_dim).transpose(-2, -3)

    output, _ = scaled_dot_product_attention(Q, K, V)
    output = output.transpose(-2, -3).contiguous().view(*Q.shape[:-2], -1)
    return output
```

### 2.3 Mask Types

```python
# Causal mask (autoregressive)
def create_causal_mask(size):
    return torch.tril(torch.ones(size, size)).bool()

# Padding mask
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

# Combined mask
combined_mask = torch.max(padding_mask, causal_mask.unsqueeze(0))
```

### 2.4 Attention Patterns

```python
# Global attention (all positions)
# Best for: Global understanding, summarization

# Local attention (window-based)
# Best for: Long sequences, efficiency
window_size = 64
mask = create_local_attention_mask(seq_len, window_size)

# Strided attention (sparse)
# Best for: Very long sequences, efficiency
strides = [32, 16, 8]  # Variable stride sizes

# Chunked attention
# Best for: Memory efficiency
chunk_size = 512
```

---

## 3. Transformer Architecture Guide {#transformer-guide}

### 3.1 Complete Transformer Block

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # Pre-norm style (more stable)
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

### 3.2 Encoder Architecture

```python
class TransformerEncoder:
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Embeddings
        x = self.token_embedding(x) * sqrt(d_model)
        x = x + self.pos_embedding

        # Layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
```

### 3.3 Decoder Architecture

```python
class TransformerDecoder:
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = [TransformerDecoderBlock(d_model, num_heads) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Embeddings
        x = self.token_embedding(x) * sqrt(d_model)
        x = x + self.pos_embedding

        # Layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        return self.norm(x)
```

### 3.4 Vision Transformer (ViT)

```python
class VisionTransformer:
    def __init__(self, patch_size, d_model, num_heads, num_layers, num_classes):
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, d_model)

        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding
        self.pos_embedding = PositionalEncoding(d_model)

        # Transformer layers
        self.layers = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, d_model)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Use class token for classification
        logits = self.classifier(x[:, 0])
        return logits
```

---

## 4. Multi-modal Network Patterns {#multimodal-patterns}

### 4.1 Vision-Language Models

```python
class CLIPModel:
    """Contrastive Language-Image Pretraining"""
    def __init__(self, vision_model, text_model, embed_dim):
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_projection = LinearLayer(vision_dim, embed_dim)
        self.text_projection = LinearLayer(text_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, images, texts):
        # Get embeddings
        image_embeds = self.vision_projection(self.vision_model(images))
        text_embeds = self.text_projection(self.text_model(texts))

        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity
        logits = torch.matmul(image_embeds, text_embeds.t()) * torch.exp(self.temperature)

        return logits, image_embeds, text_embeds
```

### 4.2 Audio-Visual Models

```python
class AudioVisualModel:
    def __init__(self, audio_model, video_model, fusion_dim):
        self.audio_model = audio_model
        self.video_model = video_model
        self.fusion_layer = TransformerFusionLayer(audio_dim, video_dim, fusion_dim)

    def forward(self, audio, video):
        # Extract features
        audio_features = self.audio_model(audio)
        video_features = self.video_model(video)

        # Align temporally
        audio_features = self.temporal_alignment(audio_features, video_features.size(1))

        # Fusion
        fused_features = self.fusion_layer(audio_features, video_features)

        # Classification/regression head
        output = self.decoder(fused_features)

        return output

    def temporal_alignment(self, audio_features, target_length):
        # Upsample or downsample audio features to match video length
        if audio_features.size(1) < target_length:
            return F.interpolate(audio_features.transpose(1, 2), target_length, mode='linear').transpose(1, 2)
        else:
            return audio_features[:, :target_length]
```

### 4.3 3D Multi-modal

```python
class MultiModal3DModel:
    def __init__(self, pointcloud_model, image_model, text_model):
        self.pointcloud_encoder = PointNet()
        self.image_encoder = ResNet()
        self.text_encoder = BERT()

        # Cross-modal attention
        self.pointcloud_text_attn = CrossAttention(pointcloud_dim, text_dim)
        self.image_text_attn = CrossAttention(image_dim, text_dim)

        # Fusion
        self.fusion = AttentionFusion(combined_dim, output_dim)

    def forward(self, pointcloud, image, text):
        # Encode each modality
        pc_features = self.pointcloud_encoder(pointcloud)
        img_features = self.image_encoder(image)
        txt_features = self.text_encoder(text)

        # Cross-modal attention
        pc_txt_features = self.pointcloud_text_attn(pc_features, txt_features)
        img_txt_features = self.image_text_attn(img_features, txt_features)

        # Combine features
        combined = torch.cat([pc_txt_features, img_txt_features, txt_features], dim=1)

        # Final fusion
        output = self.fusion(combined)

        return output
```

---

## 5. Advanced Optimization Strategies {#optimization-strategies}

### 5.1 Learning Rate Schedules

```python
# Cosine annealing with warmup
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.current_step / self.warmup_steps * self.optimizer.defaults['lr']
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.optimizer.defaults['lr'] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# One-cycle policy
class OneCycleScheduler:
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.total_steps * 0.3:  # Warmup
            lr = self.max_lr * 0.1 + (self.max_lr * 0.9) * (self.step_num / (self.total_steps * 0.3))
        elif self.step_num <= self.total_steps * 0.7:  # High lr phase
            lr = self.max_lr
        else:  # Annealing
            progress = (self.step_num - self.total_steps * 0.7) / (self.total_steps * 0.3)
            lr = self.max_lr * 0.1 + (self.max_lr * 0.9) * (0.5 * (1 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 5.2 Advanced Optimizers

```python
# AdamW with weight decay scheduling
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {p: {'step': 0, 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)} for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            state = self.state[p]

            # AdamW update
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad.conj(), value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            step_size = self.lr / bias_correction1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            p.addcdiv_(exp_avg, denom, value=-step_size)

            # Weight decay (decoupled from gradient)
            p.add_(p, alpha=-self.lr * self.weight_decay)
```

### 5.3 Gradient Techniques

```python
# Gradient clipping
def clip_gradients(model, max_norm=1.0):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        clip_coefficient = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coefficient)

# Gradient accumulation
def train_step(model, inputs, targets, optimizer, accumulation_steps=4):
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        output = model(inp)
        loss = criterion(output, tgt) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# Mixed precision training
def train_with_mixed_precision(model, inputs, targets, scaler, optimizer):
    with torch.cuda.amp.autocast():
        output = model(inputs)
        loss = criterion(output, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 6. Performance Metrics & Monitoring {#performance-monitoring}

### 6.1 Model Metrics

```python
# Classification metrics
def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean()

def precision_recall_f1(y_pred, y_true, num_classes):
    # Per-class metrics
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()

        precision[c] = tp / (tp + fp + 1e-8)
        recall[c] = tp / (tp + fn + 1e-8)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-8)

    return precision, recall, f1

# Regression metrics
def mse(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

def mae(y_pred, y_true):
    return F.l1_loss(y_pred, y_true)

def r2_score(y_pred, y_true):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)
```

### 6.2 Model Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def profile_inference(model, input_shape):
    # Warmup
    dummy_input = torch.randn(input_shape).cuda()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model(dummy_input)
    end_time = time.time()

    inference_time = end_time - start_time
    throughput = input_shape[0] / inference_time

    # Memory usage
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.memory_allocated()
    memory_cached = torch.cuda.memory_reserved()

    print(f"Inference time: {inference_time:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Memory allocated: {memory_allocated / 1024**2:.2f} MB")
    print(f"Memory cached: {memory_cached / 1024**2:.2f} MB")

    return output, {
        'inference_time': inference_time,
        'throughput': throughput,
        'memory_allocated': memory_allocated,
        'memory_cached': memory_cached
    }

# Model complexity analysis
def analyze_model_complexity(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate FLOPs (simplified)
    total_flops = 0
    for p in model.parameters():
        if p.grad is not None:  # Only count parameters that affect output
            flops = p.numel() * 2  # Multiply-add operations
            total_flops += flops

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated FLOPs: {total_flops:,}")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_flops': total_flops
    }
```

### 6.3 Training Monitoring

```python
class TrainingMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.metrics = []
        self.losses = []
        self.learning_rates = []
        self.step = 0

    def log_metrics(self, metrics, loss, lr, step=None):
        if step is None:
            step = self.step

        self.metrics.append(metrics)
        self.losses.append(loss)
        self.learning_rates.append(lr)

        if step % self.log_interval == 0:
            print(f"Step {step}: Loss = {loss:.4f}, LR = {lr:.2e}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        self.step += 1

    def plot_training_curves(self, save_path="training_curves.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curve
        ax1.plot(self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Learning rate schedule
        ax2.plot(self.learning_rates)
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

## 7. Common Architectures Overview {#architectures-overview}

### 7.1 CNN Architectures

| Architecture | Key Innovation                 | Best For                    |
| ------------ | ------------------------------ | --------------------------- |
| LeNet        | First successful CNN           | Digit recognition           |
| AlexNet      | Deep CNN, ReLU, Dropout        | Image classification        |
| VGG          | Very deep uniform architecture | Feature extraction          |
| ResNet       | Residual connections           | Very deep networks          |
| Inception    | Multi-scale processing         | Efficient computation       |
| MobileNet    | Depthwise separable convs      | Mobile deployment           |
| EfficientNet | Compound scaling               | Optimal accuracy-efficiency |
| RegNet       | Regularized design space       | Scalable architectures      |

### 7.2 Transformer Variants

| Model                | Key Innovation         | Parameter Count | Use Case            |
| -------------------- | ---------------------- | --------------- | ------------------- |
| Original Transformer | Attention mechanism    | ~65M (base)     | Machine translation |
| BERT                 | Bidirectional encoder  | 110M - 340M     | NLP tasks           |
| GPT                  | Autoregressive decoder | 117M - 175B     | Text generation     |
| T5                   | Text-to-text unified   | 220M - 11B      | Transfer learning   |
| Vision Transformer   | Transformer for images | 86M - 307M      | Computer vision     |
| Swin Transformer     | Hierarchical attention | 50M - 197M      | Vision tasks        |
| Linformer            | Linear attention       | 30M - 150M      | Long sequences      |
| Performer            | Kernel-based attention | 117M - 652M     | Efficient attention |

### 7.3 Generative Models

| Model Type             | Architecture                           | Key Use Cases                          |
| ---------------------- | -------------------------------------- | -------------------------------------- |
| GAN                    | Generator + Discriminator              | Image generation, style transfer       |
| VAE                    | Encoder + Decoder + Reparameterization | Latent space modeling, data generation |
| Flow Models            | Invertible transformations             | Density modeling, likelihood           |
| Diffusion              | U-Net + Diffusion process              | High-quality image generation          |
| Autoregressive         | LSTM/Transformer                       | Text generation, sequence modeling     |
| Neural Radiance Fields | MLP + Positional encoding              | 3D scene modeling                      |

---

## 8. Troubleshooting Guide {#troubleshooting}

### 8.1 Training Issues

| Problem                 | Symptoms                              | Solutions                                                                 |
| ----------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| **Vanishing Gradients** | Loss plateaus, very slow learning     | ReLU/LeakyReLU, Residual connections, Layer normalization                 |
| **Exploding Gradients** | Loss becomes NaN, weights blow up     | Gradient clipping, Lower learning rate, Proper initialization             |
| **Overfitting**         | High training acc, low validation acc | Dropout, Weight decay, Data augmentation, Early stopping                  |
| **Underfitting**        | Low accuracy on both train/val        | Increase model capacity, Reduce regularization                            |
| **Slow Training**       | Training takes too long               | Mixed precision, Model parallel, Better hardware, Efficient architectures |
| **Memory Issues**       | Out of memory errors                  | Gradient accumulation, Model sharding, Smaller batch size                 |

### 8.2 Architecture Choices

| Task                      | Recommended Architecture  | Notes                           |
| ------------------------- | ------------------------- | ------------------------------- |
| **Image Classification**  | ResNet, EfficientNet, ViT | Use pre-trained models          |
| **Object Detection**      | Faster R-CNN, YOLO, DETR  | Real-time vs accuracy tradeoff  |
| **Semantic Segmentation** | U-Net, DeepLab, SegFormer | Pixel-level prediction          |
| **Text Classification**   | BERT, RoBERTa, DistilBERT | Fine-tune pre-trained models    |
| **Machine Translation**   | Transformer, T5           | Sequence-to-sequence            |
| **Text Generation**       | GPT, T5, GPT-Neo          | Autoregressive models           |
| **Audio Processing**      | Conformer, Wav2Vec2       | Speech recognition, synthesis   |
| **Multi-modal**           | CLIP, DALL-E, Flamingo    | Vision + language understanding |

### 8.3 Hyperparameter Guidelines

```python
# General guidelines for hyperparameters

# Learning rates (rough guidelines)
SGD: 0.1
Adam: 1e-3 to 1e-4
AdamW: 1e-3
RMSprop: 1e-3

# Batch sizes (depends on hardware)
Small models (CPU): 16-64
Medium models (GPU): 128-512
Large models (multi-GPU): 1024-4096

# Dropout rates
Dense layers: 0.2-0.5
Convolutional layers: 0.1-0.3
Recurrent layers: 0.1-0.5

# Weight decay (L2 regularization)
AdamW: 0.01-0.1
SGD: 1e-4 to 1e-2

# Architecture depth guidelines
CNNs: 16-152 layers
Transformers: 6-24 layers
MLPs: 2-10 layers

# Attention heads
Small models: 4-8 heads
Large models: 12-32 heads
d_model % num_heads == 0 (requirement)
```

### 8.4 Quick Debugging Checklist

- [ ] **Data loading**: Check data shapes, types, and preprocessing
- [ ] **Model architecture**: Verify input/output dimensions match
- [ ] **Loss function**: Ensure appropriate for the task
- [ ] **Optimizer**: Check learning rate and parameters
- [ ] **Regularization**: Adjust dropout, weight decay, etc.
- [ ] **Initialization**: Use proper weight initialization
- [ ] **Gradient flow**: Check for NaN/Inf in gradients
- [ ] **Overfitting**: Monitor training vs validation metrics
- [ ] **Resource usage**: Monitor GPU memory and compute
- [ ] **Reproducibility**: Set random seeds, ensure deterministic

This cheatsheet provides quick reference materials for advanced neural network architectures, optimization strategies, and practical implementation details. Use it as a quick lookup for common patterns and troubleshooting guidance.
