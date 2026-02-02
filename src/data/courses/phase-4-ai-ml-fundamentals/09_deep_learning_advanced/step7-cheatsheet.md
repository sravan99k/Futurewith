# Advanced Deep Learning Cheat Sheet

## Table of Contents

1. [Advanced Architecture Patterns](#advanced-architecture-patterns)
2. [Optimization Algorithms Comparison](#optimization-algorithms-comparison)
3. [Regularization Techniques Quick Reference](#regularization-techniques-quick-reference)
4. [Transfer Learning Strategies](#transfer-learning-strategies)
5. [Performance Optimization Checklist](#performance-optimization-checklist)
6. [Common Debugging Patterns](#common-debugging-patterns)
7. [Hardware Optimization Guide](#hardware-optimization-guide)
8. [Production Deployment Checklist](#production-deployment-checklist)

---

## Advanced Architecture Patterns

### ResNet Architecture Patterns

| Component      | ResNet-18/34           | ResNet-50/101/152      |
| -------------- | ---------------------- | ---------------------- |
| **Block Type** | Basic Block            | Bottleneck             |
| **Conv1**      | 7×7, 64, stride 2      | 7×7, 64, stride 2      |
| **MaxPool**    | 3×3, stride 2          | 3×3, stride 2          |
| **Layer1**     | [2,2] blocks           | [3,4,6] blocks         |
| **Layer2**     | [2,2] blocks           | [4,23,3] blocks        |
| **Layer3**     | [2,2] blocks           | [4,23,3] blocks        |
| **Layer4**     | [2,2] blocks           | [4,23,3] blocks        |
| **Final Pool** | AdaptiveAvgPool2d(1,1) | AdaptiveAvgPool2d(1,1) |
| **FC**         | 512 → num_classes      | 2048 → num_classes     |

**Skip Connection Formula:**

```
H(x) = F(x) + x
```

- **F(x)**: Residual function (convolutional layers)
- **x**: Identity connection
- **Benefits**: Solves vanishing gradients, enables very deep networks

### Inception Block Components

**Inception v1 Module:**

```
- 1×1 conv (64 filters) + ReLU
- 1×1 conv (96) → 3×3 conv (128) + ReLU
- 1×1 conv (16) → 5×5 conv (32) + ReLU
- 3×3 maxpool → 1×1 conv (32) + ReLU
- Concatenate all outputs
```

**Bottleneck Design (1×1 → 3×3 → 1×1):**

- First 1×1: Reduce dimensionality (e.g., 256 → 64)
- 3×3: Main feature extraction (64 → 64)
- Second 1×1: Restore dimensionality (64 → 256)

### EfficientNet Scaling Rules

**Compound Scaling Formula:**

- Depth: d = α^φ (where α > 1)
- Width: w = β^φ (where β > 1)
- Resolution: r = γ^φ (where γ > 1)
- Subject to: α · β² · γ² ≈ 2

**Common Scaling Factors:**

- α = 1.2, β = 1.1, γ = 1.15
- φ controls total model size increase

---

## Optimization Algorithms Comparison

### Algorithm Properties Summary

| Algorithm    | Learning Rate | Momentum         | Memory | Convergence               | Best For                         |
| ------------ | ------------- | ---------------- | ------ | ------------------------- | -------------------------------- |
| **SGD**      | Manual tuning | β = 0.9          | Low    | Slow, may oscillate       | Simple problems, large datasets  |
| **Momentum** | Manual tuning | β = 0.9          | Low    | Faster than SGD           | Deep networks, smooth landscapes |
| **Nesterov** | Manual tuning | β = 0.9          | Low    | Better convergence        | When momentum oscillates         |
| **AdaGrad**  | Automatic     | None             | Low    | Fast initial, then stalls | Sparse features, NLP             |
| **RMSprop**  | Automatic     | ρ = 0.99         | Low    | Good for non-stationary   | RNNs, online learning            |
| **Adam**     | Automatic     | β₁=0.9, β₂=0.999 | Medium | Fast convergence          | Default choice, most problems    |
| **AdamW**    | Automatic     | Same as Adam     | Medium | Better generalization     | Transformer models               |
| **Lion**     | Automatic     | β=0.9            | Low    | Often better than Adam    | Large language models            |

### Learning Rate Schedules

**Step Decay:**

```python
lr(t) = lr₀ × γ^(floor(t/step_size))
# Common: γ=0.1, step_size=30
```

**Exponential Decay:**

```python
lr(t) = lr₀ × e^(-kt)
# k controls decay rate
```

**Cosine Annealing:**

```python
lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(πt/T)) / 2
# T = total epochs, lr_min = 0
```

**Warmup + Decay (Transformer style):**

```python
lr(t) = lr_target × min(t^(-0.5), t × warmup_steps^(-1.5))
# Linear warmup, then power decay
```

---

## Regularization Techniques Quick Reference

### Dropout Strategies

| Type                  | Probability | Where to Apply         | Effect                  |
| --------------------- | ----------- | ---------------------- | ----------------------- |
| **Standard Dropout**  | 0.2-0.5     | Fully connected layers | Prevents co-adaptation  |
| **Dropout 2D**        | 0.1-0.3     | Convolutional layers   | Spatial dropout         |
| **DropConnect**       | 0.1-0.3     | Weight matrices        | Stronger regularization |
| **Scheduled Dropout** | 0.1→0.5     | Any layer              | Progressive increase    |
| **DropBlock**         | 0.1         | Conv layers            | Structured dropout      |

**Dropout Implementation:**

```python
# During training
y = (x * mask) / p  # where mask ~ Bernoulli(p)

# During inference (no dropout)
output = x
```

### Normalization Techniques

| Method           | Formula           | Best For                | Memory Cost | Batch Size |
| ---------------- | ----------------- | ----------------------- | ----------- | ---------- | --- | -------------------- | --- | --- |
| **BatchNorm**    | γ(x-μ)/σ + β      | ConvNets, large batches | Medium      | > 8        |
| **LayerNorm**    | γ(x-μ_L)/σ_L + β  | Transformers, RNNs      | Low         | Any        |
| **GroupNorm**    | GN(x-μ_g)/σ_g + β | Small batch ConvNets    | Low         | 1-8        |
| **InstanceNorm** | IN(x-μ_i)/σ_i + β | Style transfer          | Low         | 1          |
| **WeightNorm**   | v/                |                         | v           |            | × g | When BN not suitable | Low | Any |

**Layer Normalization (Most Common):**

```python
# For input shape [N, C, H, W]
LN = (x - mean(last_dim)) / sqrt(var(last_dim) + ε) × γ + β
# μ_L and σ_L computed across feature dimension
```

### Advanced Regularization

**Label Smoothing:**

```python
# Original loss: -log(p_true)
# With smoothing: -(1-ε)log(p_true) - ε∑log(p_i)/K
# Typical ε = 0.1
```

**MixUp Training:**

```python
# Mix two samples: x_mix = λx₁ + (1-λ)x₂
# y_mix = λy₁ + (1-λ)y₂
# Loss on (x_mix, y_mix)
```

**CutOut:**

```python
# Randomly mask square regions in images
# mask_size typically 16×16 for CIFAR-10
```

---

## Transfer Learning Strategies

### Feature Extraction vs Fine-tuning

| Strategy                | Layers Frozen          | Learning Rate        | Training Time | Risk Level |
| ----------------------- | ---------------------- | -------------------- | ------------- | ---------- |
| **Feature Extraction**  | All except final layer | High (0.01-0.1)      | Fast          | Low        |
| **Partial Fine-tuning** | First N layers         | Low for early layers | Medium        | Medium     |
| **Full Fine-tuning**    | None                   | Very low (1e-5-1e-4) | Slow          | High       |

**Discriminative Learning Rates:**

```python
# Different learning rates for different layers
param_groups = [
    {'params': final_layer.parameters(), 'lr': base_lr * 10},
    {'params': late_layers.parameters(), 'lr': base_lr * 1},
    {'params': early_layers.parameters(), 'lr': base_lr * 0.1}
]
```

### Domain Adaptation Checklist

✅ **Source Domain:**

- [ ] Large, diverse dataset
- [ ] Pre-trained on related tasks
- [ ] Model shows good feature representations

✅ **Target Domain Preparation:**

- [ ] Sufficient labeled data (minimum 1000 samples)
- [ ] Similar but different distribution
- [ ] Proper data augmentation strategy

✅ **Adaptation Strategy:**

- [ ] Start with feature extraction
- [ ] Gradually unfreeze layers
- [ ] Use lower learning rates
- [ ] Monitor for overfitting

---

## Performance Optimization Checklist

### Memory Optimization ✅

**Checklist:**

- [ ] Mixed precision training enabled
- [ ] Gradient accumulation for large effective batch size
- [ ] Memory-efficient data loading (pin_memory, num_workers)
- [ ] Gradient checkpointing for deep networks
- [ ] Model/Data parallel for very large models
- [ ] Clear unused variables (del, gc.collect())
- [ ] Monitor memory usage during training

**Memory Usage Formula:**

```
Peak Memory ≈ batch_size × (model_parameters + activations + gradients)
```

### Compute Optimization ✅

**GPU Optimization:**

- [ ] Use appropriate data types (float16, float32)
- [ ] Enable cuDNN benchmark mode for fixed input sizes
- [ ] Use torch.backends.cudnn.benchmark = True
- [ ] Optimize batch size for GPU utilization
- [ ] Use tensor cores for mixed precision (Volta+)

**Data Pipeline:**

- [ ] Pin memory for GPU transfers
- [ ] Use persistent workers in DataLoader
- [ ] Prefetch data with multiple workers
- [ ] Optimize CPU data preprocessing

**Model Architecture:**

- [ ] Use efficient layer types (depthwise separable conv)
- [ ] Consider MobileNet/EfficientNet architectures
- [ ] Use adaptive pooling instead of flatten
- [ ] Minimize unnecessary tensor operations

### Training Speed Tips ⚡

| Optimization               | Speedup  | Ease   | Risk |
| -------------------------- | -------- | ------ | ---- |
| **Mixed Precision**        | 1.5-2×   | Easy   | Low  |
| **DataLoader Workers**     | 2-3×     | Easy   | Low  |
| **cuDNN Benchmark**        | 1.1-1.3× | Easy   | Low  |
| **Gradient Checkpointing** | 0.7×     | Medium | None |
| **Model Parallel**         | N/A      | Hard   | High |

---

## Common Debugging Patterns

### Gradient Flow Issues

**Problem Detection:**

```python
# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        if grad_norm > 100:  # Exploding gradients
            print(f"Large gradient in {name}: {grad_norm}")
        if grad_norm < 1e-6:  # Vanishing gradients
            print(f"Small gradient in {name}: {grad_norm}")
```

**Solutions:**

- **Exploding**: Gradient clipping, lower learning rate
- **Vanishing**: Residual connections, proper initialization
- **Dead Neurons**: ReLU/LeakyReLU, proper initialization

### Loss and Accuracy Patterns

| Pattern                                     | Symptoms            | Likely Causes                 | Solutions                          |
| ------------------------------------------- | ------------------- | ----------------------------- | ---------------------------------- |
| **High training loss, low validation loss** | Over-regularization | Too much dropout/weight decay | Reduce regularization              |
| **Low training loss, high validation loss** | Overfitting         | Model too complex             | Add regularization, get more data  |
| **Oscillating loss**                        | Unstable training   | Learning rate too high        | Lower learning rate, add momentum  |
| **Plateau early**                           | Stops improving     | Learning rate too low         | Warm restarts, lr scheduling       |
| **NaN loss**                                | Training crashes    | Exploding gradients, bad data | Gradient clipping, data inspection |

### Model Performance Issues

**Underfitting Checklist:**

- [ ] Model capacity sufficient? (Add layers/parameters)
- [ ] Learning rate appropriate? (Try 10× higher)
- [ ] Enough training epochs? (Increase patience)
- [ ] Data quality good? (Inspect samples)
- [ ] Architecture suitable for task?

**Overfitting Checklist:**

- [ ] Add more data? (Data augmentation)
- [ ] Regularization strength? (Increase dropout/weight decay)
- [ ] Model complexity? (Reduce parameters)
- [ ] Early stopping? (Monitor validation loss)
- [ ] Cross-validation? (Better performance estimate)

---

## Hardware Optimization Guide

### GPU Memory Management

**Memory Hierarchy (Fastest → Slowest):**

1. **Registers** (0-1KB/thread)
2. **Shared Memory** (48KB/block)
3. **Local Memory** (1MB/thread)
4. **Global Memory** (10+ GB/device)

**Memory Optimization Techniques:**

**Coalesced Access:**

```python
# GOOD: Sequential memory access
for i in range(n):
    output[i] = process(input[i])

# BAD: Random memory access
for i in indices:
    output[perm[i]] = process(input[i])
```

**Shared Memory Usage:**

```python
# Load into shared memory
shared_data = torch.zeros_like(block, dtype=torch.float).cuda()
shared_data[thread_idx] = input[global_idx]
torch.cuda.syncthreads()
# Process data
output = process_block(shared_data)
```

### Mixed Precision Training

**Automatic Mixed Precision (AMP):**

```python
# Enable AMP
scaler = GradScaler()
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# Training loop
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Memory Savings:**

- **Forward pass**: 2× memory reduction
- **Backward pass**: 1.5× memory reduction
- **Overall**: 1.6-2× speedup on modern GPUs

### Distributed Training

**Data Parallel Setup:**

```python
# Single machine, multiple GPUs
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Multi-machine
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)
```

**Memory Coordination:**

- **Gradient Synchronization**: All-reduce across devices
- **Local Batch Size**: total_batch_size / num_gpus
- **Communication Overhead**: ~10-20% of compute time

---

## Production Deployment Checklist

### Model Optimization for Inference ⚡

**Optimization Pipeline:**

1. [ ] **Model Conversion**: PyTorch → ONNX → TensorRT
2. [ ] **Precision Calibration**: FP32 → FP16 → INT8
3. [ ] **Architecture Optimization**: Pruning, Quantization
4. [ ] **Runtime Optimization**: TensorRT engine, NCNN, etc.

**Common Optimizations:**

**Model Pruning:**

```python
# Unstructured pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        weight = module.weight.data
        mask = torch.rand(weight.shape) > threshold
        module.weight.data = weight * mask
```

**Quantization:**

```python
# Post-training quantization
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

**ONNX Export:**

```python
# Export model
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                 export_params=True, opset_version=11,
                 do_constant_folding=True,
                 input_names=['input'], output_names=['output'])
```

### Inference Performance Benchmarks 🎯

**Performance Targets:**

| Task                      | Latency Target | Throughput Target | Model Size |
| ------------------------- | -------------- | ----------------- | ---------- |
| **Image Classification**  | < 10ms         | > 100 FPS         | < 50MB     |
| **Object Detection**      | < 30ms         | > 30 FPS          | < 200MB    |
| **Semantic Segmentation** | < 50ms         | > 20 FPS          | < 100MB    |
| **Speech Recognition**    | < 100ms        | > 10 streams      | < 200MB    |
| **Text Classification**   | < 5ms          | > 1000 QPS        | < 10MB     |

**Monitoring Metrics:**

- [ ] **Latency**: P50, P95, P99 percentiles
- [ ] **Throughput**: Requests per second
- [ ] **Memory Usage**: Peak and average
- [ ] **GPU Utilization**: Compute and memory
- [ ] **Error Rate**: Timeout, crashes, OOM

### Deployment Architecture 🏗️

**Model Server Options:**

| Platform         | Latency | Throughput | Scaling | Cost     |
| ---------------- | ------- | ---------- | ------- | -------- |
| **TensorRT**     | 1-5ms   | Very High  | Manual  | GPU      |
| **ONNX Runtime** | 2-10ms  | High       | Auto    | Low      |
| **TorchServe**   | 5-20ms  | Medium     | Auto    | Medium   |
| **Triton**       | 1-10ms  | High       | Auto    | GPU      |
| **Edge TPU**     | <1ms    | Medium     | None    | Hardware |

**Production Checklist:**

**Model Preparation:**

- [ ] Model tested for numerical stability
- [ ] Input validation implemented
- [ ] Output post-processing correct
- [ ] Model size and memory usage acceptable
- [ ] Export to optimized format (ONNX/TensorRT)

**Infrastructure:**

- [ ] Auto-scaling configured
- [ ] Load balancing setup
- [ ] Health checks implemented
- [ ] Monitoring and alerting active
- [ ] Rollback strategy prepared

**Security & Compliance:**

- [ ] Input sanitization
- [ ] Output validation
- [ ] Model versioning
- [ ] Access control
- [ ] Data privacy compliance

---

## Quick Reference Tables

### Common Learning Rates by Task

| Task                      | Base Learning Rate | Optimizer    | Notes                                 |
| ------------------------- | ------------------ | ------------ | ------------------------------------- |
| **Image Classification**  | 0.001              | AdamW        | Scale by batch size                   |
| **Object Detection**      | 0.0025             | SGD+Momentum | Use different LR for head/backbone    |
| **Semantic Segmentation** | 0.01               | SGD+Momentum | Poly LR schedule                      |
| **Language Modeling**     | 0.0001             | Adam         | Warmup + cosine                       |
| **Object Detection**      | 0.001              | Adam         | Use different LR for different layers |

### Architecture Selection Guide

| Task                      | Recommended Architecture | Pros                                | Cons                |
| ------------------------- | ------------------------ | ----------------------------------- | ------------------- |
| **Image Classification**  | ResNet50, EfficientNet   | Well-studied, good accuracy         | Standard approach   |
| **Object Detection**      | Faster R-CNN, YOLOv5     | Real-time, good accuracy            | Complex pipeline    |
| **Semantic Segmentation** | DeepLab, U-Net           | Good spatial information            | Memory intensive    |
| **Text Classification**   | BERT, RoBERTa            | Transfer learning, good performance | Large model size    |
| **Time Series**           | LSTM, Transformer        | Handles sequences well              | Training complexity |

### Data Augmentation Strategies

| Task       | Basic Augmentation               | Advanced Augmentation              |
| ---------- | -------------------------------- | ---------------------------------- |
| **Images** | Flip, rotate, crop, color jitter | CutOut, MixUp, AutoAugment         |
| **Text**   | Random deletion, replacement     | Back-translation, EDA              |
| **Audio**  | Time stretch, pitch shift        | Speed perturbation, noise addition |

---

This cheat sheet provides a comprehensive quick reference for advanced deep learning techniques, covering all major aspects from architecture design to production deployment. Use it as a daily reference during development and review before important decisions.
