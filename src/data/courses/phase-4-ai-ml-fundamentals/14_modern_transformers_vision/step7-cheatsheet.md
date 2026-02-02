# Vision Transformers and Modern Computer Vision - Quick Reference Cheatsheet

## Table of Contents

1. [Architecture Quick Reference](#architecture-reference)
2. [Key Concepts](#key-concepts)
3. [Implementation Patterns](#implementation-patterns)
4. [Model Comparison Table](#model-comparison)
5. [Training Guidelines](#training-guidelines)
6. [Code Snippets](#code-snippets)
7. [Performance Metrics](#performance-metrics)
8. [Debugging Guide](#debugging-guide)
9. [Resource Requirements](#resource-requirements)
10. [Pre-trained Models](#pretrained-models)

## 1. Architecture Quick Reference {#architecture-reference}

### Vision Transformer Core Components

```python
# 1. Patch Embedding
patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

# 2. Position Embedding
pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

# 3. Multi-Head Attention
attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

# 4. MLP Block
mlp = nn.Sequential(nn.Linear(embed_dim, mlp_ratio * embed_dim),
                   nn.GELU(),
                   nn.Dropout(dropout),
                   nn.Linear(mlp_ratio * embed_dim, embed_dim))

# 5. Complete ViT Block
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim),
                               nn.GELU(),
                               nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Patch Size vs Performance

| Patch Size | Compute Cost | Detail Captured | Typical Use     |
| ---------- | ------------ | --------------- | --------------- |
| 16×16      | Standard     | Medium          | General use     |
| 8×8        | High         | High            | Fine details    |
| 32×32      | Low          | Low             | Coarse patterns |

## 2. Key Concepts {#key-concepts}

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Patch Embedding Process

1. Split image into non-overlapping patches
2. Flatten each patch
3. Linear projection to embedding dimension
4. Add learnable position embeddings
5. Prepend CLS token

### Vision Transformer Flow

```
Input Image → Patch Embedding → Positional Encoding → Transformer Blocks → CLS Token → Classification
```

## 3. Implementation Patterns {#implementation-patterns}

### Basic ViT Model

```python
import torch
import torch.nn as nn
import timm

# Method 1: Using timm (recommended)
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Method 2: Custom implementation
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224',
                                     pretrained=True,
                                     num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
```

### Loading Pre-trained Models

```python
# Hugging Face Models
from transformers import ViTModel, ViTConfig

config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# PyTorch Hub
model = torch.hub.load('facebookresearch/deit', 'deit_base_patch16_224')

# TIMM Library (most comprehensive)
model = timm.create_model('vit_large_patch16_224', pretrained=True)
```

### Transfer Learning Setup

```python
# Freeze parameters for fine-tuning
def freeze_model(model, freeze_ratio=0.9):
    total_layers = len(model.blocks)
    freeze_layers = int(total_layers * freeze_ratio)

    # Freeze early layers
    for i in range(freeze_layers):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    return model

# Usage
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = freeze_model(model, freeze_ratio=0.8)
```

## 4. Model Comparison Table {#model-comparison}

### ViT Variants Comparison

| Model     | Input Size | Parameters | FLOPs  | ImageNet Top-1 | Best Use Case |
| --------- | ---------- | ---------- | ------ | -------------- | ------------- |
| ViT-Tiny  | 224×224    | 5.8M       | 1.2B   | 72.2%          | Mobile/Edge   |
| ViT-Small | 224×224    | 22.1M      | 4.7B   | 80.2%          | General use   |
| ViT-Base  | 224×224    | 86.3M      | 17.6B  | 84.2%          | Balanced      |
| ViT-Large | 224×224    | 307.5M     | 63.6B  | 87.1%          | High accuracy |
| ViT-Huge  | 224×224    | 632.3M     | 130.7B | 88.5%          | Research/SOTA |

### Efficient Transformers

| Model      | Parameters | FLOPs | Memory | ImageNet Top-1 | Feature        |
| ---------- | ---------- | ----- | ------ | -------------- | -------------- |
| Swin-B     | 88M        | 47B   | Linear | 83.3%          | Hierarchical   |
| Swin-L     | 197M       | 116B  | Linear | 87.3%          | High accuracy  |
| ConvNeXt-L | 198M       | 34B   | Linear | 87.8%          | CNN-like       |
| CoAtNet-4  | 244M       | 34B   | Linear | 85.1%          | Hybrid CNN-ViT |

### Object Detection Models

| Model           | Input Size | mAP@50 | Speed (FPS) | Memory (GB) |
| --------------- | ---------- | ------ | ----------- | ----------- |
| DETR-R50        | 800×800    | 42.0   | 9.5         | 8.2         |
| DETR-R101       | 800×800    | 43.3   | 7.5         | 10.1        |
| Deformable DETR | 800×800    | 46.8   | 12.5        | 7.8         |
| EfficientDET-D7 | 1536×1536  | 54.4   | 3.2         | 8.5         |

## 5. Training Guidelines {#training-guidelines}

### Dataset Size Recommendations

| Dataset Size | Recommended Model | Augmentation Strategy |
| ------------ | ----------------- | --------------------- |
| <10K images  | CNN (ResNet50)    | Heavy augmentation    |
| 10K-100K     | ViT-Small         | Moderate augmentation |
| 100K-1M      | ViT-Base          | Standard augmentation |
| >1M          | ViT-Large+        | Light augmentation    |

### Learning Rate Schedules

```python
# Cosine Annealing with Warmup (recommended for ViT)
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                   num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda)

# Usage
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
```

### Optimizer Configuration

```python
# AdamW with appropriate settings
optimizer = optim.AdamW(model.parameters(),
                       lr=1e-4,          # Standard learning rate
                       weight_decay=0.05, # Important for ViT stability
                       betas=(0.9, 0.999))

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Data Augmentation Strategies

```python
# ViT-specific augmentation (RandAugment)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    RandAugment(n=2, m=9),      # Random augmentations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mixup for improved generalization
mixup = MixUp(alpha=1.0)
cutmix = CutMix(alpha=1.0)
```

## 6. Code Snippets {#code-snippets}

### Quick Model Creation

```python
# Create ViT with custom settings
def create_vit_model(model_name='vit_base_patch16_224',
                    num_classes=1000,
                    pretrained=True):
    return timm.create_model(model_name,
                           pretrained=pretrained,
                           num_classes=num_classes)

# Usage
model = create_vit_model('vit_large_patch16_224', num_classes=10)
```

### Inference Pipeline

```python
import torchvision.transforms as transforms
from PIL import Image

def predict_image(model, image_path, class_names):
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return class_names[predicted_class], probabilities[0][predicted_class].item()

# Usage
class_names = ['cat', 'dog', 'bird', 'car', 'plane']
prediction, confidence = predict_image(model, 'image.jpg', class_names)
```

### Mixed Precision Training

```python
def train_with_mixed_precision(model, dataloader, num_epochs=10):
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for images, targets in dataloader:
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

### Custom Vision Transformer

```python
class CustomViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x
```

## 7. Performance Metrics {#performance-metrics}

### Classification Metrics

```python
def calculate_metrics(predictions, targets):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')

    cm = confusion_matrix(targets, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
```

### Object Detection Metrics

```python
def calculate_detection_metrics(pred_boxes, pred_labels, pred_scores,
                              true_boxes, true_labels, iou_threshold=0.5):
    from collections import Counter

    # Calculate AP for each class
    ap_scores = []
    for class_id in set(true_labels):
        class_ap = calculate_ap_for_class(
            pred_boxes, pred_labels, pred_scores,
            true_boxes, true_labels, class_id, iou_threshold
        )
        ap_scores.append(class_ap)

    mAP = np.mean(ap_scores)

    # Calculate accuracy
    total_predictions = len(pred_labels)
    correct_predictions = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = correct_predictions / total_predictions

    return {
        'mAP': mAP,
        'accuracy': accuracy,
        'ap_scores': ap_scores
    }
```

## 8. Debugging Guide {#debugging-guide}

### Common Issues and Solutions

| Issue                    | Symptoms                        | Solution                                                        |
| ------------------------ | ------------------------------- | --------------------------------------------------------------- |
| **Training Instability** | NaN losses, exploding gradients | Use gradient clipping, lower learning rate                      |
| **Poor Convergence**     | Loss plateaus early             | Check data preprocessing, use warmup                            |
| **Overfitting**          | High train acc, low val acc     | Add dropout, data augmentation, regularization                  |
| **Memory Issues**        | OOM errors                      | Use gradient accumulation, mixed precision, smaller batch size  |
| **Slow Training**        | Very long training time         | Use mixed precision, efficient optimizers, distributed training |

### Debug Checklist

```python
def debug_model(model, sample_input):
    print("Model Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

    # Check input/output shapes
    print(f"Input shape: {sample_input.shape}")

    # Forward pass test
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            print(f"Output shape: {output.shape}")
            print("✅ Forward pass successful")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")

    # Check gradients
    model.train()
    try:
        loss = model(sample_input).sum()
        loss.backward()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
```

### Performance Profiling

```python
import torch.profiler

def profile_model(model, input_shape):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(5):
            model(torch.randn(input_shape).cuda())
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 9. Resource Requirements {#resource-requirements}

### Memory Requirements

| Model     | Batch Size 1 | Batch Size 8 | Batch Size 32 | Training Time (1 epoch) |
| --------- | ------------ | ------------ | ------------- | ----------------------- |
| ViT-Base  | 1.2GB        | 3.8GB        | OOM           | 45 min                  |
| ViT-Large | 4.2GB        | 14.2GB       | OOM           | 2.1 hours               |
| Swin-B    | 2.1GB        | 6.8GB        | 24.8GB        | 1.2 hours               |

### Compute Requirements

```python
def estimate_training_time(model_size_mb, dataset_size, epochs=50):
    # Rough estimates for ImageNet-sized dataset
    base_time_per_epoch = {
        'ViT-Base': 45,      # minutes
        'ViT-Large': 130,    # minutes
        'Swin-B': 70,        # minutes
        'Swin-L': 180        # minutes
    }

    # Scale based on dataset size relative to ImageNet (1.2M images)
    scale_factor = dataset_size / 1_200_000

    if model_size_mb < 100:  # Base models
        base_time = base_time_per_epoch['ViT-Base']
    elif model_size_mb < 300:  # Large models
        base_time = base_time_per_epoch['ViT-Large']
    else:  # Huge models
        base_time = base_time_per_epoch['Swin-L']

    total_time = base_time * scale_factor * epochs / 60  # Convert to hours
    return total_time

# Usage
estimated_hours = estimate_training_time(86, 1_200_000, 50)  # ViT-Base on ImageNet
print(f"Estimated training time: {estimated_hours:.1f} hours")
```

### Hardware Recommendations

| Use Case    | GPU Memory | GPU Type       | Training Time (ViT-Base) |
| ----------- | ---------- | -------------- | ------------------------ |
| Development | 8GB        | RTX 3080/4070  | 2-3 days                 |
| Research    | 24GB       | RTX 4090/A6000 | 12-18 hours              |
| Production  | 48GB       | A100/H100      | 6-8 hours                |

## 10. Pre-trained Models {#pretrained-models}

### Hugging Face Models

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load ViT model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# For other variants
models_to_try = [
    "google/vit-tiny-patch16-224",
    "google/vit-small-patch16-224",
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224"
]
```

### TIMM Library Models

```python
import timm

# List available models
model_names = timm.list_models(pretrained=True)
vision_transformers = [name for name in model_names if 'vit' in name.lower()]

print("Available ViT models:")
for name in vision_transformers:
    model = timm.create_model(name, pretrained=True)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### PyTorch Hub Models

```python
# ViT models from PyTorch Hub
vit_models = {
    'deit_tiny_patch16_224': 'facebookresearch/deit',
    'deit_small_patch16_224': 'facebookresearch/deit',
    'deit_base_patch16_224': 'facebookresearch/deit',
    'vit_small_patch16_224': 'facebookresearch/deit',
    'vit_base_patch16_224': 'facebookresearch/deit',
}

for name, repo in vit_models.items():
    model = torch.hub.load(repo, name)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Custom Model Loading

```python
def load_custom_vit_model(model_path, num_classes):
    """Load custom trained ViT model"""
    # Load model architecture
    model = timm.create_model('vit_base_patch16_224', num_classes=num_classes)

    # Load state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# Usage
model = load_custom_vit_model('path/to/checkpoint.pth', num_classes=10)
```

---

## Quick Commands Reference

### Training

```bash
# Basic training
python train.py --model vit_base_patch16_224 --batch-size 32 --epochs 100

# Mixed precision training
python train.py --model vit_large_patch16_224 --fp16 --batch-size 16

# Distributed training
python -m torch.distributed.launch --nproc_per_node=8 train.py --distributed
```

### Inference

```bash
# Single image prediction
python predict.py --model vit_base_patch16_224 --image image.jpg

# Batch prediction
python predict.py --model vit_base_patch16_224 --batch-dir /path/to/images/
```

### Model Conversion

```python
# Convert to ONNX
torch.onnx.export(model, dummy_input, "vit_model.onnx",
                 input_names=['input'], output_names=['output'])

# Convert to TorchScript
model.eval()
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "vit_model.pt")
```

---

This comprehensive cheatsheet provides quick reference for all major aspects of Vision Transformers and modern computer vision, from basic architecture to advanced training techniques and debugging strategies.
