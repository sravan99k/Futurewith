# Vision Transformers and Modern Computer Vision - Theory

## Table of Contents

1. [Introduction to Vision Transformers](#introduction)
2. [Historical Context: From CNNs to Transformers](#historical-context)
3. [Vision Transformer Architecture](#vit-architecture)
4. [Core Components](#core-components)
5. [Patch Embedding and Positional Encoding](#patch-embedding)
6. [Self-Attention Mechanism in Vision](#attention-mechanism)
7. [Vision Transformer Variants](#vit-variants)
8. [Comparison with CNNs](#cnn-comparison)
9. [Advanced Vision Transformer Architectures](#advanced-architectures)
10. [Training Strategies](#training-strategies)
11. [Real-World Applications](#applications)
12. [Mathematical Foundations](#mathematical-foundations)

## 1. Introduction to Vision Transformers {#introduction}

Vision Transformers (ViTs) represent a paradigm shift in computer vision, bringing the transformer architecture that revolutionized natural language processing to image analysis tasks. Unlike traditional Convolutional Neural Networks (CNNs) that rely on spatially local operations, ViTs treat images as sequences of patches and apply the self-attention mechanism globally across the entire image.

### Why Vision Transformers Matter

**Traditional Computer Vision Limitations:**

- **Spatial locality**: CNNs only see small local neighborhoods through convolutions
- **Limited receptive field**: Requires many layers to capture global context
- **Fixed architecture**: Hard to capture long-range dependencies
- **Computational inefficiency**: More layers needed for larger contexts

**ViT Advantages:**

- **Global receptive field**: Self-attention captures relationships across entire image
- **Flexible architecture**: Can process any sequence length
- **Better modeling of long-range dependencies**: Essential for complex visual scenes
- **Scalability**: Performance improves with more data and compute

## 2. Historical Context: From CNNs to Transformers {#historical-context}

### The CNN Era

1. **LeNet (1998)**: First successful CNN for digit recognition
2. **AlexNet (2012)**: Breakthrough performance on ImageNet, started deep learning revolution
3. **VGG, ResNet (2014-2015)**: Deeper networks with skip connections
4. **EfficientNet (2019)**: Optimized scaling of model depth, width, and resolution

### The Transformer Revolution in NLP

1. **Attention is All You Need (2017)**: Transformer architecture for sequence modeling
2. **BERT (2018)**: Bidirectional encoder for language understanding
3. **GPT series (2018-2020)**: Autoregressive language generation
4. **T5 (2019)**: Text-to-Text Transfer Transformer

### The Vision Transformer Breakthrough (2020)

**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**

- First paper to apply transformers directly to image classification
- Achieved state-of-the-art results on ImageNet
- Demonstrated that CNN inductive biases are not necessary with sufficient data

## 3. Vision Transformer Architecture {#vit-architecture}

### High-Level Overview

```
Image → Patch Embedding → Positional Encoding → Multi-Head Attention → MLP → Classification
```

The ViT architecture consists of:

1. **Patch Embedding**: Split image into fixed-size patches
2. **Positional Encoding**: Add learnable position information
3. **Transformer Encoder**: Stack of transformer blocks
4. **Classification Head**: Global average pooling + linear layer

### Detailed Architecture

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
```

## 4. Core Components {#core-components}

### 4.1 Patch Embedding

The image is divided into non-overlapping patches, each flattened and linearly projected to the embedding dimension.

**Mathematical Formulation:**

- Input: Image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$
- Patch size: $P \times P$
- Number of patches: $N = \frac{HW}{P^2}$
- Patch embedding: $\mathbf{x}_{patch} = \text{Conv2d}(C, E, P)(x)$

**Implementation:**

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B x N x E
        return x
```

### 4.2 Position Encoding

Since transformers have no inherent understanding of sequence order, position encodings are added to patch embeddings to provide spatial information.

**Types of Position Encoding:**

1. **Sine-Cosine Encoding**: Used in original transformer paper
2. **Learnable Position Embeddings**: Used in ViT (more common)

```python
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embed
```

### 4.3 Class Token

A special `[CLS]` token is prepended to the sequence and used for final classification, similar to BERT.

```python
def forward(self, x):
    B = x.shape[0]

    # Add class token
    cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x E
    x = torch.cat((cls_tokens, x), dim=1)  # B x (N+1) x E

    # Add position embedding
    x = x + self.pos_embed
```

## 5. Patch Embedding and Positional Encoding {#patch-embedding}

### 5.1 Patch Size Considerations

**Common Patch Sizes:**

- **16x16**: Standard choice, balances compute and performance
- **32x32**: Larger patches, fewer tokens, faster training
- **14x14**: Smaller patches, more detail, higher compute cost

**Trade-offs:**

- **Smaller patches**: More tokens, richer representation, higher compute
- **Larger patches**: Fewer tokens, coarser representation, faster inference

### 5.2 Positional Encoding Types

#### Learnable Position Embeddings

```python
# ViT uses learnable position embeddings
self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

# Forward pass
x = x + self.pos_embed
```

#### Sine-Cosine Position Encoding

```python
def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
```

## 6. Self-Attention Mechanism in Vision {#attention-mechanism}

### 6.1 Multi-Head Self-Attention

The core of transformer architecture, allowing each token to attend to all other tokens in the sequence.

**Mathematical Formulation:**

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 6.2 Self-Attention in Vision Context

In ViT, each patch attends to all other patches, allowing modeling of long-range spatial relationships.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x N x C_per_head
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### 6.3 Computational Complexity

**Attention Complexity Analysis:**

- **Space complexity**: $O(n^2 d)$ for sequence length n and dimension d
- **Time complexity**: $O(n^2 d)$ for attention computation
- **For images**: $n = \frac{HW}{p^2}$ where p is patch size

## 7. Vision Transformer Variants {#vit-variants}

### 7.1 Standard ViT Models

| Model     | Patch Size | Embed Dim | Depth | Heads | Parameters | ImageNet Top-1 |
| --------- | ---------- | --------- | ----- | ----- | ---------- | -------------- |
| ViT-Base  | 16x16      | 768       | 12    | 12    | 86M        | 77.9%          |
| ViT-Large | 16x16      | 1024      | 24    | 16    | 307M       | 85.2%          |
| ViT-Huge  | 14x14      | 1280      | 32    | 16    | 632M       | 88.5%          |

### 7.2 Hierarchical Transformers

#### Swin Transformer

**Key Innovation**: Hierarchical feature maps and shifted window attention

- **Linear complexity**: $O(n)$ instead of $O(n^2)$
- **Shifted windows**: Reduces attention computation
- **Hierarchical structure**: Like CNNs, generates multi-scale features

```python
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_stages = len(depths)

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Build Swin Transformer blocks
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = nn.Sequential(
                SwinTransformerBlock(
                    dim=embed_dim * 2 ** i_stage,
                    num_heads=num_heads[i_stage],
                    window_size=7,
                    shift_size=0 if (i_stage % 2 == 0) else 7 // 2,
                    mlp_ratio=4.0,
                ),
                *[SwinTransformerBlock(
                    dim=embed_dim * 2 ** i_stage,
                    num_heads=num_heads[i_stage],
                    window_size=7,
                    shift_size=0,
                    mlp_ratio=4.0,
                ) for _ in range(depths[i_stage] - 1)]
            )
            self.stages.append(stage)
```

### 7.3 Efficient Transformers

#### Vision Transformer with Downsampling

```python
class ViTWithDownsampling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # ... other components

        # Add downsampling between layers
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim * 2, 3, 2, 1)
            for _ in range(3)
        ])

    def forward(self, x):
        # Process patches
        x = self.patch_embed(x)  # B x N x E

        # Transformer blocks with intermediate downsampling
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [4, 8] and i < len(self.blocks) - 1:
                # Reshape for downsampling
                B, N, E = x.shape
                H = W = int(N ** 0.5)
                x = x.transpose(1, 2).reshape(B, E, H, W)
                x = self.downsample_layers[i // 4](x)
                N = x.shape[2] * x.shape[3]
                x = x.flatten(2).transpose(1, 2)

        return x
```

## 8. Comparison with CNNs {#cnn-comparison}

### 8.1 Architectural Differences

| Aspect                       | CNNs                               | Vision Transformers      |
| ---------------------------- | ---------------------------------- | ------------------------ |
| **Inductive Biases**         | Translation equivariance, locality | None (data-driven)       |
| **Receptive Field**          | Local, grows with depth            | Global from start        |
| **Parameter Sharing**        | Spatial weight sharing             | No spatial sharing       |
| **Computational Complexity** | Linear in image size               | Quadratic in patch count |

### 8.2 Performance Characteristics

**When ViTs Excel:**

- **Large-scale datasets** (ImageNet-21k, JFT-300M)
- **Complex long-range dependencies**
- **Transfer learning scenarios**
- **When compute is abundant**

**When CNNs Excel:**

- **Small datasets** (CIFAR-10, CIFAR-100)
- **Efficient deployment** (mobile, edge)
- **When translation equivariance is crucial**
- **Limited compute resources**

### 8.3 Hybrid Approaches

#### CNN + Transformer Combinations

```python
class HybridViT(nn.Module):
    def __init__(self, cnn_backbone='resnet50', embed_dim=768, num_heads=12):
        super().__init__()

        # CNN backbone for feature extraction
        self.cnn_backbone = timm.create_model(cnn_backbone, pretrained=True)

        # Remove final classification layer
        self.cnn_features = nn.Sequential(*list(self.cnn_backbone.children())[:-2])

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads),
            num_layers=6
        )

    def forward(self, x):
        # Extract features with CNN
        features = self.cnn_features(x)  # B x C x H x W

        # Flatten spatial dimensions
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # B x (H*W) x C

        # Apply transformer
        transformed = self.transformer(features)

        return transformed
```

## 9. Advanced Vision Transformer Architectures {#advanced-architectures}

### 9.1 Deformable Vision Transformer (Deformable ViT)

**Innovation**: Deformable attention - learns to attend to relevant regions

```python
class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=8):
        super().__init__()
        self.num_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(dim, num_heads * n_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * n_points)

    def forward(self, query, value, reference_points):
        batch_size, num_queries, _ = query.shape
        _, num_value, _ = value.shape

        # Generate offset and attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.n_points
        )

        # Apply attention (simplified)
        output = self.deformable_attention(query, value,
                                          sampling_offsets, attention_weights)
        return output
```

### 9.2 Cross-Covariance Image Transformer (XCiT)

**Innovation**: Cross-covariance attention instead of self-attention

```python
class CrossCovarianceAttention(nn.Module):
    def forward(self, x):
        # x: B x N x C
        q, k, v = x.chunk(3, dim=-1)

        # Cross-covariance: C x B x C instead of B x N x N
        q = q.permute(2, 0, 1)  # C x B x N
        k = k.permute(2, 0, 1)  # C x B x N

        # Cross-covariance computation
        attention = q @ k.transpose(-1, -2)  # C x B x C
        attention = F.softmax(attention / self.temperature, dim=-1)

        # Apply to values
        out = attention @ v.permute(2, 0, 1)  # C x B x N
        return out.permute(1, 2, 0)  # B x N x C
```

### 9.3 Convolutional Vision Transformer (CvT)

**Innovation**: Convolution before transformer to reduce token count

```python
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3):
        super().__init__()

        # Convolutional token embedding
        self.conv_token_embed = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )

        # Projection to transformer dimension
        self.conv_proj = nn.Conv2d(192, 768, 1)

        # Transformer
        self.transformer = TransformerEncoder(
            layers=nn.ModuleList([CvTBlock(768) for _ in range(12)])
        )

    def forward(self, x):
        # Convolutional processing
        x = self.conv_token_embed(x)
        x = self.conv_proj(x)
        B, C, H, W = x.shape

        # Flatten for transformer
        x = x.flatten(2).transpose(1, 2)

        # Transformer processing
        x = self.transformer(x)

        return x
```

## 10. Training Strategies {#training-strategies}

### 10.1 Data Requirements

**Why ViTs Need More Data:**

- **No inductive biases**: Must learn everything from data
- **Parameter count**: ViT-Base has 86M parameters vs ResNet-50's 25M
- **Training stability**: Requires careful optimization

**Data Scaling Recommendations:**

- **Small datasets (< 1M)**: Use CNNs or hybrid approaches
- **Medium datasets (1M-10M)**: ViT-Base with strong regularization
- **Large datasets (> 10M)**: ViT-Large/Huge with extensive augmentation

### 10.2 Advanced Augmentation

#### Mixup and CutMix

```python
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
```

#### RandAugment

```python
class RandAugment:
    def __init__(self, n, m):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations

    def __call__(self, img):
        for _ in range(self.n):
            op = np.random.choice(self.augmentations)
            img = op(img, self.m)
        return img
```

### 10.3 Optimization Techniques

#### Learning Rate Scheduling

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                   num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda)
```

#### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 10.4 Regularization Techniques

#### Dropout Variants

```python
# Standard dropout
nn.Dropout(0.1)

# Stochastic depth (drop entire layers)
if training and torch.rand(1).item() < 0.1:
    return identity  # Skip this block

# LayerScale (learnable scaling)
self.gamma = nn.Parameter(torch.ones(self.out_chans) * 1e-4)
```

## 11. Real-World Applications {#applications}

### 11.1 Image Classification

**Implementation Example:**

```python
def train_vit_classifier(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### 11.2 Object Detection

#### DETR (Detection Transformer)

```python
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.class_embed = nn.Linear(transformer.d_model, num_classes + 1)
        self.bbox_embed = MLP(transformer.d_model, transformer.d_model, 4, 3)

    def forward(self, samples):
        features = self.backbone(samples)

        # Transformer encoder-decoder
        hs = self.transformer(features, query)

        # Classification and bbox prediction
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return outputs_class, outputs_coord
```

### 11.3 Medical Image Analysis

#### Applications:

- **Radiology**: X-ray, CT, MRI analysis
- **Pathology**: Histopathology slide analysis
- **Ophthalmology**: Retinal fundus analysis

**Example: Chest X-ray Classification**

```python
class MedicalViT(nn.Module):
    def __init__(self, img_size=512, patch_size=16, num_classes=14):
        super().__init__()

        # Larger image size for medical images
        self.vit = timm.create_model('vit_base_patch16_224',
                                   img_size=img_size,
                                   patch_size=patch_size,
                                   num_classes=num_classes)

        # Add multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_classes, num_classes)
        )

    def forward(self, x):
        features = self.vit.forward_features(x)
        logits = self.vit.forward_head(features)
        return torch.sigmoid(self.classifier(logits))
```

### 11.4 Autonomous Driving

#### Tesla Vision System Integration

```python
class TeslaVisionSystem(nn.Module):
    def __init__(self):
        super().__init__()

        # Multi-camera input
        self.camera_encoders = nn.ModuleList([
            SwinTransformer() for _ in range(8)  # 8 cameras
        ])

        # Bird's eye view transformer
        self.bev_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8), num_layers=6
        )

        # Task heads
        self.detection_head = DetectionHead()
        self.segmentation_head = SegmentationHead()
        self.trajectory_head = TrajectoryHead()

    def forward(self, camera_inputs):
        # Process each camera
        features = []
        for i, encoder in enumerate(self.camera_encoders):
            feat = encoder(camera_inputs[i])
            features.append(feat)

        # Combine and transform to bird's eye view
        combined = torch.cat(features, dim=1)
        bev_features = self.bev_transformer(combined)

        # Task-specific outputs
        detections = self.detection_head(bev_features)
        segmentation = self.segmentation_head(bev_features)
        trajectory = self.trajectory_head(bev_features)

        return detections, segmentation, trajectory
```

## 12. Mathematical Foundations {#mathematical-foundations}

### 12.1 Self-Attention Formulation

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Parameters:**

- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$: Linear projections
- $W^O \in \mathbb{R}^{h \times d_v \times d_{model}}$: Output projection

### 12.2 Vision Transformer Embedding

**Patch Embedding:**

```
z_0 = [x_class; x_p^1E; x_p^2E; ...; x_p^N E] + E_pos
```

Where:

- $x_{class} \in \mathbb{R}^d$: Learnable classification token
- $x_p^i \in \mathbb{R}^{P^2 \cdot C}$: Flattened i-th patch
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$: Patch embedding projection
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$: Position embedding

**Transformer Block:**

```
z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ
```

**Layer Normalization:**

```
LN(x) = γ ⊙ (x - μ)/σ + β

where μ and σ are mean and standard deviation of x
```

### 12.3 Computational Complexity

**Attention Complexity:**

- **Time**: $O(n^2 d)$ for sequence length n and dimension d
- **Space**: $O(n^2)$ for attention matrix

**Patch Embedding Complexity:**

- **Time**: $O(n p^2 c)$ for patch size p and channels c
- **Space**: $O(n d)$ for n patches and dimension d

**For typical ViT:**

- Input: $224 \times 224$ image, $P=16$, $N=196$ patches
- Attention: $O(196^2 \times 768) = O(29.5M)$ operations
- Memory: $196 \times 196 = 38,416$ attention weights per head

### 12.4 Performance Scaling Laws

**ViT Performance vs Model Size:**

```
log(accuracy) = α * log(parameters) + β
```

**Data Efficiency:**

- ViT-Large requires ~100x more data than ResNet for similar performance
- Performance follows power law with model size and dataset size
- Critical threshold around 100M images for optimal ViT performance

---

This comprehensive theoretical foundation covers all aspects of Vision Transformers and modern computer vision. The mathematical formulations, architectural details, and practical implementations provide a complete understanding of this revolutionary paradigm in computer vision.
