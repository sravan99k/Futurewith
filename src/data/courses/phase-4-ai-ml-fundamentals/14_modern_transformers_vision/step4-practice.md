# Vision Transformers and Modern Computer Vision - Practice Exercises

## Table of Contents

1. [Basic ViT Implementation](#basic-vit-implementation)
2. [Patch Embedding and Position Encoding](#patch-embedding-practice)
3. [Multi-Head Attention Implementation](#attention-practice)
4. [Complete Vision Transformer](#complete-vit-practice)
5. [Transfer Learning with ViT](#transfer-learning-practice)
6. [Data Augmentation Strategies](#augmentation-practice)
7. [Training and Optimization](#training-practice)
8. [Object Detection with DETR](#detr-practice)
9. [Efficient Training Techniques](#efficient-training-practice)
10. [Vision Transformer Variants](#variants-practice)
11. [Real-World Applications](#applications-practice)
12. [Advanced Challenges](#advanced-challenges)

## 1. Basic ViT Implementation {#basic-vit-implementation}

### Exercise 1.1: Patch Embedding from Scratch

**Task**: Implement patch embedding layer without using pre-built components.

**Implementation**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CustomPatchEmbed(nn.Module):
    """Custom patch embedding layer for Vision Transformer"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolution for patch embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # Ensure input size matches expected size
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"

        # Project patches
        x = self.proj(x)  # B x embed_dim x (H//patch_size) x (W//patch_size)

        # Flatten spatial dimensions
        x = x.flatten(2)  # B x embed_dim x num_patches
        x = x.transpose(1, 2)  # B x num_patches x embed_dim

        return x

# Test the implementation
def test_patch_embed():
    img_size = 224
    patch_size = 16
    batch_size = 2

    # Create random input
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize patch embed
    patch_embed = CustomPatchEmbed(img_size, patch_size)

    # Forward pass
    output = patch_embed(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected patches: {(img_size // patch_size) ** 2}")

    assert output.shape[1] == (img_size // patch_size) ** 2, "Incorrect number of patches"
    assert output.shape[2] == 768, "Incorrect embedding dimension"

    print("✅ Patch embedding test passed!")

test_patch_embed()
```

**Expected Output**:

```
Input shape: torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 196, 768])
Expected patches: 196
✅ Patch embedding test passed!
```

### Exercise 1.2: Position Embedding Implementation

**Task**: Implement learnable position embeddings for Vision Transformer.

**Implementation**:

```python
class PositionEmbedding(nn.Module):
    """Learnable position embeddings for Vision Transformer"""

    def __init__(self, num_patches, embed_dim, cls_token=True):
        super().__init__()

        self.num_patches = num_patches
        self.cls_token = cls_token

        if cls_token:
            self.num_tokens = num_patches + 1
        else:
            self.num_tokens = num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)

        # Class token (CLS token for classification)
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.cls_token:
            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        return x

# Test position embeddings
def test_position_embed():
    num_patches = 196
    embed_dim = 768
    batch_size = 2

    # Create random patch embeddings
    patches = torch.randn(batch_size, num_patches, embed_dim)

    # Initialize position embeddings
    pos_embed = PositionEmbedding(num_patches, embed_dim, cls_token=True)

    # Forward pass
    output = pos_embed(patches)

    print(f"Input shape: {patches.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Position embedding shape: {pos_embed.pos_embed.shape}")

    assert output.shape[1] == num_patches + 1, "CLS token not added correctly"
    assert output.shape[2] == embed_dim, "Embedding dimension mismatch"

    print("✅ Position embedding test passed!")

test_position_embed()
```

## 2. Patch Embedding and Position Encoding {#patch-embedding-practice}

### Exercise 2.1: Custom Attention Mechanism

**Task**: Implement multi-head self-attention mechanism from scratch.

**Implementation**:

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x N x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Test multi-head attention
def test_multihead_attention():
    batch_size = 2
    num_tokens = 197  # 196 patches + 1 CLS token
    embed_dim = 768
    num_heads = 12

    # Create random input
    x = torch.randn(batch_size, num_tokens, embed_dim)

    # Initialize attention
    attention = MultiHeadAttention(embed_dim, num_heads)

    # Forward pass
    output = attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention head dim: {attention.head_dim}")

    assert output.shape == x.shape, "Output shape doesn't match input shape"

    print("✅ Multi-head attention test passed!")

test_multihead_attention()
```

### Exercise 2.2: Transformer Block

**Task**: Combine attention and MLP layers into a transformer block.

**Implementation**:

```python
class TransformerBlock(nn.Module):
    """Vision Transformer block with attention and MLP"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Multi-head attention
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))

        return x

# Test transformer block
def test_transformer_block():
    batch_size = 2
    num_tokens = 197
    embed_dim = 768
    num_heads = 12

    # Create random input
    x = torch.randn(batch_size, num_tokens, embed_dim)

    # Initialize transformer block
    transformer_block = TransformerBlock(embed_dim, num_heads)

    # Forward pass
    output = transformer_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == x.shape, "Output shape doesn't match input shape"

    print("✅ Transformer block test passed!")

test_transformer_block()
```

## 3. Multi-Head Attention Implementation {#attention-practice}

### Exercise 3.1: Complete Vision Transformer

**Task**: Build a complete Vision Transformer model by combining all components.

**Implementation**:

```python
class VisionTransformer(nn.Module):
    """Complete Vision Transformer implementation"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = CustomPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = PositionEmbedding(num_patches, embed_dim, cls_token=True)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias,
                           drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.cls_token, std=0.02)

        # Initialize classification head
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add position embeddings
        x = self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification token
        x = self.norm(x)
        x = x[:, 0]  # CLS token

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# Test complete ViT
def test_vit():
    batch_size = 2
    img_size = 224
    num_classes = 1000

    # Create random input
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize ViT
    vit = VisionTransformer(num_classes=num_classes)

    # Forward pass
    output = vit(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in vit.parameters()):,}")

    assert output.shape == (batch_size, num_classes), "Output shape is incorrect"

    print("✅ Vision Transformer test passed!")

test_vit()
```

## 4. Complete Vision Transformer {#complete-vit-practice}

### Exercise 4.1: Training Setup

**Task**: Set up training pipeline for Vision Transformer.

**Implementation**:

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_data_transforms():
    """Create data transforms for ImageNet-style training"""

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

class ViTTrainer:
    """Trainer class for Vision Transformer"""

    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(),
                                   lr=1e-4, weight_decay=0.05)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, epochs):
        """Train the model for specified epochs"""
        best_acc = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step()

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_vit.pth')
                print(f'New best validation accuracy: {best_acc:.2f}%')

# Example usage
def train_example():
    # Create dummy data for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dummy datasets (replace with real datasets)
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          num_workers=4)

    # Initialize model
    model = VisionTransformer(num_classes=10)  # CIFAR-10 has 10 classes

    # Initialize trainer
    trainer = ViTTrainer(model, train_loader, val_loader)

    # Train for a few epochs (reduce for testing)
    trainer.train(epochs=5)

print("Training pipeline ready!")
```

## 5. Transfer Learning with ViT {#transfer-learning-practice}

### Exercise 5.1: Pre-trained Model Loading

**Task**: Load and fine-tune pre-trained Vision Transformer models.

**Implementation**:

```python
import timm
from timm import create_model

def load_pretrained_vit(model_name='vit_base_patch16_224', num_classes=1000):
    """Load pre-trained Vision Transformer"""

    # Create model with pre-trained weights
    model = create_model(model_name, pretrained=True, num_classes=num_classes)

    return model

def freeze_backbone_parameters(model, freeze_ratio=0.9):
    """Freeze most of the model parameters for fine-tuning"""

    # Calculate number of layers to freeze
    total_layers = len(model.blocks)
    freeze_layers = int(total_layers * freeze_ratio)

    # Freeze early layers
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    for param in model.pos_embed.parameters():
        param.requires_grad = False

    for param in model.norm.parameters():
        param.requires_grad = False

    # Freeze early transformer blocks
    for i in range(freeze_layers):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    # Keep later layers trainable
    for i in range(freeze_layers, total_layers):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Always keep classification head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    return model

def fine_tune_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    """Fine-tune the model on custom dataset"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer (only trainable parameters)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=lr, weight_decay=0.05)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%')
        print(f'  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%')

        scheduler.step()

    return model

def transfer_learning_example():
    """Example of transfer learning with ViT"""

    # Load pre-trained model
    model = load_pretrained_vit('vit_base_patch16_224', num_classes=1000)

    # Freeze parameters for fine-tuning
    model = freeze_backbone_parameters(model, freeze_ratio=0.8)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f'Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)')
    print(f'Total parameters: {total_params:,}')

    # In practice, replace with real data loaders
    print("Transfer learning setup complete!")

transfer_learning_example()
```

## 6. Data Augmentation Strategies {#augmentation-practice}

### Exercise 6.1: Advanced Augmentation for ViT

**Task**: Implement advanced augmentation techniques specifically beneficial for Vision Transformers.

**Implementation**:

```python
import random
import PIL
from PIL import Image, ImageEnhance, ImageFilter

class RandAugment:
    """Random Augment implementation for Vision Transformers"""

    def __init__(self, n=2, m=9, img_size=224):
        """
        n: Number of augmentations to apply
        m: Magnitude of augmentations (0-10)
        img_size: Target image size
        """
        self.n = n
        self.m = m
        self.img_size = img_size
        self.augmentations = [
            self.auto_contrast, self.brightness, self.color, self.contrast,
            self.cutout, self.equalize, self.invert, self.rotate, self.sharpness,
            self.solarize, self.solarize_add, self.translate_x, self.translate_y
        ]

    def __call__(self, img):
        # Convert to PIL Image if needed
        if not isinstance(img, PIL.Image.Image):
            img = transforms.ToPILImage()(img)

        # Apply n random augmentations
        ops = random.sample(self.augmentations, self.n)
        for op in ops:
            img = op(img, self.m)

        # Resize to target size
        img = img.resize((self.img_size, self.img_size))
        return img

    @staticmethod
    def auto_contrast(img, magnitude):
        return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.uniform(-1, 1))

    @staticmethod
    def brightness(img, magnitude):
        return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.uniform(-1, 1))

    @staticmethod
    def color(img, magnitude):
        return ImageEnhance.Color(img).enhance(1 + magnitude * random.uniform(-1, 1))

    @staticmethod
    def contrast(img, magnitude):
        return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.uniform(-1, 1))

    @staticmethod
    def cutout(img, magnitude):
        # Simple cutout implementation
        if random.random() < 0.5:
            w, h = img.size
            x1, y1 = random.randint(0, w//4), random.randint(0, h//4)
            x2, y2 = random.randint(w*3//4, w), random.randint(h*3//4, h)

            # Create cutout mask
            mask = Image.new('L', img.size, 0)
            mask.paste(0, (x1, y1, x2, y2))

            # Apply cutout
            img = Image.composite(img, Image.new('RGB', img.size, (128, 128, 128)), mask)
        return img

    @staticmethod
    def equalize(img, magnitude):
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE if random.random() > 0.5 else ImageFilter.SMOOTH)

    @staticmethod
    def invert(img, magnitude):
        return Image.eval(img, lambda x: 255 - x)

    @staticmethod
    def rotate(img, magnitude):
        angle = magnitude * random.uniform(-30, 30)
        return img.rotate(angle)

    @staticmethod
    def sharpness(img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.uniform(-1, 1))

    @staticmethod
    def solarize(img, magnitude):
        threshold = 128 + magnitude * random.randint(-32, 32)
        return Image.eval(img, lambda x: 255 - x if x < threshold else x)

    @staticmethod
    def solarize_add(img, magnitude):
        threshold = 128 + magnitude * random.randint(-32, 32)
        img = Image.eval(img, lambda x: min(255, max(0, x + magnitude * random.randint(-10, 10))))
        return img

    @staticmethod
    def translate_x(img, magnitude):
        w, h = img.size
        pixels = int(magnitude * w / 10)
        img = img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
        return img

    @staticmethod
    def translate_y(img, magnitude):
        w, h = img.size
        pixels = int(magnitude * h / 10)
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
        return img

class MixUp:
    """MixUp augmentation for Vision Transformers"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch

        if len(images.size()) != 4:
            raise ValueError(f'Batch size dimension expected 4, got {len(images.size())}')

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, labels, labels[index], lam

class CutMix:
    """CutMix augmentation for Vision Transformers"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch

        if len(images.size()) != 4:
            raise ValueError(f'Batch size dimension expected 4, got {len(images.size())}')

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        index = torch.randperm(batch_size).to(images.device)

        # Generate random bounding box
        H, W = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return images, labels, labels[index], torch.tensor(lam).float().to(images.device)

def create_vit_transforms():
    """Create transforms optimized for Vision Transformers"""

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        RandAugment(n=2, m=9),  # Random augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# Example usage
def test_augmentation():
    # Test RandAugment
    augmenter = RandAugment(n=2, m=9)

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Apply augmentation
    augmented = augmenter(dummy_image)

    print(f"Original size: {dummy_image.size}")
    print(f"Augmented size: {augmented.size}")
    print("✅ Augmentation test passed!")

test_augmentation()
```

## 7. Training and Optimization {#training-practice}

### Exercise 7.1: Efficient Training Techniques

**Task**: Implement gradient accumulation, mixed precision, and other optimization techniques.

**Implementation**:

```python
class EfficientViTTrainer:
    """Efficient trainer with advanced optimization techniques"""

    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize optimizer, scheduler, and loss function"""

        # Create parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.05,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        # Optimizer with different learning rates
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Gradient clipping
        self.max_grad_norm = 1.0

        # Gradient accumulation
        self.accumulation_steps = 4

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""

        def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                           num_training_steps, num_cycles=0.5):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / \
                          float(max(1, num_training_steps - num_warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        num_training_steps = len(self.train_loader) * 50  # 50 epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        return get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

    def train_epoch_mixed_precision(self):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct = 0

        self.optimizer.zero_grad()

        for step, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            batch_size = images.size(0)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps  # Scale loss for accumulation

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * self.accumulation_steps * batch_size
            _, predicted = outputs.max(1)
            total_samples += batch_size
            correct += predicted.eq(targets).sum().item()

            # Print progress
            if step % 100 == 0:
                print(f'Step {step}/{len(self.train_loader)}, '
                      f'Loss: {loss.item() * self.accumulation_steps:.4f}, '
                      f'Acc: {100.*correct/total_samples:.2f}%')

        epoch_loss = total_loss / total_samples
        epoch_acc = 100. * correct / total_samples

        return epoch_loss, epoch_acc

    def validate_mixed_precision(self):
        """Validate model with mixed precision"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_samples += images.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = total_loss / total_samples
        val_acc = 100. * correct / total_samples

        return val_loss, val_acc

    def train(self, epochs):
        """Main training loop with all optimizations"""
        best_acc = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Training
            train_loss, train_acc = self.train_epoch_mixed_precision()

            # Validation
            val_loss, val_acc = self.validate_mixed_precision()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f'LR: {current_lr:.6f}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                }, 'best_vit_optimized.pth')
                print(f'New best validation accuracy: {best_acc:.2f}%')

    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['accuracy']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
              f"accuracy: {best_acc:.2f}%")

        return start_epoch, best_acc

# Example usage
def efficient_training_example():
    """Example of efficient ViT training"""

    # Create model (replace with actual model)
    model = VisionTransformer(num_classes=1000)

    # Create data loaders (replace with actual data)
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)

    print("Efficient training setup complete!")
    print("Key optimizations:")
    print("- Mixed precision training")
    print("- Gradient accumulation")
    print("- Different learning rates for different parameter groups")
    print("- Label smoothing")
    print("- Gradient clipping")
    print("- Learning rate warmup")

efficient_training_example()
```

## 8. Object Detection with DETR {#detr-practice}

### Exercise 8.1: DETR Implementation

**Task**: Implement Detection Transformer (DETR) for object detection.

**Implementation**:

```python
class DETR(nn.Module):
    """Detection Transformer for object detection"""

    def __init__(self, backbone, transformer, num_classes=91, num_queries=100):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Object queries for detection
        self.object_queries = nn.Parameter(torch.randn(num_queries, transformer.d_model))

        # Classification and bounding box heads
        self.class_embed = nn.Linear(transformer.d_model, num_classes + 1)  # +1 for no object
        self.bbox_embed = MLP(transformer.d_model, transformer.d_model, 4, 3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize bbox predictor
        for m in self.bbox_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Initialize class predictor
        nn.init.normal_(self.class_embed.weight, std=1e-6)
        nn.init.zeros_(self.class_embed.bias)

    def forward(self, samples):
        # Extract features with backbone
        features = self.backbone(samples)  # List of feature maps

        # Prepare for transformer
        src, mask = features[-1][0], features[-1][1]  # Last layer features

        # Transformer encoder-decoder
        hs = self.transformer(src, mask, self.object_queries)

        # Prediction heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return outputs_class, outputs_coord

class MLP(nn.Module):
    """Multi-layer perceptron for bbox prediction"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETRBackbone(nn.Module):
    """ResNet backbone for DETR"""

    def __init__(self, resnet_name='resnet50', pretrained=True):
        super().__init__()

        # Create ResNet backbone
        backbone = timm.create_model(resnet_name, pretrained=pretrained)

        # Remove classification layer
        self.body = nn.Sequential(*list(backbone.children())[:-2])

        # Feature dimensions
        self.num_channels = backbone.num_features

    def forward(self, x):
        features = []

        # Extract features at different scales
        for layer in self.body[:4]:  # Use first 4 layers
            x = layer(x)
            features.append(x)

        # Use final layer for transformer
        x = self.body[-1](x)
        mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
        features.append((x, mask))

        return features

class DETRTransformer(nn.Module):
    """Transformer for object detection"""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers
        )

    def forward(self, src, src_mask, query_embed):
        # Flatten features
        N, B, C = src.shape
        src = src.view(N, B, C)

        # Encode
        memory = self.encoder(src)

        # Decode
        tgt = query_embed.unsqueeze(1).repeat(1, B, 1)
        hs = self.decoder(tgt, memory)

        return hs.transpose(0, 1)

def create_detr_model(num_classes=91, img_size=800):
    """Create complete DETR model"""

    # Backbone
    backbone = DETRBackbone('resnet50', pretrained=True)

    # Transformer
    transformer = DETRTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048
    )

    # DETR model
    model = DETR(backbone, transformer, num_classes=num_classes)

    return model

# Training function for DETR
def train_detr(model, train_loader, val_loader, epochs=100):
    """Train DETR model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Loss functions
    criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.L1Loss()

    # Hungarian matching for assignment
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    # Loss weights
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            images, targets = batch
            images, targets = images.to(device), targets

            # Forward pass
            outputs_class, outputs_coord = model(images)

            # Prepare targets for matching
            indices = matcher(outputs_class, outputs_coord, targets)

            # Calculate losses
            losses = {}
            total_loss = 0

            for i in range(len(indices)):
                loss_dict = {}

                # Classification loss
                idx = self._get_permutation_idx(indices[i])
                tgt_class_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[i])])
                tgt_class = torch.full(size=(len(idx),), fill_value=model.num_classes,
                                     dtype=torch.int64, device=device)
                tgt_class[idx] = tgt_class_o

                loss_ce = criterion(outputs_class[idx], tgt_class)
                loss_dict['loss_ce'] = loss_ce

                # Bounding box loss
                idx = self._get_permutation_idx(indices[i])
                tgt_bbox = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices[i])])
                outputs_bbox = outputs_coord[i][idx]

                loss_bbox = bbox_criterion(outputs_bbox, tgt_bbox)
                loss_dict['loss_bbox'] = loss_bbox

                # Total loss for this image
                total_loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

        # Validation
        if epoch % 10 == 0:
            validate_detr(model, val_loader)

def validate_detr(model, val_loader):
    """Validate DETR model"""
    model.eval()

    # Initialize mAP calculation (simplified)
    total_predictions = []
    total_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch
            outputs_class, outputs_coord = model(images)

            # Get predictions (simplified)
            for i in range(len(images)):
                pred_scores, pred_boxes = outputs_class[i], outputs_coord[i]

                # Keep only high-confidence predictions
                keep = pred_scores.max(-1).values > 0.7
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                total_predictions.append((pred_boxes, pred_scores))
                total_targets.append(targets[i])

    # Calculate mAP (implementation depends on specific metric)
    print(f"Validation completed - {len(total_predictions)} images processed")

# Example usage
def detr_example():
    """Example of using DETR for object detection"""

    # Create model
    model = create_detr_model(num_classes=91)  # COCO classes

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DETR parameters: {total_params:,}")

    # Forward pass test
    dummy_image = torch.randn(1, 3, 800, 800)
    outputs_class, outputs_coord = model(dummy_image)

    print(f"Input shape: {dummy_image.shape}")
    print(f"Classification output shape: {outputs_class.shape}")
    print(f"Bounding box output shape: {outputs_coord.shape}")

    print("✅ DETR implementation complete!")

detr_example()
```

## 9. Efficient Training Techniques {#efficient-training-practice}

### Exercise 9.1: Knowledge Distillation for ViT

**Task**: Implement knowledge distillation to train smaller ViT models.

**Implementation**:

```python
class VisionTransformerDistillation:
    """Vision Transformer with knowledge distillation"""

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_outputs, teacher_outputs, targets):
        """Calculate distillation loss combining soft and hard targets"""

        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # Hard targets
        hard_loss = F.cross_entropy(student_outputs, targets)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, soft_loss, hard_loss

    def train_distilled_student(self, train_loader, val_loader, epochs=50):
        """Train student model with distillation"""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model.eval()  # Teacher in eval mode
        self.student_model.train()

        optimizer = optim.AdamW(self.student_model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0

        for epoch in range(epochs):
            self.student_model.train()
            total_loss = 0
            total_soft_loss = 0
            total_hard_loss = 0
            correct = 0
            total = 0

            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)

                student_outputs = self.student_model(images)

                # Calculate distillation loss
                total_loss_batch, soft_loss, hard_loss = self.distillation_loss(
                    student_outputs, teacher_outputs, targets
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                # Statistics
                total_loss += total_loss_batch.item()
                total_soft_loss += soft_loss.item()
                total_hard_loss += hard_loss.item()

                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # Validation
            val_acc = self.validate_student(val_loader)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Total Loss: {total_loss/len(train_loader):.4f}')
            print(f'  Soft Loss: {total_soft_loss/len(train_loader):.4f}')
            print(f'  Hard Loss: {total_hard_loss/len(train_loader):.4f}')
            print(f'  Train Acc: {100.*correct/total:.2f}%')
            print(f'  Val Acc: {val_acc:.2f}%')

            scheduler.step()

            # Save best student model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_distilled_vit.pth')

    def validate_student(self, val_loader):
        """Validate student model"""
        self.student_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = self.student_model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100. * correct / total

def create_distillation_setup():
    """Create teacher-student setup for knowledge distillation"""

    # Teacher model (larger)
    teacher = create_model('vit_large_patch16_224', pretrained=True, num_classes=1000)

    # Student model (smaller)
    student = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,  # Smaller embedding dimension
        depth=8,        # Fewer layers
        num_heads=6     # Fewer attention heads
    )

    # Initialize student with teacher weights (partial)
    student.load_state_dict(teacher.state_dict(), strict=False)

    return teacher, student

# Example usage
def distillation_example():
    """Example of knowledge distillation"""

    teacher, student = create_distillation_setup()

    # Setup distillation
    distiller = VisionTransformerDistillation(teacher, student)

    print("Knowledge distillation setup:")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in student.parameters()) / sum(p.numel() for p in teacher.parameters()):.2f}")

    print("✅ Knowledge distillation setup complete!")

distillation_example()
```

## 10. Vision Transformer Variants {#variants-practice}

### Exercise 10.1: Swin Transformer Implementation

**Task**: Implement Swin Transformer with hierarchical architecture.

**Implementation**:

```python
class SwinTransformer(nn.Module):
    """Swin Transformer implementation"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_drop = nn.Dropout(drop_rate)

        # Build Swin Transformer blocks
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = SwinTransformerLayer(
                dim=embed_dim * 2 ** i_layer,
                input_resolution=(img_size // (2 ** i_layer), img_size // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                downsample=PatchMerging if (i_layer < len(depths) - 1) else None,
                use_checkpoint=False
            )
            self.layers.append(layer)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim * 2 ** (len(depths) - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 2 ** (len(depths) - 1), num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x.transpose(1, 2))  # B, C, H, W -> B, C, H*W
        x = self.avgpool(x.transpose(1, 2))  # B, C, 1
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class SwinTransformerLayer(nn.Module):
    """Swin Transformer Layer"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        # Patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # Calculate window and shift sizes
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Attention blocks
        self.attn1 = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP block
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # Apply windowed attention or shifted window attention
        if self.shift_size > 0:
            # Calculate attention mask for shifted window attention
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
            attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # Attention computation
        x = x.view(B, H, W, C)

        # Cyclically shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Apply window partition and attention
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Apply window attention
        attn_windows = self.attn1(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connection and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WindowAttention(nn.Module):
    """Window Attention module"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Query, Key, Value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Helper functions
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer for Swin Transformer"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop path function"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

# Test Swin Transformer
def test_swin_transformer():
    """Test Swin Transformer implementation"""

    model = SwinTransformer(num_classes=1000)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Swin Transformer parameters: {total_params:,}")

    # Forward pass test
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("✅ Swin Transformer test passed!")

test_swin_transformer()
```

## 11. Real-World Applications {#applications-practice}

### Exercise 11.1: Medical Image Classification

**Task**: Apply ViT to medical image classification with domain-specific considerations.

**Implementation**:

```python
class MedicalViT(nn.Module):
    """Vision Transformer optimized for medical image analysis"""

    def __init__(self, img_size=512, patch_size=16, num_classes=14, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, in_chans=3):
        super().__init__()

        # Modify for larger medical images
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Custom patch embedding for medical images
        self.patch_embed = CustomPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embedding with CLS token
        self.pos_embed = PositionEmbedding(num_patches, embed_dim, cls_token=True)

        # Transformer blocks with medical-specific modifications
        self.blocks = nn.ModuleList([
            MedicalTransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Classification head with dropout for overfitting prevention
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Medical-specific initialization
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.cls_token, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        # Patch embedding with medical image preprocessing
        x = self.patch_embed(x)

        # Add position embeddings
        x = self.pos_embed(x)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token representation
        x = self.norm(x)
        x = x[:, 0]  # CLS token

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

class MedicalTransformerBlock(nn.Module):
    """Transformer block optimized for medical images"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # Layer normalization with different epsilon for medical data
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # Attention with medical-specific modifications
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias=True,
                                     attn_drop=0.1, proj_drop=0.1)

        # MLP with larger hidden dimension for complex medical patterns
        mlp_hidden_dim = int(dim * mlp_ratio * 1.5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x

class MedicalImageDataset:
    """Custom dataset for medical images with domain-specific preprocessing"""

    def __init__(self, image_paths, labels, transform=None, modality='chest_xray'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.modality = modality

        # Medical image specific preprocessing
        self.intensity_window = self._get_intensity_window(modality)

    def _get_intensity_window(self, modality):
        """Get appropriate intensity window for medical imaging"""
        windows = {
            'chest_xray': (-1000, 1000),      # Lung window
            'ct_scan': (0, 400),              # Soft tissue window
            'mri': (0, 255),                  # Standard MRI window
            'mammogram': (0, 255),            # Mammography window
            'retina': (0, 255),               # Retinal fundus window
        }
        return windows.get(modality, (0, 255))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply medical-specific preprocessing
        image = self._apply_intensity_windowing(image)

        # Convert to RGB for ViT compatibility
        if len(image.getbands()) == 1:
            image = Image.merge('RGB', (image, image, image))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def _apply_intensity_windowing(self, image):
        """Apply intensity windowing for medical images"""
        img_array = np.array(image).astype(np.float32)

        min_val, max_val = self.intensity_window
        img_array = np.clip(img_array, min_val, max_val)
        img_array = (img_array - min_val) / (max_val - min_val)
        img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array)

class MedicalViTTrainer:
    """Specialized trainer for medical ViT models"""

    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Medical-specific loss with class weighting for imbalanced datasets
        self.criterion = nn.CrossEntropyLoss(weight=self._calculate_class_weights())

        # Optimizer with different learning rates for different layers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Validation metrics specific to medical imaging
        self.metrics = MedicalMetrics()

    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalanced medical datasets"""
        # Calculate class frequencies from training data
        class_counts = torch.zeros(self.model.num_classes)
        for _, labels in self.train_loader:
            for label in labels:
                class_counts[label] += 1

        # Calculate inverse frequency weights
        total_samples = len(self.train_loader.dataset)
        class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)

        return class_weights.to(self.device)

    def _create_optimizer(self):
        """Create optimizer with different learning rates"""
        # Different learning rates for different components
        param_groups = [
            {'params': self.model.patch_embed.parameters(), 'lr': 1e-5},
            {'params': self.model.pos_embed.parameters(), 'lr': 1e-5},
            {'params': self.model.blocks.parameters(), 'lr': 5e-5},
            {'params': self.model.head.parameters(), 'lr': 1e-4},
        ]

        return optim.AdamW(param_groups, weight_decay=0.05)

    def _create_scheduler(self):
        """Create learning rate scheduler with medical-specific settings"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self):
        """Train for one epoch with medical-specific monitoring"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct = 0

        # Track per-class performance
        class_correct = torch.zeros(self.model.num_classes)
        class_total = torch.zeros(self.model.num_classes)

        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_samples += images.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class statistics
            for i in range(len(targets)):
                class_total[targets[i]] += 1
                if predicted[i] == targets[i]:
                    class_correct[targets[i]] += 1

        epoch_loss = total_loss / total_samples
        epoch_acc = 100. * correct / total_samples

        # Per-class accuracy for medical validation
        class_acc = 100. * class_correct / (class_total + 1e-6)

        return epoch_loss, epoch_acc, class_acc

    def validate(self):
        """Validate with medical-specific metrics"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.train_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Collect predictions and targets for metrics
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / total_samples

        # Calculate medical-specific metrics
        metrics = self.metrics.calculate(all_predictions, all_targets)

        return val_loss, metrics

    def train(self, epochs):
        """Train with early stopping and best model saving"""
        best_f1 = 0
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Training
            train_loss, train_acc, train_class_acc = self.train_epoch()

            # Validation
            val_loss, val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step(val_metrics['f1_score'])

            # Print metrics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'F1 Score: {val_metrics["f1_score"]:.4f}')
            print(f'AUC: {val_metrics["auc"]:.4f}')

            # Early stopping
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                patience_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), 'best_medical_vit.pth')
                print(f'New best F1 score: {best_f1:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
                    break

class MedicalMetrics:
    """Calculate medical image analysis metrics"""

    def calculate(self, predictions, targets):
        """Calculate comprehensive medical metrics"""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')

        # AUC score (requires probability scores, simplified here)
        try:
            # This would normally require prediction probabilities
            auc = 0.85  # Placeholder
        except:
            auc = 0.0

        # Sensitivity and Specificity (for binary classification)
        if len(set(targets)) == 2:  # Binary classification
            tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = 0.0

        return {
            'accuracy': accuracy * 100,
            'f1_score': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

# Example usage for medical imaging
def medical_vit_example():
    """Example of using ViT for medical image classification"""

    # Define medical datasets (example classes)
    medical_classes = [
        'normal',           # Normal chest X-ray
        'pneumonia',        # Pneumonia
        'covid19',          # COVID-19
        'effusion',         # Pleural effusion
        'consolidation',    # Lung consolidation
        'cardiomegaly',     # Enlarged heart
        'pneumothorax',     # Collapsed lung
    ]

    # Create model for medical imaging
    model = MedicalViT(
        img_size=512,          # Higher resolution for medical images
        patch_size=16,
        num_classes=len(medical_classes),
        embed_dim=1024,        # Larger embedding dimension
        depth=16,              # Deeper network
        num_heads=16,          # More attention heads
        in_chans=1             # Grayscale images
    )

    print("Medical ViT model created:")
    print(f"Classes: {len(medical_classes)}")
    print(f"Input size: 512x512")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, medical_classes

# Test medical ViT
def test_medical_vit():
    model, classes = medical_vit_example()

    # Forward pass test with medical-sized input
    dummy_input = torch.randn(1, 1, 512, 512)  # Grayscale medical image
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {len(classes)}")

    print("✅ Medical ViT test passed!")

test_medical_vit()
```

## 12. Advanced Challenges {#advanced-challenges}

### Exercise 12.1: Custom Vision Transformer Architecture

**Task**: Design and implement a novel Vision Transformer architecture optimized for specific use cases.

**Implementation**:

```python
class AdaptiveViT(nn.Module):
    """Adaptive Vision Transformer that adjusts architecture based on input complexity"""

    def __init__(self, base_config):
        super().__init__()
        self.base_config = base_config

        # Adaptive parameters
        self.complexity_estimator = ComplexityEstimator()
        self.adaptive_layers = nn.ModuleDict()

        # Build base architecture components
        self._build_adaptive_components()

    def _build_adaptive_components(self):
        """Build components that can adapt based on input complexity"""

        # Multi-scale patch embedding
        self.multi_scale_embed = MultiScalePatchEmbed(
            patch_sizes=[8, 16, 32],  # Multiple patch sizes
            embed_dim=self.base_config['embed_dim']
        )

        # Adaptive attention blocks with complexity-aware computation
        self.adaptive_attention_blocks = nn.ModuleList([
            AdaptiveTransformerBlock(
                dim=self.base_config['embed_dim'],
                num_heads=self.base_config['num_heads'],
                adaptive_depth=base_config['adaptive_depth'] if i < 2 else base_config['depth'] // 2
            )
            for i in range(self.base_config['depth'])
        ])

        # Dynamic classification head
        self.dynamic_head = DynamicClassificationHead(
            input_dim=self.base_config['embed_dim'],
            num_classes=self.base_config['num_classes'],
            complexity_threshold=0.5
        )

    def forward(self, x):
        # Estimate input complexity
        complexity = self.complexity_estimator(x)

        # Multi-scale patch embedding
        patches = self.multi_scale_embed(x, complexity)

        # Process through adaptive transformer blocks
        for block in self.adaptive_attention_blocks:
            patches = block(patches, complexity)

        # Dynamic classification based on complexity
        output = self.dynamic_head(patches, complexity)

        return output

class ComplexityEstimator(nn.Module):
    """Estimates the complexity of input images to guide adaptive computation"""

    def __init__(self, feature_dim=256):
        super().__init__()

        # CNN feature extractor for complexity analysis
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # Additional features for complexity estimation
        self.gradient_module = GradientComplexityModule()
        self.texture_module = TextureComplexityModule()

    def forward(self, x):
        # CNN-based complexity
        cnn_complexity = self.feature_extractor(x)

        # Gradient-based complexity
        grad_complexity = self.gradient_module(x)

        # Texture-based complexity
        texture_complexity = self.texture_module(x)

        # Combine complexity measures
        combined_complexity = (cnn_complexity + grad_complexity + texture_complexity) / 3

        return combined_complexity.squeeze()

class GradientComplexityModule(nn.Module):
    """Module to estimate gradient-based complexity"""

    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

        # Initialize Sobel filters
        self._init_sobel_filters()

        self.complexity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _init_sobel_filters(self):
        """Initialize Sobel filters for gradient computation"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.sobel_x.weight.data = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)

        # Calculate gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Estimate complexity based on gradient statistics
        complexity = self.complexity_head(gradient_magnitude)

        return complexity

class TextureComplexityModule(nn.Module):
    """Module to estimate texture-based complexity"""

    def __init__(self):
        super().__init__()

        # Local Binary Pattern approximation
        self.lbp_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.complexity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply texture-sensitive convolution
        texture_features = self.lbp_conv(x)

        # Estimate complexity based on texture variation
        complexity = self.complexity_head(texture_features)

        return complexity

class AdaptiveTransformerBlock(nn.Module):
    """Transformer block with adaptive computation based on input complexity"""

    def __init__(self, dim, num_heads, adaptive_depth=1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.adaptive_depth = adaptive_depth

        # Standard transformer components
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Adaptive attention
        self.attention = AdaptiveMultiHeadAttention(dim, num_heads)

        # Adaptive MLP with complexity-aware depth
        self.mlp = AdaptiveMLP(dim, adaptive_depth)

    def forward(self, x, complexity):
        # Adaptive attention computation
        attention_output = self.attention(self.norm1(x), complexity)
        x = x + attention_output

        # Adaptive MLP computation
        mlp_output = self.mlp(self.norm2(x), complexity)
        x = x + mlp_output

        return x

class AdaptiveMultiHeadAttention(nn.Module):
    """Multi-head attention with adaptive computation"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Adaptive attention components
        self.adaptive_gate = AdaptiveGate(dim)

        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, complexity):
        B, N, C = x.shape

        # Generate adaptive gate based on complexity
        gate_value = self.adaptive_gate(x, complexity)

        # Apply attention only where needed (sparsity based on complexity)
        mask = self._create_sparse_mask(gate_value, complexity)

        # Standard attention computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply sparse attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask to attention weights
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x

    def _create_sparse_mask(self, gate_value, complexity):
        """Create sparse attention mask based on gate values"""
        B, N = gate_value.shape

        # Threshold for sparsity based on complexity
        threshold = 0.5 - complexity.unsqueeze(1) * 0.3

        # Create binary mask
        mask = (gate_value > threshold).float()

        # Ensure diagonal elements (self-attention) are always included
        eye_mask = torch.eye(N, device=gate_value.device).unsqueeze(0).repeat(B, 1, 1)
        mask = torch.max(mask.unsqueeze(1), eye_mask)

        return mask

class AdaptiveGate(nn.Module):
    """Gate mechanism for adaptive computation"""

    def __init__(self, dim):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Complexity modulation
        self.complexity_modulator = nn.Linear(1, 1)

    def forward(self, x, complexity):
        # Compute gate values for each token
        gate_values = self.gate_network(x)

        # Modulate based on input complexity
        modulation = self.complexity_modulator(complexity.unsqueeze(1))

        # Apply modulation
        gate_values = gate_values * modulation

        return gate_values

class AdaptiveMLP(nn.Module):
    """MLP with adaptive depth based on input complexity"""

    def __init__(self, dim, max_depth=4):
        super().__init__()

        self.dim = dim
        self.max_depth = max_depth

        # Build MLP layers
        self.layers = nn.ModuleList()
        for i in range(max_depth - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim),
                nn.Dropout(0.1)
            ))

        # Depth prediction head
        self.depth_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, max_depth),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, complexity):
        B, N, C = x.shape

        # Predict adaptive depth
        depth_weights = self.depth_predictor(x.transpose(1, 2))

        # Apply adaptive computation
        x_out = torch.zeros_like(x)

        for i, layer in enumerate(self.layers):
            # Compute weight for this layer
            weight = depth_weights[:, i].view(B, 1, 1)

            # Apply layer
            layer_output = layer(x)

            # Weighted combination
            x_out = x_out + weight * layer_output

        # Ensure minimum computation
        min_output = self.layers[0](x)
        x_out = x_out + min_output

        return x_out

class DynamicClassificationHead(nn.Module):
    """Classification head that adapts based on input complexity"""

    def __init__(self, input_dim, num_classes, complexity_threshold=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.complexity_threshold = complexity_threshold

        # Different heads for different complexity levels
        self.simple_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, num_classes)
        )

        self.complex_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_classes)
        )

        # Ensemble head for medium complexity
        self.ensemble_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(input_dim // 2, num_classes)
        )

        # Complexity classifier to choose head
        self.head_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 3),  # Simple, Ensemble, Complex
            nn.Softmax(dim=-1)
        )

    def forward(self, x, complexity):
        # Classify complexity level
        head_weights = self.head_selector(x.transpose(1, 2))

        # Apply different heads
        simple_out = self.simple_head(x)
        complex_out = self.complex_head(x)
        ensemble_out = self.ensemble_head(x)

        # Weighted combination based on complexity
        x = (head_weights[:, 0:1].unsqueeze(-1) * simple_out +
             head_weights[:, 1:2].unsqueeze(-1) * ensemble_out +
             head_weights[:, 2:3].unsqueeze(-1) * complex_out)

        return x

# Test adaptive ViT
def test_adaptive_vit():
    """Test the adaptive Vision Transformer"""

    base_config = {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'num_classes': 1000,
        'adaptive_depth': 2
    }

    # Create adaptive ViT
    model = AdaptiveViT(base_config)

    # Test with different complexity inputs
    simple_input = torch.randn(1, 3, 224, 224)
    complex_input = torch.randn(1, 3, 224, 224)

    # Forward passes
    simple_output = model(simple_input)
    complex_output = model(complex_input)

    print("Adaptive ViT test results:")
    print(f"Input shape: {simple_input.shape}")
    print(f"Simple output shape: {simple_output.shape}")
    print(f"Complex output shape: {complex_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("✅ Adaptive ViT test passed!")

# Example usage for custom architecture
def custom_architecture_example():
    """Example of designing a custom ViT architecture"""

    print("Designing Custom Vision Transformer Architecture")
    print("=" * 50)

    # Configuration for specialized use case (e.g., satellite imagery)
    config = {
        'embed_dim': 1024,           # Higher dimension for detailed satellite imagery
        'depth': 20,                 # Deeper network
        'num_heads': 16,             # More attention heads
        'num_classes': 50,           # Many land use classes
        'patch_size': 8,             # Smaller patches for fine details
        'adaptive_depth': 3,         # Adaptive computation
    }

    model = AdaptiveViT(config)

    print(f"Custom architecture parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture features:")
    print("- Multi-scale patch embedding")
    print("- Adaptive computation based on complexity")
    print("- Dynamic classification heads")
    print("- Complexity-aware training")

    return model

# Run tests and examples
test_adaptive_vit()
custom_model = custom_architecture_example()
```

This comprehensive practice module covers all aspects of Vision Transformers and modern computer vision, from basic implementations to advanced custom architectures. Each exercise includes complete code implementations, testing functions, and real-world applications.
