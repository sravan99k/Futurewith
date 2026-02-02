# Computer Vision & Image Processing Theory

_From CNN Fundamentals to Vision Transformers_

## Table of Contents

1. [Introduction to Computer Vision](#introduction)
2. [Image Fundamentals](#image-fundamentals)
3. [Traditional Image Processing](#traditional-processing)
4. [Convolutional Neural Networks (CNNs)](#cnn-foundations)
5. [CNN Architectures](#cnn-architectures)
6. [Advanced CNN Techniques](#advanced-cnn)
7. [Object Detection & Recognition](#object-detection)
8. [Vision Transformers](#vision-transformers)
9. [3D Computer Vision](#3d-vision)
10. [Specialized Applications](#specialized-applications)
11. [Performance Optimization](#optimization)
12. [Modern Trends & Future Directions](#future-trends)

## 1. Introduction to Computer Vision {#introduction}

### What is Computer Vision?

Computer Vision is the field of artificial intelligence that enables computers to derive meaningful information from digital images, videos, and other visual inputs. It's about teaching machines to "see" and understand visual information like humans do.

### Core Objectives of Computer Vision

- **Image Understanding**: Extract meaningful information from images
- **Object Recognition**: Identify and classify objects within images
- **Scene Interpretation**: Understand spatial relationships and context
- **Visual Decision Making**: Make decisions based on visual inputs
- **Visual Generation**: Create or modify visual content

### Historical Evolution

1. **1960s-1980s**: Rule-based image processing systems
2. **1990s-2000s**: Feature-based approaches (SIFT, HOG)
3. **2010s**: Deep learning revolution with CNNs
4. **2020s**: Vision Transformers and self-supervised learning

### Key Applications

- **Autonomous Vehicles**: Navigation and obstacle detection
- **Medical Imaging**: Disease diagnosis and analysis
- **Surveillance**: Security monitoring and threat detection
- **Retail**: Product recognition and inventory management
- **Entertainment**: AR/VR and content creation
- **Agriculture**: Crop monitoring and yield prediction

## 2. Image Fundamentals {#image-fundamentals}

### Digital Image Representation

```python
# Image as a matrix of pixels
# Grayscale: Single channel [Height, Width]
# Color: Multiple channels [Height, Width, Channels]
import numpy as np
import cv2

# Load image
image = cv2.imread('image.jpg')
height, width, channels = image.shape
print(f"Image dimensions: {height}x{width}x{channels}")

# Pixel representation
# Each pixel contains intensity values
# Grayscale: 0-255 (black to white)
# RGB: [R, G, B] each 0-255
```

### Color Spaces and Representations

#### RGB Color Space

```python
# RGB: Red, Green, Blue additive model
# Used in digital displays and cameras
# Each channel: 0-255

# Split RGB channels
b, g, r = cv2.split(image)
cv2.imshow('Red Channel', r)
cv2.imshow('Green Channel', g)
cv2.imshow('Blue Channel', b)
```

#### HSV Color Space

```python
# HSV: Hue, Saturation, Value
# More intuitive for color analysis
# Hue: Color type (0-179)
# Saturation: Color purity (0-255)
# Value: Brightness (0-255)

# Convert RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
```

#### Other Color Spaces

- **LAB**: Perceptually uniform color space
- **YUV**: Used in video compression
- **Grayscale**: Single intensity channel

### Image Properties and Operations

```python
# Basic image properties
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image min/max values: {image.min()}/{image.max()}")

# Image statistics
mean_pixel = np.mean(image)
std_pixel = np.std(image)

# Image normalization
normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
```

## 3. Traditional Image Processing {#traditional-processing}

### Image Filtering and Enhancement

#### Linear Filtering

```python
# Gaussian blur for noise reduction
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Sharpening filter
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel)

# Edge detection
edges = cv2.Canny(image, 50, 150)
```

#### Morphological Operations

```python
# Dilation: Expands white regions
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(binary_image, kernel, iterations=1)

# Erosion: Shrinks white regions
eroded = cv2.erode(binary_image, kernel, iterations=1)

# Opening: Erosion followed by dilation
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing: Dilation followed by erosion
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
```

### Feature Detection and Description

#### Corner Detection

```python
# Harris Corner Detection
corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
corners = cv2.dilate(corners, None)

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray_image, 25, 0.01, 10)
```

#### Feature Descriptors

```python
# SIFT (Scale-Invariant Feature Transform)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(image, None)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
```

### Image Segmentation

#### Thresholding

```python
# Simple thresholding
_, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

# Otsu's thresholding
_, otsu_thresh = cv2.threshold(gray_image, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

#### Region-Based Segmentation

```python
# Watershed algorithm
from scipy import ndimage

# Distance transform
dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

# Watershed segmentation
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
```

## 4. Convolutional Neural Networks (CNNs) - Foundations {#cnn-foundations}

### What Makes CNNs Special for Computer Vision?

CNNs are specifically designed to process grid-like data such as images. They excel at computer vision tasks because they:

1. **Preserve Spatial Relationships**: Convolution operations maintain spatial structure
2. **Learn Hierarchical Features**: From low-level edges to high-level concepts
3. **Parameter Sharing**: Same filters applied across entire image
4. **Translation Invariance**: Can detect features regardless of position

### The CNN Architecture Philosophy

#### Analogy: The Art Detective Team

```
Step 1: Detective A looks for horizontal lines
Step 2: Detective B looks for vertical lines
Step 3: Detective C looks for curves
Step 4: Detective D combines lines to find shapes
Step 5: Detective E recognizes objects from shapes
Step 6: Detective F says "This is a cat!"
```

### Core CNN Components

#### 1. Convolutional Layers

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Convolutional operation
# Input: [batch_size, channels, height, width]
# Output: [batch_size, out_channels, new_height, new_width]
```

**Convolution Mathematics:**

```
Output(i,j) = Σ Input(i+m, j+n) * Filter(m,n)
```

#### 2. Pooling Layers

```python
class MaxPool2D(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

class AvgPool2D(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)
```

#### 3. Feature Hierarchy Learning

```python
# What CNNs Learn at Different Layers:
# Layer 1: Basic patterns (edges, corners, colors)
# Layer 2: Shapes (circles, squares, triangles)
# Layer 3: Parts (eyes, wheels, leaves)
# Layer 4: Objects (faces, cars, animals)
# Layer 5: Concepts (person, vehicle, plant)
```

### Understanding Convolution Operations

#### The Convolution Process - Scanning for Patterns

```
Image (Big):
[🌟][🌟][🌟][🌟][🌟]
[🌟][🔍][🔍][🔍][🌟]
[🌟][🔍][🔍][🔍][🌟]  ← 3x3 window scanning
[🌟][🔍][🔍][🔍][🌟]
[🌟][🌟][🌟][🌟][🌟]

As window moves:
┌─ Window looks for "X" pattern
├─ Window moves right
├─ Window finds horizontal lines
└─ Creates "feature map" of what it found
```

#### Mathematical Details

```python
# Convolution Operation Example
# Input: 4x4 image, 3x3 kernel
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Manual convolution
def conv2d(input_matrix, kernel):
    kernel_h, kernel_w = kernel.shape
    output_h = input_matrix.shape[0] - kernel_h + 1
    output_w = input_matrix.shape[1] - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Element-wise multiplication and sum
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output
```

### CNN Training and Optimization

#### Loss Functions for Computer Vision

```python
# For Classification
criterion = nn.CrossEntropyLoss()

# For Object Detection
criterion = nn.MSELoss()  # For bounding box regression

# For Segmentation
criterion = nn.CrossEntropyLoss()  # For pixel-wise classification
```

#### Optimization Techniques

```python
# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam Optimizer (commonly used)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

## 5. CNN Architectures {#cnn-architectures}

### Classic CNN Architectures

#### 1. LeNet-5 - The Grandfather (1998)

```python
class LeNet5(nn.Module):
    """
    First successful CNN for handwritten digit recognition
    Architecture: Conv → Pool → Conv → Pool → FC → FC → FC
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers (find patterns)
        self.conv1 = nn.Conv2d(1, 6, 5)    # 1 input channel, 6 filters, 5x5 window
        self.conv2 = nn.Conv2d(6, 16, 5)   # 6 previous filters, 16 new ones

        # Pooling layers (simplify)
        self.pool = nn.MaxPool2d(2, 2)     # Keep biggest number in 2x2 window

        # Fully connected layers (make decisions)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Connect to decision neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)   # 10 digit classes (0-9)

    def forward(self, x):
        # 28x28 → 24x24 (conv) → 12x12 (pool) → 8x8 (conv) → 4x4 (pool) → decision
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten for fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Test with handwritten digit
model = LeNet5()
digit_image = torch.randn(1, 1, 28, 28)  # 1 image, 1 color, 28x28 pixels
prediction = model(digit_image)
print(f"Predicted digit: {prediction.argmax().item()}")
```

#### 2. AlexNet - The Breakthrough (2012)

```python
class AlexNet(nn.Module):
    """
    Breakthrough CNN that started the deep learning revolution
    Key innovations: ReLU, Dropout, GPU training, Data Augmentation
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 96, 11, stride=4),  # 96 filters, 11x11, stride 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # Second Convolutional Block
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # Three smaller convolutional blocks
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

#### 3. VGGNet - The Deep Network (2014)

```python
class VGG16(nn.Module):
    """
    Very Deep Convolutional Networks
    Innovation: Small 3x3 filters, very deep networks (16-19 layers)
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Continue for blocks 4 and 5...
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
```

#### 4. ResNet - The Residual Revolution (2015)

```python
class ResidualBlock(nn.Module):
    """
    Residual connection: F(x) + x
    Allows training of very deep networks
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet architecture with residual blocks
    Can have 18, 34, 50, 101, or 152 layers
    """
    def __init__(self, ResidualBlock, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

#### 5. EfficientNet - The Efficient Architecture (2019)

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Used in EfficientNet for efficient scaling
    """
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()

        self.stride = stride

        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1)
            self.bn1 = nn.BatchNorm2d(expanded_channels)

        # Depthwise convolution
        self.dw_conv = nn.Conv2d(expanded_channels, expanded_channels, 3,
                                stride, 1, groups=expanded_channels)
        self.bn2 = nn.BatchNorm2d(expanded_channels)

        # Squeeze and Excitation
        self.se = SEModule(expanded_channels)

        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # Expansion
        if hasattr(self, 'expand_conv'):
            out = self.expand_conv(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = x

        # Depthwise convolution
        out = self.dw_conv(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Squeeze and Excitation
        out = self.se(out)

        # Projection
        out = self.project_conv(out)
        out = self.bn3(out)

        # Add residual
        if self.stride == 1 and x.size(1) == out.size(1):
            out += residual

        return out
```

### Architecture Comparison and Evolution

| Architecture    | Year | Key Innovation       | Parameters | Accuracy (ImageNet) |
| --------------- | ---- | -------------------- | ---------- | ------------------- |
| LeNet-5         | 1998 | First successful CNN | 60K        | 99% (MNIST)         |
| AlexNet         | 2012 | ReLU, Dropout, GPU   | 60M        | 84.7%               |
| VGG-16          | 2014 | Small filters, depth | 138M       | 92.7%               |
| ResNet-50       | 2015 | Residual connections | 25M        | 96.4%               |
| EfficientNet-B7 | 2019 | Compound scaling     | 66M        | 97.4%               |

## 6. Advanced CNN Techniques {#advanced-cnn}

### Transfer Learning

```python
import torchvision.models as models

class TransferLearningCNN(nn.Module):
    """
    Use pre-trained models as feature extractors
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Load pre-trained ResNet
        self.backbone = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Fine-tuning approach
model = models.resnet50(pretrained=True)

# Unfreeze last few layers for fine-tuning
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.avgpool.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True
```

### Data Augmentation

```python
from torchvision import transforms

# Comprehensive augmentation strategy
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Attention Mechanisms in CNNs

```python
class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Learns "what" to attend to
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine and apply sigmoid
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Learns "where" to attend to
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)
```

### Batch Normalization and Regularization

```python
class BatchNorm2D(nn.Module):
    """
    Batch Normalization implementation
    Normalizes inputs to reduce internal covariate shift
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=[0, 2, 3])
            var = x.var(dim=[0, 2, 3], unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)

        # Scale and shift
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
```

## 7. Object Detection & Recognition {#object-detection}

### Object Detection vs Classification vs Localization

| Task                      | Output                   | Example                                          |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| **Image Classification**  | Class label              | "This image contains a dog"                      |
| **Object Localization**   | Class + Bounding box     | "Dog at position (x,y) with width w, height h"   |
| **Object Detection**      | Multiple classes + boxes | "Dog at (10,20), Cat at (50,60), Car at (80,90)" |
| **Semantic Segmentation** | Pixel-wise class labels  | Every pixel labeled as dog, cat, car, etc.       |

### Traditional Object Detection

#### Sliding Window Approach

```python
def sliding_window_detection(image, window_size, stride, model):
    """
    Traditional sliding window object detection
    """
    detections = []

    for y in range(0, image.shape[0] - window_size[1], stride):
        for x in range(0, image.shape[1] - window_size[0], stride):
            # Extract window
            window = image[y:y+window_size[1], x:x+window_size[0]]

            # Resize to model input size
            window_resized = cv2.resize(window, (224, 224))

            # Predict
            prediction = model.predict(window_resized)

            if prediction[1] > 0.8:  # Confidence threshold
                detections.append({
                    'bbox': (x, y, window_size[0], window_size[1]),
                    'class': prediction[0],
                    'confidence': prediction[1]
                })

    return detections
```

#### R-CNN (Region-based CNN)

```python
class RCNN(nn.Module):
    """
    Region-based Convolutional Neural Network
    Two-stage detection: region proposal + classification
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Region proposal network
        self.rpn = RPN()

        # Region of Interest pooling
        self.roi_pool = RoIPool((7, 7), 1.0/16)

        # Classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.cls_head = nn.Linear(4096, num_classes)
        self.reg_head = nn.Linear(4096, num_classes * 4)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Region proposals
        proposals = self.rpn(features)

        # ROI pooling
        pooled_features = self.roi_pool(features, proposals)

        # Classification and regression
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        pooled_features = self.classifier(pooled_features)

        class_scores = self.cls_head(pooled_features)
        bbox_regression = self.reg_head(pooled_features)

        return class_scores, bbox_regression, proposals
```

### Modern Object Detection: YOLO (You Only Look Once)

#### YOLO Architecture

```python
class YOLOv3(nn.Module):
    """
    YOLO v3: Unified object detection with single forward pass
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # Darknet-53 backbone
        self.backbone = Darknet53()

        # Detection heads at different scales
        self.detection_head_82 = DetectionHead(512, num_classes)  # Large objects
        self.detection_head_94 = DetectionHead(256, num_classes)  # Medium objects
        self.detection_head_106 = DetectionHead(128, num_classes) # Small objects

    def forward(self, x):
        # Feature extraction
        route1, route2, route3 = self.backbone(x)

        # Detection at three scales
        detections_82 = self.detection_head_82(route1)
        detections_94 = self.detection_head_94(route2)
        detections_106 = self.detection_head_106(route3)

        return detections_82, detections_94, detections_106

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)  # x,y,w,h,obj + classes

        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.output_conv = nn.Conv2d(in_channels, self.num_outputs, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.output_conv(x)

        # Reshape for detection output
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, self.num_anchors, self.num_outputs, height, width)
        x = x.permute(0, 1, 3, 4, 2)  # [batch, anchors, height, width, outputs]

        return x
```

#### YOLO Loss Function

```python
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=80, ignore_thresh=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

    def forward(self, predictions, targets):
        """
        Compute YOLO loss
        """
        device = predictions.device
        total_loss = 0

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            pred_boxes = pred[..., :4]      # [x, y, w, h]
            pred_conf = pred[..., 4]        # Objectness score
            pred_cls = pred[..., 5:]        # Class probabilities

            # Target assignment
            assigned_boxes, assigned_conf, assigned_cls = self.assign_targets(pred, target)

            # Localization loss (MSE for box coordinates)
            box_loss = self.mse_loss(pred_boxes[assigned_boxes], target['boxes'][assigned_boxes])

            # Confidence loss (BCE for objectness)
            conf_loss = self.bce_loss(pred_conf[assigned_boxes], assigned_conf)

            # Classification loss (BCE for classes)
            cls_loss = self.bce_loss(pred_cls[assigned_boxes], assigned_cls)

            # Total loss
            total_loss += box_loss + conf_loss + cls_loss

        return total_loss

    def assign_targets(self, predictions, targets):
        """
        Assign ground truth to predictions using IoU
        """
        # Implementation of target assignment logic
        pass

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='sum')

    def bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
```

### Modern Detection Architectures

#### SSD (Single Shot Detector)

```python
class SSD(nn.Module):
    """
    Single Shot Detector
    Multi-scale feature maps for detecting objects of different sizes
    """
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes

        # Base network
        self.base_network = VGG16()

        # Extra feature layers
        self.extra_layers = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),      # Extra conv layer 1
            nn.Conv2d(256, 512, 3, 2, 1), # Extra conv layer 2
            nn.Conv2d(512, 128, 1),       # Extra conv layer 3
            nn.Conv2d(128, 256, 3, 2, 1), # Extra conv layer 4
            nn.Conv2d(256, 128, 1),       # Extra conv layer 5
            nn.Conv2d(128, 256, 3, 1, 1), # Extra conv layer 6
            nn.Conv2d(256, 128, 1),       # Extra conv layer 7
            nn.Conv2d(128, 256, 3, 1, 1), # Extra conv layer 8
        ])

        # Prediction heads
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 16, 3, padding=1),   # conv4_3
            nn.Conv2d(1024, 24, 3, padding=1),  # fc7
            nn.Conv2d(512, 24, 3, padding=1),   # conv6_2
            nn.Conv2d(256, 24, 3, padding=1),   # conv7_2
            nn.Conv2d(256, 16, 3, padding=1),   # conv8_2
            nn.Conv2d(256, 16, 3, padding=1),   # conv9_2
        ])

        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, 16, 3, padding=1),   # conv4_3
            nn.Conv2d(1024, 24, 3, padding=1),  # fc7
            nn.Conv2d(512, 24, 3, padding=1),   # conv6_2
            nn.Conv2d(256, 24, 3, padding=1),   # conv7_2
            nn.Conv2d(256, 16, 3, padding=1),   # conv8_2
            nn.Conv2d(256, 16, 3, padding=1),   # conv9_2
        ])

        # Default boxes (anchors)
        self.default_boxes = self.generate_default_boxes()

    def forward(self, x):
        # Base network forward pass
        conv4_3, fc7 = self.base_network(x)

        # Extra feature layers
        features = [conv4_3, fc7]
        conv6_2, conv7_2, conv8_2, conv9_2 = self.forward_extra_layers(fc7)
        features.extend([conv6_2, conv7_2, conv8_2, conv9_2])

        # Prediction
        loc_preds = []
        conf_preds = []

        for i, feature in enumerate(features):
            loc_pred = self.loc_layers[i](feature)
            conf_pred = self.conf_layers[i](feature)

            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            conf_preds.append(conf_pred.view(conf_pred.size(0), -1, self.num_classes))

        # Concatenate predictions
        loc_pred = torch.cat(loc_preds, 1)
        conf_pred = torch.cat(conf_preds, 1)

        return loc_pred, conf_pred

    def forward_extra_layers(self, x):
        """
        Forward pass through extra feature layers
        """
        # Implementation of extra layers
        pass

    def generate_default_boxes(self):
        """
        Generate default boxes (anchors) for each feature map
        """
        # Implementation of default box generation
        pass
```

## 8. Vision Transformers {#vision-transformers}

### Introduction to Vision Transformers

Vision Transformers (ViTs) represent a paradigm shift in computer vision, bringing the transformer architecture that revolutionized natural language processing to image analysis tasks. Unlike traditional Convolutional Neural Networks (CNNs) that rely on spatially local operations, ViTs treat images as sequences of patches and apply the self-attention mechanism globally across the entire image.

#### Why Vision Transformers Matter

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

### Vision Transformer Architecture

#### Patch Embedding and Tokenization

```python
class PatchEmbedding(nn.Module):
    """
    Convert image patches to tokens
    """
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding layer
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)

        # Learnable position embeddings
        num_patches = (224 // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)  # +1 for CLS token
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.projection(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)        # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)   # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x
```

#### Multi-Head Self-Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class Attention(nn.Module):
    """
    Vision Transformer attention block
    """
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None,
                 attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
```

#### Vision Transformer Block

```python
class Block(nn.Module):
    """
    Vision Transformer block with attention and MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    """
    MLP for Vision Transformer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
```

#### Complete Vision Transformer

```python
class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer implementation
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        # Normalization and classification head
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification token
        x = self.norm(x)
        cls_token = x[:, 0]

        return cls_token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

### Vision Transformer vs CNN Comparison

#### Advantages of ViT:

- **Global context**: Can see entire image at once
- **Better long-range dependencies**: Self-attention captures relationships across entire image
- **Flexibility**: Can handle variable sequence lengths
- **Transferability**: Pre-trained ViTs transfer well to downstream tasks
- **Less inductive bias**: More data-driven learning

#### Advantages of CNN:

- **Inductive biases**: Built-in spatial locality and translation invariance
- **Data efficiency**: Works well with smaller datasets
- **Computational efficiency**: Fewer parameters and operations
- **Interpretability**: Feature maps are more interpretable
- **Local feature learning**: Excellent for extracting local patterns

### Advanced Vision Transformer Architectures

#### 1. Swin Transformer (Shifted Windows)

```python
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with shifted window attention
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = SwinAttention(dim, window_size=self.window_size,
                                 num_heads=num_heads, qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # Store original dimensions
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        # Normalize and apply attention
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                 dims=(1, 2))
        else:
            shifted_x = x

        attn_windows = window_partition(shifted_x, self.window_size)
        attn_windows = attn_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(attn_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Merge windows
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                          dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

#### 2. DeiT (Data-efficient Image Transformers)

```python
class DistilledVisionTransformer(nn.Module):
    """
    DeiT: Data-efficient Image Transformers with distillation
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 distill=True):
        super().__init__()

        self.distill = distill

        # Standard ViT components
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                 attn_drop=attn_drop_rate, drop_path=drop_path_rate)
            for _ in range(depth)
        ])

        # Classification heads
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        if self.distill:
            self.head_dist = nn.Linear(embed_dim, num_classes)
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.distill:
            nn.init.trunc_normal_(self.dist_token, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        if self.distill:
            distill_tokens = self.dist_token.expand(x.size(0), -1, -1)
            x = torch.cat((x, distill_tokens), dim=1)

        # Position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)

        if self.distill:
            cls_token, distill_token = x[:, 0], x[:, 1]
            x = self.head(cls_token)
            distill_out = self.head_dist(distill_token)

            if self.training:
                return x, distill_out
            else:
                return (x + distill_out) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
            return x
```

### Training Strategies for ViTs

#### Data Efficiency and Augmentation

```python
class ViTTrainingStrategy:
    """
    Training strategies specific to Vision Transformers
    """

    @staticmethod
    def get_vit_transforms(img_size=224, is_train=True):
        """
        Data augmentation optimized for ViTs
        """
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                     saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(img_size * 1.0)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    @staticmethod
    def get_optimizer(model, lr=1e-3, weight_decay=0.05):
        """
        Optimizer configuration for ViTs
        """
        param_dict = {}
        for param in model.parameters():
            if param.requires_grad:
                param_dict[param] = param

        optimizer = torch.optim.AdamW(param_dict, lr=lr,
                                     weight_decay=weight_decay,
                                     betas=(0.9, 0.95))
        return optimizer

    @staticmethod
    def get_scheduler(optimizer, warmup_epochs=5, total_epochs=300):
        """
        Learning rate scheduler for ViTs
        """
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) /
                                        (total_epochs - warmup_epochs)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## 9. 3D Computer Vision {#3d-vision}

### 3D Scene Understanding

3D Computer Vision extends traditional 2D image analysis to understand the three-dimensional structure of the world. This involves:

1. **Depth Estimation**: Determining distance from camera to objects
2. **3D Reconstruction**: Building 3D models from 2D images
3. **Scene Understanding**: Interpreting spatial relationships and layout
4. **3D Object Detection**: Localizing objects in 3D space

### Depth Estimation Techniques

#### Monocular Depth Estimation

```python
class DepthEstimationCNN(nn.Module):
    """
    CNN-based monocular depth estimation
    """
    def __init__(self, input_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels*2, base_channels, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, base_channels//2, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels//2, 1, 1),  # Single depth channel
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)

        # Decode to depth map
        depth_pred = self.decoder(encoded)

        return depth_pred
```

#### Stereo Vision and Disparity

```python
class StereoMatchingCNN(nn.Module):
    """
    CNN for stereo matching and disparity estimation
    """
    def __init__(self, max_disparity=128, feature_channels=32):
        super().__init__()
        self.max_disparity = max_disparity

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Correlation layer (manual implementation)
        self.corr_layer = CorrLayer(max_disparity)

        # Disparity regression
        self.disp_regressor = nn.Sequential(
            nn.Conv2d(max_disparity, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, left_image, right_image):
        # Extract features
        left_feat = self.feature_extractor(left_image)
        right_feat = self.feature_extractor(right_image)

        # Compute correlation
        correlation = self.corr_layer(left_feat, right_feat)

        # Regress disparity
        disparity_pred = self.disp_regressor(correlation)

        return disparity_pred

class CorrLayer(nn.Module):
    """
    Correlation layer for stereo matching
    """
    def __init__(self, max_disparity):
        super().__init__()
        self.max_disparity = max_disparity

    def forward(self, left_feat, right_feat):
        batch, channels, height, width = left_feat.shape

        # Compute correlation for each disparity
        correlations = []
        for d in range(self.max_disparity):
            if d == 0:
                corr = torch.sum(left_feat * right_feat, dim=1, keepdim=True)
            else:
                right_shifted = torch.zeros_like(right_feat)
                if d < width:
                    right_shifted[:, :, :, d:] = right_feat[:, :, :, :-d]
                    corr = torch.sum(left_feat * right_shifted, dim=1, keepdim=True)
                else:
                    corr = torch.zeros(batch, 1, height, width, device=left_feat.device)
            correlations.append(corr)

        # Stack correlations
        correlation_volume = torch.cat(correlations, dim=1)

        return correlation_volume
```

### 3D Object Detection

#### 3D Bounding Box Representation

```python
import torch
import math

class BoundingBox3D:
    """
    3D Bounding Box representation
    """
    def __init__(self, center=(0, 0, 0), size=(1, 1, 1), rotation=0):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.size = torch.tensor(size, dtype=torch.float32)
        self.rotation = rotation  # yaw rotation in radians

    def to_corners(self):
        """
        Convert 3D bbox to 8 corner points
        """
        # Box dimensions
        half_length = self.size[0] / 2
        half_width = self.size[1] / 2
        half_height = self.size[2] / 2

        # 8 corners in local coordinate system
        corners = torch.tensor([
            [-half_length, -half_width, -half_height],
            [-half_length, -half_width, +half_height],
            [-half_length, +half_width, -half_height],
            [-half_length, +half_width, +half_height],
            [+half_length, -half_width, -half_height],
            [+half_length, -half_width, +half_height],
            [+half_length, +half_width, -half_height],
            [+half_length, +half_width, +half_height],
        ])

        # Apply rotation
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        rotation_matrix = torch.tensor([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])

        # Rotate and translate
        rotated_corners = torch.matmul(corners, rotation_matrix.t()) + self.center

        return rotated_corners

    def iou_3d(self, other):
        """
        Compute 3D IoU between two 3D bounding boxes
        """
        # Convert to corner representation
        corners1 = self.to_corners()
        corners2 = other.to_corners()

        # Compute intersection (simplified - exact implementation is complex)
        # This is a simplified version for demonstration
        min_corner = torch.max(torch.min(corners1, dim=0)[0], torch.min(corners2, dim=0)[0])
        max_corner = torch.min(torch.max(corners1, dim=0)[0], torch.max(corners2, dim=0)[0])

        # Check if boxes intersect
        if torch.all(min_corner <= max_corner):
            intersection_volume = torch.prod(max_corner - min_corner)
        else:
            intersection_volume = torch.tensor(0.0)

        # Compute union
        volume1 = torch.prod(self.size)
        volume2 = torch.prod(other.size)
        union_volume = volume1 + volume2 - intersection_volume

        # IoU
        iou = intersection_volume / (union_volume + 1e-6)
        return iou.item()
```

#### PointNet for 3D Object Recognition

```python
class PointNet(nn.Module):
    """
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    """
    def __init__(self, num_classes=40):
        super().__init__()

        # Shared MLP for point features
        self.feat_conv1 = nn.Conv1d(3, 64, 1)
        self.feat_conv2 = nn.Conv1d(64, 128, 1)
        self.feat_conv3 = nn.Conv1d(128, 1024, 1)
        self.feat_bn1 = nn.BatchNorm1d(64)
        self.feat_bn2 = nn.BatchNorm1d(128)
        self.feat_bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling
        self.max_pool = nn.MaxPool1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Softmax for classification
        self.classifier = nn.Sequential(self.fc1, self.relu, self.dropout,
                                      self.fc2, self.relu, self.dropout,
                                      self.fc3)

    def forward(self, point_cloud):
        """
        point_cloud: [batch_size, num_points, 3]
        """
        # Transpose for conv1d: [batch_size, 3, num_points]
        x = point_cloud.transpose(2, 1)

        # Feature extraction
        x = self.relu(self.feat_bn1(self.feat_conv1(x)))
        x = self.relu(self.feat_bn2(self.feat_conv2(x)))
        x = self.relu(self.feat_bn3(self.feat_conv3(x)))

        # Global feature (max pooling)
        global_feat = self.max_pool(x)  # [batch_size, 1024, 1]
        global_feat = global_feat.squeeze(-1)  # [batch_size, 1024]

        # Classification
        logits = self.classifier(global_feat)

        return logits
```

#### 3D Object Detection with Voxels

```python
class VoxelNet(nn.Module):
    """
    VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
    """
    def __init__(self, num_classes=3, max_points=5, max_voxels=20000):
        super().__init__()
        self.num_classes = num_classes
        self.max_points = max_points
        self.max_voxels = max_voxels

        # Point-wise MLP (VFE - Voxel Feature Encoding)
        self.vfe_layers = nn.ModuleList([
            nn.Linear(7, 32),  # Input: [x, y, z, intensity, voxel coords]
            nn.Linear(32, 64),
            nn.Linear(64, 128),
        ])

        # 3D CNN for spatio-temporal feature extraction
        self.conv3d_layers = nn.ModuleList([
            # Input: [128, 10, 400, 1400] -> Output: [32, 5, 200, 700]
            nn.Conv3d(128, 32, 3, padding=1),
            nn.Conv3d(32, 16, 3, padding=1),
        ])

        # Region Proposal Network (RPN)
        self.rpn = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=2, stride=2),
            nn.ReLU(inplace=True),
            # ... more layers
        )

        # Detection heads
        self.bbox_head = nn.Conv2d(128, 7 * 2, 1)  # 7 anchors x (x, y) offset
        self.class_head = nn.Conv2d(128, 7 * num_classes, 1)  # 7 anchors x classes
        self.conf_head = nn.Conv2d(128, 7, 1)  # 7 anchors confidence

    def forward(self, point_clouds, voxel_coords, voxel_features, voxel_num_points):
        """
        Forward pass through VoxelNet
        """
        batch_size = point_clouds.size(0)

        # VFE (Voxel Feature Encoding)
        voxel_features = self.vfe_forward(voxel_features)

        # Convert to 3D tensor
        dense_voxel = self.sparse_to_dense(voxel_features, voxel_coords,
                                          batch_size, [128, 10, 400, 1400])

        # 3D CNN
        x = dense_voxel
        for conv3d in self.conv3d_layers:
            x = conv3d(x)

        # Remove Z dimension
        x = x.squeeze(1)  # [batch_size, 16, 200, 700]

        # RPN
        x = self.rpn(x)

        # Detection heads
        bbox_pred = self.bbox_head(x)
        cls_pred = self.class_head(x)
        conf_pred = self.conf_head(x)

        return bbox_pred, cls_pred, conf_pred

    def vfe_forward(self, voxel_features):
        """
        Forward pass through VFE layers
        """
        for vfe_layer in self.vfe_layers:
            voxel_features = vfe_layer(voxel_features)
            voxel_features = F.relu(voxel_features, inplace=True)

        # Max pooling across points in each voxel
        voxel_features = torch.max(voxel_features, dim=1)[0]

        return voxel_features

    def sparse_to_dense(self, sparse_features, indices, batch_size, output_shape):
        """
        Convert sparse voxel features to dense 3D tensor
        """
        dense = torch.zeros(batch_size, *output_shape[1:],
                           dtype=sparse_features.dtype,
                           device=sparse_features.device)

        for i in range(batch_size):
            batch_indices = (indices[:, 0] == i)
            batch_coords = indices[batch_indices, 1:].long()
            batch_features = sparse_features[batch_indices]

            dense[i, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]] = batch_features

        return dense
```

### Neural Radiance Fields (NeRF)

#### NeRF Architecture

```python
class NeRF(nn.Module):
    """
    Neural Radiance Fields: Representing scenes as neural functions
    """
    def __init__(self, pos_encoding_dims=10, dir_encoding_dims=4):
        super().__init__()

        # Input dimensions
        self.pos_encoding_dims = pos_encoding_dims
        self.dir_encoding_dims = dir_encoding_dims

        # Positional encoding function
        self.pos_encoding_fn = self.positional_encoding
        self.dir_encoding_fn = self.positional_encoding

        # Network architecture
        self.layers = nn.ModuleList([
            # Input: position encoding (3 * 2 * 10 = 60)
            nn.Linear(3 * 2 * pos_encoding_dims, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),

            # Skip connection
            nn.Linear(256 + 3 * 2 * pos_encoding_dims, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        ])

        # Output heads
        self.density_head = nn.Linear(256, 1)
        self.color_head = nn.Linear(256, 3)

        # Direction encoding (view direction)
        self.dir_layers = nn.ModuleList([
            nn.Linear(256 + 3 * 2 * dir_encoding_dims, 128),
            nn.Linear(128, 3),
        ])

    def positional_encoding(self, x, num_encoding_dims):
        """
        Sinusoidal positional encoding
        """
        encodings = []
        for i in range(num_encoding_dims):
            frequency = 2.0 ** i
            encodings.append(torch.sin(x * frequency))
            encodings.append(torch.cos(x * frequency))

        return torch.cat(encodings, dim=-1)

    def forward(self, positions, directions):
        """
        Forward pass through NeRF
        """
        # Encode positions and directions
        pos_encoded = self.pos_encoding_fn(positions, self.pos_encoding_dims)
        dir_encoded = self.dir_encoding_fn(directions, self.dir_encoding_dims)

        # Pass through main network
        x = pos_encoded
        for i, layer in enumerate(self.layers):
            if i == 5:  # Skip connection after 6 layers
                x = torch.cat([x, pos_encoded], dim=-1)
            x = layer(x)
            x = F.relu(x, inplace=True)

        # Density prediction
        density = self.density_head(x)

        # Color prediction
        # Combine feature with direction encoding
        color_feature = torch.cat([x, dir_encoded], dim=-1)
        for dir_layer in self.dir_layers:
            color_feature = dir_layer(color_feature)
            if dir_layer != self.dir_layers[-1]:
                color_feature = F.relu(color_feature, inplace=True)

        color = torch.sigmoid(color_feature)

        return density, color

    def render_rays(self, ray_origins, ray_directions, near=0.0, far=1.0,
                   num_samples=64):
        """
        Render rays using volume rendering equation
        """
        batch_size = ray_origins.shape[0]

        # Sample points along rays
        t_vals = torch.linspace(near, far, num_samples).to(ray_origins.device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_samples]

        # Add random jitter during training
        if self.training:
            shape = [batch_size, num_samples]
            noise = torch.rand(shape).to(ray_origins.device)
            t_vals = t_vals + (far - near) / num_samples * noise

        # Compute 3D positions along rays
        points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * t_vals.unsqueeze(2)

        # Compute directions (normalized view directions)
        directions = ray_directions.unsqueeze(1).expand_as(points)

        # Predict density and color
        density, color = self.forward(points, directions)

        # Volume rendering
        return self.volume_render(density, color, t_vals, ray_directions)

    def volume_render(self, density, color, t_vals, ray_days):
        """
        Volume rendering using quadrature
        """
        # Compute alpha values (transmittance)
        delta = t_vals[..., 1:] - t_vals[..., :-1]
        delta = torch.cat([delta, torch.ones_like(delta[..., -1:], device=delta.device)], dim=-1)

        alpha = 1 - torch.exp(-density.squeeze(-1) * delta)

        # Compute weights
        transmittance = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        weights = alpha * transmittance

        # Rendered color
        rendered_color = torch.sum(weights.unsqueeze(-1) * color, dim=-2)

        # Rendered depth
        rendered_depth = torch.sum(weights * t_vals, dim=-1)

        return rendered_color, rendered_depth
```

## 10. Specialized Applications {#specialized-applications}

### Medical Image Analysis

#### Medical Image Classification

```python
class MedicalImageClassifier(nn.Module):
    """
    Specialized CNN for medical image classification
    Features: Data augmentation for medical images, attention mechanisms
    """
    def __init__(self, num_classes, input_channels=1):
        super().__init__()

        # ResNet-based backbone with modifications for medical imaging
        from torchvision.models import resnet50
        self.backbone = resnet50()

        # Modify first layer for medical images (1 channel instead of 3)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)

        # Attention mechanism for focusing on important regions
        self.attention = nn.MultiheadAttention(2048, num_heads=8)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)

        # Apply attention across spatial dimensions
        b, c, h, w = features.shape
        features_flat = features.view(b, c, h * w).transpose(1, 2)

        attended_features, _ = self.attention(features_flat, features_flat, features_flat)
        attended_features = attended_features.transpose(1, 2).view(b, c, h, w)

        # Global average pooling
        attended_features = torch.mean(attended_features, dim=(2, 3))

        # Classification
        output = self.classifier(attended_features)

        return output

# Medical image preprocessing
def preprocess_medical_image(image_path, target_size=(224, 224)):
    """
    Preprocess medical images for analysis
    """
    import SimpleITK as sitk

    # Load image
    image = sitk.ReadImage(image_path)

    # Resample to isotropic spacing
    original_spacing = image.GetSpacing()
    new_spacing = [min(original_spacing)] * 3
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc))
                for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkBSpline)
    image = resample.Execute(image)

    # Convert to numpy
    image_array = sitk.GetArrayFromImage(image)

    # Normalize
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    # Resize
    image_resized = cv2.resize(image_array, target_size)

    return image_resized
```

#### Medical Segmentation with U-Net

```python
class UNetMedical(nn.Module):
    """
    U-Net architecture optimized for medical image segmentation
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck with attention
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)
        self.attention = AttentionGate(features[-1] * 2, features[-1] // 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.double_conv(feature * 2, feature))

        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.attention(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
```

### Face Recognition Systems

#### FaceNet Architecture

```python
class FaceNet(nn.Module):
    """
    FaceNet: Deep learning for face recognition
    Uses triplet loss for learning face embeddings
    """
    def __init__(self, embedding_size=128):
        super().__init__()

        # Inception-style backbone
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Inception blocks
        self.inception1 = InceptionBlock(64, [64, 96, 128, 16, 32, 32])
        self.inception2 = InceptionBlock(256, [128, 128, 192, 32, 96, 64])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # More inception blocks
        self.inception3 = InceptionBlock(480, [192, 96, 208, 16, 48, 64])
        self.inception4 = InceptionBlock(512, [160, 112, 224, 24, 64, 64])
        self.inception5 = InceptionBlock(512, [128, 128, 256, 24, 64, 64])
        self.inception6 = InceptionBlock(512, [112, 144, 288, 32, 64, 64])
        self.inception7 = InceptionBlock(528, [256, 160, 320, 32, 128, 128])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer
        self.embedding = nn.Linear(832, embedding_size)

        # L2 normalization
        self.l2_norm = nn.Lambda(lambda x: torch.nn.functional.normalize(x, p=2, dim=1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)

        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        embeddings = self.embedding(x)
        embeddings = self.l2_norm(embeddings)

        return embeddings

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        out_channels: [1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj]
        """
        super().__init__()
        c1, c3r, c3, c5r, c5, proj = out_channels

        self.branch1 = nn.Conv2d(in_channels, c1, 1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3r, 1),
            nn.Conv2d(c3r, c3, 3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5r, 1),
            nn.Conv2d(c5r, c5, 5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, proj, 1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)

class TripletLoss(nn.Module):
    """
    Triplet loss for face recognition
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)

        # Triplet loss
        loss = torch.nn.functional.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()
```

### Real-time Face Detection and Recognition

```python
class FaceRecognitionSystem:
    """
    Complete face recognition system
    """
    def __init__(self, face_detector_path, face_encoder_path):
        import dlib
        import face_recognition

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_encoder = face_recognition.FaceEncoder()

        # Load pre-trained models
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        self.face_encoder = dlib.face_recognition_model_v1(face_encoder_path)

        # Known faces database
        self.known_faces = {}

    def detect_faces(self, image):
        """
        Detect faces in image
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.face_detector(rgb_image, 1)

        face_locations = []
        for face in faces:
            face_locations.append((face.rect.top(), face.rect.right(),
                                 face.rect.bottom(), face.rect.left()))

        return face_locations

    def encode_faces(self, image, face_locations):
        """
        Generate face encodings
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        return face_encodings

    def recognize_faces(self, face_encodings, tolerance=0.6):
        """
        Recognize faces against database
        """
        matches = []

        for encoding in face_encodings:
            min_distance = float('inf')
            best_match = None

            for name, known_encoding in self.known_faces.items():
                # Compute distance to known faces
                distance = face_recognition.face_distance([known_encoding], encoding)[0]

                if distance < tolerance and distance < min_distance:
                    min_distance = distance
                    best_match = name

            matches.append((best_match, min_distance))

        return matches

    def add_known_face(self, name, face_encoding):
        """
        Add new face to database
        """
        self.known_faces[name] = face_encoding

    def process_video(self, video_source=0):
        """
        Real-time face recognition from video
        """
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            face_locations = self.detect_faces(frame)

            # Encode faces
            face_encodings = self.encode_faces(frame, face_locations)

            # Recognize faces
            matches = self.recognize_faces(face_encodings)

            # Draw results
            for (top, right, bottom, left), (name, distance) in zip(face_locations, matches):
                # Draw bounding box
                color = (0, 255, 0) if name else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw name
                text = f"{name} ({distance:.2f})" if name else "Unknown"
                cv2.putText(frame, text, (left, top-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
```

### Autonomous Vehicle Computer Vision

#### Lane Detection System

```python
class LaneDetector(nn.Module):
    """
    Deep learning-based lane detection for autonomous vehicles
    """
    def __init__(self, input_shape=(720, 1280, 3)):
        super().__init__()

        # Encoder backbone (ResNet)
        from torchvision.models import resnet34
        backbone = resnet34(pretrained=True)

        # Remove final layers for feature extraction
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # Decoder for lane segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),  # 128x128 -> 256x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),  # 256x256 -> 512x512
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2),   # 512x512 -> 1024x1024
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2),    # 1024x1024 -> 2048x2048
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Lane detection heads
        self.lane_seg_head = nn.Conv2d(32, 2, 1)  # Binary segmentation (lane vs background)
        self.lane_pts_head = nn.Conv2d(32, 18, 1)  # Lane points prediction

    def forward(self, x):
        # Feature extraction
        features = self.encoder(x)

        # Decode to full resolution
        decoded = self.decoder(features)

        # Lane detection predictions
        lane_seg = self.lane_seg_head(decoded)
        lane_pts = self.lane_pts_head(decoded)

        return lane_seg, lane_pts

def lane_detection_pipeline(image):
    """
    Complete lane detection pipeline
    """
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=200)

    # Separate left and right lane lines
    left_lane = []
    right_lane = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0

        if slope < 0:  # Left lane (negative slope)
            left_lane.append(line[0])
        else:  # Right lane (positive slope)
            right_lane.append(line[0])

    # Calculate lane boundaries
    left_x = [point[0] for point in left_lane]
    right_x = [point[2] for point in right_lane]

    left_boundary = np.mean(left_x) if left_x else image.shape[1] * 0.3
    right_boundary = np.mean(right_x) if right_x else image.shape[1] * 0.7

    return left_boundary, right_boundary, edges, lines
```

#### Traffic Sign Recognition

```python
class TrafficSignDetector(nn.Module):
    """
    Real-time traffic sign detection and classification
    """
    def __init__(self, num_classes=43):  # German traffic sign dataset has 43 classes
        super().__init__()

        # YOLO-based detector for traffic signs
        self.backbone = Darknet53()

        # Detection heads at different scales
        self.detect_head_1 = DetectionHead(512, num_classes, num_anchors=3)
        self.detect_head_2 = DetectionHead(256, num_classes, num_anchors=3)
        self.detect_head_3 = DetectionHead(128, num_classes, num_anchors=3)

    def forward(self, x):
        # Feature extraction
        route1, route2, route3 = self.backbone(x)

        # Detection at three scales
        detections_1 = self.detect_head_1(route1)
        detections_2 = self.detect_head_2(route2)
        detections_3 = self.detect_head_3(route3)

        return detections_1, detections_2, detections_3

# Traffic sign classification CNN
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## 11. Performance Optimization {#optimization}

### Model Optimization Techniques

#### Model Quantization

```python
import torch.quantization as quantization

class QuantizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Post-training quantization
def quantize_model(model):
    """
    Apply post-training quantization to reduce model size and improve inference speed
    """
    # Set quantization configuration
    model.qconfig = quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    quantized_model = quantization.prepare(model, inplace=False)

    # Calibrate with sample data (optional)
    sample_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        quantized_model(sample_input)

    # Convert to quantized model
    quantized_model = quantization.convert(quantized_model, inplace=False)

    return quantized_model

# Dynamic quantization
def dynamic_quantize_model(model):
    """
    Apply dynamic quantization (weights quantized, activations dynamic)
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model
```

#### Model Pruning

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.2):
    """
    Apply structured pruning to reduce model parameters
    """
    # Global unstructured pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # Prune weights
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Prune biases if they exist
            if hasattr(module, 'bias') and module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)

    return model

def structured_prune_channels(model, pruning_ratio=0.2):
    """
    Apply structured channel pruning
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            num_channels = module.weight.size(0)
            prune_channels = int(num_channels * pruning_ratio)

            # Remove channels with smallest L1 norm
            weights = module.weight.data
            channel_norms = torch.norm(weights.view(weights.size(0), -1), p=1, dim=1)
            _, keep_indices = torch.topk(channel_norms, num_channels - prune_channels, largest=True)

            # Keep selected channels
            module.weight.data = weights[keep_indices]
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]

            module.out_channels = num_channels - prune_channels

    return model
```

#### Knowledge Distillation

```python
class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining soft targets and hard targets
    """
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Soft targets loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student_logits = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student_logits, soft_targets) * (self.temperature ** 2)

        # Hard targets loss
        hard_loss = self.ce_loss(student_logits, targets)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

def distill_model(student_model, teacher_model, train_loader, epochs=100):
    """
    Train student model using knowledge distillation from teacher model
    """
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    criterion = KnowledgeDistillationLoss()

    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass through both models
            student_logits = student_model(data)
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            # Compute distillation loss
            loss = criterion(student_logits, teacher_logits, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### Hardware Acceleration

#### GPU Optimization

```python
def optimize_for_gpu(model):
    """
    Optimize model for GPU inference
    """
    # Enable mixed precision
    model = model.half()  # Convert to FP16

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    return model

class OptimizedInference:
    """
    Optimized inference pipeline for computer vision models
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # Compile model for optimized execution
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')

        # Warm up
        self.warm_up()

    def warm_up(self):
        """
        Warm up GPU to stabilize performance
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

    def batch_inference(self, images, batch_size=32):
        """
        Perform batched inference for efficiency
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                batch_results = self.model(batch_tensor)

            results.extend(batch_results.cpu().numpy())

        return results
```

#### ONNX Export and Runtime

```python
def export_to_onnx(model, input_shape, output_path):
    """
    Export PyTorch model to ONNX format for optimized inference
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(model, dummy_input, output_path,
                     export_params=True,
                     opset_version=11,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={
                         'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}
                     })

    print(f"Model exported to {output_path}")

def run_onnx_inference(onnx_path, input_data):
    """
    Run inference using ONNX Runtime
    """
    import onnxruntime as ort

    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    outputs = session.run([output_name], {input_name: input_data})

    return outputs[0]
```

## 12. Modern Trends & Future Directions {#future-trends}

### Self-Supervised Learning in Computer Vision

#### Contrastive Learning (SimCLR)

```python
class SimCLR(nn.Module):
    """
    Simple Contrastive Learning of Visual Representations
    """
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()

        self.encoder = base_encoder

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        # Encode images
        features = self.encoder(x)

        # Project to lower dimension
        projections = self.projector(features)

        return projections

class SimCLR Loss:
    """
    NT-Xent loss for contrastive learning
    """
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def __call__(self, projections):
        batch_size = projections.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.t()) / self.temperature

        # Create labels for positive pairs (i, i+batch_size)
        labels = torch.arange(batch_size).to(projections.device)
        labels = torch.cat([labels + batch_size, labels])

        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

        return loss
```

#### Masked Image Modeling (MAE)

```python
class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoders for Vision Representation Learning
    """
    def __init__(self, patch_size=16, img_size=224, embed_dim=768,
                 encoder_depth=12, decoder_depth=8, masking_ratio=0.75):
        super().__init__()
        self.masking_ratio = masking_ratio

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, 12) for _ in range(encoder_depth)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, 12) for _ in range(decoder_depth)
        ])

        # Decoder projection
        self.decoder_proj = nn.Linear(embed_dim, embed_dim)

        # Token prediction
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * 3)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.patch_embed.proj.weight, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask patches
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # Patch embedding
        x = self.patch_embed(imgs)

        # Add positional embedding
        x = x + self.patch_embed.pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(x, self.masking_ratio)

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)

        # Project to decoder dimension
        x = self.decoder_proj(x)

        # Add positional embedding for decoder
        decoder_pos_embed = self.patch_embed.pos_embed[:, :x.size(1), :]
        x = x + decoder_pos_embed

        # Decoder
        for block in self.decoder_blocks:
            x = block(x)

        # Prediction
        pred = self.decoder_pred(x)

        return pred, mask, ids_restore
```

### Multimodal Computer Vision

#### Vision-Language Models

```python
class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training
    """
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Projection layers
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        # Encode images and texts
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        # Project to common embedding space
        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

    def compute_loss(self, image_features, text_features):
        """
        Compute contrastive loss
        """
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Compute loss
        labels = torch.arange(len(image_features), device=image_features.device)
        loss = (F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)) / 2

        return loss
```

#### Vision-and-Language Navigation

```python
class VLN(nn.Module):
    """
    Vision-and-Language Navigation model
    """
    def __init__(self, vision_dim, language_dim, hidden_dim):
        super().__init__()

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Language encoder
        self.language_encoder = nn.LSTM(language_dim, hidden_dim,
                                       batch_first=True, bidirectional=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8)

        # Navigation policy
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)  # 6 possible actions
        )

    def forward(self, vision_obs, language_seq, hidden_state=None):
        # Encode vision
        vision_features = self.vision_encoder(vision_obs)

        # Encode language
        language_output, hidden_state = self.language_encoder(language_seq, hidden_state)
        language_features = language_output[:, -1, :]  # Use last hidden state

        # Cross-modal attention
        # Vision features as query, language features as key/value
        attended_vision, _ = self.attention(vision_features.unsqueeze(0),
                                           language_features.unsqueeze(0),
                                           language_features.unsqueeze(0))
        attended_vision = attended_vision.squeeze(0)

        # Combine features
        combined_features = torch.cat([attended_vision, language_features], dim=1)

        # Predict action
        action_logits = self.policy(combined_features)

        return action_logits, hidden_state
```

### Federated Learning for Computer Vision

#### Federated Average Algorithm

```python
class FederatedClient(nn.Module):
    """
    Federated learning client for computer vision
    """
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def local_training(self, local_epochs, train_loader):
        """
        Train model locally for specified epochs
        """
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

    def get_model_parameters(self):
        """
        Get current model parameters
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, global_parameters):
        """
        Set model parameters from global model
        """
        for name, param in self.model.named_parameters():
            param.data.copy_(global_parameters[name])

def federated_averaging(clients, global_parameters, client_weights):
    """
    Implement Federated Averaging (FedAvg) algorithm
    """
    # Initialize global model parameters
    global_params = {name: torch.zeros_like(param) for name, param in global_parameters.items()}

    # Accumulate parameters from clients
    for client, weight in zip(clients, client_weights):
        client_params = client.get_model_parameters()
        for name, param in global_params.items():
            global_params[name] += weight * client_params[name]

    # Update global model
    for name, param in global_parameters.items():
        param.data.copy_(global_params[name])

    # Distribute global parameters to clients
    for client in clients:
        client.set_model_parameters(global_parameters)

    return global_parameters

class FederatedTrainer:
    """
    Complete federated learning system for computer vision
    """
    def __init__(self, model_fn, client_data_loaders, num_rounds=100,
                 local_epochs=5, fraction_fit=1.0):
        self.model_fn = model_fn
        self.client_data_loaders = client_data_loaders
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.fraction_fit = fraction_fit

        # Initialize global model
        self.global_model = model_fn()

        # Initialize clients
        self.clients = []
        for data_loader in client_data_loaders:
            model = model_fn()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            client = FederatedClient(model, optimizer)
            self.clients.append(client)

    def train_round(self, round_num):
        """
        Execute one round of federated training
        """
        print(f"Starting training round {round_num + 1}/{self.num_rounds}")

        # Sample clients
        num_clients = int(self.fraction_fit * len(self.clients))
        selected_clients = random.sample(self.clients, num_clients)

        # Get client weights (based on dataset size)
        client_weights = [len(client_loader.dataset) / sum(len(loader.dataset)
                        for loader in self.client_data_loaders)
                        for client_loader in self.client_data_loaders
                        for client in selected_clients if client_loader == client._train_loader]

        # Set global parameters for all clients
        global_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_model_parameters(global_params)

        # Local training on each client
        for client in selected_clients:
            client.local_training(self.local_epochs, client._train_loader)

        # Aggregate parameters
        federated_averaging(selected_clients, global_params, client_weights)

        # Update global model
        self.global_model.load_state_dict(global_params)

        return self.global_model

    def run_federated_training(self):
        """
        Run complete federated training process
        """
        for round_num in range(self.num_rounds):
            self.train_round(round_num)

            # Evaluate global model periodically
            if (round_num + 1) % 10 == 0:
                accuracy = self.evaluate_global_model()
                print(f"Round {round_num + 1}, Global accuracy: {accuracy:.4f}")

    def evaluate_global_model(self):
        """
        Evaluate global model on test set
        """
        self.global_model.eval()
        correct = 0
        total = 0

        # Use first client's test loader for evaluation
        test_loader = self.client_data_loaders[0]

        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return accuracy
```

### Explainable AI in Computer Vision

#### Grad-CAM Visualization

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for given input
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output[0, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension

        # Compute weights
        weights = torch.mean(gradients, dim=(1, 2))

        # Compute CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def visualize_cam(model, image, target_class=None, layer_name='features'):
    """
    Complete CAM visualization pipeline
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    # Generate CAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor, target_class)
    grad_cam.remove_hooks()

    # Resize CAM to original image size
    original_size = image.size[::-1]  # (width, height)
    cam_resized = cv2.resize(cam, original_size)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    return overlay, cam
```

### Edge Computing and Real-time Processing

#### MobileNetV3 for Edge Deployment

```python
class MobileNetV3(nn.Module):
    """
    MobileNetV3: Searching for MobileNetV3
    Optimized for mobile and edge devices
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.Hardswish(inplace=True)

        # Inverted residual blocks
        self.blocks = nn.ModuleList([
            InvertedResidual(16, 16, 1, 16, 1),          # stride=1
            InvertedResidual(16, 24, 2, 72, 1),          # stride=2
            InvertedResidual(24, 24, 1, 88, 1),          # stride=1
            InvertedResidual(24, 40, 2, 96, 1),          # stride=2
            InvertedResidual(40, 40, 1, 240, 6),         # stride=1
            InvertedResidual(40, 40, 1, 240, 6),         # stride=1
            InvertedResidual(40, 40, 1, 240, 6),         # stride=1
            InvertedResidual(40, 48, 1, 120, 6),         # stride=1
            InvertedResidual(48, 48, 1, 144, 6),         # stride=1
            InvertedResidual(48, 96, 2, 288, 6),         # stride=2
            InvertedResidual(96, 96, 1, 576, 6),         # stride=1
            InvertedResidual(96, 96, 1, 576, 6),         # stride=1
            InvertedResidual(96, 96, 1, 576, 6),         # stride=1
            InvertedResidual(96, 192, 2, 960, 6),        # stride=2
            InvertedResidual(192, 192, 1, 960, 6),       # stride=1
            InvertedResidual(192, 192, 1, 960, 6),       # stride=1
            InvertedResidual(192, 192, 1, 960, 6),       # stride=1
        ])

        # Last conv layer
        self.conv2 = nn.Conv2d(192, 960, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.act2 = nn.Hardswish(inplace=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))

        for block in self.blocks:
            x = block(x)

        x = self.act2(self.bn2(self.conv2(x)))
        x = self.classifier(x)

        return x

class InvertedResidual(nn.Module):
    """
    Inverted Residual Block for MobileNetV3
    """
    def __init__(self, inp, oup, stride, expand_ratio, out_dim):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        if expand_ratio == 1:
            # When expand_ratio == 1, skip the expansion conv
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # expansion convolution
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

### Emerging Applications and Future Directions

#### Augmented Reality Computer Vision

```python
class ARComputerVision:
    """
    Computer vision system for augmented reality applications
    """
    def __init__(self):
        # Initialize face detection and tracking
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize feature detector for object tracking
        self.feature_detector = cv2.ORB_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 3D pose estimation
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))

    def detect_face_landmarks(self, image):
        """
        Detect facial landmarks for AR face tracking
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        landmarks = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Use MediaPipe for landmark detection (if available)
            # This is a simplified version - actual implementation would use MediaPipe
            landmarks.append((x, y, w, h))

        return landmarks

    def estimate_pose(self, image_points, object_points):
        """
        Estimate 3D pose of object
        """
        # Solve PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, self.camera_matrix, self.dist_coeffs
        )

        if success:
            # Convert rotation vector to matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Create transformation matrix
            pose_matrix = np.hstack((rotation_matrix, translation_vector))

            return pose_matrix
        else:
            return None

    def render_ar_object(self, image, pose_matrix, object_3d):
        """
        Render 3D object in AR
        """
        # Project 3D points to 2D
        points_2d, _ = cv2.projectPoints(object_3d,
                                        pose_matrix[:3, :3],
                                        pose_matrix[:, 3],
                                        self.camera_matrix,
                                        self.dist_coeffs)

        # Render object (simplified - would use OpenGL or similar)
        for point in points_2d:
            cv2.circle(image, tuple(point[0].astype(int)), 3, (0, 255, 0), -1)

        return image

def ar_face_filter(image, filter_image):
    """
    Apply AR face filter
    """
    ar_cv = ARComputerVision()

    # Detect faces
    faces = ar_cv.detect_face_landmarks(image)

    result_image = image.copy()

    for (x, y, w, h) in faces:
        # Resize filter to fit face
        filter_resized = cv2.resize(filter_image, (w, h))

        # Apply filter with blending
        mask = np.ones((h, w, 3), dtype=np.float32)

        # Blend filter with original image
        face_region = result_image[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 0.3, filter_resized, 0.7, 0)
        result_image[y:y+h, x:x+w] = blended

    return result_image
```

#### Quantum Computer Vision (Future Direction)

```python
class QuantumCNN(nn.Module):
    """
    Quantum Convolutional Neural Network (conceptual)
    Future application of quantum computing to computer vision
    """
    def __init__(self, num_qubits=4):
        super().__init__()
        self.num_qubits = num_qubits

        # Quantum circuit components
        self.quantum_circuit = self.create_quantum_circuit()

        # Classical post-processing
        self.classifier = nn.Linear(2**num_qubits, 10)

    def create_quantum_circuit(self):
        """
        Create quantum circuit for image processing
        Note: This is a conceptual implementation.
        Actual quantum implementation would require quantum computing framework.
        """
        # Placeholder for quantum circuit creation
        # In real implementation, would use frameworks like Qiskit, PennyLane
        pass

    def quantum_convolution(self, pixel_values):
        """
        Perform convolution using quantum operations
        """
        # Placeholder for quantum convolution
        # Would involve quantum gates and entanglement
        pass

    def forward(self, x):
        batch_size = x.size(0)

        # Flatten for quantum processing
        x = x.view(batch_size, -1)

        # Quantum convolution
        quantum_features = self.quantum_convolution(x)

        # Classical classification
        output = self.classifier(quantum_features)

        return output
```

### Computer Vision Ethics and Bias

#### Bias Detection and Mitigation

```python
class BiasDetector:
    """
    Detect and mitigate bias in computer vision models
    """
    def __init__(self, model, test_datasets):
        self.model = model
        self.test_datasets = test_datasets

    def evaluate_demographic_parity(self, predictions, demographics):
        """
        Evaluate demographic parity across groups
        """
        results = {}

        for group in np.unique(demographics):
            group_mask = demographics == group
            group_predictions = predictions[group_mask]

            # Compute positive prediction rate
            positive_rate = np.mean(group_predictions)
            results[group] = positive_rate

        return results

    def evaluate_equalized_odds(self, predictions, labels, demographics):
        """
        Evaluate equalized odds (true positive and false positive rates)
        """
        results = {}

        for group in np.unique(demographics):
            group_mask = demographics == group
            group_predictions = predictions[group_mask]
            group_labels = labels[group_mask]

            # True Positive Rate
            tp = np.sum((group_predictions == 1) & (group_labels == 1))
            actual_positive = np.sum(group_labels == 1)
            tpr = tp / actual_positive if actual_positive > 0 else 0

            # False Positive Rate
            fp = np.sum((group_predictions == 1) & (group_labels == 0))
            actual_negative = np.sum(group_labels == 0)
            fpr = fp / actual_negative if actual_negative > 0 else 0

            results[group] = {'TPR': tpr, 'FPR': fpr}

        return results

    def mitigation_strategies(self, model, training_data, sensitive_attributes):
        """
        Apply bias mitigation strategies
        """
        # 1. Pre-processing: Data augmentation for underrepresented groups
        augmented_data = self.augment_underrepresented_groups(training_data, sensitive_attributes)

        # 2. In-processing: Fairness constraints during training
        fair_model = self.train_with_fairness_constraints(model, augmented_data)

        # 3. Post-processing: Adjust prediction thresholds
        calibrated_model = self.post_process_calibration(fair_model)

        return calibrated_model

    def augment_underrepresented_groups(self, data, sensitive_attrs):
        """
        Augment data for underrepresented demographic groups
        """
        # Implementation would balance dataset across groups
        pass

    def train_with_fairness_constraints(self, model, data):
        """
        Train model with fairness constraints
        """
        # Implementation would include fairness regularization terms
        pass

    def post_process_calibration(self, model):
        """
        Calibrate model predictions to improve fairness
        """
        # Implementation would adjust decision thresholds
        pass

# Fairness-aware training
class FairLoss(nn.Module):
    """
    Custom loss function with fairness constraints
    """
    def __init__(self, base_criterion, fairness_weight=0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.fairness_weight = fairness_weight

    def forward(self, predictions, targets, sensitive_attrs):
        # Base prediction loss
        base_loss = self.base_criterion(predictions, targets)

        # Fairness regularization
        fairness_loss = self.compute_fairness_regularization(predictions, sensitive_attrs)

        # Combined loss
        total_loss = base_loss + self.fairness_weight * fairness_loss

        return total_loss

    def compute_fairness_regularization(self, predictions, sensitive_attrs):
        """
        Compute fairness regularization term
        """
        # Demographic parity regularization
        results = {}
        for group in torch.unique(sensitive_attrs):
            group_mask = sensitive_attrs == group
            group_predictions = predictions[group_mask]
            group_rate = torch.mean(group_predictions)
            results[group] = group_rate

        # Encourage similar positive rates across groups
        rates = list(results.values())
        fairness_reg = torch.var(torch.stack(rates))

        return fairness_reg
```

This comprehensive Computer Vision theory document covers the fundamental concepts, modern architectures, advanced techniques, and emerging applications in computer vision. The content progresses from basic image processing to cutting-edge topics like Vision Transformers, 3D computer vision, and quantum computer vision, providing both theoretical understanding and practical implementation details.
