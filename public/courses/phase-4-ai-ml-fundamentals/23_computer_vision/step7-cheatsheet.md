# Computer Vision & Image Processing Cheatsheet

_Quick Reference Guide for Common Operations_

## Table of Contents

1. [Image Loading and Basic Operations](#basic-operations)
2. [Color Spaces and Transformations](#color-spaces)
3. [Image Enhancement Techniques](#enhancement)
4. [Feature Detection and Description](#features)
5. [Convolutional Neural Networks](#cnn-quickref)
6. [Object Detection](#detection)
7. [Vision Transformers](#vit-ref)
8. [3D Computer Vision](#3d-vision)
9. [Performance Optimization](#optimization)
10. [Common Patterns and Utilities](#utilities)

## 1. Image Loading and Basic Operations {#basic-operations}

```python
# Load and display image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
image = cv2.imread('image.jpg')  # BGR format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save image
cv2.imwrite('output.jpg', image)

# Resize image
resized = cv2.resize(image, (width, height))
resized_aspect = cv2.resize(image, None, fx=scale, fy=scale)

# Crop image
cropped = image[y:y+h, x:x+w]

# Flip image
flipped_horizontal = cv2.flip(image, 1)
flipped_vertical = cv2.flip(image, 0)
flipped_both = cv2.flip(image, -1)

# Get image properties
height, width = image.shape[:2]
channels = image.shape[2] if len(image.shape) == 3 else 1
total_pixels = image.size
dtype = image.dtype

# Convert between OpenCV and PIL
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Quick visualization
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
```

## 2. Color Spaces and Transformations {#color-spaces}

```python
# Color space conversions
bgr_to_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_to_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
bgr_to_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_to_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
bgr_to_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_to_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
bgr_to_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab_to_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Split and merge channels
b, g, r = cv2.split(image)
merged = cv2.merge([b, g, r])

# Color thresholding in HSV
lower_hsv = np.array([h_min, s_min, v_min])
upper_hsv = np.array([h_max, s_max, v_max])
mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

# Extract specific color ranges
red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
green_mask = cv2.inRange(hsv, (50, 50, 50), (70, 255, 255))
```

## 3. Image Enhancement Techniques {#enhancement}

```python
# Basic operations
bright = cv2.convertScaleAbs(image, alpha=1.0, beta=50)  # brightness
contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)  # contrast

# Gamma correction
gamma = 1.2
table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                 for i in np.arange(0, 256)]).astype("uint8")
gamma_corrected = cv2.LUT(image, table)

# Histogram equalization
equalized = cv2.equalizeHist(gray_image)
color_equalized = cv2.equalizeHist(gray)  # Apply to each channel

# Blurring and smoothing
gaussian = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
box = cv2.boxFilter(image, -1, (5, 5))

# Sharpening
kernel_sharpen = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel_sharpen)

# Unsharp masking
blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(binary_image, kernel, iterations=1)
dilated = cv2.dilate(binary_image, kernel, iterations=1)
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
```

## 4. Feature Detection and Description {#features}

```python
# Corner detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=25,
                                 qualityLevel=0.01, minDistance=10)
harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(harris_corners, None)

# Edge detection
edges = cv2.Canny(gray, 50, 150)
edges = cv2.Canny(gray, 100, 200)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Contour detection
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_img = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
contour_img = cv2.drawContours(image, contours, contourIdx=-1,
                              color=(0, 255, 0), thickness=2)

# Feature detection (SIFT, ORB, SURF)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
kp, des = sift.compute(gray, kp)

orb = cv2.ORB_create(nfeatures=1000)
kp = orb.detect(gray, None)
kp, des = orb.compute(gray, kp)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Draw matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None)

# Perspective transformation
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = img1.shape[:2]
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

# Warp perspective
warped = cv2.warpPerspective(img1, M, (w, h))
```

## 5. Convolutional Neural Networks {#cnn-quickref}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Basic CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Training loop
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    predicted = output.argmax(dim=1)
    confidence = F.softmax(output, dim=1)

# Transfer learning
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Model evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
accuracy = 100 * correct / total
```

## 6. Object Detection {#detection}

```python
# YOLO detection
net = cv2.dnn.readNet('yolo.weights', 'yolo.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process YOLO detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

## 7. Vision Transformers {#vit-ref}

```python
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, 3, 768)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, 768))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))

        self.blocks = nn.ModuleList([
            Block(768, 12) for _ in range(12)
        ])

        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        return x
```

## 8. 3D Computer Vision {#3d-vision}

```python
import numpy as np

# Camera calibration
camera_matrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# Undistort image
undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

# Epipolar geometry
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, cv2.FM_RANSAC)

# Triangulation
points_3d = cv2.triangulatePoints(projMatrix1, projMatrix2, points1, points2)
points_3d = points_3d / points_3d[3]  # Convert from homogeneous

# 3D visualization
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
plt.show()

# Point cloud to mesh
from scipy.spatial import ConvexHull
hull = ConvexHull(points_3d)
```

## 9. Performance Optimization {#optimization}

```python
# Model quantization
import torch.quantization as quantization
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantized_model = quantization.prepare(model, inplace=False)
quantized_model = quantization.convert(quantized_model, inplace=False)

# ONNX export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                 export_params=True, opset_version=11)

# ONNX runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_data})

# GPU optimization
model = model.cuda()
model = model.half()  # FP16
torch.backends.cudnn.benchmark = True

# Batch processing
def batch_process_images(image_paths, batch_size=32):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

## 10. Common Patterns and Utilities {#utilities}

### Batch Image Processing

```python
def process_batch(image_paths, target_size=(224, 224)):
    """Process multiple images in batch"""
    processed = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed.append(img)
    return processed

def batch_augment(images):
    """Apply data augmentation to batch"""
    augmented = []
    for img in images:
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random rotation
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented.append(img)
    return augmented
```

### Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f'{name}: {end-start:.3f}s')

# Usage
with timer('Object Detection'):
    detections = detect_objects(image)

# GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
```

### Image Statistics and Analysis

```python
def analyze_image(image):
    """Get comprehensive image statistics"""
    stats = {
        'shape': image.shape,
        'dtype': image.dtype,
        'min_val': image.min(),
        'max_val': image.max(),
        'mean': image.mean(),
        'std': image.std(),
        'histogram': cv2.calcHist([image], [0], None, [256], [0, 256])
    }
    return stats

def compute_image_similarity(img1, img2):
    """Compute similarity between two images"""
    # Structural Similarity Index
    from skimage.metrics import structural_similarity as ssim
    similarity = ssim(img1, img2, data_range=255)

    # Mean Squared Error
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(img1.flatten().reshape(1, -1),
                               img2.flatten().reshape(1, -1))[0, 0]

    return {'ssim': similarity, 'mse': mse, 'cosine': cos_sim}
```

### Error Handling and Validation

```python
def safe_image_load(image_path):
    """Safely load image with error handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def validate_image(image):
    """Validate image format and properties"""
    if image is None:
        raise ValueError("Image is None")
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2D or 3D")
    if image.shape[2] not in [1, 3, 4]:
        raise ValueError("Image must have 1, 3, or 4 channels")
    if image.size == 0:
        raise ValueError("Image is empty")
    return True

def resize_with_padding(image, target_size, color=(0, 0, 0)):
    """Resize image with padding to maintain aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h))

    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded
```

### Visualization Helpers

```python
def plot_image_grid(images, titles=None, cols=4):
    """Plot multiple images in a grid"""
    num_images = len(images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.ravel() if num_images > 1 else [axes]

    for i in range(num_images):
        if len(images[i].shape) == 3:
            axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(images[i], cmap='gray')
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    # Hide extra axes
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image"""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    """Draw keypoints on image"""
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
    return image

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return overlay
```

### Data Loading and Preprocessing

```python
class ImageDataset:
    """Custom image dataset for PyTorch"""
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image

def create_data_transforms():
    """Create standard data transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
```

### Quick One-Liners

```python
# Get image center
center = (image.shape[1] // 2, image.shape[0] // 2)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Get ROI
roi = image[y:y+h, x:x+w]

# Resize maintaining aspect ratio
ratio = min(new_width/width, new_height/height)
new_dim = (int(width*ratio), int(height*ratio))
resized = cv2.resize(image, new_dim)

# Create mask from points
mask = np.zeros(image.shape[:2], dtype=np.uint8)
points = np.array([[x1, y1], [x2, y2], [x3, y3]])
cv2.fillPoly(mask, [points], 255)

# Apply mask
result = cv2.bitwise_and(image, image, mask=mask)

# Compute gradient
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Image rotation matrix
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, M, (width, height))

# Create canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Draw text
cv2.putText(image, "Text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Draw circle
cv2.circle(image, (x, y), radius, color, thickness)

# Draw line
cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Get frame from video
ret, frame = cap.read()

# Apply colormap
colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
```

This comprehensive cheatsheet provides quick references for all major computer vision operations, from basic image processing to advanced deep learning techniques. Each code snippet is designed to be directly usable in your projects.
