# Computer Vision & Image Processing Practice

_Hands-On Exercises from Fundamentals to Advanced_

## Table of Contents

1. [Basic Image Processing](#basic-image-processing)
2. [Feature Detection and Description](#feature-detection)
3. [Convolutional Neural Networks](#cnn-exercises)
4. [Object Detection and Recognition](#object-detection)
5. [Modern Architectures (Vision Transformers)](#vit-exercises)
6. [3D Computer Vision](#3d-vision)
7. [Specialized Applications](#specialized-apps)
8. [Real-world Projects](#real-world-projects)
9. [Performance Optimization](#optimization)
10. [Advanced Challenges](#advanced-challenges)

## 1. Basic Image Processing {#basic-image-processing}

### Exercise 1.1: Image Loading and Basic Operations

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageProcessor:
    def __init__(self):
        pass

    def load_and_explore_image(self, image_path):
        """
        Load image and explore its properties
        """
        # Load image using OpenCV
        image_cv = cv2.imread(image_path)

        # Load image using PIL
        image_pil = Image.open(image_path)

        print("=== Image Information ===")
        print(f"OpenCV shape: {image_cv.shape}")
        print(f"PIL size: {image_pil.size}")
        print(f"PIL mode: {image_pil.mode}")
        print(f"Data type: {image_cv.dtype}")

        # Convert color spaces
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Visualize different representations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0,0].imshow(rgb_image)
        axes[0,0].set_title('RGB Image')
        axes[0,0].axis('off')

        axes[0,1].imshow(hsv_image)
        axes[0,1].set_title('HSV Image')
        axes[0,1].axis('off')

        axes[1,0].imshow(gray_image, cmap='gray')
        axes[1,0].set_title('Grayscale Image')
        axes[1,0].axis('off')

        # Show individual channels
        channels = cv2.split(rgb_image)
        axes[1,1].imshow(channels[0], cmap='Reds')
        axes[1,1].set_title('Red Channel')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()

        return image_cv, rgb_image, gray_image, hsv_image

# Practice Task 1: Implement your own image loader
def practice_load_image(image_path):
    """
    Practice: Create your own image loading function
    """
    # TODO: Implement your own image loading with error handling
    # TODO: Check if file exists
    # TODO: Handle different image formats
    # TODO: Convert to appropriate format for processing

    pass

# Usage
processor = ImageProcessor()
image_cv, rgb_image, gray_image, hsv_image = processor.load_and_explore_image('sample_image.jpg')
```

### Exercise 1.2: Image Enhancement and Filtering

```python
class ImageEnhancement:
    def __init__(self, image):
        self.image = image

    def brightness_adjustment(self, beta=50):
        """
        Adjust image brightness
        """
        bright_image = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)
        return bright_image

    def contrast_adjustment(self, alpha=1.5):
        """
        Adjust image contrast
        """
        contrast_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
        return contrast_image

    def gamma_correction(self, gamma=1.2):
        """
        Apply gamma correction
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction
        gamma_corrected = cv2.LUT(self.image, table)
        return gamma_corrected

    def histogram_equalization(self):
        """
        Apply histogram equalization
        """
        if len(self.image.shape) == 3:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l_channel)
            lab = cv2.merge([l_eq, a, b])
            equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            equalized = cv2.equalizeHist(self.image)

        return equalized

    def gaussian_blur(self, kernel_size=(5, 5)):
        """
        Apply Gaussian blur
        """
        blurred = cv2.GaussianBlur(self.image, kernel_size, 0)
        return blurred

    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for edge-preserving smoothing
        """
        filtered = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        return filtered

    def unsharp_masking(self, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """
        Apply unsharp masking for image sharpening
        """
        blurred = cv2.GaussianBlur(self.image, kernel_size, sigma)
        sharpened = float(amount + 1) * self.image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(self.image - blurred) < threshold
            np.copyto(sharpened, self.image, where=low_contrast_mask)

        return sharpened

    def visualize_enhancements(self):
        """
        Visualize different enhancement techniques
        """
        original = self.image

        enhancements = {
            'Original': original,
            'Brightness +50': self.brightness_adjustment(50),
            'Contrast 1.5x': self.contrast_adjustment(1.5),
            'Gamma 1.2': self.gamma_correction(1.2),
            'Histogram Equalization': self.histogram_equalization(),
            'Gaussian Blur': self.gaussian_blur(),
            'Bilateral Filter': self.bilateral_filter(),
            'Unsharp Masking': self.unsharp_masking()
        }

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i, (title, enhanced) in enumerate(enhancements.items()):
            if len(enhanced.shape) == 3:
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                axes[i].imshow(enhanced_rgb)
            else:
                axes[i].imshow(enhanced, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Practice Task 2: Create your own filter
def create_custom_filter(image):
    """
    Practice: Create a custom image filter
    """
    # TODO: Design a custom kernel/mask
    # TODO: Apply it to the image
    # TODO: Compare with original

    # Example: Edge enhancement filter
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    enhanced = cv2.filter2D(image, -1, kernel)
    return enhanced

# Practice Task 3: Noise addition and removal
def add_and_remove_noise(image):
    """
    Practice: Add various types of noise and remove them
    """
    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_gaussian = cv2.add(image, noise)

    # Add salt and pepper noise
    noisy_sp = image.copy()
    noise = np.random.random(image.shape)
    noisy_sp[noise < 0.01] = 0
    noisy_sp[noise > 0.99] = 255

    # Remove noise using various methods
    denoised_gaussian = cv2.medianBlur(noisy_gaussian, 5)
    denoised_sp = cv2.medianBlur(noisy_sp, 5)

    # Non-local means denoising
    if len(image.shape) == 3:
        denoised_nlm = cv2.fastNlMeansDenoisingColored(noisy_gaussian, None, 10, 10, 7, 21)
    else:
        denoised_nlm = cv2.fastNlMeansDenoising(noisy_gaussian, None, 10, 7, 21)

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    axes[0,1].imshow(cv2.cvtColor(noisy_gaussian, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('Gaussian Noise')
    axes[0,1].axis('off')

    axes[0,2].imshow(cv2.cvtColor(noisy_sp, cv2.COLOR_BGR2RGB))
    axes[0,2].set_title('Salt & Pepper Noise')
    axes[0,2].axis('off')

    axes[1,0].imshow(cv2.cvtColor(denoised_gaussian, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('Denoised (Median)')
    axes[1,0].axis('off')

    axes[1,1].imshow(cv2.cvtColor(denoised_sp, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('Denoised (Median)')
    axes[1,1].axis('off')

    axes[1,2].imshow(cv2.cvtColor(denoised_nlm, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title('Denoised (NLM)')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

    return noisy_gaussian, noisy_sp, denoised_gaussian, denoised_sp, denoised_nlm

# Usage
enhancer = ImageEnhancement(image_cv)
enhancer.visualize_enhancements()
```

### Exercise 1.3: Morphological Operations

```python
class MorphologicalOperations:
    def __init__(self, image):
        self.image = image
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image

    def create_binary_image(self, method='threshold', thresh_value=127):
        """
        Create binary image using different methods
        """
        if method == 'threshold':
            _, binary = cv2.threshold(self.gray, thresh_value, 255, cv2.THRESH_BINARY)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def morphological_operations(self, operation='erosion', kernel_size=5):
        """
        Apply morphological operations
        """
        binary = self.create_binary_image()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if operation == 'erosion':
            result = cv2.erode(binary, kernel, iterations=1)
        elif operation == 'dilation':
            result = cv2.dilate(binary, kernel, iterations=1)
        elif operation == 'opening':
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        elif operation == 'tophat':
            result = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            result = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

        return result

    def advanced_morphological_features(self):
        """
        Extract features using morphological operations
        """
        binary = self.create_binary_image()

        # Extract connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        print(f"Number of connected components: {num_labels - 1}")

        # Extract individual components
        component_image = np.zeros_like(binary)
        for i in range(1, num_labels):
            component_mask = (labels == i)
            component_image[component_mask] = 255

        # Calculate moments
        moments = cv2.moments(binary)

        # Extract geometric features
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            print(f"Centroid: ({cx}, {cy})")

        return component_image, centroids, moments

    def visualize_morphology(self):
        """
        Visualize different morphological operations
        """
        original = self.gray
        binary = self.create_binary_image()

        operations = ['erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binary')
        axes[1].axis('off')

        for i, op in enumerate(operations):
            result = self.morphological_operations(op)
            axes[i+2].imshow(result, cmap='gray')
            axes[i+2].set_title(op.capitalize())
            axes[i+2].axis('off')

        plt.tight_layout()
        plt.show()

# Practice Task 4: Custom morphological operation
def custom_morphological_operation(image):
    """
    Practice: Design and implement a custom morphological operation
    """
    # TODO: Create a custom kernel
    # TODO: Apply it with your own logic
    # TODO: Compare with standard operations

    pass

# Usage
morph_ops = MorphologicalOperations(image_cv)
morph_ops.visualize_morphology()
```

## 2. Feature Detection and Description {#feature-detection}

### Exercise 2.1: Corner Detection

```python
class CornerDetector:
    def __init__(self, image):
        self.image = image
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image

    def harris_corner_detection(self, block_size=2, ksize=3, k=0.04):
        """
        Harris corner detection
        """
        # Convert to float32
        gray = np.float32(self.gray)

        # Compute Harris corner response
        dst = cv2.cornerHarris(gray, block_size, ksize, k)

        # Threshold for an optimal value
        dst = cv2.dilate(dst, None)

        # Create result image
        result = self.image.copy()
        result[dst > 0.01 * dst.max()] = [0, 0, 255]

        return result, dst

    def shi_tomasi_corner_detection(self, max_corners=25, quality_level=0.01, min_distance=10):
        """
        Shi-Tomasi corner detection
        """
        # Detect corners
        corners = cv2.goodFeaturesToTrack(self.gray, max_corners, quality_level, min_distance)

        # Create result image
        result = self.image.copy()

        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)

        return result, corners

    def fast_corner_detection(self):
        """
        FAST corner detection
        """
        # Create FAST detector
        fast = cv2.FastFeatureDetector_create()

        # Detect keypoints
        keypoints = fast.detect(self.gray, None)

        # Draw keypoints
        result = self.image.copy()
        cv2.drawKeypoints(self.image, keypoints, result, color=(255, 0, 0))

        return result, keypoints

    def compare_corner_methods(self):
        """
        Compare different corner detection methods
        """
        harris_result, harris_dst = self.harris_corner_detection()
        shi_tomasi_result, shi_tomasi_corners = self.shi_tomasi_corner_detection()
        fast_result, fast_keypoints = self.fast_corner_detection()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0,0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')

        axes[0,1].imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Harris Corner Detection')
        axes[0,1].axis('off')

        axes[1,0].imshow(cv2.cvtColor(shi_tomasi_result, cv2.COLOR_BGR2RGB))
        axes[1,0].set_title('Shi-Tomasi Corner Detection')
        axes[1,0].axis('off')

        axes[1,1].imshow(cv2.cvtColor(fast_result, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title('FAST Corner Detection')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()

        return harris_result, shi_tomasi_result, fast_result

# Practice Task 5: Corner detection parameters
def experiment_corner_parameters(image):
    """
    Practice: Experiment with different corner detection parameters
    """
    detector = CornerDetector(image)

    # Test different parameters
    parameters = [
        {'block_size': 2, 'ksize': 3, 'k': 0.04},
        {'block_size': 4, 'ksize': 3, 'k': 0.04},
        {'block_size': 2, 'ksize': 5, 'k': 0.04},
        {'block_size': 2, 'ksize': 3, 'k': 0.01},
        {'block_size': 2, 'ksize': 3, 'k': 0.1},
    ]

    fig, axes = plt.subplots(1, len(parameters), figsize=(20, 4))

    for i, params in enumerate(parameters):
        harris_result, _ = detector.harris_corner_detection(**params)
        axes[i].imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"block_size={params['block_size']}, ksize={params['ksize']}, k={params['k']}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
corner_detector = CornerDetector(image_cv)
corner_detector.compare_corner_methods()
```

### Exercise 2.2: Feature Detection and Description

```python
class FeatureDescriptor:
    def __init__(self, image):
        self.image = image
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image

    def sift_detection(self):
        """
        SIFT feature detection and description
        """
        # Create SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(self.gray, None)

        # Draw keypoints
        result = self.image.copy()
        cv2.drawKeypoints(self.image, keypoints, result,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return result, keypoints, descriptors

    def orb_detection(self):
        """
        ORB feature detection and description
        """
        # Create ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(self.gray, None)

        # Draw keypoints
        result = self.image.copy()
        cv2.drawKeypoints(self.image, keypoints, result, color=(0, 255, 0))

        return result, keypoints, descriptors

    def feature_matching(self, image1, image2):
        """
        Match features between two images
        """
        # Detect features in both images
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        # Create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = matcher.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches
        result = cv2.drawMatches(image1, kp1, image2, kp2,
                                matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return result, matches

    def homography_estimation(self, image1, image2):
        """
        Estimate homography and warp perspective
        """
        # Detect and match features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        # Create BFMatcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Sort matches
        matches = sorted(matches, key=lambda x: x.distance)

        # Use good matches
        good_matches = matches[:10]

        if len(good_matches) >= 4:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get image dimensions
            h, w = image1.shape[:2]

            # Warp image
            warped = cv2.warpPerspective(image1, M, (w, h))

            # Create panorama
            panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
            panorama[:, :w] = image2
            panorama[:, w:] = warped

            return panorama, M, good_matches
        else:
            print("Not enough matches found")
            return None, None, None

    def visualize_all_features(self):
        """
        Visualize different feature detection methods
        """
        sift_result, sift_kp, sift_des = self.sift_detection()
        orb_result, orb_kp, orb_des = self.orb_detection()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'SIFT Features (Found: {len(sift_kp)}, Descriptors: {sift_des.shape if sift_des is not None else 0})')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(orb_result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'ORB Features (Found: {len(orb_kp)}, Descriptors: {orb_des.shape if orb_des is not None else 0})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        return sift_result, orb_result

# Practice Task 6: Panorama creation
def create_panorama(image1_path, image2_path):
    """
    Practice: Create a panorama from two images
    """
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error loading images")
        return None

    # Resize images to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    target_height = min(h1, h2)
    scale1 = target_height / h1
    scale2 = target_height / h2

    image1 = cv2.resize(image1, (int(w1 * scale1), target_height))
    image2 = cv2.resize(image2, (int(w2 * scale2), target_height))

    # Create feature descriptor
    descriptor = FeatureDescriptor(image1)

    # Estimate homography and create panorama
    panorama, homography, matches = descriptor.homography_estimation(image1, image2)

    if panorama is not None:
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title('Panorama')
        plt.axis('off')
        plt.show()

        print(f"Found {len(matches)} good matches")
        print(f"Homography matrix:\n{homography}")

        return panorama
    else:
        print("Failed to create panorama")
        return None

# Usage
feature_desc = FeatureDescriptor(image_cv)
feature_desc.visualize_all_features()
```

## 3. Convolutional Neural Networks {#cnn-exercises}

### Exercise 3.1: Building CNNs from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class CustomCNN(nn.Module):
    """
    Custom CNN implementation with various architectures
    """
    def __init__(self, num_classes=10, architecture='simple'):
        super(CustomCNN, self).__init__()

        if architecture == 'simple':
            # Simple CNN architecture
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        elif architecture == 'deep':
            # Deeper CNN architecture
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block for ResNet-like architectures
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet implementation
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

class CNNTrainer:
    """
    Trainer class for CNN models
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs=10):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 30)

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print()

    def plot_training_curves(self):
        """
        Plot training and validation curves
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy curves
        axes[1].plot(self.train_accuracies, label='Train Accuracy')
        axes[1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_filters(self, layer_idx=0):
        """
        Visualize filters from a convolutional layer
        """
        # Get the first convolutional layer
        conv_layer = None
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                layer_idx -= 1
                if layer_idx == 0:
                    break

        if conv_layer is None:
            print("No convolutional layer found")
            return

        # Get filters
        filters = conv_layer.weight.data.cpu()

        # Normalize for visualization
        filters = filters - filters.min()
        filters = filters / filters.max()

        # Plot filters
        n_filters = min(filters.shape[0], 16)  # Show max 16 filters
        cols = 4
        rows = (n_filters + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_filters):
            row = i // cols
            col = i % cols

            filter_img = filters[i].mean(dim=0)  # Average across channels
            axes[row, col].imshow(filter_img, cmap='gray')
            axes[row, col].set_title(f'Filter {i+1}')
            axes[row, col].axis('off')

        # Hide empty subplots
        for i in range(n_filters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

# Practice Task 7: Custom CNN Architecture
def create_custom_cnn():
    """
    Practice: Design your own CNN architecture
    """
    # TODO: Experiment with different layer configurations
    # TODO: Try different activation functions
    # TODO: Experiment with different pooling strategies
    # TODO: Add skip connections or attention mechanisms

    class MyCustomCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Your custom architecture here

            # Example: Try a U-Net like architecture for encoder-decoder
            self.encoder = nn.Sequential(
                # TODO: Design encoder layers
            )

            self.decoder = nn.Sequential(
                # TODO: Design decoder layers
            )

            self.classifier = nn.Sequential(
                # TODO: Design classifier head
            )

        def forward(self, x):
            # TODO: Implement forward pass
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.classifier(x)
            return x

    return MyCustomCNN()

# Usage
def train_cnn_experiment():
    """
    Complete CNN training experiment
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Create model
    model = CustomCNN(num_classes=10, architecture='simple')

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    trainer = CNNTrainer(model, device)
    trainer.train(train_loader, test_loader, optimizer, criterion, num_epochs=20)

    # Visualize results
    trainer.plot_training_curves()
    trainer.visualize_filters()

    return model, trainer

# Practice Task 8: Transfer Learning
def transfer_learning_experiment():
    """
    Practice: Implement transfer learning
    """
    import torchvision.models as models

    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # 10 classes for CIFAR-10

    # Unfreeze last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.avgpool.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

# Run experiments
# cnn_model, cnn_trainer = train_cnn_experiment()
```

### Exercise 3.2: Advanced CNN Techniques

```python
class AdvancedCNN:
    """
    Advanced CNN techniques and architectures
    """

    @staticmethod
    def create_attention_module(in_channels, reduction=16):
        """
        Create attention module (Squeeze-and-Excitation)
        """
        class SEBlock(nn.Module):
            def __init__(self, in_channels, reduction):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)
                return x * y

        return SEBlock(in_channels, reduction)

    @staticmethod
    def create_depthwise_separable_conv(in_channels, out_channels, stride=1):
        """
        Create depthwise separable convolution
        """
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def create_mobilenet_block(in_channels, out_channels, stride):
        """
        Create MobileNet inverted residual block
        """
        hidden_dim = int(round(in_channels * 6))

        if in_channels == hidden_dim:
            use_res_connect = (stride == 1)
            layers = []
        else:
            use_res_connect = False
            layers = [
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ]

        if use_res_connect:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        else:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        if not use_res_connect:
            layers.append(nn.Identity())

        return nn.Sequential(*layers), use_res_connect

class EfficientNet:
    """
    EfficientNet implementation
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # This is a simplified EfficientNet-B0
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MBConv blocks with different scaling factors
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)

        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes)
        )

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Make MBConv block for EfficientNet
        """
        hidden_dim = int(round(in_channels * expand_ratio))
        use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        return nn.Sequential(*layers) if not use_res_connect else nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

class DataAugmentation:
    """
    Advanced data augmentation techniques
    """

    @staticmethod
    def auto_augment_transforms():
        """
        Define auto-augment transforms
        """
        import random

        policies = [
            # (operation, probability, magnitude)
            ('ShearX', 0.9, 4),    # Shear in X direction
            ('ShearY', 0.9, 4),    # Shear in Y direction
            ('TranslateX', 0.9, 4), # Translate in X
            ('TranslateY', 0.9, 4), # Translate in Y
            ('Rotate', 0.9, 30),   # Rotate
            ('Brightness', 0.9, 0.2), # Adjust brightness
            ('Color', 0.9, 0.2),   # Adjust color
            ('Contrast', 0.9, 0.2), # Adjust contrast
            ('Sharpness', 0.9, 0.2), # Adjust sharpness
            ('Posterize', 0.9, 4), # Posterize
            ('Solarize', 0.9, 128), # Solarize
            ('AutoContrast', 0.9, 0), # Auto contrast
            ('Equalize', 0.9, 0),  # Equalize
            ('Invert', 0.9, 0),    # Invert
        ]

        return policies

    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        """
        Apply mixup augmentation
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def cutmix_data(x, y, alpha=1.0):
        """
        Apply cutmix augmentation
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        y_a, y_b = y, y[index]

        # Generate bounding box
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """
    Generate random bounding box for cutmix
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Practice Task 9: Custom Data Augmentation
class CustomAugmentation:
    """
    Practice: Implement custom augmentation techniques
    """

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        """
        Apply elastic deformation
        """
        height, width = image.shape[:2]

        # Create displacement fields
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        transformed = map_coordinates(image, indices, order=1, mode='reflect')
        transformed = transformed.reshape(image.shape)

        return transformed

    @staticmethod
    def random_erase(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        Randomly erase patches from image
        """
        if np.random.random() > probability:
            return image

        for _ in range(1):
            area = image.shape[0] * image.shape[1]
            target_area = np.random.uniform(sl, sh) * area

            aspect_ratio = np.random.uniform(r1, 1/r1)

            h = int(np.sqrt(target_area * aspect_ratio))
            w = int(np.sqrt(target_area / aspect_ratio))

            if h < image.shape[0] and w < image.shape[1]:
                x1 = np.random.randint(0, image.shape[0] - h)
                y1 = np.random.randint(0, image.shape[1] - w)

                if len(image.shape) == 3:
                    image[x1:x1+h, y1:y1+w] = 0
                else:
                    image[x1:x1+h, y1:y1+w] = 0

        return image

# Practice Task 10: Neural Architecture Search
def simple_nas_experiment():
    """
    Practice: Simple Neural Architecture Search
    """
    from itertools import product

    # Define search space
    conv_layers = [1, 2, 3]  # Number of conv layers
    filters = [32, 64, 128]  # Number of filters
    kernel_sizes = [3, 5]    # Kernel sizes
    use_batchnorm = [True, False]  # Use batch norm
    use_dropout = [True, False]    # Use dropout

    # Generate all possible architectures
    architectures = list(product(conv_layers, filters, kernel_sizes, use_batchnorm, use_dropout))

    print(f"Total architectures to test: {len(architectures)}")

    # Example of creating a model from architecture parameters
    def create_model_from_config(config):
        conv_layers, num_filters, kernel_size, use_bn, use_dropout = config

        layers = []
        in_channels = 3

        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
            if use_bn:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = num_filters

        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 10)
        ])

        if use_dropout:
            layers.insert(-1, nn.Dropout(0.5))

        return nn.Sequential(*layers)

    # This would be used in a real NAS framework to evaluate architectures
    return architectures

# Usage
# efficient_model = EfficientNet(num_classes=10)
```

## 4. Object Detection and Recognition {#object-detection}

### Exercise 4.1: Traditional Object Detection

```python
class TraditionalObjectDetector:
    """
    Traditional object detection using sliding window and SVM
    """

    def __init__(self):
        self.hog_detector = None
        self.svm_classifier = None

    def create_hog_descriptor(self):
        """
        Create HOG descriptor
        """
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        self.hog_detector = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        return self.hog_detector

    def sliding_window_detection(self, image, window_size, step_size, classifier):
        """
        Sliding window object detection
        """
        detections = []

        for y in range(0, image.shape[0] - window_size[1], step_size):
            for x in range(0, image.shape[1] - window_size[0], step_size):
                # Extract window
                window = image[y:y + window_size[1], x:x + window_size[0]]

                # Resize to fixed size
                window_resized = cv2.resize(window, (64, 64))

                # Compute HOG features
                if self.hog_detector is not None:
                    hog_features = self.hog_detector.compute(window_resized)

                    # Predict using classifier
                    if classifier:
                        prediction = classifier.predict([hog_features])[0]
                        confidence = classifier.decision_function([hog_features])[0]

                        if prediction == 1 and confidence > 0:  # Positive detection
                            detections.append({
                                'bbox': (x, y, window_size[0], window_size[1]),
                                'confidence': confidence
                            })

        return detections

    def non_maximum_suppression(self, detections, overlap_threshold=0.3):
        """
        Apply non-maximum suppression
        """
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        suppressed = []
        while detections:
            # Take the most confident detection
            current = detections.pop(0)
            suppressed.append(current)

            # Remove overlapping detections
            remaining = []
            for detection in detections:
                iou = self.compute_iou(current['bbox'], detection['bbox'])
                if iou < overlap_threshold:
                    remaining.append(detection)

            detections = remaining

        return suppressed

    def compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

class RCNNDetector:
    """
    R-CNN object detector
    """

    def __init__(self):
        self.svm_classifier = None
        self.cnn_extractor = None

    def selective_search(self, image):
        """
        Generate region proposals using selective search
        """
        # Initialize selective search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchQuality()

        # Generate proposals
        rectangles = ss.process()

        proposals = []
        for rect in rectangles:
            x, y, w, h = rect
            proposals.append((x, y, w, h))

        return proposals

    def extract_roi_features(self, image, proposals):
        """
        Extract features for region proposals
        """
        features = []

        for (x, y, w, h) in proposals:
            # Extract ROI
            roi = image[y:y+h, x:x+w]

            # Resize to fixed size (224x224 for CNN)
            roi_resized = cv2.resize(roi, (224, 224))

            # Convert to tensor and extract features
            roi_tensor = self.preprocess_image(roi_resized)
            with torch.no_grad():
                feature_vector = self.cnn_extractor(roi_tensor)

            features.append(feature_vector.cpu().numpy().flatten())

        return np.array(features)

    def preprocess_image(self, image):
        """
        Preprocess image for CNN
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0)

    def train_rcnn(self, dataset_path):
        """
        Train R-CNN detector
        """
        # Load pre-trained CNN for feature extraction
        self.cnn_extractor = models.resnet50(pretrained=True)
        self.cnn_extractor = torch.nn.Sequential(*list(self.cnn_extractor.children())[:-1])

        # Extract features for all training images
        all_features = []
        all_labels = []

        # Training logic here...
        # This is a simplified version

        # Train SVM classifier
        self.svm_classifier = svm.SVC(kernel='rbf', probability=True)
        self.svm_classifier.fit(all_features, all_labels)

        return self.svm_classifier

    def detect_objects(self, image, confidence_threshold=0.5):
        """
        Detect objects in image
        """
        # Generate proposals
        proposals = self.selective_search(image)

        # Extract features
        features = self.extract_roi_features(image, proposals)

        # Predict
        predictions = self.svm_classifier.predict_proba(features)

        # Filter by confidence
        detections = []
        for i, (x, y, w, h) in enumerate(proposals):
            confidence = predictions[i].max()
            class_id = predictions[i].argmax()

            if confidence > confidence_threshold:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class_id': class_id
                })

        # Apply NMS
        detections = self.apply_nms(detections)

        return detections

    def apply_nms(self, detections):
        """
        Apply non-maximum suppression
        """
        # Convert to format suitable for cv2.dnn.NMSBoxes
        boxes = [det['bbox'] for det in detections]
        scores = [det['confidence'] for det in detections]
        class_ids = [det['class_id'] for det in detections]

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.4)

        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []

# Practice Task 11: Custom Region Proposal Generator
def custom_region_proposal(image, num_proposals=1000):
    """
    Practice: Implement custom region proposal algorithm
    """
    proposals = []

    # Multi-scale proposals
    scales = [0.5, 1.0, 2.0]

    for scale in scales:
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
        h, w = scaled_image.shape[:2]

        # Grid-based proposals
        grid_size = 32
        for y in range(0, h - grid_size, grid_size):
            for x in range(0, w - grid_size, grid_size):
                # Different aspect ratios
                aspect_ratios = [1.0, 2.0, 0.5]

                for ratio in aspect_ratios:
                    proposal_w = grid_size
                    proposal_h = int(grid_size * ratio)

                    if y + proposal_h < h and x + proposal_w < w:
                        # Scale back to original coordinates
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(proposal_w / scale)
                        orig_h = int(proposal_h / scale)

                        proposals.append((orig_x, orig_y, orig_w, orig_h))

    # Keep only top proposals
    proposals = proposals[:num_proposals]

    return proposals

# Usage
detector = TraditionalObjectDetector()
detector.create_hog_descriptor()
```

### Exercise 4.2: Modern Object Detection (YOLO)

```python
class YOLODetector:
    """
    YOLO object detector implementation
    """

    def __init__(self, config_path, weights_path, class_names):
        self.class_names = class_names
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess_image(self, image):
        """
        Preprocess image for YOLO
        """
        # Get image dimensions
        height, width, channels = image.shape

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set input blob
        self.net.setInput(blob)

        return blob, width, height

    def get_output_layers(self):
        """
        Get output layers
        """
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def detect_objects(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Detect objects in image
        """
        blob, width, height = self.preprocess_image(image)

        # Forward pass
        output_layers = self.get_output_layers()
        outs = self.net.forward(output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                class_name = self.class_names[class_id]

                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        return detections

    def draw_detections(self, image, detections):
        """
        Draw detections on image
        """
        result_image = image.copy()

        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f'{class_name}: {confidence:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(result_image, (x, y - label_size[1] - 10),
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(result_image, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return result_image

class SimpleYOLO:
    """
    Simple YOLO implementation for educational purposes
    """

    def __init__(self, num_classes=80, input_size=416):
        self.num_classes = num_classes
        self.input_size = input_size

        # YOLO architecture components
        self.darknet = self.build_darknet()
        self.yolo_layers = self.build_yolo_layers()

    def build_darknet(self):
        """
        Build Darknet backbone
        """
        layers = []

        # First layer
        layers.append(self.conv_block(3, 32, 3, 1))

        # Downsample
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(self.conv_block(32, 64, 3, 1))

        # Downsample
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(self.conv_block(64, 128, 3, 1))
        layers.append(self.conv_block(128, 64, 1, 1))
        layers.append(self.conv_block(64, 128, 3, 1))

        # Downsample
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(self.conv_block(128, 256, 3, 1))
        layers.append(self.conv_block(256, 128, 1, 1))
        layers.append(self.conv_block(128, 256, 3, 1))

        # Continue building...
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(self.conv_block(256, 512, 3, 1))
        layers.append(self.conv_block(512, 256, 1, 1))
        layers.append(self.conv_block(256, 512, 3, 1))
        layers.append(self.conv_block(512, 256, 1, 1))
        layers.append(self.conv_block(256, 512, 3, 1))

        # Downsample
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(self.conv_block(512, 1024, 3, 1))
        layers.append(self.conv_block(1024, 512, 1, 1))
        layers.append(self.conv_block(512, 1024, 3, 1))
        layers.append(self.conv_block(1024, 512, 1, 1))
        layers.append(self.conv_block(512, 1024, 3, 1))

        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        """
        Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def build_yolo_layers(self):
        """
        Build YOLO detection layers
        """
        # YOLO layer for different scales
        yolo_layers = nn.ModuleList([
            YOLOLayer(1024, self.num_classes),  # Large objects
            YOLOLayer(512, self.num_classes),   # Medium objects
            YOLOLayer(256, self.num_classes),   # Small objects
        ])

        return yolo_layers

    def forward(self, x):
        """
        Forward pass
        """
        # Feature extraction
        features = self.darknet(x)

        # Detection at different scales
        detections = []
        for yolo_layer in self.yolo_layers:
            detections.append(yolo_layer(features))
            features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)

        return detections

class YOLOLayer(nn.Module):
    """
    YOLO detection layer
    """

    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)  # x, y, w, h, obj + classes

        # Prediction layers
        self.conv = nn.Conv2d(in_channels, self.num_outputs, 1, 1, 0)

        # Anchor boxes (predefined)
        self.anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],  # For small objects
            [30, 61], [62, 45], [59, 119], # For medium objects
            [116, 90], [156, 198], [373, 326]  # For large objects
        ]).float()

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Prediction
        pred = self.conv(x)
        pred = pred.view(batch_size, self.num_anchors, self.num_outputs, height, width)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        return pred

# Practice Task 12: Custom Anchor Generation
def generate_anchors(kmeans_results, num_anchors=9):
    """
    Practice: Generate custom anchors using k-means clustering
    """
    # This would analyze bounding box distributions in your dataset
    # and generate anchors that fit your specific use case

    import numpy as np
    from sklearn.cluster import KMeans

    # Example: using bounding box dimensions
    bbox_widths = np.array([w for w, h in kmeans_results])
    bbox_heights = np.array([h for w, h in kmeans_results])

    # Combine width and height
    bbox_aspects = np.column_stack((bbox_widths, bbox_heights))

    # K-means clustering to find anchor shapes
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    anchors = kmeans.fit(bbox_aspects).cluster_centers_

    return anchors

# Practice Task 13: Loss Function Implementation
class YOLOLoss(nn.Module):
    """
    Custom YOLO loss function
    """

    def __init__(self, num_classes=80, anchors=None, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else torch.tensor([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ]).float()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Compute YOLO loss
        """
        batch_size = predictions.size(0)
        total_loss = 0

        for i in range(3):  # Three scales
            pred = predictions[i]
            target = targets[i]

            # Loss computation
            loss = self.compute_scale_loss(pred, target, i)
            total_loss += loss

        return total_loss / batch_size

    def compute_scale_loss(self, pred, target, scale_idx):
        """
        Compute loss for one scale
        """
        batch_size, num_anchors, height, width, num_outputs = pred.size()
        num_classes = self.num_classes

        # Objectness loss
        obj_loss = self.bce_loss(
            pred[..., 4].sigmoid(),
            target[..., 4]
        )

        # No objectness loss
        noobj_loss = self.bce_loss(
            (1 - target[..., 4]) * pred[..., 4].sigmoid(),
            1 - target[..., 4]
        )

        # Classification loss
        cls_loss = self.bce_loss(
            pred[..., 5:].sigmoid(),
            target[..., 5:]
        )

        # Box coordinate loss (only for objects)
        box_loss = self.mse_loss(
            pred[..., :4][target[..., 4] == 1],
            target[..., :4][target[..., 4] == 1]
        )

        total_loss = (self.lambda_coord * box_loss +
                     obj_loss +
                     self.lambda_noobj * noobj_loss +
                     cls_loss)

        return total_loss

# Usage
def yolo_training_example():
    """
    Example of training YOLO
    """
    # Initialize model
    model = SimpleYOLO(num_classes=80)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    criterion = YOLOLoss(num_classes=80)

    # Training loop (simplified)
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         images, targets = batch
    #
    #         optimizer.zero_grad()
    #         predictions = model(images)
    #         loss = criterion(predictions, targets)
    #         loss.backward()
    #         optimizer.step()
    pass
```

This practice file provides hands-on exercises for computer vision, covering everything from basic image processing to advanced deep learning architectures. Each exercise includes both code examples and practice tasks to help you understand and implement computer vision concepts effectively.
