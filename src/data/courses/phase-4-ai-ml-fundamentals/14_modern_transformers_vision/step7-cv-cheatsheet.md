# Computer Vision Cheat Sheet

## Table of Contents

1. [OpenCV Operations](#opencv-operations)
2. [Image Preprocessing](#image-preprocessing)
3. [Object Detection](#object-detection)
4. [Face Recognition](#face-recognition)
5. [Segmentation Techniques](#segmentation-techniques)
6. [Computer Vision Pipeline Patterns](#computer-vision-pipeline-patterns)

---

## OpenCV Operations

### Basic Image Operations

```python
import cv2
import numpy as np

# Read and Display Image
img = cv2.imread('image.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get Image Properties
height, width, channels = img.shape
print(f"Height: {height}, Width: {width}, Channels: {channels}")

# Color Space Conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Resize Image
resized = cv2.resize(img, (800, 600))
resized = cv2.resize(img, None, fx=0.5, fy=0.5)  # Scale factors

# Rotate Image
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees rotation
rotated = cv2.warpAffine(img, M, (w, h))

# Crop Image
cropped = img[100:300, 200:400]  # y_start:y_end, x_start:x_end

# Flip Image
flipped_h = cv2.flip(img, 1)  # Horizontal flip
flipped_v = cv2.flip(img, 0)  # Vertical flip
```

### Drawing Operations

```python
# Draw Rectangle
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

# Draw Circle
cv2.circle(img, (center_x, center_y), radius, (255, 0, 0), thickness)

# Draw Line
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

# Add Text
cv2.putText(img, "Text", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, color, thickness, cv2.LINE_AA)

# Draw Multiple Shapes
for i in range(5):
    cv2.rectangle(img, (i*50, i*50), (i*50+100, i*50+100), (0, 255, i*50), -1)
```

### Matrix Operations

```python
# Image Arithmetic
added = cv2.add(img1, img2)
weighted = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

# Bitwise Operations
and_result = cv2.bitwise_and(img1, img2)
or_result = cv2.bitwise_or(img1, img2)
xor_result = cv2.bitwise_xor(img1, img2)
not_result = cv2.bitwise_not(img1)

# Create Blank Image
blank = np.zeros((height, width, 3), dtype=np.uint8)

# Add Borders
bordered = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))
```

### Filtering and Convolution

```python
# Gaussian Blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur
median_blurred = cv2.medianBlur(img, 5)

# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Custom Kernel Convolution
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
convolved = cv2.filter2D(img, -1, kernel)

# Sobel Edges
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Canny Edge Detection
edges = cv2.Canny(img, 50, 150)
```

---

## Image Preprocessing

### Noise Reduction and Enhancement

```python
# Noise Removal
# Gaussian Noise
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Salt and Pepper Noise
def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy[coords] = 255

    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy[coords] = 0

    return noisy

# Denoising Techniques
fastnl = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
fastnl_gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
```

### Image Enhancement

```python
# Brightness and Contrast
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    alpha: contrast control (1.0-3.0)
    beta: brightness control (0-100)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Histogram Equalization
equalized = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_applied = clahe.apply(gray)

# Gamma Correction
def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Sharpening
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, sharpen_kernel)
```

### Geometric Transformations

```python
# Affine Transformation
points1 = np.float32([[50, 50], [200, 50], [50, 200]])
points2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(points1, points2)
affined = cv2.warpAffine(img, M_affine, (width, height))

# Perspective Transformation
points1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
points2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M_perspective = cv2.getPerspectiveTransform(points1, points2)
warped = cv2.warpPerspective(img, M_perspective, (300, 300))

# Translation
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M_translation, (width, height))

# Skewing
skewed = cv2.transform(img, M_affine)
```

### Thresholding Techniques

```python
# Global Thresholding
_, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

# Otsu's Method
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Multiple Thresholds
_, thresh_trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

# Adaptive Mean Threshold
thresh_adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
```

---

## Object Detection

### Template Matching

```python
# Template Matching
template = cv2.imread('template.jpg', 0)
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Draw Rectangle Around Best Match
top_left = max_loc
h, w = template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

# Multiple Template Matches
locations = np.where(res >= 0.8)
for pt in zip(*locations[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
```

### Contour Detection

```python
# Find Contours
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw Contours
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Contour Analysis
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Get Bounding Rectangle
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Get Rotated Rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(contour_img, [box], 0, (255, 0, 0), 2)

    # Convex Hull
    hull = cv2.convexHull(contour)
    cv2.drawContours(contour_img, [hull], -1, (255, 255, 0), 2)
```

### Corner Detection

```python
# Harris Corner Detection
gray_float = np.float32(gray)
corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
corners = cv2.dilate(corners, None)
img[corners > 0.01 * corners.max()] = [0, 0, 255]

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100,
                                 qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
```

### Feature Detection and Matching

```python
# SIFT Features
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw Keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ORB Features
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw ORB Keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None,
                                 color=(0, 255, 0), flags=0)

# Feature Matching
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# Sort and Draw Best Matches
matches = sorted(matches, key=lambda x: x.distance)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
```

---

## Face Recognition

### Basic Face Detection

```python
# Load Pre-trained Classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Detect Faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw Faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Detect Eyes in Face Region
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```

### Advanced Face Detection

```python
# Load DNN Face Detector
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Face Detection with DNN
def detect_faces_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX-startX, endY-startY))

    return faces

# Multi-scale Face Detection
faces_multi = face_cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
```

### Face Recognition Pipeline

```python
# Face Recognition with FaceNet-like Pipeline
import face_recognition

# Load Known Faces
known_faces = []
known_names = []

# Load and Encode Face
def load_face_encoding(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(name)

# Recognize Faces
def recognize_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
        face_distance = face_recognition.face_distance(known_faces, encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            names.append(known_names[best_match_index])
        else:
            names.append("Unknown")

    return face_locations, names

# Draw Recognition Results
def draw_results(image, locations, names):
    for (top, right, bottom, left), name in zip(locations, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
```

---

## Segmentation Techniques

### Thresholding-Based Segmentation

```python
# Simple Thresholding
_, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Adaptive Thresholding
adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# Otsu Thresholding
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### K-Means Clustering Segmentation

```python
# Color Segmentation with K-Means
def kmeans_segmentation(image, k=5):
    data = image.reshape((-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    return segmented_image, labels, centers

# Apply K-Means Segmentation
segmented, labels, centers = kmeans_segmentation(img, k=5)
```

### Watershed Algorithm

```python
# Watershed Segmentation
def watershed_segmentation(image):
    # Convert to grayscale and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]

    return markers

# Apply Watershed
watershed_result = watershed_segmentation(img.copy())
```

### GrabCut Algorithm

```python
# GrabCut Segmentation
def grabcut_segmentation(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Initialize rectangle for GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply mask to image
    result = image * mask2[:, :, np.newaxis]

    return result, mask2

# Example usage
rect = (x, y, width, height)  # Rectangle defining the object
grabcut_result, final_mask = grabcut_segmentation(img, rect)
```

### Mean Shift Segmentation

```python
# Mean Shift Filtering and Segmentation
def mean_shift_segmentation(image, spatial_radius=10, range_radius=10, min_density=100):
    shifted = cv2.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=range_radius)

    # Convert to grayscale and apply threshold
    gray_shift = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_shift, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw segmented regions
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result, shifted

# Apply Mean Shift Segmentation
segmented_image, mean_shift_result = mean_shift_segmentation(img)
```

---

## Computer Vision Pipeline Patterns

### Complete Object Detection Pipeline

```python
class ObjectDetectionPipeline:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def preprocess_image(self, image):
        # Resize image
        height, width = image.shape[:2]
        if max(height, width) > 800:
            scale = 800 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, gray

    def detect_objects(self, gray_image):
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

        detections = []
        for (x, y, w, h) in faces:
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_color = gray_image[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            detections.append({
                'face': (x, y, w, h),
                'eyes': eyes
            })

        return detections

    def draw_detections(self, image, detections):
        result = image.copy()

        for detection in detections:
            face = detection['face']
            x, y, w, h = face

            # Draw face rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw eyes
            for (ex, ey, ew, eh) in detection['eyes']:
                cv2.rectangle(result, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        return result

    def process_image(self, image):
        processed_img, gray_img = self.preprocess_image(image)
        detections = self.detect_objects(gray_img)
        result = self.draw_detections(processed_img, detections)

        return result, detections

# Usage
pipeline = ObjectDetectionPipeline()
result, detections = pipeline.process_image(input_image)
```

### Face Recognition Pipeline

```python
class FaceRecognitionPipeline:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.known_faces = []
        self.known_names = []

    def add_known_face(self, image_path, name):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        self.known_faces.append(encoding)
        self.known_names.append(name)

    def preprocess_face(self, image):
        # Face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Extract and align faces
        face_images = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_images.append(face_img)

        return face_images

    def recognize_faces(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            if self.known_faces:
                matches = face_recognition.compare_faces(self.known_faces, encoding)
                face_distance = face_recognition.face_distance(self.known_faces, encoding)
                best_match_index = np.argmin(face_distance)

                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distance[best_match_index]
                else:
                    name = "Unknown"
                    confidence = 0
            else:
                name = "Unknown"
                confidence = 0

            results.append({
                'name': name,
                'confidence': confidence,
                'location': location
            })

        return results

    def draw_recognition_results(self, image, results):
        result = image.copy()

        for face_info in results:
            top, right, bottom, left = face_info['location']
            name = face_info['name']
            confidence = face_info['confidence']

            # Draw face rectangle
            cv2.rectangle(result, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.rectangle(result, (left, top-25), (right, top), (0, 0, 255), cv2.FILLED)
            cv2.putText(result, label, (left+5, top-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def process_image(self, image):
        faces = self.preprocess_face(image)
        results = self.recognize_faces(image)
        result = self.draw_recognition_results(image, results)

        return result, results

# Usage
face_recognizer = FaceRecognitionPipeline()
face_recognizer.add_known_face('person1.jpg', 'Person 1')
face_recognizer.add_known_face('person2.jpg', 'Person 2')

result, recognition_results = face_recognizer.process_image(input_image)
```

### Image Processing Pipeline

```python
class ImageProcessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, name, func, *args, **kwargs):
        self.steps.append({
            'name': name,
            'func': func,
            'args': args,
            'kwargs': kwargs
        })

    def execute(self, image):
        result = image.copy()
        step_results = []

        for step in self.steps:
            try:
                if step['args'] or step['kwargs']:
                    step_result = step['func'](result, *step['args'], **step['kwargs'])
                else:
                    step_result = step['func'](result)

                step_results.append({
                    'step': step['name'],
                    'result': step_result,
                    'success': True
                })

                if step_result is not None:
                    result = step_result

            except Exception as e:
                step_results.append({
                    'step': step['name'],
                    'error': str(e),
                    'success': False
                })

        return result, step_results

# Predefined Processing Functions
def resize_image(image, width=None, height=None, scale=None):
    if scale:
        return cv2.resize(image, None, fx=scale, fy=scale)
    elif width and height:
        return cv2.resize(image, (width, height))
    return image

def enhance_contrast(image, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low_threshold, high_threshold)

def apply_threshold(image, thresh_value=127, max_value=255):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY)
    return thresh

# Usage Example
pipeline = ImageProcessingPipeline()

pipeline.add_step("Resize", resize_image, width=640, height=480)
pipeline.add_step("Enhance Contrast", enhance_contrast, alpha=1.3, beta=20)
pipeline.add_step("Apply Blur", apply_gaussian_blur, kernel_size=(3, 3))
pipeline.add_step("Edge Detection", detect_edges, low_threshold=30, high_threshold=100)

result, step_results = pipeline.execute(input_image)

for step_result in step_results:
    if step_result['success']:
        print(f"✓ {step_result['step']}: Success")
    else:
        print(f"✗ {step_result['step']}: {step_result['error']}")
```

### Video Processing Pipeline

```python
class VideoProcessingPipeline:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.processors = []

    def add_processor(self, processor_func):
        self.processors.append(processor_func)

    def process_frame(self, frame):
        result = frame.copy()

        for processor in self.processors:
            result = processor(result)
            if result is None:
                break

        return result

    def run(self, output_path=None, show_preview=True):
        cap = cv2.VideoCapture(self.video_source)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)

            if output_path:
                out.write(processed_frame)

            if show_preview:
                cv2.imshow('Video Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

# Example Video Processors
def face_detection_processor(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

def motion_detection_processor(image):
    global previous_frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if 'previous_frame' not in globals():
        previous_frame = gray
        return image

    frame_delta = cv2.absdiff(previous_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    previous_frame = gray
    return image

# Usage
video_processor = VideoProcessingPipeline('video.mp4')
video_processor.add_processor(face_detection_processor)
video_processor.add_processor(motion_detection_processor)

# Process video and save output
video_processor.run('output.mp4', show_preview=True)
```

### Real-time Camera Processing

```python
class CameraProcessor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.is_running = False
        self.frame_callback = None

    def set_frame_callback(self, callback):
        self.frame_callback = callback

    def start(self):
        self.is_running = True
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Camera started. Press 'q' to quit, 's' to save frame")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_callback:
                processed_frame = self.frame_callback(frame)
            else:
                processed_frame = frame

            cv2.imshow('Camera Feed', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'frame_{timestamp}.jpg', processed_frame)
                print(f"Frame saved as frame_{timestamp}.jpg")

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.is_running = False

# Example Camera Processing Functions
def face_recognition_camera(frame):
    face_recognizer = FaceRecognitionPipeline()
    result, recognition_results = face_recognizer.process_image(frame)
    return result

def object_tracking_camera(frame):
    # Simple object tracking implementation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return frame

# Usage
camera = CameraProcessor()
camera.set_frame_callback(face_recognition_camera)
camera.start()
```

---

## Performance Optimization Tips

### Memory Management

```python
import gc

# Release memory after processing
def process_large_image(image_path):
    image = cv2.imread(image_path)

    try:
        # Process image
        result = process_image_pipeline(image)
        return result
    finally:
        # Explicitly release memory
        del image
        gc.collect()

# Use generators for large datasets
def image_generator(image_paths):
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            yield image
        # Memory is freed when image goes out of scope

# Process images in batches
def process_images_batch(image_paths, batch_size=10):
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = []

        for path in batch:
            image = cv2.imread(path)
            processed = process_image(image)
            batch_results.append(processed)
            del image  # Free memory for this image

        yield batch_results

        # Force garbage collection between batches
        gc.collect()
```

### Parallel Processing

```python
import concurrent.futures
import threading

# Multi-threaded image processing
def process_image_parallel(image_paths, num_workers=4):
    results = []

    def process_single_image(path):
        image = cv2.imread(path)
        return process_image_pipeline(image)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(process_single_image, path): path
                         for path in image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Error processing {future_to_path[future]}: {exc}')

    return results

# GPU-accelerated processing (if OpenCV with CUDA is available)
def process_with_cuda(image):
    # Check if CUDA is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # Apply GPU operations
        gpu_blurred = cv2.cuda.bilateralFilter(gpu_image, 9, 75, 75)

        result = gpu_blurred.download()
        return result
    else:
        # Fallback to CPU processing
        return cv2.bilateralFilter(image, 9, 75, 75)
```

### Optimization Patterns

```python
# Use NumPy operations instead of loops
def optimized_pixel_processing(image):
    # Slow: Using loops
    # result = np.zeros_like(image)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         result[i, j] = process_pixel(image[i, j])

    # Fast: Vectorized operations
    result = np.vectorize(process_pixel)(image)
    return result

# Pre-allocate arrays
def preallocate_optimization(width, height, channels=3):
    # Pre-allocate memory for performance
    image = np.zeros((height, width, channels), dtype=np.uint8)
    result = np.zeros_like(image)
    return image, result

# Use in-place operations
def inplace_operations(image):
    # Avoid creating new arrays when possible
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=image)  # In-place conversion
    cv2.GaussianBlur(image, (5, 5), 0, dst=image)  # In-place blur
    return image

# Optimize contour operations
def optimized_contour_processing(binary_image):
    # Find contours on the smaller image
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area before expensive operations
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    # Process only relevant contours
    for contour in large_contours:
        # Calculate features once
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip expensive operations for small contours
        if area < 5000:
            continue

        # Apply expensive operations
        hull = cv2.convexHull(contour)
        # ... other operations
```

---

## Conclusion

This comprehensive computer vision cheat sheet covers the essential operations and techniques for image processing, object detection, face recognition, and segmentation. Use these patterns and code snippets as a reference for building robust computer vision applications.

### Key Points to Remember:

1. **Preprocessing is crucial** - Always prepare images before analysis
2. **Choose appropriate algorithms** - Different tasks require different approaches
3. **Optimize for performance** - Use vectorization and parallel processing
4. **Handle edge cases** - Check for empty results and unexpected inputs
5. **Use pipeline patterns** - Structure code for reusability and maintainability

### Recommended Tools and Libraries:

- **OpenCV** - Core computer vision operations
- **face_recognition** - Simplified face recognition
- **scikit-image** - Additional image processing algorithms
- **NumPy** - Efficient array operations
- **scipy** - Scientific computing and advanced filtering

### Performance Considerations:

- Always resize large images for processing
- Use appropriate data types (uint8 for images)
- Implement proper memory management
- Consider GPU acceleration when available
- Use parallel processing for batch operations

Keep this cheat sheet handy for quick reference while developing computer vision applications!
