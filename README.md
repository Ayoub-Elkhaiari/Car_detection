# YOLO Car Detection Implementation

This repository contains an implementation of the YOLO (You Only Look Once) object detection algorithm using TensorFlow, specifically optimized for car detection in images. This implementation processes images to detect and localize vehicles with bounding boxes and confidence scores.

## Overview

The system uses a YOLO-based neural network to detect cars in images. It provides accurate car detection with bounding box coordinates and confidence scores, making it suitable for various automotive applications, traffic monitoring, and parking management systems.

## Features

- Specialized car detection using YOLO architecture
- Bounding box generation around detected vehicles
- Confidence scoring for each detection
- Non-max suppression to prevent duplicate detections
- Real-time processing capabilities
- Visualization of detection results

## Dependencies

```
- tensorflow
- numpy
- pandas
- PIL (Python Imaging Library)
- matplotlib
- scipy
- YAD2K (Yet Another Darknet 2 Keras)
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install tensorflow numpy pandas pillow matplotlib scipy
```
3. Ensure you have the following model data files in the `model_data` directory:
   - `coco_classes.txt`: Contains class names
   - `yolo_anchors.txt`: Contains anchor box coordinates
   - YOLO model weights

## Usage

The main prediction function can be used as follows:

```python
# Import the prediction function
from yolo_detection import predict

# Run prediction on an image
out_scores, out_boxes, out_classes = predict("test.jpg")
```

The function will:
1. Load and preprocess the input image
2. Run car detection
3. Draw bounding boxes around detected vehicles
4. Return detection scores, boxes, and classes

## Key Components

### 1. `yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6)`
Filters detection boxes based on confidence scores, specially tuned for car detection.

### 2. `iou(box1, box2)`
Calculates the Intersection over Union between detected car boxes.

### 3. `yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5)`
Removes overlapping car detections to prevent duplicate counts.

### 4. `yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5)`
Processes YOLO model output to generate final car predictions.

### 5. `predict(image_file)`
Main function for running car detection on images.

## Model Configuration

- Input Image Size: 608 x 608 pixels
- Default max boxes: 10
- Default score threshold: 0.3
- Default IoU threshold: 0.5
- Optimized for vehicle detection

## Directory Structure

```
.
├── model_data/
│   ├── coco_classes.txt
│   ├── yolo_anchors.txt
│   └── model weights
└── images/
    └── (input images)
```

## Output Format

The prediction function returns three values:
- `out_scores`: Confidence scores for each car detection
- `out_boxes`: Coordinates of car bounding boxes [y_min, x_min, y_max, x_max]
- `out_classes`: Class indices for each detection

## Customization

You can modify the following parameters to optimize for your specific use case:
- `score_threshold`: Minimum confidence score for car detection
- `iou_threshold`: IoU threshold for overlapping car detections
- `max_boxes`: Maximum number of cars to detect in each image
- `model_image_size`: Input image size for the model

## Notes

- The model is specifically optimized for car detection
- Images should be placed in the `images` directory
- The model uses pre-trained weights optimized for vehicle detection
- Best results are achieved with clear, well-lit images

## Results

![Screenshot 2024-11-09 224509](https://github.com/user-attachments/assets/a195abd7-3785-4b39-8c61-d8917c9e20d7)


