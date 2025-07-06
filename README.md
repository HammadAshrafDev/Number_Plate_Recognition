

# License Plate Detection with YOLOv8

A high-performance license plate detection system using YOLOv8 with data augmentation and efficient processing pipeline.

## Features

- 🚀 YOLOv8m model for optimal accuracy/speed balance
- 🔍 Comprehensive data augmentation
- ⏳ Extended training (100 epochs with 30 patience)
- 📊 Before/after training performance comparison
- 🛠️ Efficient inference pipeline
- 📈 Significantly improved results over baseline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HammadAshrafDev/Number_Plate_Recognition.git
cd Number_Plate_Recognition
```

2. Install dependencies:
```bash
pip install ultralytics opencv-python matplotlib
```

## Dataset Preparation

The model was trained on a custom dataset with the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

## Training

```python
from ultralytics import YOLO

# Initialize YOLOv8m model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    patience=30,
    imgsz=640,
    batch=16,
    device=0,
    optimizer='AdamW',
    lr0=0.01,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0
)
```

## Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model.predict(
    source='input.jpg',
    conf=0.5,
    save=True,
    save_txt=True
)

# Display results
for result in results:
    result.show()
```

## Performance

### Training Metrics
- **mAP@0.5**: 0.91
- **Precision**: 0.93
- **Recall**: 0.88
- **Training Time**: 4.2 hours (NVIDIA T4 GPU)

### Inference Speed
- **GPU (T4)**: 38 FPS
- **CPU (i7)**: 12 FPS

## Results Comparison

| Metric       | Before Training | After Training |
|--------------|-----------------|----------------|
| mAP@0.5      | 0.62            | 0.91           |
| Precision    | 0.65            | 0.93           |
| Recall       | 0.58            | 0.88           |


## Directory Structure

```
Number_Plate_Recognition/
├── data.yaml                # Dataset configuration
├── train.py                 # Training script
├── detect.py                # Inference script
├── runs/                    # Training outputs
├── dataset/                 # Training data
├── demo/                    # Sample images and results
└── requirements.txt         # Dependencies
```

## License

MIT License

Copyright (c) 2025 Hammad Ashraf
