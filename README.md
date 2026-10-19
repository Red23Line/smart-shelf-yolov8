# 🛒 Smart Shelf Monitoring System using YOLOv8

An automated computer vision system built with **Ultralytics YOLOv8** to monitor retail shelf inventory, detect product placement, and identify out-of-stock (`empty_shelf`) areas in real time.

---

## 📌 Features
- **Object Detection**: Detects both `product` items and `empty_shelf` gaps.
- **Few-Shot Learning & Augmentation**: Trained on custom shelf imagery with data augmentation (mosaic, rotation, scaling, HSV adjustment).
- **Inference Optimization**: Filtered detection using tuned Confidence (`conf=0.65`) and IoU (`iou=0.30`) thresholds to eliminate double bounding boxes.
- **Inventory Analytics & Alerting**: Includes automated stock evaluation (`analytics.py`) to calculate shelf fill-rate percentage and trigger **Restock Alerts**.

---

## 📁 Project Structure
```text
smart-shelf/
├── dataset/
│   ├── images/
│   │   └── train/          # Training images (shelf_full.jpg, shelf_empty.jpg)
│   ├── labels/
│   │   └── train/          # YOLO format annotation files (.txt)
│   └── data.yaml           # Dataset configuration mapping
├── train.py                # Model training script with custom augmentation
├── predict.py              # Filtered inference script for clean bounding boxes
├── analytics.py            # Inventory report & automated restock alert logic
├── .gitignore              # Ignored large files (runs/, weights, cache)
└── README.md               # Project documentation
```
🛠️ Requirements & Installation
1. Python 3.12+
2. Install required packages:
  pip install ultralytics torch

🚀 How to Run
1. Training the Model
To start training YOLOv8n on the custom shelf dataset:
  python train.py
2. Running Inference (Detection Visuals)
To generate bounding box predictions on test images:
  python predict.py
Output images with detection boxes will be saved to runs/predict_test/.

⚙️ Dataset Configuration (data.yaml)
path: C:/Games/smart_shelf/dataset
train: images/train
val: images/train

names:
  0: product
  1: empty_shelf

👤 Author
Developed by Red23Line
