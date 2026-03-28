# Human Motion Detection

An object-detection style project that detects human motion and objects in a room using YOLOv8 and computer vision techniques. Trained on multiple custom datasets including safety helmets, weapons, and people.

## Overview

- **Model:** YOLOv8n (Ultralytics)
- **Datasets:** Safety helmets, weapons (guns/knives/grenades), person+weapon combinations
- **Goal:** Detect and localise objects in images/video using YOLO-format annotated datasets

## Project Structure

```
human-motion-detection/
├── datasets/
│   ├── helmet_data.yaml    # Safety helmet dataset config (1 class: Helmets)
│   ├── weapon_data.yaml    # Weapon dataset config (6 classes)
│   └── wpn_data.yaml       # Person+weapon dataset config (2 classes)
├── rename.py               # Utility to rename dataset images sequentially
├── requirements.txt
└── README.md
```

## Dataset Download

The image datasets are not included in this repo due to size.  
Download them from Roboflow using the links in each YAML file:

| Dataset | Classes | Roboflow Link |
|---------|---------|---------------|
| Safety Helmet | `Helmets` | https://universe.roboflow.com/vincent-tay-2aion/safety-helmet-kurbz/dataset/3 |
| Weapons | `Grenade, Gun, Knife, Pistol, handgun, rifle` | https://universe.roboflow.com/testing-kfsrv/guns-l4rap/dataset/3 |

After downloading, place each dataset so the structure matches:
```
datasets/
└── helmet/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a model
```bash
yolo detect train data=datasets/helmet_data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 3. Run prediction on new images
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/your/images save=True
```

### 4. Run prediction on webcam (live)
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0 show=True
```

## Training Results

After 50 epochs on the helmet dataset, the model learns to detect safety helmets with improving mAP scores. Results are saved to `runs/detect/`.

## Tech Stack

- Python
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Roboflow datasets
