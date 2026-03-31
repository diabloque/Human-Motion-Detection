# Human Motion Detection

Standalone YOLOv8 project for motion/object-style detection with custom datasets
(helmet, weapon, and person+weapon classes).

## Overview

- **Model family:** YOLOv8 (Ultralytics)
- **Task:** object detection for room/scene monitoring
- **Datasets:** custom YOLO-format datasets
- **Entry scripts:** `train.py` and `predict.py`

## Project Structure

```text
human-motion-detection/
├── datasets/
│   ├── helmet_data.yaml
│   ├── weapon_data.yaml
│   └── wpn_data.yaml
├── results/
│   ├── README.md
│   └── sample_predictions/
├── predict.py
├── train.py
├── rename.py
├── requirements.txt
└── README.md
```

## Dataset Setup

This repo excludes raw image/label datasets due to size. Download datasets from
the links in YAML metadata and place them locally with this structure:

```text
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

Do the same for `weapon` and `wpn` datasets.

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train

Helmet training (default):

```bash
python train.py --data datasets/helmet_data.yaml --epochs 50 --imgsz 640 --batch 8 --name helmet_50ep
```

Weapon training:

```bash
python train.py --data datasets/weapon_data.yaml --epochs 50 --imgsz 640 --batch 8 --name weapon_50ep
```

### 3) Predict on folder/images/video

```bash
python predict.py --model runs/helmet_50ep/weights/best.pt --source path/to/images --name helmet_predict
```

### 4) Predict from webcam

```bash
python predict.py --model runs/helmet_50ep/weights/best.pt --source 0 --show --name webcam_predict
```

## One-Command Demo Scripts

From repo root:

### PowerShell (Windows)

```powershell
# Predict on sample images
.\demo.ps1 -Mode predict -RunName demo_predict

# Train a new run
.\demo.ps1 -Mode train -RunName demo_train

# Live webcam
.\demo.ps1 -Mode webcam -Model "runs/helmet_50ep/weights/best.pt" -RunName demo_webcam
```

### Bash (Linux/macOS/Git Bash)

```bash
# Predict on sample images
bash demo.sh predict

# Train a new run
bash demo.sh train

# Live webcam
bash demo.sh webcam
```

## Output Locations

- Training outputs: `runs/<run_name>/`
- Best weights: `runs/<run_name>/weights/best.pt`
- Predictions: `runs/<predict_name>/`
- Sample qualitative outputs committed in: `results/sample_predictions/`

## Notes

- `rename.py` is a helper script for bulk-renaming dataset files.
- If you want to version `.pt` weights in GitHub, use Git LFS.
- For better accuracy, train longer or use a larger model (`yolov8s.pt`, `yolov8m.pt`).
