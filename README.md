# CIVICLEAR CV Training Repo — Fixed for Nested Roboflow/COCO Folders

This repo is for training the CIVICLEAR computer vision model. It is designed for your Google Drive dataset structure:

```text
CIVICLEAR DATASETS/
├── construction materials/
│   ├── sidewalk construction.coco/
│   └── Construction Debris.coco/
├── garbage debris/
│   └── Garbage - Litter Detector.coco/
├── illegal parking/
│   ├── illegal parking.coco/
│   └── Illegal Parking Detection.coco/
├── no violation/
│   ├── Selfie.coco/
│   └── Bdd100k.coco/
├── road obstruction/
│   ├── Road Debris.coco/
│   └── Fallen Tree.coco/
└── sidewalk obstruction/
    ├── traffic cone.coco/
    ├── Barricade.coco/
    ├── sidewalk.coco-segmentation/
    ├── Utility Poles.coco/
    └── pedestrian-obstruction-detection.coco/
```

## Important

Your previous issue happened because the script probably expected this simple structure:

```text
sidewalk obstruction/
├── train/
├── valid/
└── test/
```

But your real dataset has nested Roboflow/COCO folders inside each parent class. This repo scans recursively, so it should include folders like `Barricade.coco`, `traffic cone.coco`, `Utility Poles.coco`, and other nested datasets.

## What this trains

This trains a YOLO detection model with these final classes:

```text
0 construction_materials
1 garbage_debris
2 illegal_parking
3 road_obstruction
4 sidewalk_obstruction
```

`no violation` is treated as background/negative images, so it creates empty label files. This helps the model learn when no violation is present.

## Files

```text
count_dataset.py                    Check recursive counts before training
prepare_nested_detection_dataset.py Merge nested COCO/YOLO datasets into one YOLO dataset
train.py                            Train YOLO
test_model.py                       Test best.pt and save predictions
requirements.txt                    Python dependencies
.gitignore                          Prevent datasets/models from being pushed
```

## Google Colab Guide

### Step 1 — Upload repo to GitHub or Google Drive

Upload this repo. Do not upload the dataset into GitHub.

### Step 2 — Add dataset shortcut to MyDrive

Your classmate should open the shared dataset folder in Google Drive, then:

```text
Right click CIVICLEAR DATASETS → Organize → Add shortcut → My Drive
```

The expected path in Colab is:

```text
/content/drive/MyDrive/CIVICLEAR DATASETS
```

### Step 3 — Install requirements

```python
!pip install -q -r requirements.txt
```

Or directly:

```python
!pip install -q ultralytics pyyaml pandas
```

### Step 4 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5 — Count dataset first

Run this before training:

```bash
python count_dataset.py
```

Check that all classes appear:

```text
construction materials
garbage debris
illegal parking
no violation
road obstruction
sidewalk obstruction
```

If `sidewalk obstruction` still shows only 2 images, stop. The dataset path is wrong or the script cannot access all Drive folders.

### Step 6 — Prepare merged YOLO dataset

```bash
python prepare_nested_detection_dataset.py --reset --make-valid-if-missing
```

This creates:

```text
/content/drive/MyDrive/CIVICLEAR_MERGED_YOLO/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
├── test/labels
└── data.yaml
```

### Step 7 — Train

Start with YOLOv8n:

```bash
python train.py --epochs 50 --imgsz 640 --batch 16
```

If GPU memory error:

```bash
python train.py --epochs 50 --imgsz 640 --batch 8
```

or:

```bash
python train.py --epochs 50 --imgsz 640 --batch 4
```

### Step 8 — Test model

```bash
python test_model.py
```

## Expected output

Your classmate should send back:

```text
best.pt
last.pt
results.png
confusion_matrix.png
F1_curve.png
PR_curve.png
sample prediction images
```

Most important file:

```text
best.pt
```

## Do not push to GitHub

Do not push:

```text
CIVICLEAR DATASETS/
CIVICLEAR_MERGED_YOLO/
runs/
*.pt
*.zip
*.jpg
*.png
```

These are ignored in `.gitignore`.

## Message for classmate

```text
Bro, use this fixed CV training repo. The dataset has nested Roboflow/COCO folders, especially inside sidewalk obstruction, so please run count_dataset.py first.

If sidewalk_obstruction shows only 2 images, stop and tell me. Do not train yet.

If the count is okay, run prepare_nested_detection_dataset.py, then train.py. After training, send me best.pt, results.png, confusion_matrix.png, PR_curve.png, and sample prediction images.
```
