# Data

This repository does not redistribute the original NEU-DET and GC10-DET datasets.

Please download the datasets from their official/public sources and convert them into YOLO format.

Recommended directory structure:

data/
└── industrial_defect_yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── industrial_defect.yaml

The `industrial_defect.yaml` file should define train/val/test image paths and class names in YOLO format.
