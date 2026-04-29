# metal-defect-threshold-calibration
Zero-FLOPs operating-point calibration for lightweight YOLOv8 on metal surface defect detection. Five-seed evaluation on merged NEU-DET and GC10-DET, comparing baseline, P2, and CLAHE variants. Includes training, threshold sweep, statistics, and reproducible scripts for industrial inspection.
# Zero-FLOPs Operating-Point Calibration for Lightweight YOLOv8 in Metal Surface Defect Detection

This repository contains the code and configuration files for the paper:

**Zero-FLOPs Operating-Point Calibration for Lightweight YOLOv8 in Metal Surface Defect Detection**

## Overview

This project studies industrial metal surface defect detection using a lightweight YOLOv8-based detector. The main idea is to improve deployment robustness by calibrating the confidence threshold on the validation split, without adding any inference modules or computational cost.

## Key points

- Zero FLOPs increase
- Five-seed evaluation
- Validation-only threshold selection
- Test-only final evaluation
- Comparison with P2 and P2 + CLAHE exploratory controls

## Repository structure

```text
configs/    # YAML configs for baseline, P2, and P2+CLAHE
scripts/    # Training, evaluation, statistics, and plotting scripts
data/       # Data preparation notes
results/    # Output tables and seed-level results
paper/      # Manuscript and supplementary material
