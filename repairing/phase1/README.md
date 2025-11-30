# Phase 1: Shape Prior Learning with NeAR

## Overview
In Phase 1, we train a **Shape-Only NeAR (Neural Surface Reconstruction)** model for each of the 10 cardiac classes. The goal is to learn a smooth, continuous implicit surface representation for each structure, effectively removing the aliasing and noise present in the original manual annotations.

## Key Features
- **Class-Specific Training**: Separate models for LV, RV, LA, RA, Myocardium, Aorta, PA, LAA, Coronary, and PV.
- **Biased Sampling**: We use a boundary-biased sampling strategy (50% boundary, 50% uniform) to ensure fine structures like coronary arteries are captured accurately.
- **Dynamic Scheduling**: The boundary bias is gradually reduced during training to help the model generalize from "boundary memorization" to "shape learning".

## Directory Structure
```
phase1/
├── configs/          # Configuration files for each class (e.g., LV.py, Coronary.py)
├── checkpoints/      # Saved model weights
├── results/          # Inference results (probability maps/masks)
├── train.py          # Unified training script
├── lightning_module.py # PyTorch Lightning model definition
└── run_job.sh        # Helper script to launch training
```

## Usage

### Training
To train a model for a specific class (e.g., Left Ventricle):

```bash
# Syntax: ./run_job.sh <config_name> [num_gpus]
./run_job.sh LV 4
```

This will:
1. Load the configuration from `configs/LV.py`.
2. Train the NeAR model using `train.py`.
3. Save checkpoints to `checkpoints/`.

### Inference
Use the unified inference script (located in `phase1/` or root `inference_and_evaluate.py` if applicable) to generate probability maps from the trained models.
