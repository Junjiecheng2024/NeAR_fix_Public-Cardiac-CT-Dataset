# NeAR v2.0 Phase1 - Shape + Appearance Training

This directory contains the NeAR v2.0 Phase1 training code. The maintained mainline uses the **Shape + Appearance** model to segment coronary arteries and other small cardiac structures.

## Directory Layout

```
phase1/
├── config.py              # Unified configuration based on dataclasses
├── train.py               # Main training entry point
├── inference.py           # Inference script
├── lightning_module.py    # PyTorch Lightning module
└── README.md
```

The maintained HPC submission scripts are located in:

```
scripts/hpc/phase1/
├── run_class_sbatch.sh      # Train one class
├── run_inference_sbatch.sh  # Run inference for one class
├── submit_all_training.sh   # Submit all training jobs
└── submit_all_inference.sh  # Submit all inference jobs
```

Supporting files have been reorganized into:

- Archived documentation: `docs/archive/`
- Visualization tools: `tools/vis/`
- Debug scripts: `tools/debug/`

## Common Environment Variables

| Variable | Purpose |
|------|------|
| `NEAR_DATA_ROOT` | Dataset root. Defaults to `dataset/` under the repository root. |
| `NEAR_OUTPUT_ROOT` | Output root. Defaults to `outputs/` under the repository root. |
| `NEAR_PHASE1_CHECKPOINT_ROOT` | Phase1 checkpoint root. Defaults to `outputs/phase1/checkpoints/`. |
| `NEAR_LOGGER` | Logging backend. Supported values: `csv`, `wandb`, `none`. |
| `NEAR_CONTAINER` | Optional Apptainer image path. |

## Data Format

Each sample directory contains:
```
sample_xxx/
├── crop_params.json    # Crop parameters used for mapping back to global space
├── ct.npy              # CT volume (D, H, W)
├── mask_target.npy     # Binary mask for the current target class
├── mask_context.npy    # Context mask (for example Myo + Aorta)
└── seg_full.npy        # Full segmentation
```

## Quick Start

### 1. SLURM Submission

```bash
export NEAR_DATA_ROOT=/path/to/dataset
export NEAR_OUTPUT_ROOT=/path/to/outputs
sbatch scripts/hpc/phase1/run_class_sbatch.sh coronary
```

### 2. Local Debugging

```bash
cd repairing/phase1
python train.py --config config.py --devices 1
```

## Configuration System

Configurations are implemented with Python dataclasses:

```python
from config import CoronaryConfig, get_config

# Option 1: instantiate directly
cfg = CoronaryConfig()

# Option 2: fetch by class name
cfg = get_config("coronary")
```

Supported classes: `coronary`, `aorta`, `myocardium`, `la`, `lv`, `ra`, `rv`, `pa`, `pv`, `laa`

## Model Architecture

```
Shape + Appearance Model
├── Latent Embedding (per-sample)
├── Appearance Encoder (CT feature extraction)
├── Context Encoder (optional, e.g. Myo+Aorta mask)
└── Implicit Decoder (fused decoding)
```

## Hyperparameters

| Parameter | Default | Description |
|------|--------|------|
| `n_epochs` | 600 | Number of training epochs |
| `batch_size` | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `lr` | 5e-4 | Learning rate |
| `dice_weight` | 0.30 | Dice loss weight |
| `boundary_dice_weight` | 0.20 | Boundary Dice loss weight |
| `focal_weight` | 0.10 | Focal loss weight |

## Training Monitoring

CSV logging is used by default.

To enable WandB:

- Set `NEAR_LOGGER=wandb`
- Provide `WANDB_API_KEY` through the environment, or configure offline mode yourself

To inspect logs:
```bash
tail -f outputs/logs/<job>.out
```

## Inference

```bash
python inference.py \
    --config config.py \
    --checkpoint /path/to/best.ckpt \
    --output_dir /path/to/output
```
