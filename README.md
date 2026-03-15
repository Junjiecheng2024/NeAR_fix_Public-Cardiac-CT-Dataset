# NeAR Fix Public Cardiac CT Dataset

This repository contains a NeAR-based repair pipeline for public cardiac CT annotations.
The maintained workflow is a three-stage single-class repair pipeline built around the final `shape + appearance + context` implementation.

## What This Repository Does

The project is designed to refine existing labels rather than train a standard fully feed-forward segmenter for unseen cases.

The current mainline workflow is:

1. Prepare per-class Tier2 crops from the original cardiac CT dataset
2. Train a single-class NeAR model with shape, CT appearance, and anatomical context
3. Run single-class inference and map predictions back to a global volume
4. Apply per-class morphological cleaning
5. Fuse all classes and enforce a small set of anatomical consistency rules

## Current Mainline

The core files for the maintained pipeline are:

- `data_prepare/prepare_all_classes_tier2.py`
- `repairing/phase1/config.py`
- `repairing/phase1/train.py`
- `repairing/phase1/inference.py`
- `repairing/phase2/perform_morphology_v2.py`
- `repairing/phase3/phase3.py`
- `repairing/phase3/evaluate_repair_quality.py`

## Repository Layout

```text
data_prepare/   Data preparation utilities for Tier2 crops
docs/archive/   Archived implementation notes rewritten against the current code
near/           Dataset, model, loss, and utility code
repairing/      Main Phase1 / Phase2 / Phase3 pipeline code
scripts/hpc/    Portable SLURM / Apptainer wrappers for the current pipeline
surface_distance/ Vendored surface-distance metric implementation
tools/          Visualization and debugging helpers
```

## Data Expectations

The repository expects the original dataset to contain CT images and multi-class segmentations.
The maintained Tier2 format used by Phase1 stores one directory per case with files such as:

- `ct.npy`
- `mask_target.npy`
- `mask_context.npy`
- `crop_params.json`

By default, scripts assume data lives under:

- `${NEAR_DATA_ROOT}/original/images`
- `${NEAR_DATA_ROOT}/original/segmentations`

If `NEAR_DATA_ROOT` is not set, the code defaults to `./dataset` relative to the repository root.

## Environment

This repository is still research code.
A minimal dependency list is provided in `requirements.txt`, but the environment is not yet fully pinned for exact reproduction across machines.

Install the baseline Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

At minimum, the current mainline relies on:

- `python`
- `torch`
- `pytorch-lightning`
- `numpy`
- `scipy`
- `scikit-image`
- `nibabel`
- `pandas`
- `tqdm`
- `connected-components-3d`

Optional:

- `wandb` for experiment tracking
- `apptainer` for HPC execution

## Quick Start

### 1. Set data and output roots

```bash
export NEAR_DATA_ROOT=/path/to/dataset
export NEAR_OUTPUT_ROOT=/path/to/outputs
```

### 2. Prepare Tier2 data

```bash
python3 data_prepare/prepare_all_classes_tier2.py \
  --all \
  --images_dir "${NEAR_DATA_ROOT}/original/images" \
  --labels_dir "${NEAR_DATA_ROOT}/original/segmentations" \
  --output_dir "${NEAR_DATA_ROOT}" \
  --target_resolution 128
```

### 3. Train one class

```bash
python3 repairing/phase1/train.py \
  --config repairing/phase1/config.py \
  --class_name coronary \
  --devices 1 \
  --logger csv
```

### 4. Run Phase1 inference

```bash
python3 repairing/phase1/inference.py \
  --config repairing/phase1/config.py \
  --class_name coronary \
  --checkpoint /path/to/best.ckpt \
  --output_dir "${NEAR_DATA_ROOT}/coronary_global" \
  --no_sliding_window \
  --inference_resolution 128 \
  --global_shape 256
```

### 5. Run Phase2 morphology for one class

```bash
python3 repairing/phase2/perform_morphology_v2.py \
  --input_dir "${NEAR_DATA_ROOT}/coronary_global" \
  --output_dir "${NEAR_DATA_ROOT}/coronary_morph" \
  --target_class 9 \
  --ref_dir "${NEAR_DATA_ROOT}/original"
```

### 6. Run Phase3 fusion and evaluation

```bash
python3 repairing/phase3/phase3.py \
  --data_root "${NEAR_DATA_ROOT}" \
  --output_dir "${NEAR_DATA_ROOT}/repaired_phase3"

python3 repairing/phase3/evaluate_repair_quality.py \
  --data_root "${NEAR_DATA_ROOT}" \
  --gt_root "${NEAR_DATA_ROOT}/original/segmentations" \
  --output_csv "${NEAR_OUTPUT_ROOT}/phase3/evaluation_results_full.csv" \
  --skip_hd95
```

## HPC Usage

The maintained cluster wrappers live in `scripts/hpc/`.
They now share a common environment file and can be configured through environment variables instead of hardcoded CSC-specific paths.

Useful variables:

- `NEAR_REPO_ROOT`
- `NEAR_DATA_ROOT`
- `NEAR_OUTPUT_ROOT`
- `NEAR_PHASE1_CHECKPOINT_ROOT`
- `NEAR_CONTAINER`
- `NEAR_LOGGER`
- `NEAR_GT_ROOT`

Example:

```bash
export NEAR_DATA_ROOT=/path/to/dataset
export NEAR_OUTPUT_ROOT=/path/to/outputs
export NEAR_CONTAINER=/path/to/pytorch.sif
sbatch scripts/hpc/phase1/run_class_sbatch.sh coronary
```

## Notes and Limitations

- The maintained Phase1 model uses one learnable latent embedding per sample index.
- Validation in Phase1 is reconstruction-style evaluation on the same case set with augmentation disabled, not a separate held-out benchmark.
- The repository is being cleaned for open-source release, so paths and packaging are still being simplified.

## Additional Documentation

- `repairing/phase1/README.md`
- `repairing/phase2/README.md`
- `repairing/phase3/README.md`
- `docs/archive/`

## Upstream Acknowledgment

This repository was adapted from the NeAR project by HINTLab:
<https://github.com/HINTLab/NeAR>

The original NeAR project introduced Neural Annotation Refinement for medical annotation repair.
This repository restructures and extends that codebase for cardiac CT annotation repair, including cardiac-specific data preparation, the maintained Shape + Appearance + Context workflow, and the current Phase2/Phase3 post-processing pipeline.

The upstream NeAR repository is distributed under the Apache-2.0 license.
If this project is useful in academic work, please also cite the original NeAR paper and repository.

## License

This repository is released under the license included in `LICENSE`.
Vendored third-party code is documented in `THIRD_PARTY.md`.
