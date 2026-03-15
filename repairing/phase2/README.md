# Phase 2: Morphological Processing & Cleaning

## Overview
Phase 2 takes the per-class global masks produced by Phase 1 and applies simple morphological cleanup before multi-class fusion.
The maintained implementation is centered on `perform_morphology_v2.py`.

## What the Current Code Does

For one target class at a time, the script:

1. Applies binary closing
2. Optionally fills holes for selected large-organ classes
3. Runs connected-component filtering with a per-class `top_k` policy
4. Applies a small number of class-specific post-processing rules

The current special handling is limited to:

- `Myocardium`: an extra closing step after component filtering
- `Coronary`: an extra closing step to reconnect thin broken branches

## Inputs and Outputs

Input files are expected under one class-specific directory, typically:

```text
${NEAR_DATA_ROOT}/{class}_global/{case_id}_mask.npy
```

Outputs are written to the corresponding morphology directory, typically:

```text
${NEAR_DATA_ROOT}/{class}_morph/
```

Each run may produce:

- cleaned `*_mask.npy` files
- `{case_id}_clean.nii.gz` visualization volumes
- `morphology_stats.csv`

## Per-Class Strategy

The exact behavior comes from the `CONFIG` table in `perform_morphology_v2.py`.
In broad terms:

- large organs use radius-2 closing and usually keep the largest 1 or 2 components
- coronary keeps the largest 2 components
- pulmonary veins keep the largest 4 components
- fine structures such as LAA do not use hole filling

## Scripts

- `perform_morphology_v2.py`: main morphology and connected-component cleanup script
- `calculate_phase2_cc.py`: helper script to inspect connected-component counts
- `calculate_phase2_ratios.py`: helper script to inspect size ratios across saved outputs

## Cluster Usage

The maintained HPC wrapper is:

```bash
sbatch scripts/hpc/phase2/run_phase2_sbatch.sh
```

This wrapper processes one class per array task and reads/writes the standard `{class}_global` and `{class}_morph` directories under `NEAR_DATA_ROOT`.
