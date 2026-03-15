# Phase 3: Multi-Class Fusion & Anatomical Correction

## Overview
Phase 3 fuses the 10 single-class outputs into one final multi-class segmentation and then applies a small set of post-fusion anatomical cleanup rules.
The maintained implementation is centered on `phase3.py`.

## Fusion Strategy

Fusion is priority-based.
The current priority order in code is:

```text
Coronary > PV > LAA > LV/RV/LA/RA > Myocardium > Aorta > PA
```

The implementation writes lower-priority classes first and lets higher-priority classes overwrite them.

## Anatomical Rules Currently Implemented

After fusion, the current code applies the following rules:

1. `PV -> LA` connectivity enforcement
2. `LAA -> LA` connectivity enforcement
3. coronary attachment filtering against `Myocardium ∪ Aorta`
4. chamber fragment cleanup for `LA`, `LV`, `RA`, and `RV`

These rules are intentionally limited.
The current code does not implement a broader generic rule engine.

## Inputs and Outputs

For each case, `phase3.py` loads masks in this order:

1. `${NEAR_DATA_ROOT}/{class}_morph/{case_id}_mask.npy`
2. `${NEAR_DATA_ROOT}/{class}_global/{case_id}_mask.npy`

Outputs are written to the selected Phase3 output directory, usually:

```text
${NEAR_DATA_ROOT}/repaired_phase3/
```

Each processed case produces:

- `{case_id}_phase3.npy`
- `{case_id}_phase3.nii.gz`

## Evaluation

`evaluate_repair_quality.py` compares Phase1, Phase2, and Phase3 outputs against GT on a per-class basis.
The current script computes:

- Dice
- HD95
- ASD
- predicted and GT volumes
- connected-component counts
- Phase1 -> Phase2 and Phase2 -> Phase3 volume change ratios

If a Phase2 output is missing for a class, the evaluator falls back to the corresponding Phase1 mask for that class.

## Usage

### Run Phase 3 Fusion

```bash
sbatch scripts/hpc/phase3/run_phase3_sbatch.sh
```

### Run Evaluation

```bash
sbatch scripts/hpc/phase3/run_evaluation_sbatch.sh
```

### Visualization

```bash
python tools/vis/phase3_visualize_phase3.py --output_dir output_vis --data_root /path/to/dataset
```
