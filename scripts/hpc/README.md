# HPC Scripts

This directory contains the SLURM, Apptainer, and cluster-environment scripts that are still worth keeping.

- `data_prepare/`: Tier2 data preparation and `crop_params` generation
- `phase1/`: training, inference, and bulk submission
- `phase2/`: morphological cleanup
- `phase3/`: multi-class fusion and evaluation

Notes:

- All maintained scripts share environment configuration through `common.sh`, so private cluster paths, SLURM account names, and container paths are no longer hardcoded.
- This directory only documents the wrappers that are still present in the cleaned repository.

Common environment variables:

- `NEAR_REPO_ROOT`: repository root, inferred automatically by default
- `NEAR_DATA_ROOT`: dataset root, default `${NEAR_REPO_ROOT}/dataset`
- `NEAR_OUTPUT_ROOT`: output root, default `${NEAR_REPO_ROOT}/outputs`
- `NEAR_CONTAINER`: optional Apptainer image path; if unset, the scripts use the local Python interpreter
- `NEAR_LOGGER`: `csv`, `wandb`, or `none`
- `NEAR_GT_ROOT`: ground-truth root used by Phase3 evaluation

Example:

```bash
export NEAR_DATA_ROOT=/path/to/dataset
export NEAR_OUTPUT_ROOT=/path/to/outputs
export NEAR_CONTAINER=/path/to/pytorch.sif
sbatch scripts/hpc/phase1/run_class_sbatch.sh coronary
```
