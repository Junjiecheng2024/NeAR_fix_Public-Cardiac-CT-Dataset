# Debug Tools

This directory contains debugging and troubleshooting scripts:

- `phase1_inference_single_case.py`: run Phase1 inference for one case and report per-class Dice scores
- `phase3_run_case.py`: rerun Phase3 for one case and save intermediate fusion outputs
- `diagnose_dice.py`: directly compare saved predictions against GT Dice calculations
- `diagnose_evaluation.py`: inspect GT labels and verify expected Phase1 outputs

These scripts mainly support development and experiment reproduction and are not recommended as primary open-source entry points.
