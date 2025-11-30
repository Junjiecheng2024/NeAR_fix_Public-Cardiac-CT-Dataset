# Phase 3: Multi-Class Fusion & Anatomical Correction

## Overview
Phase 3 is the final assembly stage. We fuse the 10 cleaned single-class masks from Phase 2 into a unified, non-overlapping multi-class segmentation map. Crucially, we enforce strict anatomical rules to ensure topological correctness.

## Fusion Strategy
We use a **Priority-Based Fusion** to resolve voxel conflicts (where multiple classes claim the same voxel):

**Priority Order**:
`Coronary > PV > LAA > Chambers (LV/LA/RA/RV) > Myocardium > Aorta > PA`

*   **Rationale**: Small, thin structures (Coronary) are given highest priority to prevent them from being "eaten" by larger organs. Chambers are prioritized over Myocardium to ensure the endocardial border is defined by the blood pool.

## Anatomical Rules
After fusion, we apply specific rules to fix topological errors:

1.  **PV-LA Connectivity**: Pulmonary Veins (PV) MUST connect to the Left Atrium (LA). Any PV component not connected to the LA is removed.
2.  **LAA-LA Connectivity**: The Left Atrial Appendage (LAA) MUST connect to the LA.
3.  **Coronary Attachment**: Coronary arteries MUST attach to the Myocardium or Aorta. Floating vessel segments are removed.
4.  **Single-Connectivity Enforcement**: For major chambers, we ensure (again) that they remain as single connected components after fusion.

## Verification
We verify the final output using a unified script `verify_all.py` that calculates:
*   **Fidelity**: Dice Score, Hausdorff Distance (HD95), ASD vs. Original Ground Truth.
*   **Topology**: Connectivity rates (PV->LA, LAA->LA, Cor->Myo) and Connected Component counts.
*   **Geometry**: Isoperimetric Ratio (Surface Smoothness).

## Usage

### Run Phase 3 Fusion
```bash
./run_phase3.sh
```

### Run Comprehensive Verification
```bash
# Runs verify_all.py to generate final_verification_report.csv
./run_verification.sh
```

### Visualization
To generate side-by-side comparisons with CT overlay:
```bash
python visualize_phase3.py --phase3_dir output --original_dir ...
```
