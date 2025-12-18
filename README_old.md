# NeAR_fix_Public-Cardiac-CT-Dataset

## Project Overview
This project aims to **repair** and **refine** the Public-Cardiac-CT-Dataset using **NeAR (Neural Surface Reconstruction)** technology. The original dataset suffers from topological errors (broken vessels), anatomical inconsistencies (floating veins), and surface noise (aliasing).

Our goal is **Repair, not Copy**. We aim to produce a dataset that is:
1.  **Topologically Correct**: Broken coronary arteries and pulmonary veins are reconnected.
2.  **Geometrically Smooth**: Surface noise and aliasing are removed.
3.  **Anatomically Consistent**: All structures follow biological connectivity rules (e.g., PV connects to LA).

## Pipeline Structure
The project is organized into three sequential phases:

*   **[Phase 1: Shape Prior Learning](repairing/phase1/README.md)**
    *   Train implicit neural representations for each of the 10 cardiac classes.
    *   Generates smooth probability maps.
    *   **Key Files**: `train.py`, `configs/`

*   **[Phase 2: Morphological Processing](repairing/phase2/README.md)**
    *   Binarize and clean the masks using morphological operations.
    *   Filter connected components (CC) to remove noise.
    *   **Key Files**: `perform_morphology_v2.py`

*   **[Phase 3: Fusion & Correction](repairing/phase3/README.md)**
    *   Fuse single-class masks into a multi-class map using priority rules.
    *   Enforce strict anatomical connectivity (PV-LA, LAA-LA, Coronary-Myo).
    *   **Key Files**: `phase3.py`, `verify_all.py`

## Key Results
| Metric | Original | **Repaired (Phase 3)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Coronary Breaks (Mean CC)** | 2.12 | **1.36** | **~40% Reconnected** |
| **PV -> LA Connectivity** | 98.6% | **100.0%** | **Perfect Connectivity** |
| **Surface Smoothness (IsoRatio)** | 32.0 | **20.1** | **37% Smoother** |

## Directory Layout
```
.
├── dataset/                  # Original and intermediate data
├── repairing/
│   ├── phase1/               # NeAR Training (Unified train.py & configs)
│   ├── phase2/               # Morphology & Cleaning
│   └── phase3/               # Fusion & Comprehensive Verification
├── data_prepare/             # Data preprocessing scripts
└── project_summary.md        # Detailed project report (English)
```
