#!/bin/bash
# Run full verification for Phase 3 (Dice, CC, Connectivity, HD95, ASD, Smoothness)

PHASE3_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase3/output"
GT_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/near_format_data/shape/"

echo "Running Unified Verification (Dice, Topology, Geometry, Fidelity)..."
python verify_all.py \
    --phase3_dir "$PHASE3_DIR" \
    --gt_dir "$GT_DIR" \
    --output "final_verification_report.csv"
