#!/bin/bash

# Wrapper script for Stage 3 Verification
# Runs verification on all 998 samples against the 'near_format_data' ground truth.

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STAGE3_DIR="${BASE_DIR}/output"
GT_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset_backup/dataset/near_format_data/shape"

echo "Starting Full Verification..."
echo "Stage 3 Output: ${STAGE3_DIR}"
echo "Ground Truth:   ${GT_DIR}"

python verify_stage3.py \
    --stage3_dir "$STAGE3_DIR" \
    --original_dir "$GT_DIR"

echo "Verification Complete. Results saved to stage3_verification_full.csv"
