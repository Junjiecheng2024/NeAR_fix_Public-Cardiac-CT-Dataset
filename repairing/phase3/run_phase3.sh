#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STAGE2_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/phase2"
OUTPUT_DIR="${BASE_DIR}/output"

echo "Starting Phase 3 Fusion..."
echo "Phase 2 Dir: ${STAGE2_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"

python phase3.py \
    --phase2_dir "$STAGE2_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Done."
