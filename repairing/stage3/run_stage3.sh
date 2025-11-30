#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STAGE2_DIR="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/repairing/stage2"
OUTPUT_DIR="${BASE_DIR}/output"

echo "Starting Stage 3 Fusion..."
echo "Stage 2 Dir: ${STAGE2_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"

python stage3.py \
    --stage2_dir "$STAGE2_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Done."
