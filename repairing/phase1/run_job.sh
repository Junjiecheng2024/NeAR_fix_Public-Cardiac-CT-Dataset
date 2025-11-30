#!/bin/bash
# Usage: ./run_job.sh <config_name> [devices]
# Example: ./run_job.sh LV 4

CONFIG=$1
DEVICES=${2:-1}

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config_name> [devices]"
    echo "Available configs:"
    ls configs/*.py | xargs -n 1 basename | sed 's/.py//'
    exit 1
fi

# Ensure we are in the script directory
cd "$(dirname "$0")"

echo "Running training for $CONFIG with $DEVICES devices..."
python train.py --config configs/${CONFIG}.py --devices $DEVICES --strategy ddp
