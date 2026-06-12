#!/bin/bash
set -e

CONFIG=${1:-configs/default.yaml}
CHECKPOINT=${2:-checkpoints/meamtnet_best.pth}

python test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --save_results \
    --tta
