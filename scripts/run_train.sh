#!/bin/bash
set -e

CONFIG=${1:-configs/default.yaml}
RESUME=${2:-""}

if [ -n "$RESUME" ]; then
    python train.py --config "$CONFIG" --resume "$RESUME"
else
    python train.py --config "$CONFIG"
fi
