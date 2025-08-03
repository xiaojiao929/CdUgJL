#!/bin/bash
#Amber
# Copyright (c) 2025 Amber Xiao

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG=./configs/default.yaml
CHECKPOINT=./checkpoints/meamtnet_best.pth
OUTPUT_DIR=./results
LOG_DIR=./logs

# Create directories if not exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Start testing/inference
python test.py \
  --config $CONFIG \
  --checkpoint $CHECKPOINT \
  --output_dir $OUTPUT_DIR \
  --save_preds True \
  > $LOG_DIR/test.log 2>&1 &

echo "Testing started. Results are being saved to $OUTPUT_DIR and logs to $LOG_DIR/test.log"
