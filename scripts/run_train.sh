#!/bin/bash
#Amber
# Copyright (c) 2025 Amber Xiao

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG=./configs/default.yaml
OUTPUT_DIR=./checkpoints
LOG_DIR=./logs

# Create directories if not exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Start training
python train.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --seed 42 \
  --use_amp True \
  --resume False \
  > $LOG_DIR/train.log 2>&1 &

echo "Training started. Logs are being saved to $LOG_DIR/train.log"
