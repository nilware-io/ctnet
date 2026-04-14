#!/bin/bash
set -e

# CTNet-50: Train, export, and evaluate
# Usage: ./resnet50.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-50 ==="
echo "Dataset: $DATA_DIR"
echo ""

# Train
echo "--- Training ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet50 --epochs 128 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-6 --qstep 0.1

# Export to H.265
echo ""
echo "--- Encoding to H.265 ---"
python export_h265.py encode \
    --arch resnet50 \
    --crf 0 --bit-depth 8 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode \
    --h265-dir ./h265_out --data "$DATA_DIR"
