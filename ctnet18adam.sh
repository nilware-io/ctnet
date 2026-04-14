#!/bin/bash
set -e

# CTNet-18 with AdamW optimizer, long training
# Usage: ./ctnet18adam.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18 (AdamW, 1024 epochs) ==="
echo "Dataset: $DATA_DIR"
echo ""

# Train
echo "--- Training ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 1024 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-4 --qstep 0.1 \
    --cache-dataset

# Export to H.265
echo ""
echo "--- Encoding to H.265 ---"
python export_h265.py encode \
    --arch resnet18 \
    --crf 0 --bit-depth 8 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode \
    --h265-dir ./h265_out --data "$DATA_DIR"
