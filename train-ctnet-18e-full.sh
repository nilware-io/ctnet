#!/bin/bash
set -e

# CTNet-18e-full: 8-bit quantization-aware training on full ImageNet-1K
# Usage: ./train-ctnet-18e-full.sh [data_dir]

DATA_DIR="${1:-./imagenet}"

echo "=== CTNet-18e-full (8-bit aware, ImageNet-1K) ==="
echo "Dataset: $DATA_DIR"
echo ""

python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 1024 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --lambda-alpha 0.5 \
    --no-train-noise --dct-dropout 0 \
    --rate-warmup-epochs 4 \
    --pixel-bit-depth 8 \
    --no-imagenet-remap

# Export
echo ""
echo "--- Encoding to H.265 (8-bit) ---"
python export_h265.py encode --arch resnet18 \
    --crf 0 --bit-depth 8 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode --h265-dir ./h265_out --data "$DATA_DIR"
