#!/bin/bash
set -e

# Extract ImageNet-1K WebDataset shards to standard ImageFolder layout
# Usage: ./extract_imagenet_wds.sh [wds_dir] [output_dir]

WDS_DIR="${1:-./imagenet-1k-wds}"
OUT_DIR="${2:-./imagenet}"

echo "Extracting ImageNet-1K from $WDS_DIR to $OUT_DIR"
mkdir -p "$OUT_DIR/train" "$OUT_DIR/val"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

count=0
total_train=$(ls "$WDS_DIR"/imagenet1k-train-*.tar 2>/dev/null | wc -l)
total_val=$(ls "$WDS_DIR"/imagenet1k-validation-*.tar 2>/dev/null | wc -l)

echo "Extracting $total_train training shards..."
for tar in "$WDS_DIR"/imagenet1k-train-*.tar; do
    count=$((count + 1))
    printf "\r  Shard %d/%d" "$count" "$total_train"

    tar xf "$tar" -C "$TMPDIR"

    for jpg in "$TMPDIR"/*.jpg; do
        [ -f "$jpg" ] || continue
        wnid="${jpg##*/}"
        wnid="${wnid%%_*}"
        mkdir -p "$OUT_DIR/train/$wnid"
        mv "$jpg" "$OUT_DIR/train/$wnid/"
    done

    rm -f "$TMPDIR"/*.cls "$TMPDIR"/*.json 2>/dev/null || true
done
echo ""

count=0
echo "Extracting $total_val validation shards..."
for tar in "$WDS_DIR"/imagenet1k-validation-*.tar; do
    count=$((count + 1))
    printf "\r  Shard %d/%d" "$count" "$total_val"

    tar xf "$tar" -C "$TMPDIR"

    for jpg in "$TMPDIR"/*.jpg; do
        [ -f "$jpg" ] || continue
        wnid="${jpg##*/}"
        wnid="${wnid%%_*}"
        mkdir -p "$OUT_DIR/val/$wnid"
        mv "$jpg" "$OUT_DIR/val/$wnid/"
    done

    rm -f "$TMPDIR"/*.cls "$TMPDIR"/*.json 2>/dev/null || true
done
echo ""

TRAIN_CLASSES=$(ls -1 "$OUT_DIR/train" | wc -l)
TRAIN_IMAGES=$(find "$OUT_DIR/train" -name "*.jpg" | wc -l)
VAL_CLASSES=$(ls -1 "$OUT_DIR/val" | wc -l)
VAL_IMAGES=$(find "$OUT_DIR/val" -name "*.jpg" | wc -l)

echo "Done!"
echo "  Train: $TRAIN_IMAGES images in $TRAIN_CLASSES classes"
echo "  Val:   $VAL_IMAGES images in $VAL_CLASSES classes"
echo "  Output: $OUT_DIR"
