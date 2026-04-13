#!/usr/bin/env bash
# Download and extract ImageNette2-320 (10-class ImageNet subset, ~320 MB).
# Usage: ./download_imagenette.sh [destination_dir]

set -euo pipefail

DEST="${1:-./imagenette2-320}"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
TARBALL="$(mktemp --suffix=.tgz)"

cleanup() { rm -f "$TARBALL"; }
trap cleanup EXIT

if [ -d "$DEST/train" ] && [ -d "$DEST/val" ]; then
    echo "Dataset already exists at $DEST — skipping download."
    exit 0
fi

echo "Downloading ImageNette2-320..."
wget -q --show-progress -O "$TARBALL" "$URL"

echo "Extracting to $DEST..."
mkdir -p "$(dirname "$DEST")"
tar -xzf "$TARBALL" -C "$(dirname "$DEST")"

# Rename if the extracted dir doesn't match the destination
EXTRACTED="$(dirname "$DEST")/imagenette2-320"
if [ "$EXTRACTED" != "$DEST" ] && [ -d "$EXTRACTED" ]; then
    mv "$EXTRACTED" "$DEST"
fi

echo "Done. Dataset at $DEST (train: $(ls "$DEST/train" | wc -l) classes, val: $(ls "$DEST/val" | wc -l) classes)"
