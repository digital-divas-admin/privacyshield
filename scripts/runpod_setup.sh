#!/bin/bash
# RunPod RTX 4090 setup script for PrivacyShield encoder training
#
# Usage:
#   # On RunPod terminal after SSH in:
#   bash scripts/runpod_setup.sh [--subset 5000]
#
# Prerequisites:
#   - RunPod GPU pod with RTX 4090 (pytorch template recommended)
#   - Repo cloned to /workspace/privacyshield
#   - ArcFace weights at weights/arcface_r100.pth (copy via scp)

set -e

SUBSET_SIZE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --subset) SUBSET_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== PrivacyShield RunPod Setup ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

# Verify ArcFace weights
if [ ! -f weights/arcface_r100.pth ]; then
    echo ""
    echo "ERROR: ArcFace weights not found at weights/arcface_r100.pth"
    echo "Copy from local machine:"
    echo "  scp local:privacyshield/weights/arcface_r100.pth ./weights/"
    exit 1
fi

# Download data
echo ""
echo "=== Downloading FFHQ ==="
if [ -n "$SUBSET_SIZE" ]; then
    echo "Downloading $SUBSET_SIZE image subset..."
    python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq --max-images "$SUBSET_SIZE"
else
    echo "Downloading full dataset (70k images)..."
    python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq
fi

TRAIN_COUNT=$(find ./data/faces/train -type f | wc -l)
VAL_COUNT=$(find ./data/faces/val -type f | wc -l)
echo "Data ready: $TRAIN_COUNT train, $VAL_COUNT val images"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps — run training phases:"
echo ""
echo "  # 1. Generate PGD pairs (~1s/image)"
echo "  python scripts/train_encoder.py --phase generate \\"
echo "    --data-dir ./data/faces/train --output-dir ./data/pairs"
echo ""
echo "  # 2a. Train U-Net encoder"
echo "  python scripts/train_encoder.py --phase distill \\"
echo "    --encoder-type unet --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50"
echo "  python scripts/train_encoder.py --phase v2_e2e \\"
echo "    --encoder-type unet --data-dir ./data/faces \\"
echo "    --checkpoint ./checkpoints/distill_best.pt --use-mask --epochs 50"
echo ""
echo "  # 2b. Train ViT encoder (reuses same PGD pairs)"
echo "  python scripts/train_encoder.py --phase distill \\"
echo "    --encoder-type vit --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50"
echo "  python scripts/train_encoder.py --phase v2_e2e \\"
echo "    --encoder-type vit --data-dir ./data/faces \\"
echo "    --checkpoint ./checkpoints/vit_distill_best.pt --use-mask --epochs 50"
echo ""
echo "  # Monitor training"
echo "  tensorboard --logdir ./checkpoints/logs/ --bind_all"
echo ""
echo "  # Copy checkpoints back when done"
echo "  # scp ./checkpoints/best.pt local:privacyshield/checkpoints/"
echo "  # scp ./checkpoints/vit_best.pt local:privacyshield/checkpoints/"
