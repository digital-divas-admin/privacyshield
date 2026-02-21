#!/bin/bash
# Full unattended training pipeline for RunPod
#
# Runs all phases for both U-Net and ViT encoders.
# Kick this off and come back when it's done.
#
# Usage:
#   # After running runpod_setup.sh:
#   nohup bash scripts/train_all.sh > training.log 2>&1 &
#   tail -f training.log

set -e

EPOCHS=${EPOCHS:-50}
DATA_DIR=${DATA_DIR:-./data/faces}
PAIRS_DIR=${PAIRS_DIR:-./data/pairs}
CKPT_DIR=${CKPT_DIR:-./checkpoints}
PGD_BATCH=${PGD_BATCH:-8}

echo "=== PrivacyShield Full Training Pipeline ==="
echo "Epochs: $EPOCHS | Data: $DATA_DIR | Checkpoints: $CKPT_DIR"
echo "Started: $(date)"
echo ""

# Phase 0: Generate PGD pairs (shared between U-Net and ViT)
if [ ! -d "$PAIRS_DIR/clean" ]; then
    echo "=== Phase 0: Generating PGD pairs ==="
    python scripts/train_encoder.py --phase generate \
        --data-dir "$DATA_DIR/train" --output-dir "$PAIRS_DIR" \
        --pgd-batch-size "$PGD_BATCH"
    echo "PGD pairs done: $(date)"
else
    echo "=== Skipping PGD pair generation (already exists) ==="
fi

# Phase 1: U-Net distill
echo ""
echo "=== Phase 1a: U-Net Distillation ==="
python scripts/train_encoder.py --phase distill \
    --encoder-type unet --pairs-dir "$PAIRS_DIR" --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" --checkpoint-dir "$CKPT_DIR"
echo "U-Net distill done: $(date)"

# Phase 2: U-Net v2 E2E
echo ""
echo "=== Phase 2a: U-Net V2 E2E ==="
python scripts/train_encoder.py --phase v2_e2e \
    --encoder-type unet --data-dir "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/distill_best.pt" \
    --use-mask --epochs 20 --checkpoint-dir "$CKPT_DIR" \
    --batch-size 2 --eot-samples 3
echo "U-Net v2_e2e done: $(date)"

# Phase 3: ViT distill
echo ""
echo "=== Phase 1b: ViT Distillation ==="
python scripts/train_encoder.py --phase distill \
    --encoder-type vit --pairs-dir "$PAIRS_DIR" --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" --checkpoint-dir "$CKPT_DIR"
echo "ViT distill done: $(date)"

# Phase 4: ViT v2 E2E
echo ""
echo "=== Phase 2b: ViT V2 E2E ==="
python scripts/train_encoder.py --phase v2_e2e \
    --encoder-type vit --data-dir "$DATA_DIR" \
    --checkpoint "$CKPT_DIR/vit_distill_best.pt" \
    --use-mask --epochs 20 --checkpoint-dir "$CKPT_DIR" \
    --batch-size 2 --eot-samples 3
echo "ViT v2_e2e done: $(date)"

echo ""
echo "=== All training complete ==="
echo "Finished: $(date)"
echo ""
echo "Checkpoints:"
ls -lh "$CKPT_DIR"/*.pt 2>/dev/null || echo "No checkpoints found"
echo ""
echo "Copy to local machine:"
echo "  scp $(hostname):$(pwd)/$CKPT_DIR/best.pt local:privacyshield/checkpoints/"
echo "  scp $(hostname):$(pwd)/$CKPT_DIR/vit_best.pt local:privacyshield/checkpoints/"
