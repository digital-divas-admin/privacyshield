# PrivacyShield

Adversarial perturbation engine for facial privacy protection against
recognition systems (ArcFace, CLIP) and deepfake pipelines (Roop, IP-Adapter).

## Architecture
- `core/pipeline.py` — Unified attack orchestrator (start here)
- `core/attacks.py` — PGD + ASPL iterative attacks
- `core/losses.py` — Unified loss: ArcFace + CLIP dual-targeting + LPIPS
- `core/diff_align.py` — Differentiable face alignment via grid_sample
- `core/semantic_mask.py` — Face parsing mask (noise in hair/brows, not skin)
- `core/eot.py` — Expectation over Transformation (JPEG/resize/blur robustness)
- `core/diff_jpeg.py` — Differentiable JPEG with STE
- `core/encoder.py` — U-Net single-pass encoder
- `core/vit_encoder.py` — ViT-S/8 encoder (IDProtector-style, production)
- `core/face_model.py` — ArcFace IResNet-100 wrapper
- `api/main.py` — FastAPI backend (modes: pgd, encoder, vit, v2, v2_full)
- `scripts/train_encoder.py` — 3-phase training (generate → distill → v2_e2e)
- `scripts/evaluate.py` — Evaluation across JPEG/resize/blur conditions

## Setup
- Requires: PyTorch 2.0+, CUDA GPU, ArcFace IResNet-100 weights
- `pip install -r requirements.txt`
- Download ArcFace weights from InsightFace model zoo (iresnet100)
- `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Training pipeline
```bash
# Phase 1: Generate PGD perturbation pairs
python scripts/train_encoder.py --phase generate --data-dir ./data/faces --output-dir ./data/pairs

# Phase 2: Distill PGD into encoder
python scripts/train_encoder.py --phase distill --pairs-dir ./data/pairs --epochs 50

# Phase 3: V2 end-to-end with LPIPS + CLIP + semantic mask
python scripts/train_encoder.py --phase v2_e2e --checkpoint ./checkpoints/best.pt --use-mask --epochs 50
```

## Key technical decisions
- All transforms are differentiable (grid_sample for alignment, DiffJPEG with STE)
- Semantic mask concentrates perturbation in textured regions (hair, eyebrows)
- LPIPS prevents visible artifacts even at full ε budget
- CLIP dual-targeting blinds IP-Adapter FaceID Plus v2's style channel
- ViT-S/8 encoder is the production inference path (~170ms per image)
- EoT averages gradients over random JPEG/resize/blur for social media robustness

## API modes
- `pgd` — Iterative PGD attack (slow, best quality, ~5-10s)
- `encoder` — U-Net single-pass (fast, ~50ms, requires training)
- `vit` — ViT single-pass (fast, ~170ms, requires training)
- `v2` — Full pipeline: PGD + LPIPS + CLIP + semantic mask on aligned face
- `v2_full` — Full pipeline on full-size image with differentiable alignment

## Current status
- All code compiles (4,393 lines across 19 Python files)
- Architecture complete, all components wired end-to-end
- Needs: ArcFace weights downloaded, training data (CelebA/VGGFace2), GPU training run

## What to build next
- Download and wire ArcFace IResNet-100 weights
- Set up training data pipeline with CelebA-HQ
- Run training phases 1-3
- Build React demo frontend with before/after slider
- Benchmark against Roop, InstantID, IP-Adapter FaceID Plus v2
