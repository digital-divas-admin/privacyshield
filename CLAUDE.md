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
- `core/face_model.py` — ArcFace IResNet-100 wrapper + FaceNet/AdaFace ensemble
- `core/adaface_backbone.py` — AdaFace IR-101 backbone for ensemble
- `core/deepfake_test.py` — Deepfake tool testing (inswapper + IP-Adapter FaceID Plus v2)
- `api/main.py` — FastAPI backend (modes: pgd, encoder, vit, v2, v2_full)
- `scripts/train_encoder.py` — 4-phase training (generate → distill → e2e → v2_e2e), U-Net + ViT
- `scripts/prepare_data.py` — Download FFHQ from HuggingFace, resize, train/val split
- `scripts/runpod_setup.sh` — Cloud GPU setup (install deps, download data, verify weights)
- `scripts/train_all.sh` — Unattended full training pipeline (all phases, both encoders)
- `scripts/evaluate.py` — Evaluation across JPEG/resize/blur conditions
- `scripts/test_deepfake.py` — CLI script for deepfake tool testing

## Frontend
- `frontend/src/app/deepfake/page.tsx` — Deepfake tool testing page
- `frontend/src/components/deepfake/DeepfakeResults.tsx` — Deepfake test results display
- `frontend/src/components/health/SystemStatus.tsx` — System/model health dashboard
- `frontend/src/components/layout/Sidebar.tsx` — Navigation sidebar

## Setup
- Requires: PyTorch 2.0+, CUDA GPU, ArcFace IResNet-100 weights
- `pip install -r requirements.txt`
- Download ArcFace weights from InsightFace model zoo (iresnet100)
- Download inswapper weights: `weights/inswapper_128.onnx` from HuggingFace
- `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Training pipeline

### Data preparation
```bash
# Full FFHQ (70k images)
python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq

# 5k subset for validation runs
python scripts/prepare_data.py --output-dir ./data/faces --dataset ffhq --max-images 5000
```

### U-Net encoder training (~50ms inference)
```bash
# Generate PGD pairs
python scripts/train_encoder.py --phase generate --data-dir ./data/faces/train --output-dir ./data/pairs

# Distill PGD into U-Net
python scripts/train_encoder.py --phase distill --encoder-type unet --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50

# V2 E2E fine-tune (saves checkpoints/best.pt)
python scripts/train_encoder.py --phase v2_e2e --encoder-type unet --data-dir ./data/faces \
  --checkpoint ./checkpoints/distill_best.pt --use-mask --epochs 50
```

### ViT encoder training (~170ms inference)
```bash
# Distill PGD into ViT (reuses same PGD pairs)
python scripts/train_encoder.py --phase distill --encoder-type vit --pairs-dir ./data/pairs --data-dir ./data/faces --epochs 50

# V2 E2E fine-tune (saves checkpoints/vit_best.pt)
python scripts/train_encoder.py --phase v2_e2e --encoder-type vit --data-dir ./data/faces \
  --checkpoint ./checkpoints/vit_distill_best.pt --use-mask --epochs 50
```

### Training features
- `--encoder-type {unet, vit}` — architecture selection with smart defaults (batch, lr)
- `--arcface-weights` — path to ArcFace weights (default: `./weights/arcface_r100.pth`)
- `--patience N` — early stopping after N epochs without val improvement (default: 10)
- `--no-tensorboard` — disable TensorBoard logging (logs to `checkpoints/logs/`)
- `--num-workers N` — DataLoader workers (auto: 0 on Windows, 4 on Linux)
- Validation split: uses `data/faces/train/` + `data/faces/val/` if present, else 90/10 split
- ViT gets linear warmup (5 epochs) before cosine annealing

### Cloud training (RunPod RTX 4090, $0.50/hr)
5k subset: ~15 hrs (~$8). Full 70k: ~60 hrs (~$30).
```bash
# SSH into RunPod, clone repo, copy ArcFace weights
git clone <repo> && cd privacyshield
scp local:privacyshield/weights/arcface_r100.pth ./weights/

# Automated setup (install deps + download data)
bash scripts/runpod_setup.sh --subset 5000   # 5k subset
bash scripts/runpod_setup.sh                  # full 70k

# Run all training unattended
nohup bash scripts/train_all.sh > training.log 2>&1 &
tail -f training.log

# Monitor
tensorboard --logdir ./checkpoints/logs/ --bind_all

# Copy checkpoints back
scp ./checkpoints/best.pt local:privacyshield/checkpoints/
scp ./checkpoints/vit_best.pt local:privacyshield/checkpoints/
```

## Deepfake tool testing
```bash
# Test against inswapper (Roop) only
python scripts/test_deepfake.py \
  --clean-image face.jpg --protected-image face_protected.png \
  --target-image face.jpg --output-dir ./test_outputs --skip-ipadapter

# Test against both inswapper + IP-Adapter FaceID Plus v2
python scripts/test_deepfake.py \
  --clean-image face.jpg --protected-image face_protected.png \
  --target-image face.jpg --output-dir ./test_outputs
```

API endpoint: `POST /test-deepfake` accepts multipart form with clean_image, protected_image,
target_image (optional), run_inswapper, run_ipadapter, prompt, threshold.

## Key technical decisions
- All transforms are differentiable (grid_sample for alignment, DiffJPEG with STE)
- Semantic mask concentrates perturbation in textured regions (hair, eyebrows)
- LPIPS prevents visible artifacts even at full ε budget
- CLIP dual-targeting blinds IP-Adapter FaceID Plus v2's style channel
- ViT-S/8 encoder is the production inference path (~170ms per image)
- EoT averages gradients over random JPEG/resize/blur for social media robustness
- Cross-model ensemble (ArcFace + FaceNet + AdaFace) improves transferability
- Deepfake models are lazy-loaded on first use (not at startup) to avoid heavy memory cost

## API modes
- `pgd` — Iterative PGD attack (slow, best quality, ~5-10s)
- `encoder` — U-Net single-pass (fast, ~50ms, requires training)
- `vit` — ViT single-pass (fast, ~170ms, requires training)
- `v2` — Full pipeline: PGD + LPIPS + CLIP + semantic mask on aligned face
- `v2_full` — Full pipeline on full-size image with differentiable alignment

## Deepfake testing internals
- `core/deepfake_test.py` contains `DeepfakeTestRegistry` with lazy-loaded models
- Inswapper: loads `weights/inswapper_128.onnx` via insightface, uses FaceAnalysis for detection
- IP-Adapter FaceID Plus v2: loads SD 1.5 + LoRA + IP-Adapter weights from h94/IP-Adapter-FaceID
  - Uses `image_encoder_folder=None` in `load_ip_adapter()` to skip auto CLIP loading
  - CLIP ViT-H loaded separately from laion/CLIP-ViT-H-14-laion2B-s32B-b79K
  - Projection layer expects face ID embeds (512-dim) via `ip_adapter_image_embeds`
  - CLIP hidden_states[-2] (257x1280) set on `proj_layer.clip_embeds` as 4D tensor
  - CFG requires negative (zero) + positive embeddings concatenated on dim=0

## Current status
- All code compiles, architecture complete, all components wired end-to-end
- Deepfake testing verified: both inswapper and IP-Adapter defeated by PGD protection
  - Inswapper: clean cos_sim=0.94, protected cos_sim=-0.01 → PROTECTED
  - IP-Adapter: clean cos_sim=-0.01, protected cos_sim=0.10 → PROTECTED
- Needs: training data (CelebA/VGGFace2), GPU training run for encoder modes

## What to build next
- Run encoder training on GPU cloud (5k subset first, then full 70k)
- Test with real-world face photos (not padded crops) for more representative metrics
- Add InstantID testing to deepfake test registry
- Build React demo frontend with before/after slider
