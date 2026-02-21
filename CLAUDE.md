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
- `scripts/train_encoder.py` — 3-phase training (generate → distill → v2_e2e)
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
```bash
# Phase 1: Generate PGD perturbation pairs
python scripts/train_encoder.py --phase generate --data-dir ./data/faces --output-dir ./data/pairs

# Phase 2: Distill PGD into encoder
python scripts/train_encoder.py --phase distill --pairs-dir ./data/pairs --epochs 50

# Phase 3: V2 end-to-end with LPIPS + CLIP + semantic mask
python scripts/train_encoder.py --phase v2_e2e --checkpoint ./checkpoints/best.pt --use-mask --epochs 50
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
- Set up training data pipeline with CelebA-HQ
- Run training phases 1-3
- Test with real-world face photos (not padded crops) for more representative metrics
- Add InstantID testing to deepfake test registry
- Build React demo frontend with before/after slider
