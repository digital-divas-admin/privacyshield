# PrivacyShield

Adversarial perturbation engine for facial privacy protection against
recognition systems (ArcFace, CLIP) and deepfake pipelines (Roop, IP-Adapter, InstantID,
Nano Banana Pro, SeedReam 4.5).

## Architecture
- `core/pipeline.py` — Unified attack orchestrator (start here)
- `core/attacks.py` — PGD + ASPL iterative attacks
- `core/losses.py` — Unified loss: ArcFace + CLIP dual-targeting + CLIP patch tokens + LPIPS
- `core/diff_align.py` — Differentiable face alignment via grid_sample + inverse grid for full-image hybrid
- `core/semantic_mask.py` — Face parsing mask (noise in hair/brows, not skin)
- `core/eot.py` — Expectation over Transformation (JPEG/resize/blur robustness)
- `core/diff_jpeg.py` — Differentiable JPEG with STE
- `core/encoder.py` — U-Net single-pass encoder
- `core/vit_encoder.py` — ViT-S/8 encoder (IDProtector-style, production)
- `core/face_model.py` — ArcFace IResNet-100 wrapper + FaceNet/AdaFace/Antelopev2 ensemble
- `core/adaface_backbone.py` — AdaFace IR-101 backbone for ensemble
- `core/openrouter.py` — OpenRouter API client for cloud deepfake testing (Nano Banana Pro, SeedReam 4.5)
- `core/prompt_injection.py` — Defensive prompt injection for cloud model resistance
- `core/deepfake_test.py` — Deepfake tool testing (inswapper + IP-Adapter FaceID Plus v2 + InstantID + cloud models)
- `api/main.py` — FastAPI backend (modes: pgd, encoder, vit, v2, v2_full, encoder_refined)
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

# Test against InstantID (SDXL + antelopev2, ~8GB VRAM)
python scripts/test_deepfake.py \
  --clean-image face.jpg --protected-image face_protected.png \
  --target-image face.jpg --output-dir ./test_outputs --run-instantid
```

API endpoint: `POST /test-deepfake` accepts multipart form with clean_image, protected_image,
target_image (optional), run_inswapper, run_ipadapter, run_instantid, run_nano_banana_pro,
run_seedream, prompt, threshold.

### Cloud model testing (OpenRouter)
Requires `OPENROUTER_API_KEY` env var and `openai>=1.0.0` package.
- Nano Banana Pro (Gemini 3 Pro Image Preview): `run_nano_banana_pro=true`
- SeedReam 4.5 (ByteDance): `run_seedream=true`
- Pipeline: send reference image + prompt → cloud model generates output → detect face → compare identity

## Key technical decisions
- All transforms are differentiable (grid_sample for alignment, DiffJPEG with STE)
- Semantic mask (BiSeNet) allows high perturbation everywhere (skin=0.9) — LPIPS constrains artifacts
- LPIPS prevents visible artifacts even at full ε budget
- CLIP dual-targeting blinds IP-Adapter FaceID Plus v2's style channel
- ViT-S/8 encoder is the production inference path (~170ms per image)
- EoT averages gradients over random JPEG/resize/blur for social media robustness
- Cross-model ensemble (ArcFace + FaceNet + AdaFace + Antelopev2) improves transferability
- CLIP patch token attack (gamma_clip_patch=0.3) disrupts hidden_states[-2] that one-shot generators consume
- Antelopev2 is eval-only (ONNX, no grad) — measures transferability to InstantID's FR model
- Deepfake models are lazy-loaded on first use (not at startup) to avoid heavy memory cost
- Hybrid full-image mode: encoder seed is blurred (sigma=4) and scaled to 0.5x before
  inverse-warping to full-image space, then blended with random noise everywhere via soft
  feathered mask (sigma=20). PGD refinement uses normalized gradient (not sign()) for smooth
  noise. This eliminates the visible face-region boundary that naive encoder seeding creates.
- Inverse grid (build_inverse_grid in DifferentiableAligner) maps full-image pixels to aligned
  face coordinates, enabling grid_sample to scatter aligned-space encoder deltas back to
  full-image space with natural zero-padding outside the face region
- Defensive prompt injection embeds near-invisible text instructions in protected images
  that cause multimodal cloud generators (Gemini 3 Pro) to refuse or distort face
  reproduction. Applied post-protection as independent compositing step. Uses BiSeNet
  for safe placement in hair/background. Text survives JPEG via DCT-aware rendering.

## API modes
- `pgd` — Iterative PGD attack (slow, best quality, ~5-10s)
- `encoder` — U-Net single-pass (fast, ~50ms, requires training)
- `vit` — ViT single-pass (fast, ~170ms, requires training)
- `v2` — Full pipeline: PGD + LPIPS + CLIP + semantic mask on aligned face
- `v2_full` — Full pipeline on full-size image with differentiable alignment
- `encoder_refined` — Hybrid: encoder seed + PGD refinement. Auto-detects full-size images
  and uses full-image pipeline with normalized gradient for near-invisible noise (~4s)

## Deepfake testing internals
- `core/deepfake_test.py` contains `DeepfakeTestRegistry` with lazy-loaded models
- `core/openrouter.py` contains `OpenRouterClient` wrapping OpenRouter's OpenAI-compatible API
- Cloud models (Nano Banana Pro, SeedReam 4.5) use `run_openrouter_test()` — sends reference
  image + prompt to cloud API, extracts generated image from response, compares identity
- Inswapper: loads `weights/inswapper_128.onnx` via insightface, uses FaceAnalysis for detection
- IP-Adapter FaceID Plus v2: loads SD 1.5 + LoRA + IP-Adapter weights from h94/IP-Adapter-FaceID
  - Uses `image_encoder_folder=None` in `load_ip_adapter()` to skip auto CLIP loading
  - CLIP ViT-H loaded separately from laion/CLIP-ViT-H-14-laion2B-s32B-b79K
  - Projection layer expects face ID embeds (512-dim) via `ip_adapter_image_embeds`
  - CLIP hidden_states[-2] (257x1280) set on `proj_layer.clip_embeds` as 4D tensor
  - CFG requires negative (zero) + positive embeddings concatenated on dim=0
- InstantID: loads SDXL + ControlNet + IP-Adapter weights from InstantX/InstantID
  - Uses antelopev2 (not buffalo_l) for face embedding extraction (512-dim)
  - CLIP ViT-H patch tokens fed through IP-Adapter cross-attention
  - ControlNet conditioned on face keypoint visualization
  - Reuses existing CLIP ViT-H encoder from IP-Adapter test if already loaded

## Current status
- All code compiles, architecture complete, all components wired end-to-end
- Deepfake testing verified: both inswapper and IP-Adapter defeated by PGD protection
  - Inswapper: clean cos_sim=0.94, protected cos_sim=-0.01 → PROTECTED
  - IP-Adapter: clean cos_sim=-0.01, protected cos_sim=0.10 → PROTECTED
- Encoder training complete (5k FFHQ subset, RunPod RTX 4090)
  - U-Net checkpoint: `checkpoints/best.pt` (218MB)
  - ViT checkpoint: `checkpoints/vit_best.pt` (87MB)
  - BiSeNet face parser: `weights/bisenet_face.pth` (53MB)
- Hybrid mode (encoder_refined) tested and tuned:
  - Hybrid aligned: ArcFace=-0.01, robust=0.18, LPIPS=0.12, PSNR=34.4, ~2.3s
  - Hybrid full-image: ArcFace=-0.34, FaceNet=-0.29, AdaFace=-0.09, ~4.1s
  - vs v2_full 50 steps: ~48s — hybrid full is 12x faster with near-invisible noise
  - vs original PGD 50: ArcFace=-0.52, LPIPS=0.16, PSNR=32, ~10s
- Frontend fully wired: encoder_refined mode with refine_steps slider, 3-column
  results view (Original aligned / Protected / Zoomed comparison), dynamic mask mode

## Cloud model brute force ceiling test (2024-02-24)
Tested whether extreme adversarial perturbation can break Gemini 3 Pro identity
preservation. Script: `scripts/test_perturbation_brute.py`. Mode: v2_full PGD,
100 steps (200 for nuclear), 2 images, 3 Gemini trials each.

### Results (Gemini 3 Pro, avg across 2 images, 3 trials each)
| Config | Eps | PSNR | Gemini Avg Sim | Local ArcFace | Verdict |
|---------------|---------|-------|----------------|---------------|---------|
| eps8_100 | 8/255 | 34.7 | 0.428 | -0.356 | VULNERABLE |
| eps16_100 | 16/255 | 28.8 | 0.298 | -0.259 | borderline |
| eps32_100 | 32/255 | 22.8 | 0.228 | -0.312 | PROTECTED (img0), VULNERABLE (img1) |
| eps64_100 | 64/255 | 17.1 | 0.038 | -0.268 | PROTECTED (both) |
| eps128_100 | 128/255 | 11.9 | 0.001 | -0.209 | PROTECTED (both) |
| allloss_eps32 | 32/255 | 22.8 | 0.266 | -0.225 | mostly PROTECTED |
| nuclear | 128/255 | 11.9 | 0.122 | -0.123 | PROTECTED (both) |

### Key findings
- **Breakpoint at eps=64/255**: reliably breaks Gemini for both images, but PSNR=17dB
  means heavy visible noise (rainbow mosaic on face). Not usable for imperceptible protection.
- **eps=8/255 (default) is invisible but ineffective**: Gemini avg_sim=0.428, all local FR
  models show negative cos_sim. Attack transfers to local models but not cloud.
- **eps=32/255 is image-dependent**: image 0 protected (sim=0.107), image 1 vulnerable
  (sim=0.348). Some faces are harder for Gemini to reconstruct.
- **allloss_eps32 (DINOv2+multi-layer CLIP+highfreq+attack_lpips) at eps=32 slightly
  outperforms default eps=32**: flipped image 1 from 0/3 to 2/3 protected. DINOv2
  features provide marginal improvement.
- **Nuclear (eps=128, all losses, 200 steps) produces weaker local protection (-0.123)
  than eps=8 (-0.356)**: extreme epsilon causes optimizer thrashing. More isn't better.
- **Zero refusals across all configs**: Gemini never refuses to generate, even with
  heavily corrupted input (PSNR=11.9dB). It always produces a face.
- **Conclusion**: current loss targets (ArcFace+FaceNet+AdaFace+CLIP+DINOv2) fundamentally
  cannot break cloud models at imperceptible perturbation levels. The problem is what we're
  attacking, not how hard. Need fundamentally different attack vectors.

## Cloud model loss ablation test (2026-02-24)
Systematic ablation of 5 disabled loss components against BOTH Gemini 3 Pro AND
SeedReam 4.5. Script: `scripts/test_cloud_ablation.py`. Mode: v2_full PGD,
50 steps, 2 images, 3 trials x 2 cloud models each (120 API calls total).

### Results (avg across 2 images, 3 trials each)
| Config | Eps | PSNR | Gemini Sim | SR4.5 Sim | Local Arc | Components |
|---|---|---|---|---|---|---|
| baseline_default | 8/255 | 34.7 | 0.395 | 0.617 | -0.397 | default |
| +dino_cls | 8/255 | 34.7 | 0.403 | 0.697 | -0.310 | +DINOv2 CLS |
| +dino_patch | 8/255 | 34.7 | 0.460 | 0.662 | -0.248 | +DINOv2 patch |
| +dino_both | 8/255 | 34.7 | 0.452 | 0.681 | -0.367 | +DINOv2 both |
| +clip_multi | 8/255 | 34.7 | 0.389 | 0.619 | -0.262 | +CLIP layers -3,-4 |
| +highfreq | 8/255 | 34.7 | 0.424 | 0.613 | -0.367 | +Laplacian |
| +atk_lpips | 8/255 | 34.7 | 0.404 | 0.651 | -0.367 | +attack LPIPS |
| kitchen_sink_8 | 8/255 | 34.7 | 0.379 | 0.711 | -0.236 | all components |
| kitchen_sink_12 | 12/255 | 31.2 | 0.334 | 0.579 | -0.191 | all components |
| kitchen_sink_16 | 16/255 | 28.8 | 0.346 | 0.536 | -0.281 | all components |

### Key findings
- **No single loss component helps at eps=8/255.** Every addition is either within
  noise or actively makes things worse due to gradient competition.
- **DINOv2 actively hurts**: dino_patch makes Gemini worse (+0.065), dino_cls makes
  SeedReam worse (+0.080). Budget spent on DINOv2 is stolen from ArcFace/CLIP which
  are doing the actual work. The hypothesis that cloud models rely on DINOv2 is falsified.
- **+clip_multi is the only neutral addition** (Gemini d=-0.006, SR d=+0.002). Multi-layer
  CLIP doesn't help but doesn't steal gradient budget either.
- **SeedReam 4.5 is substantially harder than Gemini 3 Pro** (avg sim 0.637 vs 0.399).
  SeedReam preserves identity at sim=0.82 on image 1. Different identity encoding pipeline.
- **Image 1 is impossible across ALL configs and BOTH models.** Even kitchen_sink_16
  only gets Gemini to 0.494 and SeedReam to 0.722 — both VULNERABLE.
- **Kitchen sink at elevated epsilon helps marginally**: kitchen_sink_12 drops Gemini
  by -0.061 and SR by -0.037, but still nowhere near threshold (0.3).
- **Gradient competition is real**: kitchen_sink configs show weaker local ArcFace
  (-0.236 vs -0.397 baseline) despite more loss terms. The optimizer can't serve
  6+ masters within the same L-inf budget.
- **Zero refusals** from either model across all 120 trials.
- **Conclusion**: confirmed that the problem is WHAT we're attacking, not HOW.
  None of {DINOv2, multi-layer CLIP, Laplacian highfreq, attack LPIPS} transfer
  to cloud models. Cloud models likely use proprietary identity encoders that don't
  share feature space with any public model we can differentiate through.

## What to build next
- Find attack targets that transfer to cloud models at low epsilon (frequency-domain,
  VAE latent disruption, JPEG-DCT attacks, or model-agnostic texture destruction)
- Investigate cloud model preprocessor attacks (adversarial JPEG artifacts, VAE
  latent space disruption) since feature-space attacks are confirmed ineffective
- Train on full 70k FFHQ (current checkpoints from 5k subset)
- Test InstantID protection end-to-end: `python scripts/test_deepfake.py --run-instantid`
- Add PhotoMaker / PuLID to deepfake test registry
- Make antelopev2 differentiable (PyTorch re-implementation) for direct gradient attack
