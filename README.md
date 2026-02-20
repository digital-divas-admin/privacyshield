# PrivacyShield

**Adversarial perturbation engine for facial privacy protection against recognition systems.**

Adds imperceptible noise to photos so that face recognition models (InsightFace/ArcFace) fail to match identities — while the image looks identical to human eyes. Perturbations survive social media compression (JPEG, resizing, re-encoding).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                     │
│  POST /protect   POST /protect/batch   GET /health   │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
       ┌───────▼───────┐     ┌────────▼────────┐
       │  PGD Attack   │     │  Noise Encoder  │
       │  (iterative)  │     │ (single-pass)   │
       │  ~50 steps    │     │  U-Net amortized│
       └───────┬───────┘     └────────┬────────┘
               │                      │
       ┌───────▼──────────────────────▼────────┐
       │       EoT (Expectation over           │
       │         Transformation)               │
       │  ┌──────────┬───────────┬──────────┐  │
       │  │DiffJPEG  │ Resize   │ Gaussian  │  │
       │  │q∈[50,95] │ s∈[0.5,1]│ σ∈[0,1]  │  │
       │  └──────────┴───────────┴──────────┘  │
       └───────────────────┬───────────────────┘
                           │
               ┌───────────▼───────────┐
               │   InsightFace/ArcFace │
               │   (frozen target)     │
               │   embedding: R^512    │
               └───────────────────────┘
```

## How It Works

### Attack Objective
Given image `x`, find perturbation `δ` (‖δ‖∞ ≤ ε) that **maximizes** cosine distance between the face embedding of the clean image and the perturbed image:

```
max_δ  𝔼_t~T [ 1 - cos_sim( F(t(x)), F(t(x + δ)) ) ]
s.t.   ‖δ‖∞ ≤ ε
```

Where:
- `F` = InsightFace ArcFace embedding model (frozen)
- `t ~ T` = random transformation (JPEG compression, resize, Gaussian blur)
- `ε` = perturbation budget (default 8/255)

### Key Components

1. **PGD Attack** (`core/attacks.py`): Projected Gradient Descent to iteratively craft the perturbation. Supports both targeted (push toward different identity) and untargeted (push away from original) modes.

2. **EoT Wrapper** (`core/eot.py`): Averages gradients over `N` random transformations per step so the perturbation survives real-world image processing. Includes differentiable JPEG, random resize, center crop, Gaussian blur.

3. **Differentiable JPEG** (`core/diff_jpeg.py`): Differentiable approximation of JPEG compression — DCT → quantization (with straight-through estimator) → IDCT. Allows gradients to flow through the compression.

4. **Noise Encoder** (`core/encoder.py`): U-Net that predicts `δ` in a single forward pass (amortized attack). Train on PGD-generated pairs, then deploy for real-time inference.

5. **Face Model Wrapper** (`core/face_model.py`): Wraps InsightFace ArcFace into a differentiable PyTorch module with face detection, alignment, and embedding extraction.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Protect a single image (PGD mode)
curl -X POST http://localhost:8000/protect \
  -F "image=@photo.jpg" \
  -F "epsilon=0.031" \
  -F "steps=50" \
  --output protected.png

# Protect with trained noise encoder (fast mode)
curl -X POST http://localhost:8000/protect \
  -F "image=@photo.jpg" \
  -F "mode=encoder" \
  --output protected.png
```

## Training the Noise Encoder

```bash
# Generate PGD training pairs
python scripts/generate_pairs.py --data-dir ./faces --output-dir ./pairs

# Train the encoder
python scripts/train_encoder.py \
  --pairs-dir ./pairs \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4

# Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/best.pt
```

## Configuration

See `config.py` for all hyperparameters. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 8/255 | L∞ perturbation budget |
| `pgd_steps` | 50 | PGD iteration count |
| `pgd_step_size` | 2/255 | Per-step perturbation |
| `eot_samples` | 10 | Transforms averaged per step |
| `jpeg_quality_range` | (50, 95) | JPEG quality factor range |
| `resize_range` | (0.5, 1.0) | Random resize scale range |

## Project Structure

```
privacyshield/
├── api/
│   ├── main.py              # FastAPI application
│   ├── routes.py            # API endpoints
│   └── schemas.py           # Request/response models
├── core/
│   ├── attacks.py           # PGD attack implementation
│   ├── diff_jpeg.py         # Differentiable JPEG codec
│   ├── encoder.py           # U-Net noise encoder
│   ├── eot.py               # Expectation over Transformation
│   └── face_model.py        # InsightFace wrapper
├── scripts/
│   ├── generate_pairs.py    # Generate PGD training data
│   ├── train_encoder.py     # Train noise encoder
│   └── evaluate.py          # Evaluation metrics
├── config.py                # Global configuration
├── requirements.txt
└── README.md
```

## References

- Salman et al. "Raising the Cost of Malicious AI-Powered Image Editing" (PhotoGuard, ICML 2023)
- Van Le et al. "Anti-DreamBooth: Protecting Users from Personalized Text-to-Image Synthesis" (ICCV 2023)
- Athalye et al. "Synthesizing Robust Adversarial Examples" (EoT, ICML 2018)
- Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- Shin & Song "JPEG-resistant Adversarial Images" (Differentiable JPEG)
