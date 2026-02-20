"""
PrivacyShield FastAPI Backend

Endpoints:
  POST /protect           — Protect a single image
  POST /protect/batch     — Protect multiple images
  GET  /health            — Health check + model status
  GET  /docs              — Auto-generated API docs (Swagger)

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ProtectMode, ProtectResponse, HealthResponse
from config import config


# ---------------------------------------------------------------------------
# Global model registry (loaded once at startup)
# ---------------------------------------------------------------------------

class ModelRegistry:
    face_model = None
    eot_wrapper = None
    noise_encoder = None
    vit_encoder = None
    pipeline_v2 = None          # Full v2 pipeline with LPIPS + CLIP + mask
    device = "cpu"


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    import os
    from core.face_model import FaceEmbedder
    from core.eot import EoTWrapper
    from core.encoder import NoiseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    registry.device = device
    print(f"Initializing PrivacyShield on {device}")

    # Load face model
    try:
        registry.face_model = FaceEmbedder(
            weights_path=None,  # Set path to ArcFace weights here
            device=device,
        )
        print("Face model loaded")
    except Exception as e:
        print(f"Warning: Face model load failed: {e}")

    # Load EoT wrapper
    if registry.face_model is not None:
        registry.eot_wrapper = EoTWrapper(
            model=registry.face_model,
            num_samples=config.eot.num_samples,
            jpeg_quality_range=config.eot.jpeg_quality_range,
            resize_scale_range=config.eot.resize_scale_range,
            gaussian_sigma_range=config.eot.gaussian_sigma_range,
            crop_fraction_range=config.eot.crop_fraction_range,
            enable_jpeg=config.eot.enable_jpeg,
            enable_resize=config.eot.enable_resize,
            enable_gaussian=config.eot.enable_gaussian,
            enable_crop=config.eot.enable_crop,
        )
        print("EoT wrapper initialized")

    # Try to load noise encoder (U-Net)
    try:
        registry.noise_encoder = NoiseEncoder(
            epsilon=config.attack.epsilon,
        ).to(device)

        ckpt_path = os.path.join(config.encoder.checkpoint_dir, "best.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            registry.noise_encoder.load_state_dict(state)
            print(f"Noise encoder loaded from {ckpt_path}")
        else:
            print("No encoder checkpoint found — encoder mode unavailable until trained")
            registry.noise_encoder = None
    except Exception as e:
        print(f"Warning: Encoder load failed: {e}")
        registry.noise_encoder = None

    # Try to load ViT noise encoder (IDProtector-style)
    try:
        from core.vit_encoder import vit_noise_encoder_small

        vit_ckpt = os.path.join(config.encoder.checkpoint_dir, "vit_best.pt")
        if os.path.exists(vit_ckpt):
            registry.vit_encoder = vit_noise_encoder_small(
                epsilon=config.attack.epsilon,
            ).to(device)
            state = torch.load(vit_ckpt, map_location=device)
            registry.vit_encoder.load_state_dict(state)
            registry.vit_encoder.eval()
            print(f"ViT encoder loaded from {vit_ckpt}")
        else:
            print("No ViT checkpoint found — vit mode unavailable until trained")
    except Exception as e:
        print(f"Warning: ViT encoder load failed: {e}")

    # --- v2 Pipeline (with LPIPS + CLIP + semantic mask) ---
    try:
        from core.pipeline import ProtectionPipeline, PipelineConfig

        pipe_cfg = PipelineConfig(
            epsilon=config.attack.epsilon,
            num_steps=config.attack.num_steps,
            eot_samples=config.eot.num_samples,
            use_semantic_mask=True,
            mask_mode="default",
        )
        registry.pipeline_v2 = ProtectionPipeline(pipe_cfg)
        registry.pipeline_v2.setup(device=device)
        print("v2 Pipeline initialized (LPIPS + CLIP + semantic mask)")
    except Exception as e:
        print(f"Warning: v2 Pipeline failed to init: {e}")
        registry.pipeline_v2 = None

    print("PrivacyShield ready!")
    yield

    # Cleanup
    print("Shutting down PrivacyShield")
    del registry.face_model
    del registry.eot_wrapper
    del registry.noise_encoder
    del registry.pipeline_v2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PrivacyShield API",
    description="Adversarial perturbation engine for facial privacy protection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image_tensor(image_bytes: bytes) -> torch.Tensor:
    """Load image bytes, detect face, align to 112x112 tensor."""
    import cv2

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Could not decode image")

    # Detect and align face
    aligned = registry.face_model.align_from_numpy(img_bgr)
    if aligned is None:
        raise HTTPException(422, "No face detected in image")

    return aligned


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert (1, 3, H, W) tensor in [0,1] to PNG bytes."""
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def apply_perturbation_to_full_image(
    original_bytes: bytes,
    delta_aligned: torch.Tensor,
) -> bytes:
    """
    Apply the perturbation back to the original full-size image.
    This maps the 112x112 perturbation back to the face region
    in the original image using the inverse affine transform.

    For the MVP, we return just the aligned protected face.
    Full-image mapping requires the affine matrix from detection.
    """
    # TODO: Implement inverse affine warp for full-image output
    return tensor_to_png_bytes(delta_aligned.unsqueeze(0) if delta_aligned.dim() == 3 else delta_aligned)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=registry.device,
        face_model_loaded=registry.face_model is not None,
        encoder_loaded=registry.noise_encoder is not None,
        vit_encoder_loaded=registry.vit_encoder is not None,
        pipeline_v2_loaded=registry.pipeline_v2 is not None,
    )


@app.post("/protect")
async def protect_image(
    image: UploadFile = File(...),
    mode: str = Form("pgd"),
    epsilon: float = Form(8/255),
    steps: int = Form(50),
    step_size: Optional[float] = Form(None),
    eot_samples: int = Form(10),
    mask_mode: str = Form("default"),
):
    """
    Protect a face image against recognition.

    Modes:
      - pgd:      Classic PGD attack (slow, reliable)
      - encoder:  U-Net single-pass (fast, requires training)
      - vit:      ViT single-pass (fast, requires training)
      - v2:       Full pipeline with LPIPS + CLIP + semantic mask
      - v2_full:  Full-image mode with differentiable alignment
      - aspl:     Alternating surrogate attack
    """
    if registry.face_model is None:
        raise HTTPException(503, "Face model not loaded")

    start_time = time.time()
    image_bytes = await image.read()

    # --- v2 pipeline modes ---
    if mode in ("v2", "v2_full"):
        if registry.pipeline_v2 is None:
            raise HTTPException(503, "v2 pipeline not initialized")

        # Update config dynamically
        registry.pipeline_v2.config.epsilon = epsilon
        registry.pipeline_v2.config.num_steps = steps
        registry.pipeline_v2.config.eot_samples = eot_samples
        registry.pipeline_v2.config.mask_mode = mask_mode
        registry.pipeline_v2.config.verbose = False

        if mode == "v2_full":
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise HTTPException(400, "Could not decode image")

            x_protected, info = registry.pipeline_v2.protect_full(img_bgr)
            if x_protected is None:
                raise HTTPException(422, info.get("error", "Protection failed"))
        else:
            x = load_image_tensor(image_bytes)
            x_protected, info = registry.pipeline_v2.protect_aligned(x)

        elapsed_ms = (time.time() - start_time) * 1000
        png_bytes = tensor_to_png_bytes(x_protected)

        headers = {
            "X-Privacy-Mode": mode,
            "X-ArcFace-Cos-Sim": f"{info.get('arcface_cos_sim', 0):.4f}",
            "X-CLIP-Cos-Sim": f"{info.get('clip_cos_sim', 0):.4f}",
            "X-LPIPS": f"{info.get('lpips', 0):.4f}",
            "X-PSNR": f"{info.get('psnr', 0):.1f}",
            "X-Delta-Linf": f"{info.get('delta_linf', 0):.4f}",
            "X-Processing-Ms": f"{elapsed_ms:.0f}",
        }

        return StreamingResponse(
            io.BytesIO(png_bytes),
            media_type="image/png",
            headers=headers,
        )

    # --- Legacy modes ---
    x = load_image_tensor(image_bytes)

    # Run protection
    if mode in ("encoder", "vit"):
        encoder = registry.noise_encoder if mode == "encoder" else registry.vit_encoder
        if encoder is None:
            raise HTTPException(503, f"{mode} encoder not trained yet. Use mode=pgd.")

        with torch.no_grad():
            x_protected = encoder.protect(x)

        # Quick evaluation
        clean_emb = registry.face_model(x)
        adv_emb = registry.face_model(x_protected)
        cos_sim = torch.nn.functional.cosine_similarity(clean_emb, adv_emb, dim=1).mean().item()
        delta = x_protected - x

        info = {
            "final_cosine_sim": cos_sim,
            "robust_cosine_sim": cos_sim,  # TODO: run EoT evaluation
            "delta_linf": delta.abs().max().item(),
            "delta_l2": delta.norm(p=2).item(),
            "num_steps": 1,
        }

    elif mode == "aspl":
        from core.attacks import ASPLAttack
        attack = ASPLAttack(
            face_model=registry.face_model,
            eot_wrapper=registry.eot_wrapper,
            epsilon=epsilon,
        )
        x_protected, info = attack.run(x)

    else:  # pgd
        from core.attacks import PGDAttack

        if step_size is None:
            step_size = epsilon / 4

        # Update EoT samples
        registry.eot_wrapper.num_samples = eot_samples

        attack = PGDAttack(
            face_model=registry.face_model,
            eot_wrapper=registry.eot_wrapper,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=steps,
        )
        x_protected, info = attack.run(x)

    elapsed_ms = (time.time() - start_time) * 1000

    # Convert to PNG
    png_bytes = tensor_to_png_bytes(x_protected)

    # Return image with metadata headers
    headers = {
        "X-Privacy-Mode": mode,
        "X-Cosine-Sim": f"{info['final_cosine_sim']:.4f}",
        "X-Robust-Cosine-Sim": f"{info.get('robust_cosine_sim', 0):.4f}",
        "X-Delta-Linf": f"{info['delta_linf']:.4f}",
        "X-Processing-Ms": f"{elapsed_ms:.0f}",
    }

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers=headers,
    )


@app.post("/analyze")
async def analyze_similarity(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Compare two face images and return cosine similarity.
    Useful for verifying that protection is working.
    """
    if registry.face_model is None:
        raise HTTPException(503, "Face model not loaded")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    x1 = load_image_tensor(bytes1)
    x2 = load_image_tensor(bytes2)

    with torch.no_grad():
        emb1 = registry.face_model(x1)
        emb2 = registry.face_model(x2)
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()

    return {
        "cosine_similarity": cos_sim,
        "is_same_person": cos_sim > 0.4,  # Typical threshold
        "threshold": 0.4,
    }
