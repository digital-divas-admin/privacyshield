"""
PrivacyShield FastAPI Backend

Endpoints:
  POST /protect           — Protect a single image
  POST /protect/batch     — Protect multiple images
  POST /evaluate          — Test robustness across transforms
  POST /analyze           — Compare two face images
  GET  /health            — Health check + model status
  GET  /docs              — Auto-generated API docs (Swagger)

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import time
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ProtectMode, ProtectResponse, HealthResponse,
    TransformResult, EvaluateResponse,
    BatchProtectResult, BatchProtectResponse,
)
from config import config


# ---------------------------------------------------------------------------
# Social media platform presets
# ---------------------------------------------------------------------------

PLATFORM_PRESETS = {
    "instagram": {"jpeg_quality": 70, "max_dim": 1080, "sharpen": True, "label": "Instagram"},
    "twitter": {"jpeg_quality": 85, "max_dim": 2048, "sharpen": False, "label": "Twitter/X"},
    "facebook": {"jpeg_quality": 71, "max_dim": 2048, "sharpen": False, "label": "Facebook"},
    "whatsapp": {"jpeg_quality": 60, "max_dim": 1600, "sharpen": False, "label": "WhatsApp"},
    "tiktok": {"jpeg_quality": 75, "max_dim": 1080, "sharpen": False, "label": "TikTok"},
}


# ---------------------------------------------------------------------------
# Global model registry (loaded once at startup)
# ---------------------------------------------------------------------------

class ModelRegistry:
    face_model = None
    eot_wrapper = None
    noise_encoder = None
    vit_encoder = None
    pipeline_v2 = None          # Full v2 pipeline with LPIPS + CLIP + mask
    diff_jpeg = None            # Reusable DiffJPEG instance for evaluation
    upscaler = None             # Real-ESRGAN for upscaler robustness testing
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
            weights_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "arcface_r100.pth"),
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
        ).to(device)
        print("EoT wrapper initialized")

    # Load DiffJPEG for evaluation
    try:
        from core.diff_jpeg import DiffJPEG
        registry.diff_jpeg = DiffJPEG().to(device)
        print("DiffJPEG loaded for evaluation")
    except Exception as e:
        print(f"Warning: DiffJPEG load failed: {e}")

    # Load Real-ESRGAN for upscaler robustness testing
    try:
        # Fix basicsr/torchvision compatibility (functional_tensor removed in torchvision >= 0.18)
        import sys, types
        import torchvision.transforms.functional as _F_tv
        if 'torchvision.transforms.functional_tensor' not in sys.modules:
            _shim = types.ModuleType('torchvision.transforms.functional_tensor')
            _shim.rgb_to_grayscale = _F_tv.rgb_to_grayscale
            sys.modules['torchvision.transforms.functional_tensor'] = _shim

        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
        esrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        registry.upscaler = RealESRGANer(
            scale=4,
            model_path=esrgan_url,
            model=esrgan_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )
        print("Real-ESRGAN loaded for upscaler robustness testing")
    except Exception as e:
        print(f"Warning: Real-ESRGAN load failed: {e}")

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
        arcface_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "arcface_r100.pth")
        registry.pipeline_v2.setup(device=device, arcface_weights=arcface_weights)
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
    expose_headers=[
        "X-Privacy-Mode", "X-ArcFace-Cos-Sim", "X-CLIP-Cos-Sim",
        "X-LPIPS", "X-PSNR", "X-Delta-Linf", "X-Processing-Ms",
        "X-Cosine-Sim", "X-Robust-Cosine-Sim",
    ],
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


def load_image_pair_aligned(clean_bytes: bytes, protected_bytes: bytes):
    """
    Load a clean/protected image pair, aligning BOTH using the clean image's
    face landmarks and the SAME alignment method (DifferentiableAligner /
    grid_sample) that protect_full() uses during optimization.

    This is critical: protect_full() optimizes perturbation through grid_sample,
    so evaluation must also use grid_sample (not cv2.warpAffine / norm_crop)
    to produce matching aligned faces.
    """
    import cv2
    from core.diff_align import DifferentiableAligner

    clean_bgr = cv2.imdecode(np.frombuffer(clean_bytes, np.uint8), cv2.IMREAD_COLOR)
    prot_bgr = cv2.imdecode(np.frombuffer(protected_bytes, np.uint8), cv2.IMREAD_COLOR)

    if clean_bgr is None or prot_bgr is None:
        raise HTTPException(400, "Could not decode image")

    # Use the v2 pipeline's aligner if available, otherwise create one
    if registry.pipeline_v2 is not None:
        aligner = registry.pipeline_v2.aligner
    else:
        aligner = DifferentiableAligner()

    # Detect landmarks on the CLEAN image only
    landmarks = aligner.detect_landmarks(clean_bgr)
    if landmarks is None:
        raise HTTPException(422, "No face detected in clean image")

    # Convert both to tensors
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    prot_rgb = cv2.cvtColor(prot_bgr, cv2.COLOR_BGR2RGB)

    clean_t = torch.from_numpy(clean_rgb).float().permute(2, 0, 1) / 255.0
    prot_t = torch.from_numpy(prot_rgb).float().permute(2, 0, 1) / 255.0

    clean_t = clean_t.unsqueeze(0).to(registry.device)
    prot_t = prot_t.unsqueeze(0).to(registry.device)

    # Build grid from clean landmarks (same as protect_full does)
    grid = aligner.build_grid(
        landmarks,
        src_size=(clean_t.shape[2], clean_t.shape[3]),
        device=registry.device,
    )

    # Warp both images using grid_sample (matching protect_full's alignment)
    with torch.no_grad():
        clean_aligned = aligner.warp(clean_t, grid)
        prot_aligned = aligner.warp(prot_t, grid)

    return clean_aligned, prot_aligned


def load_image_tensor_simple(image_bytes: bytes) -> torch.Tensor:
    """Load image bytes as 112x112 tensor without face detection (for already-aligned images)."""
    from torchvision import transforms
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    return tfm(pil).unsqueeze(0).to(registry.device)


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
    For the MVP, we return just the aligned protected face.
    """
    return tensor_to_png_bytes(delta_aligned.unsqueeze(0) if delta_aligned.dim() == 3 else delta_aligned)


def apply_gaussian_blur(imgs: torch.Tensor, sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images."""
    device = imgs.device
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-0.5 * (ax / max(sigma, 1e-6)) ** 2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.outer(kernel_1d).view(1, 1, kernel_size, kernel_size).expand(3, -1, -1, -1)
    return F.conv2d(imgs, kernel_2d, padding=kernel_size // 2, groups=3).clamp(0, 1)


def apply_resize_roundtrip(imgs: torch.Tensor, scale: float, target_size: int = 112) -> torch.Tensor:
    """Resize down then back up to simulate resolution loss."""
    h, w = int(target_size * scale), int(target_size * scale)
    if h < 1 or w < 1:
        return imgs
    down = F.interpolate(imgs, size=(h, w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(target_size, target_size), mode="bilinear", align_corners=False)


def apply_platform_preset(imgs: torch.Tensor, preset: dict) -> torch.Tensor:
    """Apply a social media platform simulation transform chain."""
    result = imgs.clone()
    max_dim = preset["max_dim"]
    # Simulate resize to max_dim then back (proportional)
    scale = min(max_dim / 112.0, 1.0)  # For 112px aligned images, most presets won't downscale
    if scale < 1.0:
        result = apply_resize_roundtrip(result, scale)
    # JPEG compression
    if registry.diff_jpeg is not None:
        result = registry.diff_jpeg(result, quality=preset["jpeg_quality"])
    # Slight sharpen for Instagram
    if preset.get("sharpen"):
        # Unsharp mask: original + alpha * (original - blurred)
        blurred = apply_gaussian_blur(result, sigma=1.0)
        result = (result + 0.3 * (result - blurred)).clamp(0, 1)
    return result


def apply_ai_upscale(imgs: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Run Real-ESRGAN AI upscaler on aligned face, then downscale back to 112x112.
    This simulates someone running the protected image through an AI upscaler,
    which can act as a denoiser that strips adversarial perturbations.
    """
    import cv2

    if registry.upscaler is None:
        return imgs

    result_tensors = []
    for i in range(imgs.shape[0]):
        # Tensor (3, 112, 112) [0,1] → numpy BGR uint8
        img_rgb = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Run Real-ESRGAN (upscales to 448x448 at 4x)
        try:
            output_bgr, _ = registry.upscaler.enhance(img_bgr, outscale=scale)
        except Exception:
            result_tensors.append(imgs[i])
            continue

        # Downscale back to 112x112
        output_bgr = cv2.resize(output_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Back to tensor
        t = torch.from_numpy(output_rgb).float().permute(2, 0, 1) / 255.0
        result_tensors.append(t)

    return torch.stack(result_tensors).to(imgs.device)


@torch.no_grad()
def run_robustness_evaluation(
    clean_tensor: torch.Tensor,
    protected_tensor: torch.Tensor,
    threshold: float = 0.3,
) -> EvaluateResponse:
    """Run full robustness evaluation on a clean/protected image pair."""
    device = registry.device
    clean = clean_tensor.to(device)
    protected = protected_tensor.to(device)

    # Ensure batch dim
    if clean.dim() == 3:
        clean = clean.unsqueeze(0)
    if protected.dim() == 3:
        protected = protected.unsqueeze(0)

    clean_emb = registry.face_model(clean)
    results: List[TransformResult] = []

    def eval_condition(category: str, name: str, params: dict, transformed: torch.Tensor):
        emb = registry.face_model(transformed)
        cos_sim = F.cosine_similarity(clean_emb, emb, dim=1).item()
        results.append(TransformResult(
            category=category,
            name=name,
            params=params,
            cosine_similarity=round(cos_sim, 4),
            is_match=cos_sim > 0.4,
            protection_holds=cos_sim < threshold,
        ))

    # 1. Clean (no transform)
    eval_condition("clean", "No transform", {}, protected)

    # 2. JPEG compression
    if registry.diff_jpeg is not None:
        for q in [95, 85, 75, 50]:
            jpeg_img = registry.diff_jpeg(protected, quality=q)
            eval_condition("jpeg", f"JPEG Q={q}", {"quality": q}, jpeg_img)

    # 3. Resize roundtrip
    for scale in [0.75, 0.5, 0.25]:
        resized = apply_resize_roundtrip(protected, scale)
        eval_condition("resize", f"Resize {scale:.0%}", {"scale": scale}, resized)

    # 4. Gaussian blur
    for sigma in [0.5, 1.0, 2.0]:
        blurred = apply_gaussian_blur(protected, sigma)
        eval_condition("blur", f"Blur σ={sigma}", {"sigma": sigma}, blurred)

    # 5. Combined JPEG Q=75 + resize 0.75
    combined = apply_resize_roundtrip(protected, 0.75)
    if registry.diff_jpeg is not None:
        combined = registry.diff_jpeg(combined, quality=75)
    eval_condition("combined", "JPEG Q=75 + Resize 75%", {"jpeg_quality": 75, "scale": 0.75}, combined)

    # 6. Social media platform presets
    for platform_key, preset in PLATFORM_PRESETS.items():
        transformed = apply_platform_preset(protected, preset)
        eval_condition("platform", preset["label"], preset, transformed)

    # 7. AI upscaler (Real-ESRGAN) — tests if perturbation survives neural upscaling
    if registry.upscaler is not None:
        upscaled = apply_ai_upscale(protected, scale=4)
        eval_condition("upscaler", "Real-ESRGAN 4x", {"scale": 4}, upscaled)

        # Also test upscale + JPEG (common pipeline: upscale then share on social media)
        if registry.diff_jpeg is not None:
            upscaled_jpeg = registry.diff_jpeg(upscaled, quality=75)
            eval_condition("upscaler", "Real-ESRGAN 4x + JPEG Q=75",
                           {"scale": 4, "jpeg_quality": 75}, upscaled_jpeg)

    # Perturbation stats
    delta = protected - clean
    linf = delta.abs().max().item()
    l2 = delta.norm(p=2).item()
    mse = (delta ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    pass_count = sum(1 for r in results if r.protection_holds)

    return EvaluateResponse(
        threshold=threshold,
        overall_pass=pass_count == len(results),
        pass_count=pass_count,
        total_count=len(results),
        perturbation_linf=round(linf, 4),
        perturbation_l2=round(l2, 4),
        psnr=round(psnr, 1),
        results=results,
    )


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
        registry.pipeline_v2.config.step_size = epsilon / 4  # Keep proportional
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
            "robust_cosine_sim": cos_sim,
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


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_robustness(
    clean_image: UploadFile = File(...),
    protected_image: UploadFile = File(...),
    threshold: float = Form(0.3),
):
    """
    Evaluate protection robustness across multiple transform conditions.
    Takes the original clean image and the protected image, runs both through
    the face model after applying various transforms (JPEG, resize, blur,
    social media platform presets), and returns per-condition pass/fail.
    """
    if registry.face_model is None:
        raise HTTPException(503, "Face model not loaded")

    clean_bytes = await clean_image.read()
    protected_bytes = await protected_image.read()

    # Detect image size to pick the right loader
    # Full-size images need face detection + alignment; 112x112 crops don't
    clean_pil = Image.open(io.BytesIO(clean_bytes))
    is_full_image = max(clean_pil.size) > 200  # Full photo vs aligned crop

    if is_full_image:
        # Align BOTH images using landmarks from the CLEAN image only.
        # This prevents alignment drift (the face detector finds slightly
        # different landmarks on the perturbed image, which would invalidate
        # the perturbation during evaluation).
        clean_tensor, protected_tensor = load_image_pair_aligned(clean_bytes, protected_bytes)
    else:
        clean_tensor = load_image_tensor_simple(clean_bytes)
        protected_tensor = load_image_tensor_simple(protected_bytes)

    return run_robustness_evaluation(clean_tensor, protected_tensor, threshold)


@app.post("/protect/batch", response_model=BatchProtectResponse)
async def protect_batch(
    images: List[UploadFile] = File(...),
    mode: str = Form("pgd"),
    epsilon: float = Form(8/255),
    steps: int = Form(50),
    eot_samples: int = Form(10),
    mask_mode: str = Form("default"),
):
    """
    Protect multiple face images. Returns base64-encoded protected images + metrics.
    """
    if registry.face_model is None:
        raise HTTPException(503, "Face model not loaded")

    results: List[BatchProtectResult] = []

    for i, upload in enumerate(images):
        start_time = time.time()
        try:
            image_bytes = await upload.read()

            if mode in ("v2", "v2_full"):
                if registry.pipeline_v2 is None:
                    raise Exception("v2 pipeline not initialized")

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
                        raise Exception("Could not decode image")
                    x_protected, info = registry.pipeline_v2.protect_full(img_bgr)
                    if x_protected is None:
                        raise Exception(info.get("error", "Protection failed"))
                else:
                    x = load_image_tensor(image_bytes)
                    x_protected, info = registry.pipeline_v2.protect_aligned(x)

                elapsed_ms = (time.time() - start_time) * 1000
                png_bytes = tensor_to_png_bytes(x_protected)

                results.append(BatchProtectResult(
                    index=i,
                    success=True,
                    protected_image_b64=base64.b64encode(png_bytes).decode(),
                    mode=mode,
                    arcface_cos_sim=info.get("arcface_cos_sim", 0),
                    clip_cos_sim=info.get("clip_cos_sim", 0),
                    lpips=info.get("lpips", 0),
                    psnr=info.get("psnr", 0),
                    delta_linf=info.get("delta_linf", 0),
                    processing_time_ms=elapsed_ms,
                ))
            else:
                # PGD / encoder / vit modes
                x = load_image_tensor(image_bytes)

                if mode in ("encoder", "vit"):
                    encoder = registry.noise_encoder if mode == "encoder" else registry.vit_encoder
                    if encoder is None:
                        raise Exception(f"{mode} encoder not trained yet")
                    with torch.no_grad():
                        x_protected = encoder.protect(x)
                    clean_emb = registry.face_model(x)
                    adv_emb = registry.face_model(x_protected)
                    cos_sim = torch.nn.functional.cosine_similarity(clean_emb, adv_emb, dim=1).mean().item()
                    delta = x_protected - x
                    info = {"final_cosine_sim": cos_sim, "delta_linf": delta.abs().max().item()}
                else:  # pgd
                    from core.attacks import PGDAttack
                    step_size = epsilon / 4
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
                png_bytes = tensor_to_png_bytes(x_protected)

                results.append(BatchProtectResult(
                    index=i,
                    success=True,
                    protected_image_b64=base64.b64encode(png_bytes).decode(),
                    mode=mode,
                    arcface_cos_sim=info.get("final_cosine_sim", info.get("arcface_cos_sim", 0)),
                    delta_linf=info.get("delta_linf", 0),
                    processing_time_ms=elapsed_ms,
                ))

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append(BatchProtectResult(
                index=i,
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms,
            ))

    succeeded = sum(1 for r in results if r.success)
    return BatchProtectResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
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
