"""
PrivacyShield v2 — Unified Attack Pipeline

Integrates ALL components into a single coherent attack:
  1. Differentiable alignment (grid_sample)
  2. Semantic mask (concentrate noise in textured regions)
  3. EoT (survive JPEG, resize, blur)
  4. Dual-target loss (ArcFace + CLIP)
  5. LPIPS visual quality guard

This replaces the basic PGDAttack for production use.

Pipeline per PGD step:
  x_full ──► +δ ──► mask(δ) ──► grid_sample(align) ──► EoT(transforms) ──► ArcFace ──► loss_arcface
                                                     └──► CLIP ──────► loss_clip
                                                     └──► LPIPS(x, x+δ) ──► loss_quality
                                                                              │
                                                         total_loss ◄────────┘
                                                              │
                                                         loss.backward()
                                                              │
                                            gradients flow through grid_sample
                                            back to δ in original pixel space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class PipelineConfig:
    """All attack hyperparameters in one place."""
    # Perturbation budget
    epsilon: float = 8 / 255
    step_size: float = 2 / 255
    num_steps: int = 50
    random_start: bool = True

    # EoT
    eot_samples: int = 10
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    resize_scale_range: Tuple[float, float] = (0.5, 1.0)
    gaussian_sigma_range: Tuple[float, float] = (0.0, 1.0)

    # Loss weights
    alpha_arcface: float = 1.0
    beta_clip: float = 0.5
    lambda_lpips: float = 0.1
    lambda_reg: float = 0.01

    # Semantic mask
    use_semantic_mask: bool = True
    mask_mode: str = "default"  # "default", "stealth", "off"

    # Full image mode
    full_image: bool = False  # If True, attack full image with diff alignment

    # Ensemble
    adaface_weights: Optional[str] = None
    ensemble_weights: Optional[Dict[str, float]] = None

    # BiSeNet face parsing
    bisenet_weights: Optional[str] = None

    # Logging
    verbose: bool = True


class ProtectionPipeline:
    """
    Production attack pipeline with all components integrated.

    Usage:
        pipeline = ProtectionPipeline(config)
        pipeline.setup(device="cuda")

        # Protect a single pre-aligned face
        x_protected, metrics = pipeline.protect_aligned(x_aligned)

        # Protect a full-size image
        x_protected, metrics = pipeline.protect_full(image_bgr)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._ready = False

    def setup(self, device: Optional[str] = None, arcface_weights: Optional[str] = None):
        """Initialize all models. Call once at startup."""
        from .face_model import FaceEmbedder, FaceNetWrapper, AdaFaceWrapper, EnsembleFaceModel
        from .eot import EoTWrapper
        from .losses import PrivacyShieldLoss, LPIPSLoss, CLIPVisionWrapper
        from .semantic_mask import SemanticMask, DEFAULT_MASK_WEIGHTS, STEALTH_MASK_WEIGHTS
        from .diff_align import DifferentiableAligner

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        cfg = self.config

        # 1. Face model (ArcFace)
        print("Loading ArcFace...")
        self.face_model = FaceEmbedder(
            weights_path=arcface_weights,
            device=device,
        )

        # 2. Ensemble models (FaceNet + AdaFace)
        self.facenet_model = FaceNetWrapper(device=device)
        self.adaface_model = AdaFaceWrapper(
            weights_path=cfg.adaface_weights,
            device=device,
        )
        self.ensemble_model = EnsembleFaceModel(
            arcface_model=self.face_model,
            facenet_model=self.facenet_model,
            adaface_model=self.adaface_model,
            weights=cfg.ensemble_weights,
        )

        # 3. CLIP model (for dual-targeting)
        print("Loading CLIP Vision...")
        self.clip_model = CLIPVisionWrapper(device=device)

        # 4. LPIPS
        print("Loading LPIPS...")
        self.lpips_loss = LPIPSLoss(device=device)

        # 5. EoT wrapper (uses face_model internally for transforms)
        self.eot = EoTWrapper(
            model=self.face_model,
            num_samples=cfg.eot_samples,
            jpeg_quality_range=cfg.jpeg_quality_range,
            resize_scale_range=cfg.resize_scale_range,
            gaussian_sigma_range=cfg.gaussian_sigma_range,
        ).to(device)

        # 6. Unified loss (with ensemble if available)
        has_ensemble = len(self.ensemble_model.active_model_names) > 1
        self.loss_fn = PrivacyShieldLoss(
            face_model=self.face_model,
            clip_model=self.clip_model if self.clip_model.is_available else None,
            lpips_loss=self.lpips_loss,
            alpha_arcface=cfg.alpha_arcface,
            beta_clip=cfg.beta_clip,
            lambda_lpips=cfg.lambda_lpips,
            lambda_reg=cfg.lambda_reg,
            ensemble_model=self.ensemble_model if has_ensemble else None,
        )

        # 7. Semantic mask
        if cfg.use_semantic_mask and cfg.mask_mode != "off":
            weights = STEALTH_MASK_WEIGHTS if cfg.mask_mode == "stealth" else DEFAULT_MASK_WEIGHTS
            self.semantic_mask = SemanticMask(
                mask_weights=weights,
                bisenet_weights=cfg.bisenet_weights,
            )
            mask_backend = "BiSeNet" if self.semantic_mask._bisenet is not None else "heuristic"
            print(f"Semantic mask: {cfg.mask_mode} mode ({mask_backend})")
        else:
            self.semantic_mask = None

        # 8. Differentiable aligner (for full-image mode)
        self.aligner = DifferentiableAligner(output_size=112)

        self._ready = True
        print("Pipeline ready!")

    def protect_aligned(
        self,
        x: torch.Tensor,
        callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Protect pre-aligned face image(s).

        Args:
            x: (B, 3, 112, 112) aligned face in [0, 1]
            callback: optional fn(step, x_adv, metrics) per iteration

        Returns:
            x_protected: (B, 3, 112, 112) protected image
            metrics: dict with loss history and final measurements
        """
        assert self._ready, "Call .setup() first"
        cfg = self.config
        device = x.device
        x = x.detach().clone()
        start_time = time.time()

        # Pre-compute clean embeddings (frozen targets)
        has_ensemble = len(self.ensemble_model.active_model_names) > 1
        with torch.no_grad():
            if has_ensemble:
                clean_arcface = self.ensemble_model.get_all_clean_embeddings(x)
            else:
                clean_arcface = self.face_model(x)
            clean_clip = None
            if self.clip_model.is_available:
                clean_clip = self.clip_model(x)

        # Pre-compute semantic mask (if enabled)
        mask = None
        if self.semantic_mask is not None:
            with torch.no_grad():
                mask = self.semantic_mask(x)  # (B, 1, H, W)

        # Initialize perturbation
        if cfg.random_start:
            delta = torch.empty_like(x).uniform_(-cfg.epsilon, cfg.epsilon)
            if mask is not None:
                delta = delta * mask  # Respect mask from the start
            delta = delta.clamp(-(x), 1.0 - x)
        else:
            delta = torch.zeros_like(x)

        delta.requires_grad_(True)

        history = {
            "loss": [], "arcface_cos": [], "clip_cos": [], "lpips": [],
        }

        iterator = range(cfg.num_steps)
        if cfg.verbose:
            iterator = tqdm(iterator, desc="PGD v2", leave=False)

        for step in iterator:
            # Apply mask to delta before constructing adversarial image
            if mask is not None:
                delta_masked = delta * mask
            else:
                delta_masked = delta

            x_adv = (x + delta_masked).clamp(0.0, 1.0)

            # --- EoT-averaged loss ---
            # We compute the full loss over multiple random transforms
            total_loss = torch.tensor(0.0, device=device)
            step_metrics = {}

            for t in range(cfg.eot_samples):
                # Random transform
                x_t = self.eot.apply_random_transform(x_adv)

                # Unified loss (ArcFace + CLIP + LPIPS + reg)
                loss_t, metrics_t = self.loss_fn(x, x_t, clean_arcface, clean_clip)
                total_loss = total_loss + loss_t

                # Accumulate metrics from last sample for logging
                if t == cfg.eot_samples - 1:
                    step_metrics = metrics_t

            avg_loss = total_loss / cfg.eot_samples

            # Backward
            avg_loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach()

                # Sign gradient descent (minimize loss = maximize distance)
                delta.data -= cfg.step_size * grad.sign()

                # Project onto ε-ball
                delta.data = delta.data.clamp(-cfg.epsilon, cfg.epsilon)

                # Apply mask to projected delta
                if mask is not None:
                    delta.data = delta.data * mask

                # Project onto valid image range
                delta.data = (x + delta.data).clamp(0.0, 1.0) - x

                # Log
                history["loss"].append(avg_loss.item())
                history["arcface_cos"].append(step_metrics.get("arcface_cos_sim", 0))
                history["clip_cos"].append(step_metrics.get("clip_cos_sim", 0))
                history["lpips"].append(step_metrics.get("lpips", 0))

                if cfg.verbose and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{avg_loss.item():.3f}",
                        arc=f"{step_metrics.get('arcface_cos_sim', 0):.3f}",
                        clip=f"{step_metrics.get('clip_cos_sim', 0):.3f}",
                        lpips=f"{step_metrics.get('lpips', 0):.4f}",
                    )

                if callback:
                    callback(step, (x + delta.data * (mask if mask is not None else 1)).clamp(0, 1), step_metrics)

            delta.grad.zero_()

        # --- Final protected image ---
        if mask is not None:
            x_protected = (x + delta.data * mask).clamp(0.0, 1.0)
        else:
            x_protected = (x + delta.data).clamp(0.0, 1.0)

        # --- Final evaluation ---
        elapsed = time.time() - start_time

        with torch.no_grad():
            # ArcFace evaluation (use raw ArcFace embedding for backward compat)
            clean_arcface_emb = clean_arcface["arcface"] if isinstance(clean_arcface, dict) else clean_arcface
            final_arcface = self.face_model(x_protected)
            cos_clean = F.cosine_similarity(clean_arcface_emb, final_arcface, dim=1).mean().item()

            # Robust evaluation (more EoT samples)
            robust_emb = self.eot.get_transformed_embedding(x_protected, num_avg=30)
            cos_robust = F.cosine_similarity(clean_arcface_emb, robust_emb, dim=1).mean().item()

            # Per-model evaluation (ensemble)
            per_model_sim = {}
            if has_ensemble:
                all_clean = clean_arcface if isinstance(clean_arcface, dict) else {"arcface": clean_arcface}
                for name in self.ensemble_model.active_model_names:
                    if name in all_clean:
                        model = self.ensemble_model._models[name]
                        adv_emb = model(x_protected)
                        sim = F.cosine_similarity(all_clean[name], adv_emb, dim=1).mean().item()
                        per_model_sim[name] = sim

            # CLIP evaluation
            cos_clip = 0.0
            if self.clip_model.is_available and clean_clip is not None:
                final_clip = self.clip_model(x_protected)
                if final_clip is not None:
                    cos_clip = F.cosine_similarity(clean_clip, final_clip, dim=1).mean().item()

            # LPIPS
            lpips_final = self.lpips_loss(x, x_protected).item()

            delta_final = x_protected - x

        final_metrics = {
            "arcface_cos_sim": cos_clean,
            "arcface_cos_sim_robust": cos_robust,
            "clip_cos_sim": cos_clip,
            "lpips": lpips_final,
            "delta_linf": delta_final.abs().max().item(),
            "delta_l2": delta_final.norm(p=2).item(),
            "psnr": (10 * torch.log10(1.0 / (delta_final ** 2).mean())).item(),
            "processing_time_s": elapsed,
            "num_steps": cfg.num_steps,
            "history": history,
        }
        if per_model_sim:
            final_metrics["per_model_similarity"] = per_model_sim

        if cfg.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Protection Complete ({elapsed:.1f}s)")
            print(f"  ArcFace cos sim:  {cos_clean:.4f} (robust: {cos_robust:.4f})")
            if per_model_sim:
                for name, sim in per_model_sim.items():
                    if name != "arcface":
                        print(f"  {name.capitalize()} cos sim: {sim:.4f}")
            print(f"  CLIP cos sim:     {cos_clip:.4f}")
            print(f"  LPIPS:            {lpips_final:.4f}")
            print(f"  PSNR:             {final_metrics['psnr']:.1f} dB")
            print(f"  δ L∞:             {final_metrics['delta_linf']:.4f}")
            print(f"{'=' * 60}")

        return x_protected.detach(), final_metrics

    def protect_full(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Protect a full-size BGR image with differentiable alignment.

        The perturbation is optimized in the ORIGINAL image space,
        with gradients flowing through the differentiable face warp.

        Args:
            image_bgr: (H, W, 3) BGR numpy image
        Returns:
            x_protected: (1, 3, H, W) full-size protected image, or None
            metrics: dict
        """
        assert self._ready, "Call .setup() first"
        cfg = self.config

        # Detect landmarks (non-differentiable, done once)
        landmarks = self.aligner.detect_landmarks(image_bgr)
        if landmarks is None:
            return None, {"error": "No face detected"}

        # Convert to tensor
        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        x = x.unsqueeze(0).to(self.device)

        # Build differentiable alignment grid (done once)
        grid = self.aligner.build_grid(
            landmarks,
            src_size=(x.shape[2], x.shape[3]),
            device=self.device,
        )

        # Pre-compute clean embedding through differentiable warp
        has_ensemble = len(self.ensemble_model.active_model_names) > 1
        with torch.no_grad():
            aligned_clean = self.aligner.warp(x, grid)
            if has_ensemble:
                clean_embs_all = self.ensemble_model.get_all_clean_embeddings(aligned_clean)
            clean_arcface = self.face_model(aligned_clean)
            clean_clip = None
            if self.clip_model.is_available:
                clean_clip = self.clip_model(aligned_clean)

        # Full-image mode: skip semantic mask (it's designed for 112x112 faces,
        # not full photos). The grid_sample warp naturally restricts perturbation
        # to the face region through gradient sparsity.
        mask = None

        # Initialize delta in full image space
        delta = torch.empty_like(x).uniform_(-cfg.epsilon, cfg.epsilon)
        delta.requires_grad_(True)

        # EoT samples per step — use what the user configured
        eot_n = cfg.eot_samples

        if cfg.verbose:
            print(f"  protect_full: eps={cfg.epsilon:.4f}, step_size={cfg.step_size:.4f}, "
                  f"steps={cfg.num_steps}, eot={eot_n}, image={x.shape}")

        start_time = time.time()
        iterator = range(cfg.num_steps)
        if cfg.verbose:
            iterator = tqdm(iterator, desc="Full-image PGD", leave=False)

        for step in iterator:
            x_adv = (x + delta).clamp(0.0, 1.0)

            # Differentiable warp → aligned face
            aligned_adv = self.aligner.warp(x_adv, grid)

            # EoT-averaged identity + clip distance loss
            # (skip LPIPS/L1 reg — they fight the attack in full-image mode)
            total_loss = torch.tensor(0.0, device=self.device)
            for _ in range(eot_n):
                x_t = self.eot.apply_random_transform(aligned_adv)
                if has_ensemble:
                    # Ensemble: weighted loss across all FR models
                    ens_loss, _ = self.ensemble_model.ensemble_cosine_loss(clean_embs_all, x_t)
                    sample_loss = ens_loss
                else:
                    # Single-model: ArcFace only
                    adv_emb = self.face_model(x_t)
                    cos_sim = F.cosine_similarity(clean_arcface, adv_emb, dim=1).mean()
                    sample_loss = cos_sim
                if self.clip_model.is_available and clean_clip is not None:
                    adv_clip = self.clip_model(x_t)
                    if adv_clip is not None:
                        clip_cos = F.cosine_similarity(clean_clip, adv_clip, dim=1).mean()
                        sample_loss = sample_loss + cfg.beta_clip * clip_cos
                total_loss = total_loss + sample_loss

            avg_loss = total_loss / eot_n
            avg_loss.backward()

            if delta.grad is None:
                if cfg.verbose:
                    print(f"  [WARN] Step {step}: delta.grad is None!")
                break

            with torch.no_grad():
                # Use normalized gradient instead of sign for smoother perturbations
                # (sign creates salt-and-pepper noise that grid_sample averages out)
                grad = delta.grad
                grad_norm = grad.norm()
                if grad_norm > 1e-10:
                    # Normalized gradient step: same L2 magnitude as sign gradient
                    # but spatially smoother — crucial for grid_sample
                    normalized_grad = grad / grad_norm * grad.numel() ** 0.5
                    delta.data -= cfg.step_size * normalized_grad
                else:
                    delta.data -= cfg.step_size * grad.sign()

                delta.data = delta.data.clamp(-cfg.epsilon, cfg.epsilon)
                delta.data = (x + delta.data).clamp(0.0, 1.0) - x

                if cfg.verbose and (step == 0 or step == cfg.num_steps - 1 or step % 10 == 0):
                    test_adv = (x + delta.data).clamp(0, 1)
                    test_aligned = self.aligner.warp(test_adv, grid)
                    test_emb = self.face_model(test_aligned)
                    cur_cos = F.cosine_similarity(clean_arcface, test_emb, dim=1).mean().item()
                    print(f"  Step {step}: loss={avg_loss.item():.4f}, "
                          f"cos_sim={cur_cos:.4f}, "
                          f"delta_linf={delta.data.abs().max().item():.4f}")

            delta.grad.zero_()

        # Final: apply delta directly (no mask in full-image mode)
        x_protected = (x + delta.data).clamp(0.0, 1.0)
        elapsed = time.time() - start_time

        with torch.no_grad():
            aligned_prot = self.aligner.warp(x_protected, grid)
            prot_emb = self.face_model(aligned_prot)
            cos_sim = F.cosine_similarity(clean_arcface, prot_emb, dim=1).mean().item()

            # Per-model evaluation
            per_model_sim = {}
            if has_ensemble:
                for name in self.ensemble_model.active_model_names:
                    if name in clean_embs_all:
                        model = self.ensemble_model._models[name]
                        adv_emb = model(aligned_prot)
                        sim = F.cosine_similarity(clean_embs_all[name], adv_emb, dim=1).mean().item()
                        per_model_sim[name] = sim

        delta_final = x_protected - x
        metrics = {
            "arcface_cos_sim": cos_sim,
            "delta_linf": delta_final.abs().max().item(),
            "processing_time_s": elapsed,
        }
        if per_model_sim:
            metrics["per_model_similarity"] = per_model_sim

        if cfg.verbose:
            print(f"\nFull-image protection: cos_sim={cos_sim:.4f}, time={elapsed:.1f}s")
            if per_model_sim:
                for name, sim in per_model_sim.items():
                    if name != "arcface":
                        print(f"  {name}: cos_sim={sim:.4f}")

        return x_protected.detach(), metrics
