"""
Enhanced Loss Functions

Fixes Gemini's Points #3 and #4:

Point #3 — LPIPS (Visual Guardrails):
  L∞ only bounds max pixel change, not perceptual quality.
  LPIPS is a neural network trained to judge how similar two images
  look to human vision. Adding it as a penalty forces the optimizer
  to hide noise in textures humans don't notice.

  New objective:  max [ ArcFace_distance - λ_lpips * LPIPS(x, x+δ) ]

Point #4 — CLIP Dual-Targeting:
  IP-Adapter FaceID Plus v2 uses BOTH ArcFace AND CLIP ViT-H/14.
  If we only blind ArcFace, CLIP can still extract enough likeness
  for a semi-accurate deepfake.

  New objective:  max [ α * ArcFace_distance + β * CLIP_distance - λ * LPIPS ]

This module provides a unified loss function that combines all three signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# LPIPS (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------------------------

class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual distance using VGG-16 features.

    Measures perceptual difference between two images using deep features
    from a pretrained VGG network. This is what makes the perturbation
    invisible to human eyes even when L∞ budget is fully used.

    If torchvision VGG is unavailable, falls back to a multi-scale
    gradient-based perceptual proxy.
    """

    def __init__(self, net: str = "vgg", device: str = "cuda"):
        super().__init__()
        self.device = device
        self._lpips = None
        self._fallback = False

        try:
            # Try loading lpips package first (gold standard)
            import lpips
            self._lpips = lpips.LPIPS(net=net).to(device)
            self._lpips.eval()
            for p in self._lpips.parameters():
                p.requires_grad = False
            print(f"LPIPS loaded ({net} backbone)")
        except ImportError:
            try:
                # Fallback: build from VGG features
                self._build_vgg_lpips(device)
                print("LPIPS using VGG feature extractor (built-in)")
            except Exception:
                self._fallback = True
                print("Warning: No LPIPS backend available. Using MSE proxy.")

    def _build_vgg_lpips(self, device: str):
        """Build a lightweight LPIPS approximation from VGG-16 features."""
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Extract features at specific layers (conv1_2, conv2_2, conv3_3, conv4_3)
        self._vgg_slices = nn.ModuleList()
        slice_points = [4, 9, 16, 23]
        prev = 0
        for sp in slice_points:
            self._vgg_slices.append(nn.Sequential(*list(vgg.children())[prev:sp]))
            prev = sp
        self._vgg_slices = self._vgg_slices.to(device)

        # Normalization for ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_features(self, x: torch.Tensor):
        """Extract multi-scale VGG features."""
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = []
        h = x
        for slice_module in self._vgg_slices:
            h = slice_module(h)
            features.append(h)
        return features

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual distance between x and y.
        Args:
            x, y: (B, 3, H, W) images in [0, 1]
        Returns:
            Scalar perceptual distance (lower = more similar)
        """
        if self._lpips is not None:
            # Scale to [-1, 1] as lpips expects
            return self._lpips(x * 2 - 1, y * 2 - 1).mean()

        if self._fallback:
            # Crude MSE proxy
            return F.mse_loss(x, y)

        # VGG feature matching
        feats_x = self._extract_features(x)
        feats_y = self._extract_features(y)

        total = torch.tensor(0.0, device=x.device)
        for fx, fy in zip(feats_x, feats_y):
            # Normalize features
            fx_norm = fx / (fx.norm(dim=1, keepdim=True) + 1e-8)
            fy_norm = fy / (fy.norm(dim=1, keepdim=True) + 1e-8)
            total = total + (fx_norm - fy_norm).pow(2).mean()

        return total / len(feats_x)


# ---------------------------------------------------------------------------
# CLIP Vision Encoder for Dual-Targeting
# ---------------------------------------------------------------------------

class CLIPVisionWrapper(nn.Module):
    """
    Frozen CLIP Vision Encoder for dual-targeting.

    IP-Adapter FaceID Plus v2 uses CLIP ViT-H/14 alongside ArcFace.
    By targeting both, we blind the full pipeline.

    The CLIP embedding captures:
      - Lighting and skin tone
      - Style and aesthetic qualities
      - High-level semantic features

    ArcFace captures:
      - Biometric geometry (eye spacing, jawline)
      - Identity-specific features

    Attacking both gives comprehensive protection.
    """

    def __init__(self, model_name: str = "ViT-H-14", device: str = "cuda"):
        super().__init__()
        self.device = device
        self._model = None
        self._preprocess = None
        self._backend = None  # "open_clip" or "hf"

        try:
            # Try open_clip (supports ViT-H/14)
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="laion2b_s32b_b79k"
            )
            self._model = model.visual.to(device).eval()
            self._input_size = 224
            self._backend = "open_clip"

            for p in self._model.parameters():
                p.requires_grad = False

            print(f"CLIP Vision loaded: {model_name}")

        except ImportError:
            try:
                # Fallback: use transformers
                from transformers import CLIPVisionModel
                self._model = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-large-patch14"
                ).to(device).eval()
                self._input_size = 224
                self._backend = "hf"

                for p in self._model.parameters():
                    p.requires_grad = False

                print("CLIP Vision loaded: clip-vit-large-patch14 (HuggingFace)")

            except ImportError:
                print("Warning: No CLIP backend available. Dual-targeting disabled.")

        # CLIP normalization
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract CLIP vision embedding.
        Args:
            x: (B, 3, H, W) in [0, 1]
        Returns:
            (B, D) L2-normalized CLIP embedding, or None if unavailable
        """
        if self._model is None:
            return None

        # Resize to CLIP input size
        if x.shape[2] != self._input_size or x.shape[3] != self._input_size:
            x = F.interpolate(x, size=(self._input_size, self._input_size),
                              mode="bicubic", align_corners=False).clamp(0, 1)

        # Normalize
        x = (x - self.clip_mean.to(x.device)) / self.clip_std.to(x.device)

        # Extract features
        if self._backend == "open_clip":
            # open_clip VisionTransformer: positional arg
            emb = self._model(x)
        else:
            # HuggingFace style
            outputs = self._model(pixel_values=x)
            emb = outputs.pooler_output

        return F.normalize(emb.float(), dim=1)


# ---------------------------------------------------------------------------
# Unified Multi-Target Loss
# ---------------------------------------------------------------------------

class PrivacyShieldLoss(nn.Module):
    """
    Unified loss function combining all four signals:

      L = α * cos_sim(ArcFace_clean, ArcFace_adv)     — identity attack
        + β * cos_sim(CLIP_clean, CLIP_adv)             — style/lighting attack
        + λ_lpips * LPIPS(x, x+δ)                      — visual quality penalty
        + λ_reg * ‖δ‖₁                                  — sparsity regularizer

    The optimizer MINIMIZES this loss, which means:
      - Minimizing ArcFace cos_sim = maximizing identity distance ✓
      - Minimizing CLIP cos_sim = maximizing style distance ✓
      - LPIPS term acts as penalty (added, not subtracted) — prevents ugly artifacts ✓
      - L1 reg encourages sparse perturbation ✓

    The balance between attack strength and visual quality is controlled by
    λ_lpips. Higher λ = prettier output but weaker attack.
    """

    def __init__(
        self,
        face_model: nn.Module,
        clip_model: Optional[CLIPVisionWrapper] = None,
        lpips_loss: Optional[LPIPSLoss] = None,
        alpha_arcface: float = 1.0,
        beta_clip: float = 0.5,
        lambda_lpips: float = 0.1,
        lambda_reg: float = 0.01,
    ):
        super().__init__()
        self.face_model = face_model
        self.clip_model = clip_model
        self.lpips_loss = lpips_loss
        self.alpha = alpha_arcface
        self.beta = beta_clip
        self.lambda_lpips = lambda_lpips
        self.lambda_reg = lambda_reg

    def forward(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        clean_arcface_emb: torch.Tensor,
        clean_clip_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute unified loss.

        Args:
            x_clean: (B, 3, H, W) clean image
            x_adv: (B, 3, H, W) perturbed image
            clean_arcface_emb: (B, 512) pre-computed clean ArcFace embedding
            clean_clip_emb: (B, D) pre-computed clean CLIP embedding (optional)

        Returns:
            loss: scalar to minimize
            metrics: dict of individual loss components
        """
        metrics = {}

        # --- ArcFace identity distance ---
        adv_arcface_emb = self.face_model(x_adv)
        arcface_cos = F.cosine_similarity(clean_arcface_emb, adv_arcface_emb, dim=1).mean()
        metrics["arcface_cos_sim"] = arcface_cos.item()

        loss = self.alpha * arcface_cos

        # --- CLIP style distance ---
        if self.clip_model is not None and self.clip_model.is_available and clean_clip_emb is not None:
            adv_clip_emb = self.clip_model(x_adv)
            if adv_clip_emb is not None:
                clip_cos = F.cosine_similarity(clean_clip_emb, adv_clip_emb, dim=1).mean()
                loss = loss + self.beta * clip_cos
                metrics["clip_cos_sim"] = clip_cos.item()

        # --- LPIPS visual quality penalty ---
        if self.lpips_loss is not None:
            lpips_val = self.lpips_loss(x_clean, x_adv)
            loss = loss + self.lambda_lpips * lpips_val
            metrics["lpips"] = lpips_val.item()

        # --- L1 sparsity regularizer ---
        delta = x_adv - x_clean
        l1_reg = delta.abs().mean()
        loss = loss + self.lambda_reg * l1_reg
        metrics["delta_l1"] = l1_reg.item()

        metrics["total_loss"] = loss.item()
        return loss, metrics
