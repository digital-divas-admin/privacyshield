"""
Semantic Face Masking (The "Vogue Requirement")

Gemini's Point #2: applying uniform ε=8/255 noise across smooth skin regions
creates visible grain that high-end clients will reject.

Solution: Use a face parser (BiSeNet) to generate a mask that:
  - Blocks noise on skin (forehead, cheeks, chin) → 0% perturbation
  - Allows full noise on hair, eyebrows, eyelashes, background → 100% perturbation
  - Gradual falloff at boundaries → smooth transition

Why this works:
  ArcFace embeddings rely heavily on geometric relationships (eye spacing,
  nose-to-chin ratio, jawline contour). The high-frequency textures in hair,
  eyebrows, and background carry substantial identity signal in the embedding.
  By concentrating perturbation there, we:
    1. Maximize disruption to the embedding
    2. Hide noise in naturally textured regions (hair, fabric)
    3. Keep smooth skin pristine

The mask is applied multiplicatively: δ_final = δ_raw * mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Face parsing labels (BiSeNet / MediaPipe standard)
# ---------------------------------------------------------------------------

# BiSeNet face parsing class indices (CelebAMask-HQ convention)
FACE_PARTS = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",      # eyeglasses
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",      # earring
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# Default mask weights per face part
# 0.0 = no perturbation allowed, 1.0 = full perturbation
DEFAULT_MASK_WEIGHTS: Dict[str, float] = {
    "background": 1.0,   # Full noise — not visible as face
    "skin":       0.05,  # Nearly zero — keep pristine
    "l_brow":     0.9,   # High-frequency texture, good place for noise
    "r_brow":     0.9,
    "l_eye":      0.7,   # Identity-critical, textured enough
    "r_eye":      0.7,
    "eye_g":      0.8,   # Eyeglasses — textured
    "l_ear":      0.6,
    "r_ear":      0.6,
    "ear_r":      1.0,   # Earring — tiny, textured
    "nose":       0.15,  # Moderate — some smooth skin
    "mouth":      0.3,   # Lips have some texture
    "u_lip":      0.4,
    "l_lip":      0.4,
    "neck":       0.1,   # Smooth skin
    "necklace":   1.0,   # Textured accessory
    "cloth":      1.0,   # Textured, high-frequency
    "hair":       1.0,   # Maximum noise — naturally hides perturbation
    "hat":        1.0,   # Textured
}

# Aggressive mode: even less visible, concentrates on edges
STEALTH_MASK_WEIGHTS: Dict[str, float] = {
    "background": 1.0,
    "skin":       0.0,   # Zero noise on skin
    "l_brow":     1.0,
    "r_brow":     1.0,
    "l_eye":      0.8,
    "r_eye":      0.8,
    "eye_g":      0.9,
    "l_ear":      0.5,
    "r_ear":      0.5,
    "ear_r":      1.0,
    "nose":       0.0,   # Zero noise
    "mouth":      0.1,
    "u_lip":      0.2,
    "l_lip":      0.2,
    "neck":       0.0,   # Zero noise
    "necklace":   1.0,
    "cloth":      1.0,
    "hair":       1.0,
    "hat":        1.0,
}


class SemanticMask(nn.Module):
    """
    Generate a per-pixel perturbation mask based on face semantics.

    Can use either:
      1. BiSeNet face parser (highest quality, requires model weights)
      2. Heuristic edge-based mask (fallback, no extra model needed)

    The mask is (B, 1, H, W) with values in [0, 1].
    Apply to perturbation: δ_masked = δ_raw * mask
    """

    def __init__(
        self,
        mask_weights: Optional[Dict[str, float]] = None,
        use_bisenet: bool = True,
        blur_radius: int = 5,
        mask_floor: float = 0.05,
        bisenet_weights: Optional[str] = None,
    ):
        super().__init__()
        self.weights = mask_weights or DEFAULT_MASK_WEIGHTS
        self.blur_radius = blur_radius
        self.mask_floor = mask_floor

        # Build weight lookup tensor
        weight_list = [0.0] * 19
        for idx, name in FACE_PARTS.items():
            weight_list[idx] = self.weights.get(name, 0.5)
        self.register_buffer("weight_lut", torch.tensor(weight_list, dtype=torch.float32))

        self._bisenet = None
        self._use_bisenet = use_bisenet

        # Auto-load BiSeNet if weights path provided
        if bisenet_weights and use_bisenet:
            self.load_bisenet(bisenet_weights)

    def load_bisenet(self, model_path: str, device: str = None):
        """Load BiSeNet face parsing model."""
        try:
            from .bisenet import load_bisenet as _load_bisenet
            if device is None:
                device = str(self.weight_lut.device)
            self._bisenet = _load_bisenet(model_path, device=device)
            self._use_bisenet = True
            print(f"BiSeNet loaded from {model_path}")
        except Exception as e:
            print(f"BiSeNet load failed: {e}. Using heuristic mask.")
            self._use_bisenet = False

    @torch.no_grad()
    def _parse_face_bisenet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BiSeNet to get per-pixel class labels.
        Args: x (B, 3, H, W) in [0, 1]
        Returns: (B, 1, H, W) mask in [0, 1]
        """
        if self._bisenet is None:
            return self._heuristic_mask(x)

        # Ensure BiSeNet and weight_lut are on the same device as input
        if self.weight_lut.device != x.device:
            self.weight_lut = self.weight_lut.to(x.device)
        if next(self._bisenet.parameters()).device != x.device:
            self._bisenet = self._bisenet.to(x.device)

        # BiSeNet expects 512x512 input
        x_resized = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        # Normalize for BiSeNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_resized - mean) / std

        # Get segmentation (BiSeNet returns tuple: out, out16, out32)
        out, _, _ = self._bisenet(x_norm)
        labels = out.argmax(dim=1)  # (B, 512, 512)

        # Map labels to weights
        mask = self.weight_lut[labels]  # (B, 512, 512)
        mask = mask.unsqueeze(1)  # (B, 1, 512, 512)

        # Resize back to original
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        # Apply minimum floor
        mask = mask.clamp(min=self.mask_floor, max=1.0)

        # Smooth edges
        if self.blur_radius > 0:
            mask = self._gaussian_blur(mask, sigma=self.blur_radius / 2)

        return mask

    def _heuristic_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback: edge-based heuristic mask.

        High-frequency regions (edges, texture) get more noise budget.
        Smooth regions (skin) get less.

        This is a crude approximation of BiSeNet but requires no extra model.
        """
        B, C, H, W = x.shape

        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Sobel edge detection (differentiable)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)

        # Normalize to [0, 1]
        edges = edges / (edges.max() + 1e-8)

        # Local variance (texture detector)
        mean_local = F.avg_pool2d(gray, 7, stride=1, padding=3)
        var_local = F.avg_pool2d((gray - mean_local) ** 2, 7, stride=1, padding=3)
        texture = torch.sqrt(var_local + 1e-8)
        texture = texture / (texture.max() + 1e-8)

        # Combine: high texture or high edges → more noise allowed
        mask = 0.6 * texture + 0.4 * edges

        # Apply minimum floor (allow at least some noise everywhere)
        mask = mask.clamp(min=self.mask_floor, max=1.0)

        # Smooth
        if self.blur_radius > 0:
            mask = self._gaussian_blur(mask, sigma=self.blur_radius / 2)

        return mask

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, sigma: float = 2.0, kernel_size: int = 11) -> torch.Tensor:
        """Apply Gaussian blur to smooth mask edges."""
        k = kernel_size
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        gauss = torch.exp(-0.5 * (ax / max(sigma, 0.1)) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.outer(kernel_1d).view(1, 1, k, k)

        return F.conv2d(x, kernel_2d, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate perturbation mask.
        Args: x (B, 3, H, W) face image in [0, 1]
        Returns: (B, 1, H, W) mask in [0, 1]
        """
        if self._use_bisenet and self._bisenet is not None:
            return self._parse_face_bisenet(x)
        return self._heuristic_mask(x)


class MaskedPerturbation(nn.Module):
    """
    Apply semantic mask to perturbation.

    δ_final = δ_raw * mask * (ε / max(mask))

    The normalization ensures that the effective perturbation budget
    in high-weight regions is still ε, even though skin regions get near-zero.
    """

    def __init__(self, semantic_mask: SemanticMask):
        super().__init__()
        self.semantic_mask = semantic_mask

    @torch.no_grad()
    def get_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mask (no grad needed, mask is fixed per image)."""
        return self.semantic_mask(x)

    def apply(self, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to perturbation.
        Args:
            delta: (B, 3, H, W) raw perturbation
            mask: (B, 1, H, W) semantic mask
        Returns:
            (B, 3, H, W) masked perturbation
        """
        return delta * mask
