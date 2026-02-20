"""
Expectation over Transformation (EoT)

From Athalye et al. 2018 "Synthesizing Robust Adversarial Examples":

Instead of optimizing:
    max_δ  L(F(x + δ))

We optimize:
    max_δ  𝔼_{t ~ T} [ L(F(t(x + δ))) ]

By averaging the loss (and thus gradients) over N random transformations
per PGD step, the resulting perturbation δ is robust to the kinds of
image processing that social media platforms apply (JPEG, resize, blur).

Supported transformations:
  - Differentiable JPEG compression (random quality)
  - Random resize + crop back to original size
  - Gaussian blur (random σ)
  - Random horizontal flip

All transformations are differentiable so gradients flow through.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Optional

from .diff_jpeg import DiffJPEG


class RandomDiffJPEG(nn.Module):
    """Apply differentiable JPEG with random quality factor."""

    def __init__(self, quality_range: Tuple[int, int] = (50, 95)):
        super().__init__()
        self.quality_range = quality_range
        self.diff_jpeg = DiffJPEG()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = random.randint(self.quality_range[0], self.quality_range[1])
        return self.diff_jpeg(x, quality=q)


class RandomResize(nn.Module):
    """
    Resize to random smaller scale, then resize back to original.
    Simulates the resolution loss from social media re-encoding.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.0)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_h, new_w = max(8, int(H * scale)), max(8, int(W * scale))

        # Downsample then upsample (both differentiable)
        down = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
        return up


class RandomGaussianBlur(nn.Module):
    """Differentiable Gaussian blur with random sigma."""

    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 1.0), kernel_size: int = 5):
        super().__init__()
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        if sigma < 0.01:
            return x  # Skip near-zero blur

        # Create Gaussian kernel
        k = self.kernel_size
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, k, k).expand(x.shape[1], -1, -1, -1)

        padding = k // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])


class RandomCrop(nn.Module):
    """
    Random crop to a fraction of the image, then resize back.
    Simulates cropping that might occur on social media.
    """

    def __init__(self, crop_fraction_range: Tuple[float, float] = (0.8, 1.0)):
        super().__init__()
        self.crop_fraction_range = crop_fraction_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        frac = random.uniform(self.crop_fraction_range[0], self.crop_fraction_range[1])
        if frac >= 0.99:
            return x

        crop_h, crop_w = int(H * frac), int(W * frac)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        cropped = x[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# EoT Wrapper
# ---------------------------------------------------------------------------

class EoTWrapper(nn.Module):
    """
    Wraps a model with Expectation over Transformation.

    Usage in the attack loop:
        eot = EoTWrapper(face_model, config.eot)
        for step in range(num_steps):
            loss = eot(x_adv)  # Averages loss over N transforms
            loss.backward()     # Gradients are EoT-averaged
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        jpeg_quality_range: Tuple[int, int] = (50, 95),
        resize_scale_range: Tuple[float, float] = (0.5, 1.0),
        gaussian_sigma_range: Tuple[float, float] = (0.0, 1.0),
        crop_fraction_range: Tuple[float, float] = (0.8, 1.0),
        enable_jpeg: bool = True,
        enable_resize: bool = True,
        enable_gaussian: bool = True,
        enable_crop: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples

        # Build transformation list
        self.transforms: nn.ModuleList = nn.ModuleList()
        if enable_jpeg:
            self.transforms.append(RandomDiffJPEG(jpeg_quality_range))
        if enable_resize:
            self.transforms.append(RandomResize(resize_scale_range))
        if enable_gaussian:
            self.transforms.append(RandomGaussianBlur(gaussian_sigma_range))
        if enable_crop:
            self.transforms.append(RandomCrop(crop_fraction_range))

        if len(self.transforms) == 0:
            raise ValueError("At least one EoT transformation must be enabled")

    def apply_random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random subset of transforms (1-3 chosen at random)."""
        n = random.randint(1, min(3, len(self.transforms)))
        chosen = random.sample(list(self.transforms), n)
        for t in chosen:
            x = t(x)
        return x.clamp(0.0, 1.0)

    def forward(
        self,
        x_adv: torch.Tensor,
        x_clean_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute EoT-averaged embedding distance loss.

        Args:
            x_adv: (B, 3, 112, 112) perturbed image in [0, 1]
            x_clean_emb: (B, 512) pre-computed clean embedding (if None, just returns avg embedding)

        Returns:
            Scalar loss (negative cosine similarity — minimize to push embeddings apart)
        """
        total_loss = torch.tensor(0.0, device=x_adv.device)

        for _ in range(self.num_samples):
            # Apply random transforms
            x_t = self.apply_random_transform(x_adv)

            # Get embedding through transformed image
            emb = self.model(x_t)

            if x_clean_emb is not None:
                # Loss: negative cosine distance (we want to MAXIMIZE distance)
                # So we MINIMIZE negative distance = maximize distance
                cos_sim = F.cosine_similarity(emb, x_clean_emb, dim=1)
                total_loss = total_loss + cos_sim.mean()
            else:
                # If no clean embedding, return just the embedding
                total_loss = total_loss + emb.norm(dim=1).mean()

        return total_loss / self.num_samples

    def get_transformed_embedding(
        self, x: torch.Tensor, num_avg: int = 20
    ) -> torch.Tensor:
        """
        Get a robust embedding by averaging over many transformations.
        Useful for evaluation: how stable is the embedding after transforms?
        """
        embeddings = []
        with torch.no_grad():
            for _ in range(num_avg):
                x_t = self.apply_random_transform(x)
                emb = self.model(x_t)
                embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)
