"""
ViT-based Noise Encoder (following IDProtector, CVPR 2025)

This is the architecture Gemini was pointing to: a Vision Transformer
that predicts adversarial perturbations in a single forward pass.

Why ViT over U-Net for this task:
  1. Global receptive field from layer 1 — every patch token attends to
     every other patch token, so perturbation at one face region is
     immediately informed by the entire face structure.
  2. Better generalization to unseen faces — ViTs learn more transferable
     features than CNNs for adversarial tasks.
  3. Scales better with data — ViT performance keeps climbing with more
     training data, which matters for a SaaS product.

Architecture:
  Input (B, 3, 224, 224) → PatchEmbed(8×8) → 12× Transformer blocks
  → Reshape to spatial → ConvHead → tanh → ε·δ → Resize to original

IDProtector numbers (CVPR 2025):
  - 0.173s per image (single pass)
  - ISM reduction >0.4 on InstantID
  - PSNR 32.94 dB (better than PGD baselines)
  - Robust to JPEG, crop, resize, affine transforms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to sequence of patch embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, embed_dim)"""
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Standard ViT block: LayerNorm → Attention → LayerNorm → MLP"""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Pixel Shuffle Decoder
# ---------------------------------------------------------------------------

class PixelShuffleHead(nn.Module):
    """
    Convert patch tokens back to a spatial perturbation map.
    Uses sub-pixel convolution (pixel shuffle) for high-res output.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 8,
        out_channels: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Each token needs to produce patch_size² × out_channels values
        total_out = out_channels * patch_size * patch_size
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, total_out),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) transformer output tokens
            grid_size: sqrt(N) — spatial grid dimension
        Returns:
            (B, out_channels, H, W) spatial output
        """
        B, N, _ = x.shape
        x = self.head(x)  # (B, N, C*P*P)

        # Reshape to spatial grid
        x = x.reshape(B, grid_size, grid_size, self.out_channels, self.patch_size, self.patch_size)
        # (B, grid, grid, C, P, P) → (B, C, grid*P, grid*P)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.out_channels, grid_size * self.patch_size, grid_size * self.patch_size
        )
        return x


# ---------------------------------------------------------------------------
# ViT Noise Encoder
# ---------------------------------------------------------------------------

class ViTNoiseEncoder(nn.Module):
    """
    Vision Transformer noise encoder for single-pass adversarial perturbation.

    Architecture: ViT-S/8 (Small, patch size 8)
    - embed_dim=384, depth=12, heads=6
    - Input: 224×224 (resized from arbitrary input)
    - Output: perturbation δ at 224×224, resized to match input

    IDProtector-style: the output is added to the input image for protection.

    Key design choices from IDProtector:
    1. Patch size 8 (not 16) — finer spatial granularity for the perturbation
    2. No CLS token — all tokens are spatial, all contribute to output
    3. Learned positional embeddings
    4. Pixel shuffle decoder — reconstructs full-res perturbation from tokens
    5. tanh + ε scaling — hard-bounds the perturbation
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        epsilon: float = 9 / 255,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.epsilon = epsilon

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        # Positional embedding (learned, no CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: tokens → spatial perturbation
        self.decoder = PixelShuffleHead(embed_dim, patch_size, out_channels=3)
        self.grid_size = grid_size

        # Initialize output to near-zero
        self._init_output_small()

    def _init_output_small(self):
        """Initialize so initial perturbation ≈ 0."""
        with torch.no_grad():
            for p in self.decoder.head[-1].parameters():
                p.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict perturbation for input image.

        Args:
            x: (B, 3, H, W) face image in [0, 1].
                If H, W ≠ 224, will be resized internally.
        Returns:
            delta: (B, 3, H, W) perturbation in [-ε, ε] at original resolution
        """
        B, C, H, W = x.shape

        # Resize to model input size if needed
        if H != self.img_size or W != self.img_size:
            x_resized = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        else:
            x_resized = x

        # Patch embed + positional encoding
        tokens = self.patch_embed(x_resized)  # (B, N, D)
        tokens = tokens + self.pos_embed

        # Transformer
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Decode to spatial perturbation
        delta = self.decoder(tokens, self.grid_size)  # (B, 3, 224, 224)

        # Bound with tanh
        delta = torch.tanh(delta) * self.epsilon

        # Resize perturbation to match original input resolution
        if H != self.img_size or W != self.img_size:
            delta = F.interpolate(
                delta, size=(H, W),
                mode="bilinear", align_corners=False,
            )

        return delta

    def protect(self, x: torch.Tensor) -> torch.Tensor:
        """Add perturbation to image. Returns protected image in [0, 1]."""
        delta = self.forward(x)
        return (x + delta).clamp(0.0, 1.0)

    def param_count(self) -> str:
        """Human-readable parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def vit_noise_encoder_small(epsilon: float = 9/255, **kwargs) -> ViTNoiseEncoder:
    """ViT-S/8 — matches IDProtector's architecture."""
    return ViTNoiseEncoder(
        img_size=224,
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        epsilon=epsilon,
        **kwargs,
    )


def vit_noise_encoder_tiny(epsilon: float = 9/255, **kwargs) -> ViTNoiseEncoder:
    """ViT-Ti/8 — lighter variant for faster inference / edge deployment."""
    return ViTNoiseEncoder(
        img_size=224,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=3,
        epsilon=epsilon,
        **kwargs,
    )


def vit_noise_encoder_base(epsilon: float = 9/255, **kwargs) -> ViTNoiseEncoder:
    """ViT-B/8 — heavier variant for maximum protection quality."""
    return ViTNoiseEncoder(
        img_size=224,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        epsilon=epsilon,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = vit_noise_encoder_small()
    print(f"ViT-S/8 Noise Encoder: {model.param_count()}")

    x = torch.randn(2, 3, 224, 224).clamp(0, 1)
    delta = model(x)
    print(f"Input:  {x.shape}")
    print(f"Delta:  {delta.shape}")
    print(f"Delta range: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"Epsilon: {model.epsilon:.4f}")

    # Test with non-standard resolution
    x2 = torch.randn(1, 3, 112, 112).clamp(0, 1)
    delta2 = model(x2)
    print(f"\nNon-standard input: {x2.shape}")
    print(f"Delta output:       {delta2.shape}")

    # Speed test
    import time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(x)

        start = time.time()
        for _ in range(100):
            _ = model(x)
        elapsed = (time.time() - start) / 100
        print(f"\nAverage inference time (CPU, batch=2): {elapsed*1000:.1f}ms")
