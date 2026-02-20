"""
Differentiable JPEG Compression

Approximates JPEG compression/decompression in a way that allows gradients
to flow through. This is critical for EoT — without it, we can't optimize
perturbations that survive JPEG compression on social media platforms.

Pipeline:  RGB → YCbCr → Chroma subsample → 8×8 DCT → Quantize(STE) → IDCT → Upsample → RGB

The quantization step is non-differentiable, so we use a Straight-Through
Estimator (STE): forward pass rounds normally, backward pass passes
gradients through as if rounding were the identity function.

Reference: Shin & Song, "JPEG-resistant Adversarial Images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Standard JPEG luminance and chrominance quantization tables
# ---------------------------------------------------------------------------

LUMINANCE_TABLE = torch.tensor([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=torch.float32)

CHROMINANCE_TABLE = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _quality_to_scale(quality: int) -> float:
    """Convert JPEG quality factor (1-100) to quantization scale."""
    if quality < 50:
        return 5000.0 / quality
    else:
        return 200.0 - 2.0 * quality


def _scale_quant_table(table: torch.Tensor, quality: int) -> torch.Tensor:
    """Scale a quantization table by JPEG quality factor."""
    s = _quality_to_scale(quality)
    scaled = torch.floor((table * s + 50.0) / 100.0)
    return scaled.clamp(min=1.0)


# ---------------------------------------------------------------------------
# DCT / IDCT for 8×8 blocks
# ---------------------------------------------------------------------------

def _dct_matrix() -> torch.Tensor:
    """Precompute the 8×8 DCT-II basis matrix."""
    n = 8
    D = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                D[k, i] = 1.0 / np.sqrt(n)
            else:
                D[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    return D


# Precompute once
_DCT = _dct_matrix()


def block_dct(blocks: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D DCT to 8×8 blocks.
    Args: blocks (B, C, H//8, W//8, 8, 8)
    Returns: DCT coefficients, same shape
    """
    D = _DCT.to(blocks.device)
    # 2D DCT = D @ block @ D^T
    return torch.einsum("ij,...jk,lk->...il", D, blocks, D)


def block_idct(coeffs: torch.Tensor) -> torch.Tensor:
    """Inverse 2D DCT."""
    D = _DCT.to(coeffs.device)
    # Inverse: D^T @ coeffs @ D
    return torch.einsum("ji,...jk,kl->...il", D, coeffs, D)


# ---------------------------------------------------------------------------
# Straight-Through Estimator for quantization
# ---------------------------------------------------------------------------

class STERound(torch.autograd.Function):
    """Round in forward pass, identity in backward pass."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return STERound.apply(x)


# ---------------------------------------------------------------------------
# Color space conversion
# ---------------------------------------------------------------------------

def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [0, 255] to YCbCr.
    Args: x (B, 3, H, W) in [0, 255]
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr to RGB [0, 255].
    Args: x (B, 3, H, W)
    """
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 128.0, x[:, 2:3] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# Main DiffJPEG module
# ---------------------------------------------------------------------------

class DiffJPEG(nn.Module):
    """
    Differentiable JPEG compression/decompression.

    Usage:
        jpeg = DiffJPEG()
        compressed = jpeg(image_tensor, quality=75)  # [0, 1] -> [0, 1]

    The quality parameter can change per call, enabling random quality
    sampling during EoT optimization.
    """

    def __init__(self):
        super().__init__()
        # Register quantization tables as buffers (move with .to())
        self.register_buffer("lum_table", LUMINANCE_TABLE)
        self.register_buffer("chrom_table", CHROMINANCE_TABLE)

    def forward(
        self,
        x: torch.Tensor,
        quality: int = 75,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [0, 1]
            quality: JPEG quality factor 1-100
        Returns:
            (B, 3, H, W) compressed image in [0, 1]
        """
        B, C, H, W = x.shape
        assert C == 3, "Expected RGB input"

        # Pad to multiple of 16 (for chroma subsampling + 8x8 blocks)
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, Hp, Wp = x.shape

        # Scale to [0, 255]
        x_255 = x * 255.0

        # RGB -> YCbCr
        ycbcr = rgb_to_ycbcr(x_255)

        # Process each channel
        y_ch = ycbcr[:, 0:1]    # (B, 1, H, W)
        cb_ch = ycbcr[:, 1:2]
        cr_ch = ycbcr[:, 2:3]

        # Chroma subsampling (2x average pool, differentiable)
        cb_sub = F.avg_pool2d(cb_ch, 2, stride=2)
        cr_sub = F.avg_pool2d(cr_ch, 2, stride=2)

        # Get scaled quantization tables
        q_lum = _scale_quant_table(self.lum_table, quality)
        q_chrom = _scale_quant_table(self.chrom_table, quality)

        # Compress each channel
        y_comp = self._compress_channel(y_ch, q_lum)
        cb_comp = self._compress_channel(cb_sub, q_chrom)
        cr_comp = self._compress_channel(cr_sub, q_chrom)

        # Chroma upsampling (bilinear, differentiable)
        cb_up = F.interpolate(cb_comp, size=(Hp, Wp), mode="bilinear", align_corners=False)
        cr_up = F.interpolate(cr_comp, size=(Hp, Wp), mode="bilinear", align_corners=False)

        # Reassemble YCbCr
        ycbcr_out = torch.cat([y_comp, cb_up, cr_up], dim=1)

        # YCbCr -> RGB
        rgb_out = ycbcr_to_rgb(ycbcr_out)

        # Scale back to [0, 1] and clamp
        result = (rgb_out / 255.0).clamp(0.0, 1.0)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :H, :W]

        return result

    def _compress_channel(
        self, ch: torch.Tensor, quant_table: torch.Tensor
    ) -> torch.Tensor:
        """
        DCT -> quantize -> dequantize -> IDCT for a single channel.
        Args:
            ch: (B, 1, H, W) — H, W must be multiples of 8
            quant_table: (8, 8) scaled quantization table
        """
        B, _, H, W = ch.shape
        assert H % 8 == 0 and W % 8 == 0

        # Reshape into 8x8 blocks: (B, 1, H//8, 8, W//8, 8) -> (B, 1, H//8, W//8, 8, 8)
        blocks = ch.view(B, 1, H // 8, 8, W // 8, 8)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)  # (B, 1, H//8, W//8, 8, 8)

        # Shift by -128 (center around 0)
        blocks = blocks - 128.0

        # DCT
        dct_coeffs = block_dct(blocks)

        # Quantize with STE
        q = quant_table.view(1, 1, 1, 1, 8, 8)
        quantized = ste_round(dct_coeffs / q) * q

        # IDCT
        reconstructed = block_idct(quantized) + 128.0

        # Reshape back: (B, 1, H//8, W//8, 8, 8) -> (B, 1, H, W)
        out = reconstructed.permute(0, 1, 2, 4, 3, 5)  # (B, 1, H//8, 8, W//8, 8)
        out = out.reshape(B, 1, H, W)

        return out
