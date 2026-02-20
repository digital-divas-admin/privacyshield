"""
Noise Encoder (Amortized Attack)

Instead of running 50-step PGD for every image (~5-10s per image on GPU),
we train a U-Net to predict the perturbation δ in a single forward pass (~50ms).

Architecture:  x → U-Net → δ̂ → tanh → ε·δ̂ → x + δ̂

Training:
  1. Generate (x, δ*) pairs using PGD attack
  2. Train U-Net with combined loss:
     L = λ_mse · ‖δ̂ - δ*‖² + λ_emb · cos_sim(F(x+δ̂), F(x)) + λ_eot · EoT_loss

The embedding loss ensures the encoder learns the attack objective, not just
pixel-matching the PGD output. The EoT loss ensures compression robustness
is preserved even when the encoder makes different perturbation choices than PGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BatchNorm → LeakyReLU × 2"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample: MaxPool → ConvBlock"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample: Bilinear up → concat skip → ConvBlock"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = F.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([x, skip], dim=1))


class SelfAttention(nn.Module):
    """Self-attention at the bottleneck for global context."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(1, 2) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# U-Net Noise Encoder
# ---------------------------------------------------------------------------

class NoiseEncoder(nn.Module):
    """
    U-Net that predicts perturbation δ for a given face image.

    Input:  (B, 3, 112, 112) — aligned face image in [0, 1]
    Output: (B, 3, 112, 112) — perturbation δ scaled to [-ε, ε]

    The output goes through tanh and is scaled by epsilon, so the
    perturbation is automatically bounded.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_attention: bool = True,
        dropout: float = 0.1,
        epsilon: float = 8 / 255,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.depth = depth
        ch = base_channels

        # Encoder path
        self.enc_in = ConvBlock(in_channels, ch)
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_c = ch * (2 ** i) if i > 0 else ch
            out_c = ch * (2 ** (i + 1)) if i < depth - 1 else ch * (2 ** i) * 2
            self.encoders.append(DownBlock(in_c, out_c, dropout))

        # Bottleneck
        bottleneck_ch = ch * (2 ** depth)
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, dropout)
        self.attention = SelfAttention(bottleneck_ch) if use_attention else nn.Identity()

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_c = ch * (2 ** (i + 1)) + ch * (2 ** i) if i > 0 else ch * 2 + ch
            # After concat with skip, we have 2x channels
            dec_in = ch * (2 ** (i + 1)) * 2 if i < depth - 1 else bottleneck_ch + ch * (2 ** i)
            out_c = ch * (2 ** i) if i > 0 else ch
            self.decoders.append(UpBlock(dec_in, out_c, dropout))

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch // 2, out_channels, 1),
            nn.Tanh(),  # Output in [-1, 1], scaled to [-ε, ε]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights so initial δ ≈ 0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale down the final conv so initial perturbation is tiny
        with torch.no_grad():
            self.out_conv[-2].weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) face image in [0, 1]
        Returns:
            delta: (B, 3, 112, 112) perturbation in [-ε, ε]
        """
        # Encoder
        skips = []
        h = self.enc_in(x)
        skips.append(h)

        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)
        h = self.attention(h)

        # Decoder (reverse order, skip from encoder)
        for i, dec in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i
            h = dec(h, skips[skip_idx])

        # Output: tanh scaled to epsilon
        delta = self.out_conv(h) * self.epsilon
        return delta

    def protect(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: add perturbation to image.
        Returns: x_protected in [0, 1]
        """
        delta = self.forward(x)
        return (x + delta).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Training loss for the encoder
# ---------------------------------------------------------------------------

class EncoderLoss(nn.Module):
    """
    Combined training loss for the noise encoder.

    L = λ_mse · ‖δ̂ - δ*‖²          (match PGD output)
      + λ_emb · cos_sim(F(x+δ̂), F(x))  (embedding distance)
      + λ_per · LPIPS(x+δ̂, x)         (perceptual imperceptibility)

    The embedding loss is the most important — it ensures the encoder
    learns the actual attack objective, not just pixel-matching PGD.
    """

    def __init__(
        self,
        face_model: nn.Module,
        lambda_mse: float = 1.0,
        lambda_emb: float = 5.0,
        lambda_per: float = 0.5,
    ):
        super().__init__()
        self.face_model = face_model
        self.lambda_mse = lambda_mse
        self.lambda_emb = lambda_emb
        self.lambda_per = lambda_per

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
        x_clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            delta_pred: (B, 3, 112, 112) encoder output
            delta_target: (B, 3, 112, 112) PGD-generated perturbation
            x_clean: (B, 3, 112, 112) clean image
        """
        losses = {}

        # MSE loss (match PGD perturbation)
        mse = F.mse_loss(delta_pred, delta_target)
        losses["mse"] = mse

        # Embedding distance loss
        with torch.no_grad():
            clean_emb = self.face_model(x_clean)

        adv_emb = self.face_model((x_clean + delta_pred).clamp(0, 1))
        cos_sim = F.cosine_similarity(clean_emb, adv_emb, dim=1).mean()
        losses["cos_sim"] = cos_sim

        # Perceptual loss (simple L1 on the perturbation as a proxy)
        # Replace with LPIPS if available
        perceptual = delta_pred.abs().mean()
        losses["perceptual"] = perceptual

        total = (
            self.lambda_mse * mse
            + self.lambda_emb * cos_sim  # Minimize cosine sim
            + self.lambda_per * perceptual
        )
        losses["total"] = total

        return total, losses
