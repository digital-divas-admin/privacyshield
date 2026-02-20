"""
PrivacyShield Configuration
All hyperparameters for attack, EoT, encoder, and API.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AttackConfig:
    """PGD attack parameters."""
    epsilon: float = 8 / 255          # L∞ perturbation budget
    step_size: float = 2 / 255        # Per-step perturbation size
    num_steps: int = 50               # PGD iteration count
    targeted: bool = False            # If True, push toward target_embedding
    loss_type: str = "cosine"         # "cosine" | "l2" — distance metric in embedding space
    random_start: bool = True         # Initialize δ with uniform noise


@dataclass
class EoTConfig:
    """Expectation over Transformation parameters."""
    num_samples: int = 10             # Transforms averaged per PGD step
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    resize_scale_range: Tuple[float, float] = (0.5, 1.0)
    gaussian_sigma_range: Tuple[float, float] = (0.0, 1.0)
    enable_jpeg: bool = True
    enable_resize: bool = True
    enable_gaussian: bool = True
    enable_crop: bool = True
    crop_fraction_range: Tuple[float, float] = (0.8, 1.0)


@dataclass
class DiffJPEGConfig:
    """Differentiable JPEG approximation parameters."""
    default_quality: int = 75
    use_straight_through: bool = True  # STE for quantization step


@dataclass
class EncoderConfig:
    """U-Net noise encoder parameters."""
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    depth: int = 4                     # U-Net depth
    use_attention: bool = True         # Self-attention at bottleneck
    dropout: float = 0.1
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    checkpoint_dir: str = "./checkpoints"


@dataclass
class FaceModelConfig:
    """InsightFace/ArcFace wrapper parameters."""
    model_name: str = "buffalo_l"     # InsightFace model pack
    det_size: Tuple[int, int] = (640, 640)
    embedding_dim: int = 512
    device: str = "cuda"


@dataclass
class APIConfig:
    """FastAPI server parameters."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_image_size: int = 4096        # Max dimension in pixels
    default_mode: str = "pgd"         # "pgd" | "encoder"
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    attack: AttackConfig = field(default_factory=AttackConfig)
    eot: EoTConfig = field(default_factory=EoTConfig)
    diff_jpeg: DiffJPEGConfig = field(default_factory=DiffJPEGConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    face_model: FaceModelConfig = field(default_factory=FaceModelConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global singleton
config = Config()
