"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ProtectMode(str, Enum):
    PGD = "pgd"
    ENCODER = "encoder"
    VIT = "vit"
    ASPL = "aspl"
    V2 = "v2"
    V2_FULL = "v2_full"


class ProtectRequest(BaseModel):
    """Query parameters for the /protect endpoint."""
    mode: ProtectMode = ProtectMode.PGD
    epsilon: float = Field(default=8/255, ge=0, le=0.1, description="L∞ perturbation budget")
    steps: int = Field(default=50, ge=1, le=500, description="PGD iterations")
    step_size: Optional[float] = Field(default=None, description="Per-step size (auto = ε/4)")
    eot_samples: int = Field(default=10, ge=1, le=50, description="EoT samples per step")
    jpeg_quality_min: int = Field(default=50, ge=10, le=100)
    jpeg_quality_max: int = Field(default=95, ge=10, le=100)


class ProtectResponse(BaseModel):
    """Metadata returned alongside the protected image."""
    success: bool
    mode: str
    final_cosine_sim: float = Field(description="Cosine sim between clean and protected embeddings")
    robust_cosine_sim: float = Field(description="Cosine sim after random transforms")
    delta_linf: float = Field(description="Actual L∞ norm of perturbation")
    delta_l2: float = Field(description="L2 norm of perturbation")
    num_steps: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    face_model_loaded: bool
    encoder_loaded: bool
    vit_encoder_loaded: bool
    pipeline_v2_loaded: bool = False


class BatchProtectRequest(BaseModel):
    mode: ProtectMode = ProtectMode.PGD
    epsilon: float = Field(default=8/255, ge=0, le=0.1)
    steps: int = Field(default=50, ge=1, le=500)
    eot_samples: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Evaluate / Robustness schemas
# ---------------------------------------------------------------------------

class TransformResult(BaseModel):
    """Result for a single transform condition."""
    category: str = Field(description="Transform category: clean, jpeg, resize, blur, combined, platform")
    name: str = Field(description="Human-readable condition name")
    params: dict = Field(default_factory=dict, description="Transform parameters")
    cosine_similarity: float = Field(description="Cosine similarity after transform")
    is_match: bool = Field(description="Whether face recognition still matches (cos_sim > threshold)")
    protection_holds: bool = Field(description="Whether protection survives this transform (cos_sim < threshold)")


class EvaluateResponse(BaseModel):
    """Response from /evaluate endpoint."""
    threshold: float = Field(default=0.3, description="Cosine similarity threshold for protection")
    overall_pass: bool = Field(description="True if protection holds across ALL conditions")
    pass_count: int = Field(description="Number of conditions where protection holds")
    total_count: int = Field(description="Total number of conditions tested")
    perturbation_linf: float = Field(description="L∞ norm of perturbation")
    perturbation_l2: float = Field(description="L2 norm of perturbation")
    psnr: float = Field(description="Peak Signal-to-Noise Ratio in dB")
    results: List[TransformResult] = Field(description="Per-condition results")


class BatchProtectResult(BaseModel):
    """Result for a single image in a batch protect request."""
    index: int
    success: bool
    protected_image_b64: Optional[str] = None
    error: Optional[str] = None
    mode: str = ""
    arcface_cos_sim: float = 0.0
    clip_cos_sim: float = 0.0
    lpips: float = 0.0
    psnr: float = 0.0
    delta_linf: float = 0.0
    processing_time_ms: float = 0.0


class BatchProtectResponse(BaseModel):
    """Response from /protect/batch endpoint."""
    total: int
    succeeded: int
    failed: int
    results: List[BatchProtectResult]
