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
