"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
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
    facenet_loaded: bool = False
    adaface_loaded: bool = False
    ensemble_models: List[str] = Field(default_factory=list, description="Active ensemble model names")
    inswapper_loaded: bool = False
    ipadapter_loaded: bool = False


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
    per_model_similarity: Optional[Dict[str, float]] = Field(default=None, description="Per-model cosine similarities when ensemble is active")


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


# ---------------------------------------------------------------------------
# Deepfake tool testing schemas
# ---------------------------------------------------------------------------

class DeepfakeToolResultSchema(BaseModel):
    """Result from a single deepfake tool test."""
    tool_name: str = Field(description="Tool tested: inswapper or ipadapter")
    clean_output_b64: Optional[str] = Field(default=None, description="Base64 PNG of output from clean input")
    protected_output_b64: Optional[str] = Field(default=None, description="Base64 PNG of output from protected input")
    clean_similarity: float = Field(default=0.0, description="Cosine sim: clean output vs clean identity")
    protected_similarity: float = Field(default=0.0, description="Cosine sim: protected output vs clean identity")
    protection_effective: bool = Field(default=False, description="True if protected_similarity < threshold")
    error: Optional[str] = Field(default=None, description="Error message if test failed")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class DeepfakeTestResponse(BaseModel):
    """Combined response from deepfake tool testing."""
    inswapper: Optional[DeepfakeToolResultSchema] = None
    ipadapter: Optional[DeepfakeToolResultSchema] = None
    overall_verdict: str = Field(default="untested", description="protected, partial, vulnerable, or error")
