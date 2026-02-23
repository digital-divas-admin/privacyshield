// ---------------------------------------------------------------------------
// API types matching backend schemas
// ---------------------------------------------------------------------------

export type ProtectMode = "pgd" | "encoder" | "vit" | "v2" | "v2_full" | "aspl" | "encoder_refined";

export interface ProtectParams {
  mode: ProtectMode;
  epsilon: number;
  steps: number;
  eot_samples: number;
  mask_mode: string;
  refine_steps?: number;
}

export interface ProtectionMetrics {
  mode: string;
  arcfaceCosSim: number;
  clipCosSim: number;
  lpips: number;
  psnr: number;
  deltaLinf: number;
  processingMs: number;
  // Legacy mode fields
  cosineSim?: number;
  robustCosineSim?: number;
  // Ensemble per-model similarities
  perModelSimilarity?: Record<string, number>;
}

export interface ProtectionResult {
  protectedImageBlob: Blob;
  protectedImageUrl: string;
  originalAlignedUrl?: string;
  metrics: ProtectionMetrics;
}

// ---------------------------------------------------------------------------
// Evaluate / Robustness types
// ---------------------------------------------------------------------------

export interface TransformResult {
  category: string;
  name: string;
  params: Record<string, unknown>;
  cosine_similarity: number;
  is_match: boolean;
  protection_holds: boolean;
  per_model_similarity?: Record<string, number>;
}

export interface EvaluateResult {
  threshold: number;
  overall_pass: boolean;
  pass_count: number;
  total_count: number;
  perturbation_linf: number;
  perturbation_l2: number;
  psnr: number;
  results: TransformResult[];
}

// ---------------------------------------------------------------------------
// Batch types
// ---------------------------------------------------------------------------

export interface BatchProtectItemResult {
  index: number;
  success: boolean;
  protected_image_b64?: string;
  error?: string;
  mode: string;
  arcface_cos_sim: number;
  clip_cos_sim: number;
  lpips: number;
  psnr: number;
  delta_linf: number;
  processing_time_ms: number;
}

export interface BatchResult {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchProtectItemResult[];
}

// ---------------------------------------------------------------------------
// Analyze types
// ---------------------------------------------------------------------------

export interface AnalyzeResult {
  cosine_similarity: number;
  is_same_person: boolean;
  threshold: number;
}

// ---------------------------------------------------------------------------
// Health types
// ---------------------------------------------------------------------------

export interface HealthStatus {
  status: string;
  device: string;
  face_model_loaded: boolean;
  encoder_loaded: boolean;
  vit_encoder_loaded: boolean;
  pipeline_v2_loaded: boolean;
  hybrid_mode_available: boolean;
  facenet_loaded: boolean;
  adaface_loaded: boolean;
  ensemble_models: string[];
  inswapper_loaded: boolean;
  ipadapter_loaded: boolean;
}

// ---------------------------------------------------------------------------
// Deepfake tool testing types
// ---------------------------------------------------------------------------

export interface DeepfakeToolResult {
  tool_name: string;
  clean_output_b64?: string;
  protected_output_b64?: string;
  clean_similarity: number;
  protected_similarity: number;
  protection_effective: boolean;
  error?: string;
  processing_time_ms: number;
}

export interface DeepfakeTestResult {
  inswapper?: DeepfakeToolResult;
  ipadapter?: DeepfakeToolResult;
  overall_verdict: string;
}

// ---------------------------------------------------------------------------
// Run history
// ---------------------------------------------------------------------------

export interface Run {
  id: string;
  timestamp: number;
  params: ProtectParams;
  originalImageUrl: string;
  protectedImageUrl: string;
  metrics: ProtectionMetrics;
  robustness?: EvaluateResult;
  pinned?: boolean;
}
