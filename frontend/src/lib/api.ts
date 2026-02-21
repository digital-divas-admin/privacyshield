import { API_URL } from "./constants";
import type {
  ProtectParams,
  ProtectionResult,
  ProtectionMetrics,
  EvaluateResult,
  BatchResult,
  AnalyzeResult,
  HealthStatus,
  DeepfakeTestResult,
} from "./types";

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Protect single image
// ---------------------------------------------------------------------------

export async function protectImage(
  file: File | Blob,
  params: ProtectParams
): Promise<ProtectionResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("mode", params.mode);
  form.append("epsilon", String(params.epsilon));
  form.append("steps", String(params.steps));
  form.append("eot_samples", String(params.eot_samples));
  form.append("mask_mode", params.mask_mode);

  const res = await fetch(`${API_URL}/protect`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Protect failed: ${res.status}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  // Parse metrics from headers — handle both v2 and legacy header formats
  const h = (name: string) => parseFloat(res.headers.get(name) || "0");

  const mode = res.headers.get("X-Privacy-Mode") || params.mode;
  const isV2 = mode.startsWith("v2");

  // Parse per-model similarity JSON header
  let perModelSimilarity: Record<string, number> | undefined;
  const perModelRaw = res.headers.get("X-Per-Model-Similarity");
  if (perModelRaw) {
    try {
      perModelSimilarity = JSON.parse(perModelRaw);
    } catch {
      // ignore parse errors
    }
  }

  const metrics: ProtectionMetrics = {
    mode,
    arcfaceCosSim: isV2 ? h("X-ArcFace-Cos-Sim") : h("X-Cosine-Sim"),
    clipCosSim: h("X-CLIP-Cos-Sim"),
    lpips: h("X-LPIPS"),
    psnr: h("X-PSNR"),
    deltaLinf: h("X-Delta-Linf"),
    processingMs: h("X-Processing-Ms"),
    cosineSim: h("X-Cosine-Sim"),
    robustCosineSim: h("X-Robust-Cosine-Sim"),
    perModelSimilarity,
  };

  return { protectedImageBlob: blob, protectedImageUrl: url, metrics };
}

// ---------------------------------------------------------------------------
// Batch protect
// ---------------------------------------------------------------------------

export async function protectBatch(
  files: File[],
  params: ProtectParams
): Promise<BatchResult> {
  const form = new FormData();
  files.forEach((f) => form.append("images", f));
  form.append("mode", params.mode);
  form.append("epsilon", String(params.epsilon));
  form.append("steps", String(params.steps));
  form.append("eot_samples", String(params.eot_samples));
  form.append("mask_mode", params.mask_mode);

  const res = await fetch(`${API_URL}/protect/batch`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Batch protect failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Evaluate robustness
// ---------------------------------------------------------------------------

export async function evaluateRobustness(
  cleanBlob: Blob,
  protectedBlob: Blob,
  threshold = 0.3
): Promise<EvaluateResult> {
  const form = new FormData();
  form.append("clean_image", cleanBlob, "clean.png");
  form.append("protected_image", protectedBlob, "protected.png");
  form.append("threshold", String(threshold));

  const res = await fetch(`${API_URL}/evaluate`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Evaluate failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Analyze similarity
// ---------------------------------------------------------------------------

export async function analyzeImages(
  image1: File | Blob,
  image2: File | Blob
): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image1", image1, "image1.png");
  form.append("image2", image2, "image2.png");

  const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Analyze failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Deepfake tool testing
// ---------------------------------------------------------------------------

export async function testDeepfake(
  cleanImage: File | Blob,
  protectedImage: File | Blob,
  options: {
    targetImage?: File | Blob;
    runInswapper?: boolean;
    runIpadapter?: boolean;
    prompt?: string;
    threshold?: number;
  } = {}
): Promise<DeepfakeTestResult> {
  const form = new FormData();
  form.append("clean_image", cleanImage, "clean.png");
  form.append("protected_image", protectedImage, "protected.png");
  if (options.targetImage) {
    form.append("target_image", options.targetImage, "target.png");
  }
  form.append("run_inswapper", String(options.runInswapper ?? true));
  form.append("run_ipadapter", String(options.runIpadapter ?? false));
  form.append("prompt", options.prompt ?? "a photo of a person");
  form.append("threshold", String(options.threshold ?? 0.3));

  const res = await fetch(`${API_URL}/test-deepfake`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Deepfake test failed: ${res.status}`);
  }
  return res.json();
}
