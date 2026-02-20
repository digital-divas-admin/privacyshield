import type { ProtectMode } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const PROTECT_MODES: {
  value: ProtectMode;
  label: string;
  description: string;
  speed: string;
}[] = [
  { value: "pgd", label: "PGD", description: "Iterative PGD attack — slow but reliable", speed: "~5-10s" },
  { value: "v2", label: "V2 Pipeline", description: "PGD + LPIPS + CLIP + semantic mask on aligned face", speed: "~5-10s" },
  { value: "v2_full", label: "V2 Full Image", description: "Full pipeline on full-size image with differentiable alignment", speed: "~10-20s" },
  { value: "encoder", label: "U-Net Encoder", description: "Single-pass encoder (requires training)", speed: "~50ms" },
  { value: "vit", label: "ViT Encoder", description: "ViT-S/8 single-pass (requires training)", speed: "~170ms" },
];

export const EPSILON_PRESETS = [
  { value: 4 / 255, label: "4/255 (subtle)" },
  { value: 8 / 255, label: "8/255 (default)" },
  { value: 12 / 255, label: "12/255 (strong)" },
  { value: 16 / 255, label: "16/255 (maximum)" },
];

export const SIMILARITY_THRESHOLD = 0.3;
export const MATCH_THRESHOLD = 0.4;

export const PLATFORM_PRESETS = [
  { key: "instagram", label: "Instagram", jpeg: 70, maxDim: 1080 },
  { key: "twitter", label: "Twitter/X", jpeg: 85, maxDim: 2048 },
  { key: "facebook", label: "Facebook", jpeg: 71, maxDim: 2048 },
  { key: "whatsapp", label: "WhatsApp", jpeg: 60, maxDim: 1600 },
  { key: "tiktok", label: "TikTok", jpeg: 75, maxDim: 1080 },
];

export const ROBUSTNESS_CATEGORIES = [
  { key: "clean", label: "Clean" },
  { key: "jpeg", label: "JPEG" },
  { key: "resize", label: "Resize" },
  { key: "blur", label: "Blur" },
  { key: "combined", label: "Combined" },
  { key: "platform", label: "Platform" },
  { key: "upscaler", label: "AI Upscaler" },
];
