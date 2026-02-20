import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Color for cosine similarity: green if low (protected), red if high (matched) */
export function simColor(cosSim: number): string {
  if (cosSim < 0.2) return "text-green-500";
  if (cosSim < 0.3) return "text-yellow-500";
  return "text-red-500";
}

export function simBgColor(cosSim: number): string {
  if (cosSim < 0.2) return "bg-green-500/10 border-green-500/30";
  if (cosSim < 0.3) return "bg-yellow-500/10 border-yellow-500/30";
  return "bg-red-500/10 border-red-500/30";
}

export function simCellBg(cosSim: number): string {
  if (cosSim < 0.2) return "bg-green-600";
  if (cosSim < 0.3) return "bg-yellow-600";
  return "bg-red-600";
}

/** Format epsilon as fraction of 255 */
export function formatEpsilon(eps: number): string {
  return `${Math.round(eps * 255)}/255`;
}

/** Format milliseconds as human-readable */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Generate a unique ID */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Convert a Blob to a data URL */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/** Convert a data URL to a Blob */
export function dataUrlToBlob(dataUrl: string): Blob {
  const parts = dataUrl.split(",");
  const mime = parts[0].match(/:(.*?);/)?.[1] || "image/png";
  const raw = atob(parts[1]);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  return new Blob([arr], { type: mime });
}
