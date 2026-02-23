"use client";

import type { ProtectMode, HealthStatus } from "@/lib/types";
import { PROTECT_MODES } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface ModeSelectorProps {
  value: ProtectMode;
  onChange: (mode: ProtectMode) => void;
  health: HealthStatus | null;
}

function isModeAvailable(mode: ProtectMode, health: HealthStatus | null): boolean {
  if (!health) return false;
  if (!health.face_model_loaded) return false;
  switch (mode) {
    case "pgd":
    case "aspl":
      return true;
    case "encoder":
      return health.encoder_loaded;
    case "vit":
      return health.vit_encoder_loaded;
    case "v2":
    case "v2_full":
      return health.pipeline_v2_loaded;
    case "encoder_refined":
      return health.hybrid_mode_available;
    default:
      return false;
  }
}

export function ModeSelector({ value, onChange, health }: ModeSelectorProps) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {PROTECT_MODES.map((mode) => {
        const available = isModeAvailable(mode.value, health);
        const selected = value === mode.value;
        return (
          <button
            key={mode.value}
            disabled={!available}
            onClick={() => onChange(mode.value)}
            className={cn(
              "rounded-lg border p-3 text-left transition-colors",
              selected
                ? "border-primary bg-primary/10"
                : available
                ? "border-border hover:border-muted-foreground/50"
                : "border-border opacity-40 cursor-not-allowed"
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">{mode.label}</span>
              <span className="text-xs text-muted-foreground">{mode.speed}</span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{mode.description}</p>
            {!available && (
              <p className="mt-1 text-xs text-yellow-500">Not available</p>
            )}
          </button>
        );
      })}
    </div>
  );
}
