"use client";

import { Slider } from "@/components/ui/slider";
import { formatEpsilon } from "@/lib/utils";
import type { ProtectParams } from "@/lib/types";

interface ParameterPanelProps {
  params: ProtectParams;
  onChange: (params: ProtectParams) => void;
}

export function ParameterPanel({ params, onChange }: ParameterPanelProps) {
  const set = <K extends keyof ProtectParams>(key: K, value: ProtectParams[K]) =>
    onChange({ ...params, [key]: value });

  return (
    <div className="space-y-4">
      {/* Epsilon */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label className="text-sm font-medium">Epsilon (perturbation budget)</label>
          <span className="text-xs font-mono text-muted-foreground">{formatEpsilon(params.epsilon)}</span>
        </div>
        <Slider
          min={1}
          max={16}
          step={1}
          value={[Math.round(params.epsilon * 255)]}
          onValueChange={([v]) => set("epsilon", v / 255)}
        />
        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
          <span>Subtle</span>
          <span>Maximum</span>
        </div>
      </div>

      {/* Steps */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label className="text-sm font-medium">PGD Steps</label>
          <span className="text-xs font-mono text-muted-foreground">{params.steps}</span>
        </div>
        <Slider
          min={10}
          max={200}
          step={10}
          value={[params.steps]}
          onValueChange={([v]) => set("steps", v)}
        />
      </div>

      {/* EoT Samples */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label className="text-sm font-medium">EoT Samples</label>
          <span className="text-xs font-mono text-muted-foreground">{params.eot_samples}</span>
        </div>
        <Slider
          min={1}
          max={30}
          step={1}
          value={[params.eot_samples]}
          onValueChange={([v]) => set("eot_samples", v)}
        />
      </div>

      {/* Refine Steps (hybrid mode only) */}
      {params.mode === "encoder_refined" && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <label className="text-sm font-medium">Refinement Steps</label>
            <span className="text-xs font-mono text-muted-foreground">{params.refine_steps ?? 10}</span>
          </div>
          <Slider
            min={5}
            max={50}
            step={5}
            value={[params.refine_steps ?? 10]}
            onValueChange={([v]) => set("refine_steps", v)}
          />
          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>Fast (~2s)</span>
            <span>Strong (~4s)</span>
          </div>
        </div>
      )}

      {/* Mask mode */}
      <div>
        <label className="mb-2 block text-sm font-medium">Semantic Mask</label>
        <div className="flex gap-2">
          {["default", "hair_only", "none"].map((m) => (
            <button
              key={m}
              onClick={() => set("mask_mode", m)}
              className={`rounded border px-3 py-1 text-xs transition-colors ${
                params.mask_mode === m
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border text-muted-foreground hover:border-muted-foreground/50"
              }`}
            >
              {m === "default" ? "Default" : m === "hair_only" ? "Hair Only" : "None"}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
