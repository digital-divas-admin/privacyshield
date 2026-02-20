"use client";

import type { EvaluateResult } from "@/lib/types";
import { RobustnessCell } from "./RobustnessCell";

interface PlatformPresetsProps {
  result: EvaluateResult;
}

export function PlatformPresets({ result }: PlatformPresetsProps) {
  const platforms = result.results.filter((r) => r.category === "platform");
  if (platforms.length === 0) return null;

  return (
    <div>
      <p className="mb-2 text-sm font-medium">Social Media Platforms</p>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5">
        {platforms.map((p) => (
          <div key={p.name} className="space-y-1">
            <p className="text-xs font-medium text-center">{p.name}</p>
            <RobustnessCell result={p} />
          </div>
        ))}
      </div>
    </div>
  );
}
