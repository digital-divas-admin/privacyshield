"use client";

import { Badge } from "@/components/ui/badge";
import type { ProtectionMetrics } from "@/lib/types";
import { SIMILARITY_THRESHOLD } from "@/lib/constants";

interface ProtectionScoreProps {
  metrics: ProtectionMetrics;
}

export function ProtectionScore({ metrics }: ProtectionScoreProps) {
  const cosSim = metrics.arcfaceCosSim || metrics.cosineSim || 0;
  const pass = cosSim < SIMILARITY_THRESHOLD;

  return (
    <div className="flex items-center gap-3">
      <Badge
        variant={pass ? "success" : "destructive"}
        className="px-4 py-1.5 text-sm"
      >
        {pass ? "PROTECTED" : "NOT PROTECTED"}
      </Badge>
      <span className="text-sm text-muted-foreground">
        Cosine similarity: {cosSim.toFixed(4)}
        {pass
          ? ` (below ${SIMILARITY_THRESHOLD} threshold)`
          : ` (above ${SIMILARITY_THRESHOLD} threshold)`}
      </span>
    </div>
  );
}
