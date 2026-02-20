"use client";

import type { BatchResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

interface BatchSummaryProps {
  result: BatchResult;
}

export function BatchSummary({ result }: BatchSummaryProps) {
  const passCount = result.results.filter((r) => r.success && r.arcface_cos_sim < 0.3).length;

  return (
    <div className="flex items-center gap-4 rounded-lg border p-4">
      <Badge
        variant={result.failed === 0 && passCount === result.total ? "success" : "warning"}
        className="px-3 py-1 text-sm"
      >
        {passCount}/{result.total} Protected
      </Badge>
      <div className="flex gap-4 text-sm text-muted-foreground">
        <span>{result.succeeded} succeeded</span>
        {result.failed > 0 && <span className="text-red-400">{result.failed} failed</span>}
      </div>
    </div>
  );
}
