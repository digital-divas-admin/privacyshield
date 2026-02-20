"use client";

import type { EvaluateResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

interface OverallVerdictProps {
  result: EvaluateResult;
}

export function OverallVerdict({ result }: OverallVerdictProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border p-4">
      <Badge
        variant={result.overall_pass ? "success" : "destructive"}
        className="px-3 py-1 text-sm"
      >
        {result.overall_pass ? "ALL PASS" : "SOME FAIL"}
      </Badge>
      <span className="text-sm text-muted-foreground">
        {result.pass_count}/{result.total_count} conditions protected
        (threshold: cos_sim &lt; {result.threshold})
      </span>
      <span className="ml-auto text-xs text-muted-foreground">
        L-inf: {result.perturbation_linf.toFixed(4)} | PSNR: {result.psnr.toFixed(1)} dB
      </span>
    </div>
  );
}
