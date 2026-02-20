"use client";

import type { Run } from "@/lib/types";
import { cn, simColor, formatEpsilon, formatDuration } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { SIMILARITY_THRESHOLD } from "@/lib/constants";

interface RunComparisonProps {
  runA: Run;
  runB: Run;
}

function MetricRow({ label, a, b, lower }: { label: string; a: number; b: number; lower?: boolean }) {
  const better = lower ? (a < b ? "a" : a > b ? "b" : "tie") : (a > b ? "a" : a < b ? "b" : "tie");
  return (
    <tr className="border-b border-border">
      <td className="py-1.5 text-xs text-muted-foreground">{label}</td>
      <td className={cn("py-1.5 font-mono text-xs text-right", better === "a" && "text-green-400 font-bold")}>
        {a.toFixed(4)}
      </td>
      <td className={cn("py-1.5 font-mono text-xs text-right", better === "b" && "text-green-400 font-bold")}>
        {b.toFixed(4)}
      </td>
    </tr>
  );
}

export function RunComparison({ runA, runB }: RunComparisonProps) {
  const cosA = runA.metrics.arcfaceCosSim || runA.metrics.cosineSim || 0;
  const cosB = runB.metrics.arcfaceCosSim || runB.metrics.cosineSim || 0;
  const passA = cosA < SIMILARITY_THRESHOLD;
  const passB = cosB < SIMILARITY_THRESHOLD;

  return (
    <div className="space-y-4">
      {/* Images side by side */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant={passA ? "success" : "destructive"}>Run A</Badge>
            <span className="text-xs text-muted-foreground">{runA.params.mode}</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            <img src={runA.originalImageUrl} alt="Original A" className="rounded border border-border" />
            <img src={runA.protectedImageUrl} alt="Protected A" className="rounded border border-border" />
          </div>
          <p className="text-[10px] text-muted-foreground">
            {formatEpsilon(runA.params.epsilon)} | {runA.params.steps} steps
          </p>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant={passB ? "success" : "destructive"}>Run B</Badge>
            <span className="text-xs text-muted-foreground">{runB.params.mode}</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            <img src={runB.originalImageUrl} alt="Original B" className="rounded border border-border" />
            <img src={runB.protectedImageUrl} alt="Protected B" className="rounded border border-border" />
          </div>
          <p className="text-[10px] text-muted-foreground">
            {formatEpsilon(runB.params.epsilon)} | {runB.params.steps} steps
          </p>
        </div>
      </div>

      {/* Metrics table */}
      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            <th className="py-1 text-left text-xs text-muted-foreground">Metric</th>
            <th className="py-1 text-right text-xs text-muted-foreground">Run A</th>
            <th className="py-1 text-right text-xs text-muted-foreground">Run B</th>
          </tr>
        </thead>
        <tbody>
          <MetricRow label="ArcFace Cos Sim" a={cosA} b={cosB} lower />
          <MetricRow label="CLIP Cos Sim" a={runA.metrics.clipCosSim} b={runB.metrics.clipCosSim} lower />
          <MetricRow label="LPIPS" a={runA.metrics.lpips} b={runB.metrics.lpips} lower />
          <MetricRow label="PSNR (dB)" a={runA.metrics.psnr} b={runB.metrics.psnr} />
          <MetricRow label="Delta L-inf" a={runA.metrics.deltaLinf} b={runB.metrics.deltaLinf} lower />
          <MetricRow label="Processing (ms)" a={runA.metrics.processingMs} b={runB.metrics.processingMs} lower />
        </tbody>
      </table>

      {/* Robustness comparison */}
      {runA.robustness && runB.robustness && (
        <div>
          <p className="mb-2 text-sm font-medium">Robustness Comparison</p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted-foreground mb-1">
                Run A: {runA.robustness.pass_count}/{runA.robustness.total_count} pass
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">
                Run B: {runB.robustness.pass_count}/{runB.robustness.total_count} pass
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
