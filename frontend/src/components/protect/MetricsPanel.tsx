"use client";

import type { ProtectionMetrics } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { cn, simColor, formatDuration, formatEpsilon } from "@/lib/utils";

interface MetricsPanelProps {
  metrics: ProtectionMetrics;
}

function MetricCard({
  label,
  value,
  colorClass,
  detail,
}: {
  label: string;
  value: string;
  colorClass?: string;
  detail?: string;
}) {
  return (
    <Card>
      <CardContent className="p-3">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={cn("mt-0.5 text-lg font-bold font-mono", colorClass)}>{value}</p>
        {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
      </CardContent>
    </Card>
  );
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const cosSim = metrics.arcfaceCosSim || metrics.cosineSim || 0;
  const isV2 = metrics.mode?.startsWith("v2");

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
      <MetricCard
        label="ArcFace Cos Sim"
        value={cosSim.toFixed(4)}
        colorClass={simColor(cosSim)}
        detail={cosSim < 0.3 ? "Protected" : cosSim < 0.4 ? "Borderline" : "Not protected"}
      />
      {isV2 && metrics.clipCosSim > 0 && (
        <MetricCard
          label="CLIP Cos Sim"
          value={metrics.clipCosSim.toFixed(4)}
          colorClass={simColor(metrics.clipCosSim)}
        />
      )}
      {isV2 && metrics.lpips > 0 && (
        <MetricCard
          label="LPIPS"
          value={metrics.lpips.toFixed(4)}
          detail="Lower = less visible"
        />
      )}
      {metrics.psnr > 0 && (
        <MetricCard
          label="PSNR"
          value={`${metrics.psnr.toFixed(1)} dB`}
          detail="Higher = less visible"
        />
      )}
      <MetricCard
        label="Delta L-inf"
        value={formatEpsilon(metrics.deltaLinf)}
      />
      <MetricCard
        label="Processing Time"
        value={formatDuration(metrics.processingMs)}
      />
    </div>
  );
}
