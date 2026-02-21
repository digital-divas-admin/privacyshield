"use client";

import type { DeepfakeToolResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn, simColor } from "@/lib/utils";

interface ToolResultCardProps {
  result: DeepfakeToolResult;
  title: string;
}

function ToolResultCard({ result, title }: ToolResultCardProps) {
  if (result.error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-red-400">{result.error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
        <Badge variant={result.protection_effective ? "success" : "destructive"}>
          {result.protection_effective ? "Protected" : "Vulnerable"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Side-by-side output images */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">From Clean</p>
            {result.clean_output_b64 ? (
              <img
                src={`data:image/png;base64,${result.clean_output_b64}`}
                alt="Output from clean input"
                className="w-full rounded border border-border"
              />
            ) : (
              <div className="flex h-32 items-center justify-center rounded border border-border text-xs text-muted-foreground">
                No output
              </div>
            )}
          </div>
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">From Protected</p>
            {result.protected_output_b64 ? (
              <img
                src={`data:image/png;base64,${result.protected_output_b64}`}
                alt="Output from protected input"
                className="w-full rounded border border-border"
              />
            ) : (
              <div className="flex h-32 items-center justify-center rounded border border-border text-xs text-muted-foreground">
                No output
              </div>
            )}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 rounded-lg border p-3">
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Clean Similarity</p>
            <p className={cn("text-xl font-bold font-mono", simColor(result.clean_similarity))}>
              {result.clean_similarity.toFixed(4)}
            </p>
            <p className="text-xs text-muted-foreground">Should be high</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Protected Similarity</p>
            <p className={cn("text-xl font-bold font-mono", simColor(result.protected_similarity))}>
              {result.protected_similarity.toFixed(4)}
            </p>
            <p className="text-xs text-muted-foreground">Should be low</p>
          </div>
        </div>

        <p className="text-xs text-muted-foreground text-right">
          {result.processing_time_ms.toFixed(0)}ms
        </p>
      </CardContent>
    </Card>
  );
}

interface DeepfakeResultsProps {
  inswapper?: DeepfakeToolResult;
  ipadapter?: DeepfakeToolResult;
  overallVerdict: string;
}

export function DeepfakeResults({ inswapper, ipadapter, overallVerdict }: DeepfakeResultsProps) {
  const verdictBadge = {
    protected: { variant: "success" as const, label: "Protected" },
    partial: { variant: "outline" as const, label: "Partial Protection" },
    vulnerable: { variant: "destructive" as const, label: "Vulnerable" },
    error: { variant: "destructive" as const, label: "Error" },
    untested: { variant: "outline" as const, label: "Untested" },
  }[overallVerdict] ?? { variant: "outline" as const, label: overallVerdict };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Results</h3>
        <Badge variant={verdictBadge.variant} className="px-3 py-1">
          {verdictBadge.label}
        </Badge>
      </div>

      {inswapper && (
        <ToolResultCard result={inswapper} title="Inswapper (Roop)" />
      )}

      {ipadapter && (
        <ToolResultCard result={ipadapter} title="IP-Adapter FaceID Plus v2" />
      )}
    </div>
  );
}
