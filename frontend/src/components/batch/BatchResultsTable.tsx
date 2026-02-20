"use client";

import type { BatchProtectItemResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { cn, simColor, formatDuration } from "@/lib/utils";

interface BatchResultsTableProps {
  results: BatchProtectItemResult[];
  fileNames: string[];
}

export function BatchResultsTable({ results, fileNames }: BatchResultsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="py-2 text-left text-xs text-muted-foreground">#</th>
            <th className="py-2 text-left text-xs text-muted-foreground">Image</th>
            <th className="py-2 text-left text-xs text-muted-foreground">Preview</th>
            <th className="py-2 text-left text-xs text-muted-foreground">Mode</th>
            <th className="py-2 text-right text-xs text-muted-foreground">Cos Sim</th>
            <th className="py-2 text-right text-xs text-muted-foreground">Time</th>
            <th className="py-2 text-right text-xs text-muted-foreground">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.index} className="border-b border-border">
              <td className="py-2 text-xs text-muted-foreground">{r.index + 1}</td>
              <td className="py-2 text-xs max-w-[120px] truncate">{fileNames[r.index] || `Image ${r.index + 1}`}</td>
              <td className="py-2">
                {r.protected_image_b64 && (
                  <img
                    src={`data:image/png;base64,${r.protected_image_b64}`}
                    alt={`Protected ${r.index + 1}`}
                    className="h-10 w-10 rounded object-cover"
                  />
                )}
              </td>
              <td className="py-2 text-xs">{r.mode}</td>
              <td className={cn("py-2 text-right font-mono text-xs", simColor(r.arcface_cos_sim))}>
                {r.success ? r.arcface_cos_sim.toFixed(4) : "—"}
              </td>
              <td className="py-2 text-right text-xs text-muted-foreground">
                {formatDuration(r.processing_time_ms)}
              </td>
              <td className="py-2 text-right">
                {r.success ? (
                  <Badge variant={r.arcface_cos_sim < 0.3 ? "success" : "destructive"} className="text-[10px]">
                    {r.arcface_cos_sim < 0.3 ? "Pass" : "Fail"}
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="text-[10px]">Error</Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
