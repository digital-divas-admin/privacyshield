"use client";

import type { Run } from "@/lib/types";
import { cn, simColor, formatDuration, formatEpsilon } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { SIMILARITY_THRESHOLD } from "@/lib/constants";

interface RunCardProps {
  run: Run;
  onPin: () => void;
  onDelete: () => void;
  onSelect: () => void;
  selected?: boolean;
}

export function RunCard({ run, onPin, onDelete, onSelect, selected }: RunCardProps) {
  const cosSim = run.metrics.arcfaceCosSim || run.metrics.cosineSim || 0;
  const pass = cosSim < SIMILARITY_THRESHOLD;
  const date = new Date(run.timestamp);

  return (
    <div
      className={cn(
        "cursor-pointer rounded-lg border p-3 transition-colors",
        selected ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/50"
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Badge variant={pass ? "success" : "destructive"} className="text-[10px] px-1.5 py-0">
              {pass ? "Pass" : "Fail"}
            </Badge>
            <span className="text-xs text-muted-foreground">{run.params.mode}</span>
            {run.pinned && <span className="text-xs">📌</span>}
          </div>
          <p className={cn("mt-1 font-mono text-sm font-bold", simColor(cosSim))}>
            {cosSim.toFixed(4)}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {formatEpsilon(run.params.epsilon)} | {run.params.steps} steps | {formatDuration(run.metrics.processingMs)}
          </p>
          {run.robustness && (
            <p className="text-[10px] text-muted-foreground">
              Robustness: {run.robustness.pass_count}/{run.robustness.total_count}
            </p>
          )}
        </div>
        <div className="flex flex-col gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onPin(); }}
            className="text-xs text-muted-foreground hover:text-foreground"
            title={run.pinned ? "Unpin" : "Pin as baseline"}
          >
            {run.pinned ? "📌" : "📍"}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className="text-xs text-muted-foreground hover:text-red-400"
            title="Delete run"
          >
            ✕
          </button>
        </div>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        {date.toLocaleTimeString()}
      </p>
    </div>
  );
}
