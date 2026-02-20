"use client";

import { useState } from "react";
import { useRunHistory } from "@/hooks/useRunHistory";
import { RunCard } from "./RunCard";
import { RunComparison } from "./RunComparison";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export function RunHistoryPanel() {
  const { runs, togglePin, deleteRun, clearAll, compareIds, setCompareIds } = useRunHistory();
  const [showComparison, setShowComparison] = useState(false);

  const selectedRuns = runs.filter((r) => compareIds.includes(r.id));

  const handleSelect = (runId: string) => {
    const prev = compareIds;
    if (prev[0] === runId) { setCompareIds([null, prev[1]]); return; }
    if (prev[1] === runId) { setCompareIds([prev[0], null]); return; }
    if (!prev[0]) { setCompareIds([runId, prev[1]]); return; }
    setCompareIds([prev[0], runId]);
  };

  if (runs.length === 0) {
    return (
      <div className="p-4 text-center text-xs text-muted-foreground">
        No runs yet. Protect an image to start.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <span className="text-xs font-medium text-muted-foreground">
          {runs.length} run{runs.length !== 1 ? "s" : ""}
        </span>
        <div className="flex gap-1">
          {selectedRuns.length === 2 && (
            <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={() => setShowComparison(true)}>
              Compare
            </Button>
          )}
          <Button variant="ghost" size="sm" className="h-6 text-[10px]" onClick={clearAll}>
            Clear
          </Button>
        </div>
      </div>
      <div className="space-y-2">
        {runs.map((run) => (
          <RunCard
            key={run.id}
            run={run}
            onPin={() => togglePin(run.id)}
            onDelete={() => deleteRun(run.id)}
            onSelect={() => handleSelect(run.id)}
            selected={compareIds.includes(run.id)}
          />
        ))}
      </div>

      <Dialog open={showComparison} onOpenChange={setShowComparison}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Run Comparison</DialogTitle>
          </DialogHeader>
          {selectedRuns.length === 2 && (
            <RunComparison runA={selectedRuns[0]} runB={selectedRuns[1]} />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
