"use client";

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import type { Run, EvaluateResult } from "@/lib/types";
import { generateId } from "@/lib/utils";
import type { ProtectParams, ProtectionMetrics } from "@/lib/types";

interface RunHistoryContextValue {
  runs: Run[];
  addRun: (params: ProtectParams, originalImageUrl: string, protectedImageUrl: string, metrics: ProtectionMetrics) => string;
  updateRobustness: (runId: string, robustness: EvaluateResult) => void;
  togglePin: (runId: string) => void;
  deleteRun: (runId: string) => void;
  clearAll: () => void;
  compareIds: [string | null, string | null];
  setCompareIds: (ids: [string | null, string | null]) => void;
}

const RunHistoryContext = createContext<RunHistoryContextValue>({
  runs: [],
  addRun: () => "",
  updateRobustness: () => {},
  togglePin: () => {},
  deleteRun: () => {},
  clearAll: () => {},
  compareIds: [null, null],
  setCompareIds: () => {},
});

const STORAGE_KEY = "privacyshield-run-history";

export function RunHistoryProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [compareIds, setCompareIds] = useState<[string | null, string | null]>([null, null]);
  const [loaded, setLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setRuns(JSON.parse(stored));
      }
    } catch {}
    setLoaded(true);
  }, []);

  // Persist to localStorage on change
  useEffect(() => {
    if (!loaded) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
    } catch {
      // localStorage full — drop oldest non-pinned run
      const nonPinned = runs.filter((r) => !r.pinned);
      if (nonPinned.length > 0) {
        const oldest = nonPinned[nonPinned.length - 1];
        const trimmed = runs.filter((r) => r.id !== oldest.id);
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
          setRuns(trimmed);
        } catch {}
      }
    }
  }, [runs, loaded]);

  const addRun = useCallback(
    (params: ProtectParams, originalImageUrl: string, protectedImageUrl: string, metrics: ProtectionMetrics) => {
      const id = generateId();
      const run: Run = {
        id,
        timestamp: Date.now(),
        params,
        originalImageUrl,
        protectedImageUrl,
        metrics,
      };
      setRuns((prev) => [run, ...prev]);
      return id;
    },
    []
  );

  const updateRobustness = useCallback((runId: string, robustness: EvaluateResult) => {
    setRuns((prev) => prev.map((r) => (r.id === runId ? { ...r, robustness } : r)));
  }, []);

  const togglePin = useCallback((runId: string) => {
    setRuns((prev) => prev.map((r) => (r.id === runId ? { ...r, pinned: !r.pinned } : r)));
  }, []);

  const deleteRun = useCallback((runId: string) => {
    setRuns((prev) => prev.filter((r) => r.id !== runId));
  }, []);

  const clearAll = useCallback(() => {
    setRuns([]);
  }, []);

  return (
    <RunHistoryContext.Provider
      value={{ runs, addRun, updateRobustness, togglePin, deleteRun, clearAll, compareIds, setCompareIds }}
    >
      {children}
    </RunHistoryContext.Provider>
  );
}

export function useRunHistoryContext() {
  return useContext(RunHistoryContext);
}
