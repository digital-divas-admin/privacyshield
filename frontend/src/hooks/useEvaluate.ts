"use client";

import { useState, useCallback } from "react";
import type { EvaluateResult } from "@/lib/types";
import { evaluateRobustness } from "@/lib/api";

type EvalState = "idle" | "evaluating" | "done" | "error";

export function useEvaluate() {
  const [state, setState] = useState<EvalState>("idle");
  const [result, setResult] = useState<EvaluateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const evaluate = useCallback(async (cleanBlob: Blob, protectedBlob: Blob, threshold = 0.3) => {
    setState("evaluating");
    setError(null);
    try {
      const res = await evaluateRobustness(cleanBlob, protectedBlob, threshold);
      setResult(res);
      setState("done");
      return res;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Evaluation failed";
      setError(msg);
      setState("error");
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState("idle");
    setResult(null);
    setError(null);
  }, []);

  return { state, result, error, evaluate, reset };
}
