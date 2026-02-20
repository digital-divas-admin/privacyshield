"use client";

import { useState, useCallback } from "react";
import type { ProtectParams, ProtectionResult } from "@/lib/types";
import { protectImage } from "@/lib/api";

type ProtectState = "idle" | "protecting" | "success" | "error";

export function useProtect() {
  const [state, setState] = useState<ProtectState>("idle");
  const [result, setResult] = useState<ProtectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const protect = useCallback(async (file: File | Blob, params: ProtectParams) => {
    setState("protecting");
    setError(null);
    try {
      const res = await protectImage(file, params);
      setResult(res);
      setState("success");
      return res;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Protection failed";
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

  return { state, result, error, protect, reset };
}
