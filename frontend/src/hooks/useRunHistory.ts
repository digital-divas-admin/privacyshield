"use client";

import { useRunHistoryContext } from "@/contexts/RunHistoryContext";

export function useRunHistory() {
  return useRunHistoryContext();
}
