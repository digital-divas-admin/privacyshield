"use client";

import { useHealthContext } from "@/contexts/HealthContext";

export function useHealth() {
  return useHealthContext();
}
