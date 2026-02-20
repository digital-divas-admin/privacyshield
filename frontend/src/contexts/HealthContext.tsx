"use client";

import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from "react";
import type { HealthStatus } from "@/lib/types";
import { getHealth } from "@/lib/api";

interface HealthContextValue {
  health: HealthStatus | null;
  connected: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const HealthContext = createContext<HealthContextValue>({
  health: null,
  connected: false,
  error: null,
  refresh: async () => {},
});

export function HealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
      setConnected(true);
      setError(null);
    } catch (e) {
      setConnected(false);
      setError(e instanceof Error ? e.message : "Connection failed");
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10_000);
    return () => clearInterval(interval);
  }, [refresh]);

  return (
    <HealthContext.Provider value={{ health, connected, error, refresh }}>
      {children}
    </HealthContext.Provider>
  );
}

export function useHealthContext() {
  return useContext(HealthContext);
}
