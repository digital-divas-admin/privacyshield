"use client";

import "./globals.css";
import { useState } from "react";
import { HealthProvider } from "@/contexts/HealthContext";
import { RunHistoryProvider } from "@/contexts/RunHistoryContext";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { RunHistoryPanel } from "@/components/history/RunHistoryPanel";
import { cn } from "@/lib/utils";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [historyOpen, setHistoryOpen] = useState(false);

  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">
        <HealthProvider>
          <RunHistoryProvider>
            <div className="flex h-screen overflow-hidden">
              <Sidebar />
              <div className="flex flex-1 flex-col overflow-hidden">
                <Header
                  onToggleHistory={() => setHistoryOpen((o) => !o)}
                  historyOpen={historyOpen}
                />
                <div className="flex flex-1 overflow-hidden">
                  <main className="flex-1 overflow-y-auto p-6">{children}</main>
                  <aside
                    className={cn(
                      "overflow-y-auto border-l border-border bg-card transition-all duration-200",
                      historyOpen ? "w-64" : "w-0"
                    )}
                  >
                    {historyOpen && (
                      <div className="p-3">
                        <p className="mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                          Run History
                        </p>
                        <RunHistoryPanel />
                      </div>
                    )}
                  </aside>
                </div>
              </div>
            </div>
          </RunHistoryProvider>
        </HealthProvider>
      </body>
    </html>
  );
}
