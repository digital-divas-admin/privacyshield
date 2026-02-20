"use client";

import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";

const PAGE_TITLES: Record<string, string> = {
  "/protect": "Protect Image",
  "/batch": "Batch Testing",
  "/analyze": "Analyze Similarity",
  "/health": "System Status",
};

interface HeaderProps {
  onToggleHistory?: () => void;
  historyOpen?: boolean;
}

export function Header({ onToggleHistory, historyOpen }: HeaderProps) {
  const pathname = usePathname();
  const title = PAGE_TITLES[pathname] || "PrivacyShield";

  return (
    <header className="flex h-14 items-center justify-between border-b border-border px-6">
      <h1 className="text-lg font-semibold">{title}</h1>
      {onToggleHistory && (
        <Button variant="ghost" size="sm" onClick={onToggleHistory}>
          {historyOpen ? "Hide History" : "Show History"}
        </Button>
      )}
    </header>
  );
}
