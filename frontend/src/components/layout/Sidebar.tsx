"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useHealthContext } from "@/contexts/HealthContext";

const NAV_ITEMS = [
  { href: "/protect", label: "Protect", icon: "🛡" },
  { href: "/batch", label: "Batch", icon: "📁" },
  { href: "/analyze", label: "Analyze", icon: "🔍" },
  { href: "/deepfake", label: "Deepfake Test", icon: "🎭" },
  { href: "/health", label: "System", icon: "⚙" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { connected } = useHealthContext();

  return (
    <aside className="flex w-56 flex-col border-r border-border bg-card">
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <span className="text-lg font-bold">PrivacyShield</span>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
              pathname === item.href
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            )}
          >
            <span>{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="border-t border-border p-4">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              connected ? "bg-green-500" : "bg-red-500"
            )}
          />
          {connected ? "API Connected" : "API Offline"}
        </div>
      </div>
    </aside>
  );
}
