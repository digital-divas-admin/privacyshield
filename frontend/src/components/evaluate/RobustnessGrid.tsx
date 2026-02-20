"use client";

import type { EvaluateResult } from "@/lib/types";
import { RobustnessCell } from "./RobustnessCell";
import { ROBUSTNESS_CATEGORIES } from "@/lib/constants";

interface RobustnessGridProps {
  result: EvaluateResult;
  showPlatforms?: boolean;
}

export function RobustnessGrid({ result, showPlatforms = true }: RobustnessGridProps) {
  const categories = showPlatforms
    ? ROBUSTNESS_CATEGORIES
    : ROBUSTNESS_CATEGORIES.filter((c) => c.key !== "platform");

  return (
    <div className="space-y-3">
      {categories.map((cat) => {
        const items = result.results.filter((r) => r.category === cat.key);
        if (items.length === 0) return null;
        return (
          <div key={cat.key}>
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">{cat.label}</p>
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {items.map((item) => (
                <div key={item.name} className="space-y-0.5">
                  <p className="truncate text-[10px] text-muted-foreground">{item.name}</p>
                  <RobustnessCell result={item} />
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
