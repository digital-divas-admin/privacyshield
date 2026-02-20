"use client";

import type { TransformResult } from "@/lib/types";
import { cn, simCellBg } from "@/lib/utils";

interface RobustnessCellProps {
  result: TransformResult;
}

export function RobustnessCell({ result }: RobustnessCellProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded p-2 text-xs",
        simCellBg(result.cosine_similarity),
        "text-white"
      )}
      title={`${result.name}: cos_sim=${result.cosine_similarity}`}
    >
      <span className="font-mono font-bold">{result.cosine_similarity.toFixed(3)}</span>
      <span className="mt-0.5 text-[10px] opacity-80">
        {result.protection_holds ? "Pass" : "Fail"}
      </span>
    </div>
  );
}
