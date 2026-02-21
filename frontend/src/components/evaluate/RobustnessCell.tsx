"use client";

import type { TransformResult } from "@/lib/types";
import { cn, simCellBg } from "@/lib/utils";

interface RobustnessCellProps {
  result: TransformResult;
}

export function RobustnessCell({ result }: RobustnessCellProps) {
  const perModel = result.per_model_similarity;

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
      {perModel && Object.keys(perModel).length > 1 && (
        <div className="mt-1 flex flex-col gap-0.5 w-full">
          {Object.entries(perModel).map(([name, sim]) => (
            <div key={name} className="flex justify-between text-[9px] opacity-70">
              <span>{name}</span>
              <span className="font-mono">{sim.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
