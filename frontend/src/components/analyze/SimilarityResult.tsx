"use client";

import type { AnalyzeResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { cn, simColor } from "@/lib/utils";

interface SimilarityResultProps {
  result: AnalyzeResult;
}

export function SimilarityResult({ result }: SimilarityResultProps) {
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border p-6 text-center">
      <Badge
        variant={result.is_same_person ? "destructive" : "success"}
        className="px-4 py-2 text-lg"
      >
        {result.is_same_person ? "Same Person" : "Different Person"}
      </Badge>
      <div>
        <p className="text-sm text-muted-foreground">Cosine Similarity</p>
        <p className={cn("text-3xl font-bold font-mono", simColor(result.cosine_similarity))}>
          {result.cosine_similarity.toFixed(4)}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Threshold: {result.threshold} (above = same person)
        </p>
      </div>
    </div>
  );
}
