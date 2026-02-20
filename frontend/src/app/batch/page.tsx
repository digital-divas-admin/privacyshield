"use client";

import { useState, useCallback } from "react";
import type { ProtectMode, ProtectParams, BatchResult } from "@/lib/types";
import { protectBatch } from "@/lib/api";
import { useHealth } from "@/hooks/useHealth";
import { BatchUploader } from "@/components/batch/BatchUploader";
import { BatchResultsTable } from "@/components/batch/BatchResultsTable";
import { BatchSummary } from "@/components/batch/BatchSummary";
import { ModeSelector } from "@/components/protect/ModeSelector";
import { ParameterPanel } from "@/components/protect/ParameterPanel";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function BatchPage() {
  const { health } = useHealth();
  const [files, setFiles] = useState<File[]>([]);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<ProtectParams>({
    mode: "pgd",
    epsilon: 8 / 255,
    steps: 50,
    eot_samples: 10,
    mask_mode: "default",
  });

  const handleBatch = useCallback(async () => {
    if (files.length === 0) return;
    setProcessing(true);
    setError(null);
    try {
      const res = await protectBatch(files, params);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch processing failed");
    } finally {
      setProcessing(false);
    }
  }, [files, params]);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>1. Upload Images</CardTitle>
        </CardHeader>
        <CardContent>
          <BatchUploader
            onFilesSelect={(newFiles) => setFiles((prev) => [...prev, ...newFiles])}
            files={files}
            disabled={processing}
          />
          {files.length > 0 && (
            <div className="mt-2 flex gap-2">
              <Button variant="ghost" size="sm" onClick={() => setFiles([])}>
                Clear All
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {files.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>2. Configure & Run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ModeSelector
              value={params.mode}
              onChange={(mode: ProtectMode) => setParams((p) => ({ ...p, mode }))}
              health={health}
            />
            <Separator />
            <ParameterPanel params={params} onChange={setParams} />
            <Button
              onClick={handleBatch}
              disabled={processing || !health?.face_model_loaded}
              className="w-full"
              size="lg"
            >
              {processing ? `Processing ${files.length} images...` : `Protect ${files.length} Images`}
            </Button>
            {error && <p className="text-sm text-red-400">{error}</p>}
          </CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>3. Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <BatchSummary result={result} />
            <Separator />
            <BatchResultsTable
              results={result.results}
              fileNames={files.map((f) => f.name)}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
