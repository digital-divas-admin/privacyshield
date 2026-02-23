"use client";

import { useState, useCallback } from "react";
import type { ProtectMode, ProtectParams, ProtectionResult } from "@/lib/types";
import { useProtect } from "@/hooks/useProtect";
import { useEvaluate } from "@/hooks/useEvaluate";
import { useHealth } from "@/hooks/useHealth";
import { useRunHistory } from "@/hooks/useRunHistory";
import { blobToDataUrl, dataUrlToBlob } from "@/lib/utils";
import { ImageUploader } from "@/components/protect/ImageUploader";
import { ModeSelector } from "@/components/protect/ModeSelector";
import { ParameterPanel } from "@/components/protect/ParameterPanel";
import { BeforeAfterSlider } from "@/components/protect/BeforeAfterSlider";
import { PerturbationViewer } from "@/components/protect/PerturbationViewer";
import { MetricsPanel } from "@/components/protect/MetricsPanel";
import { ProtectionScore } from "@/components/protect/ProtectionScore";
import { RobustnessGrid } from "@/components/evaluate/RobustnessGrid";
import { OverallVerdict } from "@/components/evaluate/OverallVerdict";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function ProtectPage() {
  const { health } = useHealth();
  const { state: protectState, result, error: protectError, protect } = useProtect();
  const { state: evalState, result: evalResult, error: evalError, evaluate } = useEvaluate();
  const { addRun, updateRobustness } = useRunHistory();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [params, setParams] = useState<ProtectParams>({
    mode: "pgd",
    epsilon: 8 / 255,
    steps: 50,
    eot_samples: 10,
    mask_mode: "default",
  });

  const handleImageSelect = useCallback((f: File) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const handleProtect = useCallback(async () => {
    if (!file) return;
    const res = await protect(file, params);
    if (res) {
      // Save to run history
      const origUrl = await blobToDataUrl(file);
      const protUrl = await blobToDataUrl(res.protectedImageBlob);
      const runId = addRun(params, origUrl, protUrl, res.metrics);
      setCurrentRunId(runId);
    }
  }, [file, params, protect, addRun]);

  const handleEvaluate = useCallback(async () => {
    if (!file || !result) return;
    const evalRes = await evaluate(file, result.protectedImageBlob);
    if (evalRes && currentRunId) {
      updateRobustness(currentRunId, evalRes);
    }
  }, [file, result, currentRunId, evaluate, updateRobustness]);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>1. Upload Face Image</CardTitle>
        </CardHeader>
        <CardContent>
          <ImageUploader
            onImageSelect={handleImageSelect}
            preview={previewUrl}
            disabled={protectState === "protecting"}
          />
        </CardContent>
      </Card>

      {/* Configuration */}
      {file && (
        <Card>
          <CardHeader>
            <CardTitle>2. Configure Protection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="mb-2 text-sm font-medium">Mode</p>
              <ModeSelector
                value={params.mode}
                onChange={(mode: ProtectMode) => setParams((p) => ({
                  ...p,
                  mode,
                  refine_steps: mode === "encoder_refined" ? (p.refine_steps ?? 10) : undefined,
                  eot_samples: mode === "encoder_refined" ? 2 : (p.mode === "encoder_refined" ? 10 : p.eot_samples),
                }))}
                health={health}
              />
            </div>
            <Separator />
            <ParameterPanel params={params} onChange={setParams} />
            <Button
              onClick={handleProtect}
              disabled={protectState === "protecting" || !health?.face_model_loaded}
              className="w-full"
              size="lg"
            >
              {protectState === "protecting" ? "Protecting..." : "Protect Image"}
            </Button>
            {protectError && (
              <p className="text-sm text-red-400">{protectError}</p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && previewUrl && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>3. Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ProtectionScore metrics={result.metrics} />
              <Separator />
              <MetricsPanel metrics={result.metrics} />
              <Separator />
              {(() => {
                const origSrc = result.originalAlignedUrl || previewUrl;
                return (
                  <>
                    <div className="grid gap-4 grid-cols-3">
                      <div>
                        <p className="mb-2 text-sm font-medium">Original</p>
                        <div className="overflow-hidden rounded-lg border border-border">
                          <img
                            src={origSrc}
                            alt="Original"
                            className="w-full h-auto"
                            draggable={false}
                          />
                        </div>
                      </div>
                      <div>
                        <p className="mb-2 text-sm font-medium">Protected</p>
                        <div className="overflow-hidden rounded-lg border border-border">
                          <img
                            src={result.protectedImageUrl}
                            alt="Protected"
                            className="w-full h-auto"
                            draggable={false}
                          />
                        </div>
                      </div>
                      <div>
                        <p className="mb-2 text-sm font-medium">Zoomed Comparison</p>
                        <BeforeAfterSlider
                          beforeSrc={origSrc}
                          afterSrc={result.protectedImageUrl}
                          zoom={2.5}
                        />
                      </div>
                    </div>
                    <div className="grid gap-4 grid-cols-3">
                      <div>
                        <PerturbationViewer
                          originalSrc={origSrc}
                          protectedSrc={result.protectedImageUrl}
                        />
                      </div>
                    </div>
                  </>
                );
              })()}
              <div className="flex gap-2">
                <a href={result.protectedImageUrl} download="protected.png">
                  <Button variant="outline">
                    Download Protected Image
                  </Button>
                </a>
                <Button
                  variant="secondary"
                  onClick={handleEvaluate}
                  disabled={evalState === "evaluating"}
                >
                  {evalState === "evaluating" ? "Testing..." : "Test Robustness"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Robustness Results */}
          {evalResult && (
            <Card>
              <CardHeader>
                <CardTitle>4. Robustness Test</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <OverallVerdict result={evalResult} />
                <Separator />
                <RobustnessGrid result={evalResult} />
              </CardContent>
            </Card>
          )}
          {evalError && (
            <p className="text-sm text-red-400">{evalError}</p>
          )}
        </>
      )}
    </div>
  );
}
