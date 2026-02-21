"use client";

import { useState, useCallback, useRef } from "react";
import type { DeepfakeTestResult } from "@/lib/types";
import { testDeepfake } from "@/lib/api";
import { DeepfakeResults } from "@/components/deepfake/DeepfakeResults";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

function DropZone({
  label,
  onFile,
  preview,
  disabled,
}: {
  label: string;
  onFile: (f: File) => void;
  preview?: string | null;
  disabled?: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("image/")) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      className={cn(
        "flex min-h-[160px] cursor-pointer items-center justify-center rounded-lg border-2 border-dashed transition-colors",
        dragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/50",
        disabled && "pointer-events-none opacity-50"
      )}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
      {preview ? (
        <img src={preview} alt={label} className="max-h-[180px] rounded object-contain" />
      ) : (
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <p className="mt-1 text-xs text-muted-foreground">Drop or click</p>
        </div>
      )}
    </div>
  );
}

export default function DeepfakePage() {
  const [cleanFile, setCleanFile] = useState<File | null>(null);
  const [protectedFile, setProtectedFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [cleanPreview, setCleanPreview] = useState<string | null>(null);
  const [protectedPreview, setProtectedPreview] = useState<string | null>(null);
  const [targetPreview, setTargetPreview] = useState<string | null>(null);

  const [runInswapper, setRunInswapper] = useState(true);
  const [runIpadapter, setRunIpadapter] = useState(false);
  const [prompt, setPrompt] = useState("a photo of a person");

  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<DeepfakeTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClean = useCallback((f: File) => {
    setCleanFile(f);
    setCleanPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleProtected = useCallback((f: File) => {
    setProtectedFile(f);
    setProtectedPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleTarget = useCallback((f: File) => {
    setTargetFile(f);
    setTargetPreview(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleTest = useCallback(async () => {
    if (!cleanFile || !protectedFile) return;
    setTesting(true);
    setError(null);
    try {
      const res = await testDeepfake(cleanFile, protectedFile, {
        targetImage: targetFile ?? undefined,
        runInswapper,
        runIpadapter,
        prompt,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Deepfake test failed");
    } finally {
      setTesting(false);
    }
  }, [cleanFile, protectedFile, targetFile, runInswapper, runIpadapter, prompt]);

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Deepfake Tool Testing</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Test protection against real deepfake pipelines. Upload a clean face, its protected
            version, and optionally a target image for face swapping.
          </p>

          {/* Image upload zones */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="mb-2 text-sm font-medium">Clean Face</p>
              <DropZone label="Drop clean face" onFile={handleClean} preview={cleanPreview} disabled={testing} />
            </div>
            <div>
              <p className="mb-2 text-sm font-medium">Protected Face</p>
              <DropZone label="Drop protected face" onFile={handleProtected} preview={protectedPreview} disabled={testing} />
            </div>
            <div>
              <p className="mb-2 text-sm font-medium">Target (optional)</p>
              <DropZone label="Drop target image" onFile={handleTarget} preview={targetPreview} disabled={testing} />
            </div>
          </div>

          {/* Options */}
          <div className="flex flex-wrap items-center gap-4 rounded-lg border p-3">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={runInswapper}
                onChange={(e) => setRunInswapper(e.target.checked)}
                disabled={testing}
                className="rounded"
              />
              Inswapper (Roop)
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={runIpadapter}
                onChange={(e) => setRunIpadapter(e.target.checked)}
                disabled={testing}
                className="rounded"
              />
              IP-Adapter FaceID
            </label>
            {runIpadapter && (
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Generation prompt"
                className="flex-1 rounded border border-border bg-background px-2 py-1 text-sm"
                disabled={testing}
              />
            )}
          </div>

          <Button
            onClick={handleTest}
            disabled={!cleanFile || !protectedFile || (!runInswapper && !runIpadapter) || testing}
            className="w-full"
            size="lg"
          >
            {testing ? "Running Deepfake Tests..." : "Run Test"}
          </Button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </CardContent>
      </Card>

      {result && (
        <DeepfakeResults
          inswapper={result.inswapper}
          ipadapter={result.ipadapter}
          overallVerdict={result.overall_verdict}
        />
      )}
    </div>
  );
}
