"use client";

import { useState, useCallback } from "react";
import type { AnalyzeResult } from "@/lib/types";
import { analyzeImages } from "@/lib/api";
import { DualImageUploader } from "@/components/analyze/DualImageUploader";
import { SimilarityResult } from "@/components/analyze/SimilarityResult";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function AnalyzePage() {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [preview1, setPreview1] = useState<string | null>(null);
  const [preview2, setPreview2] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImage1 = useCallback((f: File) => {
    setFile1(f);
    setPreview1(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleImage2 = useCallback((f: File) => {
    setFile2(f);
    setPreview2(URL.createObjectURL(f));
    setResult(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file1 || !file2) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await analyzeImages(file1, file2);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  }, [file1, file2]);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Compare Two Face Images</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <DualImageUploader
            onImage1Select={handleImage1}
            onImage2Select={handleImage2}
            preview1={preview1}
            preview2={preview2}
            disabled={analyzing}
          />
          <Button
            onClick={handleAnalyze}
            disabled={!file1 || !file2 || analyzing}
            className="w-full"
            size="lg"
          >
            {analyzing ? "Analyzing..." : "Compare Faces"}
          </Button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardContent className="pt-6">
            <SimilarityResult result={result} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
