"use client";

import { useEffect, useRef } from "react";

interface PerturbationViewerProps {
  originalSrc: string;
  protectedSrc: string;
  amplification?: number;
}

export function PerturbationViewer({
  originalSrc,
  protectedSrc,
  amplification = 10,
}: PerturbationViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const origImg = new Image();
    const protImg = new Image();
    let loaded = 0;

    const onLoad = () => {
      loaded++;
      if (loaded < 2) return;

      const w = origImg.naturalWidth;
      const h = origImg.naturalHeight;
      canvas.width = w;
      canvas.height = h;

      // Draw original to get pixel data
      ctx.drawImage(origImg, 0, 0);
      const origData = ctx.getImageData(0, 0, w, h);

      // Draw protected to get pixel data
      ctx.drawImage(protImg, 0, 0);
      const protData = ctx.getImageData(0, 0, w, h);

      // Compute amplified delta
      const out = ctx.createImageData(w, h);
      for (let i = 0; i < origData.data.length; i += 4) {
        for (let c = 0; c < 3; c++) {
          const diff = protData.data[i + c] - origData.data[i + c];
          out.data[i + c] = Math.min(255, Math.max(0, 128 + diff * amplification));
        }
        out.data[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    };

    origImg.onload = onLoad;
    protImg.onload = onLoad;
    origImg.src = originalSrc;
    protImg.src = protectedSrc;
  }, [originalSrc, protectedSrc, amplification]);

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">
        Perturbation ({amplification}x amplified)
      </p>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-border"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}
