"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  preview?: string | null;
  disabled?: boolean;
}

export function ImageUploader({ onImageSelect, preview, disabled }: ImageUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      className={cn(
        "relative flex min-h-[200px] cursor-pointer items-center justify-center rounded-lg border-2 border-dashed transition-colors",
        dragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/50",
        disabled && "pointer-events-none opacity-50"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
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
        <img
          src={preview}
          alt="Selected face"
          className="max-h-[300px] rounded object-contain"
        />
      ) : (
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">
            Drop a face image here or click to browse
          </p>
          <p className="mt-1 text-xs text-muted-foreground">PNG, JPG up to 10MB</p>
        </div>
      )}
    </div>
  );
}
