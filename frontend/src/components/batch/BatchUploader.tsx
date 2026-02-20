"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface BatchUploaderProps {
  onFilesSelect: (files: File[]) => void;
  files: File[];
  disabled?: boolean;
}

export function BatchUploader({ onFilesSelect, files, disabled }: BatchUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (fileList: FileList) => {
      const imageFiles = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
      if (imageFiles.length > 0) onFilesSelect(imageFiles);
    },
    [onFilesSelect]
  );

  return (
    <div className="space-y-3">
      <div
        className={cn(
          "flex min-h-[120px] cursor-pointer items-center justify-center rounded-lg border-2 border-dashed transition-colors",
          dragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/50",
          disabled && "pointer-events-none opacity-50"
        )}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">
            Drop multiple face images here or click to browse
          </p>
          <p className="mt-1 text-xs text-muted-foreground">PNG, JPG — select multiple files</p>
        </div>
      </div>
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-1 rounded border border-border px-2 py-1 text-xs">
              <span className="max-w-[120px] truncate">{f.name}</span>
              <span className="text-muted-foreground">({(f.size / 1024).toFixed(0)}KB)</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
