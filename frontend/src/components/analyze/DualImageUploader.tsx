"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface DualImageUploaderProps {
  onImage1Select: (file: File) => void;
  onImage2Select: (file: File) => void;
  preview1?: string | null;
  preview2?: string | null;
  disabled?: boolean;
}

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
        "flex min-h-[200px] cursor-pointer items-center justify-center rounded-lg border-2 border-dashed transition-colors",
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
        <img src={preview} alt={label} className="max-h-[250px] rounded object-contain" />
      ) : (
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <p className="mt-1 text-xs text-muted-foreground">Drop or click</p>
        </div>
      )}
    </div>
  );
}

export function DualImageUploader({
  onImage1Select,
  onImage2Select,
  preview1,
  preview2,
  disabled,
}: DualImageUploaderProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <p className="mb-2 text-sm font-medium">Image 1</p>
        <DropZone label="Drop Image 1" onFile={onImage1Select} preview={preview1} disabled={disabled} />
      </div>
      <div>
        <p className="mb-2 text-sm font-medium">Image 2</p>
        <DropZone label="Drop Image 2" onFile={onImage2Select} preview={preview2} disabled={disabled} />
      </div>
    </div>
  );
}
