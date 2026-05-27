"use client";

import { useState, useRef } from "react";

const CLASS_LABELS: Record<string, string> = {
  apple_level_0: "Apple Fresh",
  apple_level_1: "Apple Medium",
  apple_level_2: "Apple Rotten",
  potato_level_0: "Potato Fresh",
  potato_level_1: "Potato Medium",
  potato_level_2: "Potato Rotten",
};

const CLASS_EMOJIS: Record<string, string> = {
  apple_level_0: "\uD83C\uDF4E",
  apple_level_1: "\uD83C\uDF4E",
  apple_level_2: "\uD83C\uDF4E",
  potato_level_0: "\uD83E\uDD54",
  potato_level_1: "\uD83E\uDD54",
  potato_level_2: "\uD83E\uDD54",
};

type Prediction = {
  class: string;
  confidence: number;
};

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) return;
    setPrediction(null);
    setError(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      setPrediction(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (c: number) =>
    c > 0.8 ? "text-green-600" : c > 0.5 ? "text-yellow-600" : "text-red-600";

  return (
    <div className="flex flex-col items-center min-h-screen p-6 bg-zinc-50 dark:bg-black">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Produce Quality</h1>
        <p className="text-zinc-500 mt-1">Upload an apple or potato photo</p>
      </header>

      <div
        onClick={() => inputRef.current?.click()}
        onDrop={(e) => {
          e.preventDefault();
          if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        }}
        onDragOver={(e) => e.preventDefault()}
        className="relative w-full max-w-sm h-64 rounded-2xl border-2 border-dashed border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 flex items-center justify-center cursor-pointer hover:border-zinc-400 transition-colors overflow-hidden"
      >
        {preview ? (
          <img
            src={preview}
            alt="preview"
            className="max-w-full max-h-full object-contain p-2"
          />
        ) : (
          <p className="text-zinc-400 text-sm px-6 text-center">
            Drop image here or tap to select
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) =>
            e.target.files?.[0] && handleFile(e.target.files[0])
          }
        />
      </div>

      {file && (
        <button
          onClick={predict}
          disabled={loading}
          className="mt-6 px-10 py-3 rounded-full bg-black text-white dark:bg-white dark:text-black font-medium hover:opacity-80 transition-opacity disabled:opacity-40"
        >
          {loading ? "Analyzing\u2026" : prediction ? "Re-predict" : "Predict"}
        </button>
      )}

      {loading && (
        <p className="mt-4 text-zinc-400 text-sm">Processing image\u2026</p>
      )}

      {error && (
        <p className="mt-4 px-4 py-2 rounded-xl bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-sm">
          {error}
        </p>
      )}

      {prediction && (
        <div className="mt-6 w-full max-w-sm p-5 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center">
          <span className="text-4xl">
            {CLASS_EMOJIS[prediction.class]}
          </span>
          <p className="text-xl font-semibold mt-2">
            {CLASS_LABELS[prediction.class] || prediction.class}
          </p>
          <p
            className={`text-lg font-mono ${confidenceColor(prediction.confidence)}`}
          >
            {(prediction.confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </div>
  );
}
