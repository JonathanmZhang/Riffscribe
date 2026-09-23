"use client";

import { useState, type FormEvent } from "react";
import { createJobFromFile, createJobFromUrl } from "@/app/lib/api";

type Mode = "file" | "url";

interface UploadFormProps {
  onJobCreated: (jobId: string) => void;
}

export default function UploadForm({ onJobCreated }: UploadFormProps) {
  const [mode, setMode] = useState<Mode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    if (mode === "file" && !file) {
      setError("Choose an audio file first.");
      return;
    }
    if (mode === "url" && !url.trim()) {
      setError("Enter a URL first.");
      return;
    }

    setSubmitting(true);
    try {
      const response =
        mode === "file"
          ? await createJobFromFile(file as File)
          : await createJobFromUrl(url.trim());
      onJobCreated(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit job.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex w-full max-w-md flex-col gap-4">
      <div className="flex gap-6">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            name="mode"
            value="file"
            checked={mode === "file"}
            onChange={() => setMode("file")}
          />
          Upload file
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            name="mode"
            value="url"
            checked={mode === "url"}
            onChange={() => setMode("url")}
          />
          Paste URL
        </label>
      </div>

      {mode === "file" ? (
        <input
          type="file"
          accept=".mp3,.wav,.m4a"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          className="rounded border p-2"
        />
      ) : (
        <input
          type="text"
          placeholder="https://..."
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          className="rounded border p-2"
        />
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-black p-2 text-white disabled:opacity-50"
      >
        {submitting ? "Submitting..." : "Submit"}
      </button>
    </form>
  );
}
