"use client";

import { useEffect, useState, type FormEvent } from "react";
import {
  createJobFromFile,
  createJobFromUrl,
  getSeparationCapabilities,
  URL_INGESTION_ENABLED,
  type SeparationCapabilities,
  type SeparationQuality,
} from "@/app/lib/api";

type Mode = "file" | "url";

const SEPARATION_QUALITIES: { value: SeparationQuality; label: string; description: string }[] = [
  { value: "standard", label: "Standard", description: "Demucs. Works on any computer." },
  {
    value: "high",
    label: "High quality",
    description: "Mega 53. Needs an NVIDIA GPU; takes a few minutes per song.",
  },
];

const README_LOCAL_SETUP_URL = "https://github.com/JonathanmZhang/Riffscribe#running-locally";

interface UploadFormProps {
  onJobCreated: (jobId: string) => void;
}

export default function UploadForm({ onJobCreated }: UploadFormProps) {
  const [mode, setMode] = useState<Mode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [isolateGuitar, setIsolateGuitar] = useState(false);
  const [quality, setQuality] = useState<SeparationQuality>("standard");
  // null until the API answers; High quality stays disabled meanwhile.
  const [capabilities, setCapabilities] = useState<SeparationCapabilities | null>(null);
  const [capabilitiesError, setCapabilitiesError] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSeparationCapabilities()
      .then((value) => !cancelled && setCapabilities(value))
      .catch(() => !cancelled && setCapabilitiesError(true));
    return () => {
      cancelled = true;
    };
  }, []);

  const highQualityAvailable = capabilities?.high_quality_available ?? false;
  const highQualityNote = highQualityAvailable
    ? `Using ${capabilities?.gpu ?? "the GPU"}.`
    : capabilities
      ? `Not available: ${capabilities.high_quality_unavailable_reason ?? "no NVIDIA GPU was found"}.`
      : capabilitiesError
        ? "Not available: couldn't check for a GPU."
        : "Checking for a GPU…";
  // Never submit "high" while it's disabled.
  const effectiveQuality: SeparationQuality = highQualityAvailable ? quality : "standard";

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
          ? await createJobFromFile(file as File, isolateGuitar, effectiveQuality)
          : await createJobFromUrl(url.trim(), isolateGuitar, effectiveQuality);
      onJobCreated(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit job.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex w-full flex-col gap-4">
      {/* With URL ingestion disabled there's only one mode, so the toggle
          is hidden entirely and mode stays "file". */}
      {URL_INGESTION_ENABLED && (
        <div className="inline-flex w-fit rounded-lg bg-slate-100 p-1 text-sm font-medium">
          {(["file", "url"] as const).map((value) => (
            <label
              key={value}
              className={`cursor-pointer rounded-md px-4 py-1.5 transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-indigo-500 ${
                mode === value ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <input
                type="radio"
                name="mode"
                value={value}
                checked={mode === value}
                onChange={() => setMode(value)}
                className="sr-only"
              />
              {value === "file" ? "Upload file" : "Paste URL"}
            </label>
          ))}
        </div>
      )}

      {mode === "file" ? (
        <>
          <input
            key="file-input"
            type="file"
            accept=".mp3,.wav,.m4a,.mp4,.webm,.mov"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="rounded-lg border border-slate-300 bg-white p-2 text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-indigo-50 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-indigo-700 hover:file:bg-indigo-100"
          />
          <p className="-mt-2 text-xs text-slate-500">
            Audio (MP3, WAV, M4A) or video (MP4, WebM, MOV), up to 200 MB. For video, only the sound is used.
          </p>
          {!URL_INGESTION_ENABLED && (
            <p className="text-xs text-slate-500">
              Transcribing from a YouTube or SoundCloud link works when you run Riffscribe locally. See the{" "}
              <a
                href={README_LOCAL_SETUP_URL}
                target="_blank"
                rel="noreferrer"
                className="font-medium text-indigo-600 underline-offset-2 hover:underline"
              >
                README
              </a>
              .
            </p>
          )}
        </>
      ) : (
        <input
          key="url-input"
          type="text"
          placeholder="https://..."
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
        />
      )}

      {/* Applies to both file and URL mode. */}
      <label className="flex w-fit cursor-pointer items-start gap-2.5 text-sm">
        <input
          type="checkbox"
          checked={isolateGuitar}
          onChange={(event) => setIsolateGuitar(event.target.checked)}
          className="mt-0.5 h-4 w-4 rounded border-slate-300 accent-indigo-600"
        />
        <span className="flex flex-col">
          <span className="font-medium text-slate-800">Isolate guitar</span>
          <span className="text-xs text-slate-500">
            Separates the guitar from the rest of the mix first. Slower: can take a few minutes.
          </span>
        </span>
      </label>

      {isolateGuitar && (
        <fieldset className="ml-[1.625rem] flex flex-col gap-2 text-sm">
          <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Separation quality
          </legend>
          {SEPARATION_QUALITIES.map((option) => {
            const disabled = option.value === "high" && !highQualityAvailable;
            return (
              <label
                key={option.value}
                className={`flex w-fit items-start gap-2.5 ${disabled ? "cursor-not-allowed" : "cursor-pointer"}`}
              >
                <input
                  type="radio"
                  name="separation-quality"
                  value={option.value}
                  checked={effectiveQuality === option.value}
                  disabled={disabled}
                  onChange={() => setQuality(option.value)}
                  aria-describedby={`separation-quality-${option.value}`}
                  className="mt-0.5 h-4 w-4 border-slate-300 accent-indigo-600"
                />
                <span id={`separation-quality-${option.value}`} className="flex flex-col">
                  <span className={`font-medium ${disabled ? "text-slate-400" : "text-slate-800"}`}>
                    {option.label}
                  </span>
                  <span className="text-xs text-slate-500">{option.description}</span>
                  {option.value === "high" && (
                    <span className={`text-xs ${highQualityAvailable ? "text-emerald-700" : "text-amber-700"}`}>
                      {highQualityNote}
                    </span>
                  )}
                </span>
              </label>
            );
          })}
        </fieldset>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      <button
        type="submit"
        disabled={submitting}
        className="w-fit rounded-lg bg-indigo-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? "Submitting..." : "Transcribe"}
      </button>
    </form>
  );
}
