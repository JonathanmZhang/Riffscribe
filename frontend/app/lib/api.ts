// NEXT_PUBLIC_* values are inlined into the client bundle when Next compiles
// (at build time in production), so changing them requires a rebuild/redeploy.
// Must be referenced as literal process.env.NEXT_PUBLIC_... for inlining to work.
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Hides the "Paste URL" option in the UI when "false". Defaults to enabled.
// Frontend-only: the backend's POST /jobs accepts URLs regardless.
export const URL_INGESTION_ENABLED = process.env.NEXT_PUBLIC_ENABLE_URL_INGESTION !== "false";

export type JobStatusValue = "queued" | "processing" | "done" | "failed";

export interface Note {
  string: number;
  fret: number;
  start_time: number;
  end_time: number;
  pitch: string;
  // Played with vibrato; a wavy line in the sheet-music view. Missing on
  // jobs finished before vibrato was detected.
  vibrato?: boolean;
}

// A recognized chord over [start, end) seconds (BTC, in the worker).
export interface ChordSegment {
  start: number;
  end: number;
  name: string;
}

export interface TabResult {
  job_id: string;
  duration_seconds: number;
  tempo_bpm: number;
  notes: Note[];
  // Missing on jobs finished before chord names existed.
  chords?: ChordSegment[];
  // Beat and bar-start times in seconds; empty for older jobs or if beat
  // tracking failed.
  beats?: number[];
  downbeats?: number[];
  // Bar start times as notated (4/4, after the job's overrides); empty
  // without beats.
  bars?: number[];
}

// Notation overrides (PATCH /jobs/{id}): 0.5 / 2 halve / double the tracked
// beats; bar_offset_beats moves the bar lines later by 0-3 beats.
export type TempoFactor = 0.5 | 1 | 2;
export type BarOffsetBeats = 0 | 1 | 2 | 3;

export interface JobCreateResponse {
  job_id: string;
  status: JobStatusValue;
}

// Which pipeline task is running; separate from status. null when queued or
// done, and left at the failing stage when a job fails.
export type JobStage = "ingesting" | "separating" | "transcribing" | "mapping";

// Guitar separation, for "Isolate guitar" jobs. Quality is what was asked
// for: "standard" = Demucs (CPU), "high" = Mega 53 (NVIDIA GPU). Separator
// is what produced the stem; it differs from the quality after a fallback.
export type SeparationQuality = "standard" | "high";
export type Separator = "demucs" | "mega53";

export interface JobStatusResponse {
  job_id: string;
  status: JobStatusValue;
  error: string | null;
  result: TabResult | null;
  isolate_guitar: boolean;
  stage: JobStage | null;
  stem_available: boolean;
  separation_quality: SeparationQuality | null;
  separator: Separator | null;
  // Why a "high" job was separated with Demucs; null otherwise.
  separation_note: string | null;
  tempo_factor: TempoFactor;
  bar_offset_beats: BarOffsetBeats;
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

// What the separation worker reported at its start (GET /capabilities).
export interface SeparationCapabilities {
  high_quality_available: boolean;
  high_quality_unavailable_reason: string | null;
  gpu: string | null;
  gpu_memory_mib: number | null;
}

export async function getSeparationCapabilities(): Promise<SeparationCapabilities> {
  const response = await fetch(`${API_BASE_URL}/capabilities`);
  return (await parseOrThrow<{ separation: SeparationCapabilities }>(response)).separation;
}

export async function createJobFromFile(
  file: File,
  isolateGuitar: boolean,
  separationQuality: SeparationQuality,
): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("isolate_guitar", String(isolateGuitar));
  formData.append("separation_quality", separationQuality);

  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<JobCreateResponse>(response);
}

export async function createJobFromUrl(
  url: string,
  isolateGuitar: boolean,
  separationQuality: SeparationQuality,
): Promise<JobCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, isolate_guitar: isolateGuitar, separation_quality: separationQuality }),
  });

  return parseOrThrow<JobCreateResponse>(response);
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  return parseOrThrow<JobStatusResponse>(response);
}

export function getJobAudioUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/audio`;
}

// The separated guitar (jobs run with Isolate guitar; see stem_available).
export function getJobStemUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/stem`;
}

// A new job from an existing job's source audio, with Isolate guitar on.
export async function rerunJobWithIsolation(
  jobId: string,
  separationQuality: SeparationQuality,
): Promise<JobCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/rerun`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isolate_guitar: true, separation_quality: separationQuality }),
  });
  return parseOrThrow<JobCreateResponse>(response);
}

// Finished jobs only; recomputes the bars without re-transcribing.
export async function setJobOverrides(
  jobId: string,
  overrides: { tempo_factor?: TempoFactor; bar_offset_beats?: BarOffsetBeats },
): Promise<JobStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(overrides),
  });
  return parseOrThrow<JobStatusResponse>(response);
}

// What the synth plays the sheet music with: sets the General MIDI program
// in the MusicXML export (clean electric 28, overdriven 30, distorted 31,
// acoustic steel 26). The notes are the same for every tone.
export type PlaybackTone = "clean" | "overdriven" | "distorted" | "acoustic";

// MusicXML download (notation + TAB), built with the job's current overrides.
export function getJobMusicXmlUrl(jobId: string, tone: PlaybackTone = "clean"): string {
  return `${API_BASE_URL}/jobs/${jobId}/musicxml?tone=${tone}`;
}
