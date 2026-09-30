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

export interface JobStatusResponse {
  job_id: string;
  status: JobStatusValue;
  error: string | null;
  result: TabResult | null;
  isolate_guitar: boolean;
  stage: JobStage | null;
  stem_available: boolean;
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

export async function createJobFromFile(file: File, isolateGuitar: boolean): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("isolate_guitar", String(isolateGuitar));

  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<JobCreateResponse>(response);
}

export async function createJobFromUrl(url: string, isolateGuitar: boolean): Promise<JobCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, isolate_guitar: isolateGuitar }),
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

// MusicXML download (notation + TAB), built with the job's current overrides.
export function getJobMusicXmlUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/musicxml`;
}
