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

export interface TabResult {
  job_id: string;
  duration_seconds: number;
  tempo_bpm: number;
  notes: Note[];
}

export interface JobCreateResponse {
  job_id: string;
  status: JobStatusValue;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatusValue;
  error: string | null;
  result: TabResult | null;
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

export async function createJobFromFile(file: File): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: formData,
  });

  return parseOrThrow<JobCreateResponse>(response);
}

export async function createJobFromUrl(url: string): Promise<JobCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
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
