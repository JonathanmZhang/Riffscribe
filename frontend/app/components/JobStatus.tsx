"use client";

import { useEffect, useRef, useState, type SyntheticEvent } from "react";
import { getJob, getJobAudioUrl, type JobStage, type JobStatusValue, type TabResult } from "@/app/lib/api";
import TabViewer from "@/app/components/TabViewer";

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES: JobStatusValue[] = ["done", "failed"];

// Practice speeds. The highlight sync reads audio.currentTime (media time),
// which already accounts for playbackRate, so it needs no speed adjustment.
const PLAYBACK_RATES = [0.5, 0.75, 1] as const;

const STATUS_STYLES: Record<JobStatusValue, string> = {
  queued: "bg-slate-100 text-slate-700",
  processing: "bg-indigo-100 text-indigo-700",
  done: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
};

const STAGE_LABELS: Record<JobStage, string> = {
  ingesting: "Preparing audio…",
  separating: "Separating guitar…",
  transcribing: "Detecting notes…",
  mapping: "Mapping to fretboard…",
};

interface JobStatusProps {
  jobId: string;
}

export default function JobStatus({ jobId }: JobStatusProps) {
  const [status, setStatus] = useState<JobStatusValue>("queued");
  const [jobError, setJobError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [stage, setStage] = useState<JobStage | null>(null);
  const [result, setResult] = useState<TabResult | null>(null);
  const [currentTime, setCurrentTime] = useState<number | null>(null);
  const [playbackRate, setPlaybackRate] = useState<number>(1);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Runs once the <audio> element mounts (result arrives) and on every
  // speed change. defaultPlaybackRate is set too, since browsers reset
  // playbackRate to it when the media reloads.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.defaultPlaybackRate = playbackRate;
    audio.playbackRate = playbackRate;
  }, [playbackRate, result]);

  const handleTimeUpdate = (event: SyntheticEvent<HTMLAudioElement>) => {
    setCurrentTime(event.currentTarget.currentTime);
  };

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await getJob(jobId);
        if (cancelled) return;

        setStatus(job.status);
        setStage(job.stage);
        setJobError(job.error);

        if (TERMINAL_STATUSES.includes(job.status)) {
          clearInterval(intervalId);
          if (job.status === "done") {
            setResult(job.result);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setPollError(err instanceof Error ? err.message : "Failed to poll job status.");
        }
      }
    };

    const intervalId = setInterval(poll, POLL_INTERVAL_MS);
    poll();

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [jobId]);

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Job status</h2>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${STATUS_STYLES[status]}`}
          >
            {!TERMINAL_STATUSES.includes(status) && (
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            )}
            {status}
          </span>
        </div>
        {status === "processing" && stage && (
          <p className="text-sm font-medium text-indigo-700">
            {STAGE_LABELS[stage]}
            {stage === "separating" && (
              <span className="font-normal text-slate-500"> This can take a few minutes.</span>
            )}
          </p>
        )}
        <p className="text-sm text-slate-500">
          Job <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-700">{jobId}</code>
        </p>
        {status === "failed" && jobError && (
          <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{jobError}</p>
        )}
        {pollError && <p className="text-sm text-red-600">{pollError}</p>}
      </section>

      {result && (
        <section className="flex flex-col gap-5 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Tablature</h2>
          <div className="flex flex-col gap-3">
            <audio
              ref={audioRef}
              controls
              src={getJobAudioUrl(jobId)}
              onTimeUpdate={handleTimeUpdate}
              onSeeked={handleTimeUpdate}
              // Keeps the buttons in sync if the speed is changed from the
              // browser's native audio-controls menu instead.
              onRateChange={(event) => setPlaybackRate(event.currentTarget.playbackRate)}
              className="w-full"
            />
            <div className="flex items-center gap-3 text-sm">
              <span className="text-slate-500">Speed</span>
              <div role="group" aria-label="Playback speed" className="inline-flex rounded-lg bg-slate-100 p-1 font-medium">
                {PLAYBACK_RATES.map((rate) => (
                  <button
                    key={rate}
                    type="button"
                    aria-pressed={playbackRate === rate}
                    onClick={() => setPlaybackRate(rate)}
                    className={`rounded-md px-3 py-1 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500 ${
                      playbackRate === rate ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>
          </div>
          <TabViewer result={result} currentTime={currentTime} />
        </section>
      )}
    </div>
  );
}
