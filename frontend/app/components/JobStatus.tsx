"use client";

import { useEffect, useState, type SyntheticEvent } from "react";
import { getJob, getJobAudioUrl, type JobStatusValue, type TabResult } from "@/app/lib/api";
import TabViewer from "@/app/components/TabViewer";

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES: JobStatusValue[] = ["done", "failed"];

interface JobStatusProps {
  jobId: string;
}

export default function JobStatus({ jobId }: JobStatusProps) {
  const [status, setStatus] = useState<JobStatusValue>("queued");
  const [jobError, setJobError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [result, setResult] = useState<TabResult | null>(null);
  const [currentTime, setCurrentTime] = useState<number | null>(null);

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
    <div className="flex w-full max-w-3xl flex-col gap-4">
      <p>
        Job <code className="font-mono text-sm">{jobId}</code>
      </p>
      <p>
        Status: <span className="font-semibold">{status}</span>
      </p>
      {status === "failed" && jobError && <p className="text-red-600">{jobError}</p>}
      {pollError && <p className="text-sm text-red-600">{pollError}</p>}

      {result && (
        <>
          <audio
            controls
            src={getJobAudioUrl(jobId)}
            onTimeUpdate={handleTimeUpdate}
            onSeeked={handleTimeUpdate}
            className="w-full max-w-3xl"
          />
          <TabViewer result={result} currentTime={currentTime} />
        </>
      )}
    </div>
  );
}
