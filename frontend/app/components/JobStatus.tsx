"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  getJob,
  getJobAudioUrl,
  getJobMusicXmlUrl,
  getJobStemUrl,
  getSeparationCapabilities,
  rerunJobWithIsolation,
  setJobOverrides,
  type BarOffsetBeats,
  type JobStage,
  type JobStatusValue,
  type PlaybackTone,
  type SeparationQuality,
  type Separator,
  type TabResult,
  type TempoFactor,
} from "@/app/lib/api";
import DebugOverlay from "@/app/components/DebugOverlay";
import SheetMusicView, { type PlaybackPosition, type SynthControl } from "@/app/components/SheetMusicView";
import Transport from "@/app/components/Transport";
import TabViewer, { groupIntoSteps } from "@/app/components/TabViewer";
import { readDebugParams } from "@/app/lib/debug";

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES: JobStatusValue[] = ["done", "failed"];

// Practice speeds. The highlight sync reads audio.currentTime (media time),
// which already accounts for playbackRate, so it needs no speed adjustment.
const PLAYBACK_RATES = [0.5, 0.75, 1] as const;

// What is heard. "original" and "stem" are two recordings of the same length
// played by the page's <audio>; "synth" is alphaTab playing the notation
// (sheet-music view only).
type PlaybackSource = "original" | "stem" | "synth";
const SOURCE_LABELS: Record<PlaybackSource, string> = {
  original: "Original",
  stem: "Guitar only",
  synth: "Synth",
};

// Synth only: the General MIDI guitar the notation is played with.
const PLAYBACK_TONES: { value: PlaybackTone; label: string }[] = [
  { value: "clean", label: "Clean electric" },
  { value: "overdriven", label: "Overdriven" },
  { value: "distorted", label: "Distorted" },
  { value: "acoustic", label: "Acoustic steel" },
];

// Guitar isolation's audio limit (the workers' MAX_SEPARATION_DURATION_SECONDS;
// longer audio fails in ingest with a clear message).
const ISOLATION_LIMIT_SECONDS = 120;

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

const SEPARATOR_LABELS: Record<Separator, string> = {
  demucs: "Guitar isolated with Demucs (standard quality).",
  mega53: "Guitar isolated with Mega 53 (high quality).",
};

const segmentClass = (selected: boolean) =>
  `rounded-md px-3 py-1 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500 ${
    selected ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
  } disabled:cursor-not-allowed disabled:text-slate-400 disabled:hover:text-slate-400`;

interface JobStatusProps {
  jobId: string;
  // Opens another job in this one's place (the re-run with Isolate guitar).
  onJobCreated: (jobId: string) => void;
}

export default function JobStatus({ jobId, onJobCreated }: JobStatusProps) {
  const [status, setStatus] = useState<JobStatusValue>("queued");
  const [jobError, setJobError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [stage, setStage] = useState<JobStage | null>(null);
  const [isolateGuitar, setIsolateGuitar] = useState(false);
  const [stemAvailable, setStemAvailable] = useState(false);
  const [separator, setSeparator] = useState<Separator | null>(null);
  const [separationNote, setSeparationNote] = useState<string | null>(null);
  const [result, setResult] = useState<TabResult | null>(null);
  const [playbackRate, setPlaybackRate] = useState<number>(1);
  const [view, setView] = useState<"tab" | "sheet">("tab");
  const [source, setSource] = useState<PlaybackSource>("original");
  const [tone, setTone] = useState<PlaybackTone>("clean");
  const [tempoFactor, setTempoFactor] = useState<TempoFactor>(1);
  const [barOffsetBeats, setBarOffsetBeats] = useState<BarOffsetBeats>(0);
  const [highQualityAvailable, setHighQualityAvailable] = useState(false);
  const [rerunQuality, setRerunQuality] = useState<SeparationQuality>("standard");
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  // Where playback is on the recording's time axis; carried across a change
  // of source so it continues from the same place.
  const positionRef = useRef<PlaybackPosition>({ time: 0, playing: false });
  // Set by SheetMusicView while the synth is the source; the transport uses it.
  const synthControlRef = useRef<SynthControl | null>(null);
  const [debug, setDebug] = useState(false);
  useEffect(() => {
    const params = readDebugParams();
    setDebug(params.debug);
    // With ?debug=1: the handed-over playback position, for browser tests.
    if (params.debug) (window as unknown as { riffscribePosition?: typeof positionRef }).riffscribePosition = positionRef;
  }, []);
  const stepTimes = useMemo(() => (result ? groupIntoSteps(result.notes).map((s) => s.time) : []), [result]);
  const audioSrc = source === "stem" ? getJobStemUrl(jobId) : getJobAudioUrl(jobId);
  const hasResult = result !== null;

  // Runs once the <audio> element mounts (result arrives) and on every
  // speed change. defaultPlaybackRate is set too, since browsers reset
  // playbackRate to it when the media reloads (e.g. on a change of source).
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.defaultPlaybackRate = playbackRate;
    audio.playbackRate = playbackRate;
  }, [playbackRate, result]);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await getJob(jobId);
        if (cancelled) return;

        setStatus(job.status);
        setStage(job.stage);
        setIsolateGuitar(job.isolate_guitar);
        setStemAvailable(job.stem_available);
        setSeparator(job.separator);
        setSeparationNote(job.separation_note);
        setJobError(job.error);

        if (TERMINAL_STATUSES.includes(job.status)) {
          clearInterval(intervalId);
          if (job.status === "done") {
            setResult(job.result);
            setTempoFactor(job.tempo_factor);
            setBarOffsetBeats(job.bar_offset_beats);
            if (job.stem_available && readDebugParams().src === "stem") setSource("stem");
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

  // For the re-run note: is High quality an option here?
  useEffect(() => {
    let cancelled = false;
    getSeparationCapabilities()
      .then((value) => !cancelled && setHighQualityAvailable(value.high_quality_available))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const changeSource = (next: PlaybackSource) => {
    if (next === source) return;
    const audio = audioRef.current;
    // Leaving a recording: remember where it was. (While the synth plays,
    // SheetMusicView keeps positionRef current itself.)
    if (source !== "synth" && audio) positionRef.current = { time: audio.currentTime, playing: !audio.paused };
    setSource(next);
  };

  // After a change of source to a recording: continue from the remembered
  // position, playing if it was. A change between the two recordings reloads
  // the <audio> (readyState drops to 0), so wait for its metadata; coming
  // back from the synth to the same recording, it is still loaded.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || source === "synth") return;
    const { time, playing } = positionRef.current;
    const resume = () => {
      if (Math.abs(audio.currentTime - time) > 0.05) audio.currentTime = time;
      if (playing) void audio.play();
    };
    if (audio.readyState >= 1) {
      resume();
      return;
    }
    audio.addEventListener("loadedmetadata", resume, { once: true });
    return () => audio.removeEventListener("loadedmetadata", resume);
  }, [source, hasResult]);

  const changeView = (next: "tab" | "sheet") => {
    // The synth belongs to the sheet-music view.
    if (next === "tab" && source === "synth") changeSource("original");
    setView(next);
  };

  // Bar-line overrides: the backend recomputes the bars (and the MusicXML)
  // from the stored transcription; the new result carries the new bars.
  const applyOverrides = async (overrides: { tempo_factor?: TempoFactor; bar_offset_beats?: BarOffsetBeats }) => {
    const job = await setJobOverrides(jobId, overrides);
    setResult(job.result);
    setTempoFactor(job.tempo_factor);
    setBarOffsetBeats(job.bar_offset_beats);
  };

  const rerunWithIsolation = async () => {
    setRerunning(true);
    setRerunError(null);
    try {
      const job = await rerunJobWithIsolation(jobId, highQualityAvailable ? rerunQuality : "standard");
      onJobCreated(job.job_id);
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : "Failed to start the re-run.");
      setRerunning(false);
    }
  };

  // Read out by screen readers when the job moves on (role="status" below).
  const announcement =
    status === "done"
      ? `Transcription done${result ? `: ${result.notes.length} notes` : ""}.`
      : status === "failed"
        ? `Transcription failed. ${jobError ?? ""}`
        : status === "processing"
          ? (stage ? STAGE_LABELS[stage] : "Processing…")
          : "Queued.";

  const tooLongToIsolate = result !== null && result.duration_seconds > ISOLATION_LIMIT_SECONDS;
  const sources: PlaybackSource[] = view === "sheet" ? ["original", "stem", "synth"] : ["original", "stem"];

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Job status</h2>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${STATUS_STYLES[status]}`}
          >
            {!TERMINAL_STATUSES.includes(status) && (
              <span className="h-1.5 w-1.5 rounded-full bg-current motion-safe:animate-pulse" />
            )}
            {status}
          </span>
        </div>
        <p role="status" className="sr-only">
          {announcement}
        </p>
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
        {separator && <p className="text-sm text-slate-600">{SEPARATOR_LABELS[separator]}</p>}
        {separationNote && (
          <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            {separationNote}
          </p>
        )}
        {status === "done" && !isolateGuitar && (
          <div className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm">
            <p className="text-slate-600">
              Transcribed from the whole recording. If other instruments are playing, isolating the guitar first
              gives a cleaner tab.
            </p>
            {tooLongToIsolate ? (
              <p className="text-xs text-slate-500">
                Isolate guitar works on up to 2 minutes of audio, and this recording is longer.
              </p>
            ) : (
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={rerunWithIsolation}
                  disabled={rerunning}
                  className="rounded-md border border-slate-300 bg-white px-3 py-1 text-sm font-medium text-slate-800 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {rerunning ? "Starting…" : "Re-run with Isolate guitar"}
                </button>
                {highQualityAvailable && (
                  <label className="flex items-center gap-2 text-xs text-slate-600">
                    Quality
                    <select
                      value={rerunQuality}
                      onChange={(event) => setRerunQuality(event.target.value as SeparationQuality)}
                      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                    >
                      <option value="standard">Standard</option>
                      <option value="high">High quality</option>
                    </select>
                  </label>
                )}
                <span className="text-xs text-slate-500">Uses the same audio; takes a few minutes.</span>
              </div>
            )}
            {rerunError && <p className="text-sm text-red-600">{rerunError}</p>}
          </div>
        )}
        {status === "failed" && jobError && (
          <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{jobError}</p>
        )}
        {pollError && <p className="text-sm text-red-600">{pollError}</p>}
      </section>

      {result && (
        <section className="flex flex-col gap-5 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              {view === "tab" ? "Tablature" : "Sheet music"}
            </h2>
            <div className="flex flex-wrap items-center gap-3">
              {/* The export opens in notation software; the tooltip says
                  which, on hover and on keyboard focus. */}
              <div className="group relative">
                <a
                  href={getJobMusicXmlUrl(jobId, tone)}
                  download
                  aria-describedby="musicxml-tip"
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current" strokeWidth="1.8">
                    <path d="M10 3.5v9m0 0-3.5-3.5M10 12.5l3.5-3.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M4 13.5v1.75c0 .69.56 1.25 1.25 1.25h9.5c.69 0 1.25-.56 1.25-1.25V13.5" strokeLinecap="round" />
                  </svg>
                  Download MusicXML
                </a>
                <span
                  id="musicxml-tip"
                  role="tooltip"
                  className="pointer-events-none absolute right-0 top-full z-20 mt-2 whitespace-nowrap rounded-md bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-white opacity-0 shadow-md group-focus-within:opacity-100 group-hover:opacity-100 motion-safe:transition-opacity"
                >
                  Open in MuseScore or Guitar Pro
                </span>
              </div>
              <div role="group" aria-label="View" className="inline-flex rounded-lg bg-slate-100 p-1 text-sm font-medium">
                {(["tab", "sheet"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    aria-pressed={view === v}
                    onClick={() => changeView(v)}
                    className={segmentClass(view === v)}
                  >
                    {v === "tab" ? "Tab" : "Sheet music"}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
              <span className="text-slate-500">Listen to</span>
              <div
                role="group"
                aria-label="Playback source"
                className="inline-flex rounded-lg bg-slate-100 p-1 font-medium"
              >
                {sources.map((value) => {
                  const unavailable = value === "stem" && !stemAvailable;
                  return (
                    <button
                      key={value}
                      type="button"
                      aria-pressed={source === value}
                      disabled={unavailable}
                      title={unavailable ? "Needs a job run with Isolate guitar" : undefined}
                      onClick={() => changeSource(value)}
                      className={segmentClass(source === value)}
                    >
                      {SOURCE_LABELS[value]}
                    </button>
                  );
                })}
              </div>
              {source === "synth" && (
                <label className="flex items-center gap-2">
                  <span className="text-slate-500">Tone</span>
                  <select
                    value={tone}
                    onChange={(event) => setTone(event.target.value as PlaybackTone)}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-sm font-medium text-slate-700 shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-500"
                  >
                    {PLAYBACK_TONES.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <span className="text-xs text-slate-500">
                {source === "synth"
                  ? "The transcription itself, played by a synth."
                  : source === "stem"
                    ? "The guitar separated from the recording."
                    : stemAvailable
                      ? "The recording as submitted."
                      : "Guitar only needs a job run with Isolate guitar."}
              </span>
            </div>
            {/* The recordings play here; the visible control is the
                transport, the same for every source and both views. */}
            <audio ref={audioRef} src={audioSrc} preload="metadata" className="hidden" />
            <Transport
              audioRef={audioRef}
              synth={source === "synth"}
              positionRef={positionRef}
              synthControlRef={synthControlRef}
              fallbackDuration={result.duration_seconds}
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
                    className={segmentClass(playbackRate === rate)}
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>
          </div>
          {view === "tab" ? (
            <TabViewer result={result} audioRef={audioRef} />
          ) : (
            <SheetMusicView
              jobId={jobId}
              audioRef={audioRef}
              bars={result.bars ?? []}
              tempoFactor={tempoFactor}
              barOffsetBeats={barOffsetBeats}
              playbackRate={playbackRate}
              playback={source === "synth" ? "synth" : "recording"}
              tone={tone}
              positionRef={positionRef}
              synthControlRef={synthControlRef}
              onOverrides={applyOverrides}
            />
          )}
        </section>
      )}
      {result && debug && <DebugOverlay audioRef={audioRef} stepTimes={stepTimes} />}
    </div>
  );
}
