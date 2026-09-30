"use client";

import { type RefObject, useEffect, useRef, useState } from "react";
import type * as AlphaTab from "@coderline/alphatab";
import { getJobMusicXmlUrl, type BarOffsetBeats, type TempoFactor } from "@/app/lib/api";
import { readDebugParams } from "@/app/lib/debug";

// alphaTab (MPL-2.0) renders the job's MusicXML export: standard notation +
// TAB, chord names above. Its UMD build is served from public/alphatab/
// (copied from node_modules by scripts/copy-alphatab.mjs), because it starts
// its render worker and audio worklet from its own script URL.
declare global {
  interface Window {
    alphaTab?: typeof AlphaTab;
    // With ?debug=1 (lib/debug.ts): the live alphaTab instance, for testing
    // the cursor against the audio from the browser console / Playwright.
    riffscribeSheet?: AlphaTab.AlphaTabApi | null;
  }
}

const ALPHATAB_ROOT = "/alphatab";
let scriptPromise: Promise<typeof AlphaTab> | null = null;

function loadAlphaTab(): Promise<typeof AlphaTab> {
  if (window.alphaTab) return Promise.resolve(window.alphaTab);
  scriptPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${ALPHATAB_ROOT}/alphaTab.min.js`;
    script.async = true;
    script.onload = () => (window.alphaTab ? resolve(window.alphaTab) : reject(new Error("alphaTab failed to load")));
    script.onerror = () => {
      scriptPromise = null;
      reject(new Error("alphaTab failed to load"));
    };
    document.head.appendChild(script);
  });
  return scriptPromise;
}

// "recording": alphaTab's cursor follows the page's <audio> (original audio
// or stem) through sync points, one per bar at the time the backend placed
// it. "synth": alphaTab plays the notation itself with its SoundFont synth.
export type SheetPlayback = "recording" | "synth";

interface SheetMusicViewProps {
  jobId: string;
  audioRef: RefObject<HTMLAudioElement>;
  // result.bars: start time (s) of each measure of the export, for the
  // current overrides. Also the reload trigger after an override changes.
  bars: number[];
  tempoFactor: TempoFactor;
  barOffsetBeats: BarOffsetBeats;
  playbackRate: number;
  onOverrides: (overrides: { tempo_factor?: TempoFactor; bar_offset_beats?: BarOffsetBeats }) => Promise<void>;
}

const buttonClass =
  "rounded-md border border-slate-200 bg-white px-3 py-1 text-sm font-medium text-slate-700 shadow-sm transition-colors " +
  "hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 " +
  "focus-visible:outline-indigo-500";

export default function SheetMusicView({
  jobId,
  audioRef,
  bars,
  tempoFactor,
  barOffsetBeats,
  playbackRate,
  onOverrides,
}: SheetMusicViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<AlphaTab.AlphaTabApi | null>(null);
  const barsRef = useRef(bars);
  barsRef.current = bars;
  const [playback, setPlayback] = useState<SheetPlayback>("recording");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playerReady, setPlayerReady] = useState(false);
  const [synthPlaying, setSynthPlaying] = useState(false);
  const [updating, setUpdating] = useState(false);

  // One alphaTab instance per playback mode (the player mode is fixed at
  // creation); the score is (re)loaded by the effect below.
  useEffect(() => {
    let cancelled = false;
    let api: AlphaTab.AlphaTabApi | null = null;
    const cleanups: (() => void)[] = [];
    setPlayerReady(false);
    setSynthPlaying(false);

    loadAlphaTab()
      .then((at) => {
        if (cancelled || !containerRef.current) return;
        const origin = window.location.origin;
        api = new at.AlphaTabApi(containerRef.current, {
          core: {
            scriptFile: `${origin}${ALPHATAB_ROOT}/alphaTab.min.js`,
            fontDirectory: `${origin}${ALPHATAB_ROOT}/font/`,
          },
          display: { layoutMode: at.LayoutMode.Page, scale: 0.9 },
          player: {
            playerMode: playback === "synth" ? at.PlayerMode.EnabledSynthesizer : at.PlayerMode.EnabledExternalMedia,
            soundFont: `${origin}${ALPHATAB_ROOT}/soundfont/sonivox.sf2`,
            enableCursor: true,
            enableAnimatedBeatCursor: true,
            enableUserInteraction: true,
            scrollElement: scrollRef.current ?? undefined,
            scrollMode: at.ScrollMode.Continuous,
          },
        });
        apiRef.current = api;
        if (readDebugParams().debug) window.riffscribeSheet = api;

        // Bars pinned to the recording's time axis (external media only).
        api.scoreLoaded.on((score) => {
          if (playback !== "recording") return;
          score.applyFlatSyncPoints(
            barsRef.current.slice(0, score.masterBars.length).map((t, i) => ({
              barIndex: i,
              barPosition: 0,
              barOccurence: 0,
              millisecondOffset: Math.round(t * 1000),
            })),
          );
        });
        api.renderFinished.on(() => setLoading(false));
        api.error.on((e) => setError(e.message ?? String(e)));
        api.playerReady.on(() => setPlayerReady(true));
        api.playerStateChanged.on((e) => setSynthPlaying(e.state === at.synth.PlayerState.Playing));

        if (playback === "recording") {
          const audio = audioRef.current;
          if (!audio) return;
          const output = api.player?.output as unknown as AlphaTab.synth.IExternalMediaSynthOutput | undefined;
          if (!output) return;
          output.handler = {
            get backingTrackDuration() {
              return Number.isFinite(audio.duration) ? audio.duration * 1000 : 0;
            },
            get playbackRate() {
              return audio.playbackRate;
            },
            set playbackRate(value: number) {
              audio.playbackRate = value;
            },
            get masterVolume() {
              return audio.volume;
            },
            set masterVolume(value: number) {
              audio.volume = value;
            },
            seekTo(ms: number) {
              audio.currentTime = ms / 1000;
            },
            play() {
              void audio.play();
            },
            pause() {
              audio.pause();
            },
          };
          // The audio element drives: its position feeds the cursor every
          // frame while playing, and its play/pause sets alphaTab's state.
          let frame = 0;
          const push = () => output.updatePosition(audio.currentTime * 1000);
          const tick = () => {
            push();
            frame = requestAnimationFrame(tick);
          };
          const onPlay = () => {
            if (api?.playerState !== at.synth.PlayerState.Playing) api?.play();
            cancelAnimationFrame(frame);
            frame = requestAnimationFrame(tick);
          };
          const onPause = () => {
            cancelAnimationFrame(frame);
            push();
            if (api?.playerState === at.synth.PlayerState.Playing) api?.pause();
          };
          audio.addEventListener("play", onPlay);
          audio.addEventListener("pause", onPause);
          audio.addEventListener("seeked", push);
          audio.addEventListener("timeupdate", push);
          if (!audio.paused) onPlay();
          cleanups.push(() => {
            cancelAnimationFrame(frame);
            audio.removeEventListener("play", onPlay);
            audio.removeEventListener("pause", onPause);
            audio.removeEventListener("seeked", push);
            audio.removeEventListener("timeupdate", push);
          });
        } else {
          audioRef.current?.pause(); // one sound source at a time
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));

    return () => {
      cancelled = true;
      cleanups.forEach((fn) => fn());
      api?.destroy();
      apiRef.current = null;
    };
  }, [playback, audioRef]);

  // (Re)load the score: on mount, on a playback-mode switch and after an
  // override changes (bars differ). Nothing is re-transcribed; the backend
  // rebuilds the MusicXML from the stored notes.
  const barsKey = bars.join(",");
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const load = async () => {
      const response = await fetch(getJobMusicXmlUrl(jobId));
      if (!response.ok) throw new Error(`MusicXML: ${response.status} ${response.statusText}`);
      const data = new Uint8Array(await response.arrayBuffer());
      // The API is created asynchronously (script load); wait for it.
      for (let i = 0; !apiRef.current && i < 100 && !cancelled; i++) {
        await new Promise((r) => setTimeout(r, 50));
      }
      if (!cancelled) apiRef.current?.load(data);
    };
    load().catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [jobId, playback, barsKey]);

  useEffect(() => {
    if (apiRef.current && playback === "synth") apiRef.current.playbackSpeed = playbackRate;
  }, [playbackRate, playback, playerReady]);

  const applyOverrides = async (overrides: { tempo_factor?: TempoFactor; bar_offset_beats?: BarOffsetBeats }) => {
    setUpdating(true);
    try {
      await onOverrides(overrides);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-slate-500">Playback</span>
          <div role="group" aria-label="Sheet music playback" className="inline-flex rounded-lg bg-slate-100 p-1 font-medium">
            {(["recording", "synth"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                aria-pressed={playback === mode}
                onClick={() => setPlayback(mode)}
                className={`rounded-md px-3 py-1 transition-colors ${
                  playback === mode ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {mode === "recording" ? "Recording" : "Synth"}
              </button>
            ))}
          </div>
          {playback === "synth" && (
            <button
              type="button"
              className={buttonClass}
              disabled={!playerReady}
              onClick={() => apiRef.current?.playPause()}
            >
              {!playerReady ? "Loading sounds…" : synthPlaying ? "Pause" : "Play"}
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-500">Bars</span>
          <button
            type="button"
            className={buttonClass}
            disabled={updating || tempoFactor >= 2}
            onClick={() => applyOverrides({ tempo_factor: (tempoFactor * 2) as TempoFactor })}
          >
            Tempo ×2
          </button>
          <button
            type="button"
            className={buttonClass}
            disabled={updating || tempoFactor <= 0.5}
            onClick={() => applyOverrides({ tempo_factor: (tempoFactor / 2) as TempoFactor })}
          >
            Tempo ÷2
          </button>
          <button
            type="button"
            className={buttonClass}
            disabled={updating}
            onClick={() => applyOverrides({ bar_offset_beats: ((barOffsetBeats + 1) % 4) as BarOffsetBeats })}
          >
            Shift bar start
          </button>
          <span className="text-xs text-slate-500">
            tempo ×{tempoFactor}, bar start +{barOffsetBeats} beat{barOffsetBeats === 1 ? "" : "s"}
          </span>
        </div>
      </div>
      {playback === "recording" && (
        <p className="text-xs text-slate-500">
          Play the audio above; the cursor follows it. Click a note to jump there.
        </p>
      )}
      {error && <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      <div ref={scrollRef} className="relative max-h-[70vh] overflow-y-auto rounded-lg border border-slate-200 bg-white">
        {loading && !error && <p className="p-4 text-sm text-slate-500">Rendering sheet music…</p>}
        <div ref={containerRef} />
      </div>
    </div>
  );
}
