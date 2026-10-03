"use client";

import { type MutableRefObject, type RefObject, useEffect, useState } from "react";
import type { PlaybackPosition, SynthControl } from "@/app/components/SheetMusicView";

// One transport for every playback source: play/pause, time and a seek bar
// on the recording's time axis. The recordings (original, stem) play in the
// page's <audio>; the synth is driven through SheetMusicView's SynthControl
// and reports its position through positionRef. State is polled here, every
// 100ms, so only this component re-renders while playing (not the tab).

interface TransportProps {
  audioRef: RefObject<HTMLAudioElement>;
  synth: boolean;
  positionRef: MutableRefObject<PlaybackPosition>;
  synthControlRef: MutableRefObject<SynthControl | null>;
  // Used until the audio's own duration is known.
  fallbackDuration: number;
}

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function Transport({ audioRef, synth, positionRef, synthControlRef, fallbackDuration }: TransportProps) {
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(fallbackDuration);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(true);

  useEffect(() => {
    const read = () => {
      const audio = audioRef.current;
      if (audio && Number.isFinite(audio.duration) && audio.duration > 0) setDuration(audio.duration);
      if (synth) {
        setTime(positionRef.current.time);
        setPlaying(positionRef.current.playing);
        setReady(synthControlRef.current?.ready ?? false);
      } else if (audio) {
        setTime(audio.currentTime);
        setPlaying(!audio.paused);
        setReady(true);
      }
    };
    read();
    const timer = window.setInterval(read, 100);
    return () => window.clearInterval(timer);
  }, [audioRef, synth, positionRef, synthControlRef]);

  const toggle = () => {
    if (synth) {
      synthControlRef.current?.toggle();
      return;
    }
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) void audio.play();
    else audio.pause();
  };

  const seek = (seconds: number) => {
    setTime(seconds);
    if (synth) {
      synthControlRef.current?.seek(seconds);
      positionRef.current = { ...positionRef.current, time: seconds };
    } else if (audioRef.current) {
      audioRef.current.currentTime = seconds;
    }
  };

  const label = !ready ? "Loading sounds" : playing ? "Pause" : "Play";
  return (
    <div className="flex items-center gap-3 rounded-full bg-slate-100 py-1.5 pl-1.5 pr-4">
      <button
        type="button"
        onClick={toggle}
        disabled={!ready}
        aria-label={label}
        title={label}
        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-sm transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-wait disabled:bg-indigo-300"
      >
        {playing ? (
          <svg viewBox="0 0 20 20" aria-hidden="true" className="h-5 w-5 fill-current">
            <rect x="5" y="4" width="3.5" height="12" rx="1" />
            <rect x="11.5" y="4" width="3.5" height="12" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 20 20" aria-hidden="true" className="ml-0.5 h-5 w-5 fill-current">
            <path d="M6 4.2v11.6a.8.8 0 0 0 1.2.7l9.3-5.8a.8.8 0 0 0 0-1.4L7.2 3.5a.8.8 0 0 0-1.2.7Z" />
          </svg>
        )}
      </button>
      <span className="shrink-0 text-sm tabular-nums text-slate-700">
        {ready ? (
          <>
            {formatTime(time)} <span className="text-slate-600">/ {formatTime(duration)}</span>
          </>
        ) : (
          "Loading sounds…"
        )}
      </span>
      <input
        type="range"
        min={0}
        max={Math.max(duration, 0.1)}
        step={0.1}
        value={Math.min(time, duration)}
        onChange={(event) => seek(Number(event.target.value))}
        disabled={!ready}
        aria-label="Playback position"
        aria-valuetext={`${formatTime(time)} of ${formatTime(duration)}`}
        className="h-2 min-w-0 flex-1 cursor-pointer accent-indigo-600 disabled:cursor-wait"
      />
    </div>
  );
}
