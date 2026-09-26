"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

// Debug-only (?debug=1): measures how far the tab highlight lags the audio.
// Every animation frame it reads audio.currentTime and the column the tab
// currently highlights (TabViewer's data-active-step attribute). Each time a
// new column lights up it records lag = currentTime - that column's start.
// Columns that start while playing but are never highlighted count as
// skipped. Samples are kept on window.__riffscribeLag for automated tests.

interface LagSample {
  rate: number;
  step: number;
  lagMs: number;
  t: number; // audio.currentTime when the column lit up
}

interface LagLog {
  samples: LagSample[];
  skipped: { rate: number; step: number }[];
}

declare global {
  interface Window {
    __riffscribeLag?: LagLog;
  }
}

interface DebugOverlayProps {
  audioRef: RefObject<HTMLAudioElement>;
  stepTimes: number[];
}

function stats(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return { median: sorted[Math.floor(sorted.length / 2)], max: sorted[sorted.length - 1], n: sorted.length };
}

export default function DebugOverlay({ audioRef, stepTimes }: DebugOverlayProps) {
  const [view, setView] = useState({ time: 0, activeStart: null as number | null, diffMs: null as number | null });
  const [summary, setSummary] = useState<string[]>([]);
  const lastActive = useRef(-1);
  const lastExpected = useRef(-1);

  useEffect(() => {
    const log: LagLog = { samples: [], skipped: [] };
    window.__riffscribeLag = log;
    let frame = 0;
    let tick = 0;

    const loop = () => {
      frame = requestAnimationFrame(loop);
      const audio = audioRef.current;
      const table = document.querySelector<HTMLElement>("[data-active-step]");
      if (!audio || !table) return;
      const t = audio.currentTime;
      const active = Number(table.dataset.activeStep);
      const rate = audio.playbackRate;

      // Which column should be current: the last one that has started.
      let expected = -1;
      while (expected + 1 < stepTimes.length && stepTimes[expected + 1] <= t) expected++;
      if (!audio.paused && !audio.seeking) {
        // Columns passed over since the last frame without being shown.
        for (let s = lastExpected.current + 1; s < expected; s++) {
          if (s > lastActive.current) log.skipped.push({ rate, step: s });
        }
        if (active !== lastActive.current && active > lastActive.current && active >= 0) {
          log.samples.push({ rate, step: active, lagMs: (t - stepTimes[active]) * 1000, t });
        }
      }
      if (audio.seeking || active < lastActive.current - 1) {
        lastExpected.current = expected; // a seek: don't count jumped-over columns
      } else {
        lastExpected.current = Math.max(lastExpected.current, expected);
      }
      lastActive.current = active;

      if (++tick % 6 === 0) {
        const start = active >= 0 ? stepTimes[active] : null;
        setView({ time: t, activeStart: start, diffMs: start === null ? null : (t - start) * 1000 });
        const rates = Array.from(new Set(log.samples.map((s) => s.rate))).sort();
        setSummary(rates.map((r) => {
          const st = stats(log.samples.filter((s) => s.rate === r).map((s) => s.lagMs));
          const skipped = log.skipped.filter((s) => s.rate === r).length;
          return st ? `${r}x: median ${st.median.toFixed(0)}ms, max ${st.max.toFixed(0)}ms (n=${st.n}, skipped ${skipped})` : "";
        }));
      }
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [audioRef, stepTimes]);

  return (
    <div className="fixed bottom-4 right-4 z-50 rounded-lg bg-slate-900/90 px-3 py-2 font-mono text-xs text-slate-100 shadow-lg">
      <div>currentTime {view.time.toFixed(3)}s</div>
      <div>active column start {view.activeStart === null ? "-" : `${view.activeStart.toFixed(3)}s`}</div>
      <div>diff {view.diffMs === null ? "-" : `${view.diffMs.toFixed(0)}ms`}</div>
      {summary.filter(Boolean).map((line) => (
        <div key={line} className="text-amber-300">{line}</div>
      ))}
    </div>
  );
}
