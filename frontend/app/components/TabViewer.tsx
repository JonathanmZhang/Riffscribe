"use client";

import { useEffect, useMemo, useRef, type RefObject } from "react";
import type { Note, TabResult } from "@/app/lib/api";

// Display order top-to-bottom matches conventional tab notation: high e on
// top, low E on bottom - the reverse of the string numbering (1 = high e).
const STRING_DISPLAY_ORDER = [1, 2, 3, 4, 5, 6];
const STRING_LABELS: Record<number, string> = {
  1: "e",
  2: "B",
  3: "G",
  4: "D",
  5: "A",
  6: "E",
};

// Notes starting within this window of each other are shown as one column
// (a chord). Must match CHORD_ONSET_TOLERANCE_SECONDS in the worker's
// fretboard.py. 150ms is an empirically-set value based on real strum
// testing, not a proven optimum - same status as the cost-function constants.
const STEP_TOLERANCE_SECONDS = 0.15;

// Basic Pitch places note onsets slightly after the real attack: on the
// synthetic chord set (known strum times) a column's first detected note is a
// median 22ms late (p10 -6ms; low notes can be ~100ms late, which no constant
// fixes). So a column is highlighted this much before its start_time. This is
// the only onset-offset adjustment in the frontend.
export const ONSET_LEAD_SECONDS = 0.02;

interface TabStep {
  time: number;
  notes: Note[];
}

export function groupIntoSteps(notes: Note[]): TabStep[] {
  const sorted = [...notes].sort((a, b) => a.start_time - b.start_time);
  const steps: TabStep[] = [];

  for (const note of sorted) {
    const last = steps[steps.length - 1];
    if (last && note.start_time - last.time <= STEP_TOLERANCE_SECONDS) {
      last.notes.push(note);
    } else {
      steps.push({ time: note.start_time, notes: [note] });
    }
  }

  return steps;
}

// Index of the last column that has started at media time t, or -1.
function currentStepIndex(stepTimes: number[], t: number): number {
  let lo = 0;
  let hi = stepTimes.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (stepTimes[mid] <= t) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

interface TabViewerProps {
  result: TabResult;
  // The paired <audio> element. Its currentTime is media time, so it already
  // reflects playbackRate; the highlight needs no speed adjustment.
  audioRef?: RefObject<HTMLAudioElement>;
}

export default function TabViewer({ result, audioRef }: TabViewerProps) {
  const steps = useMemo(() => groupIntoSteps(result.notes), [result.notes]);
  const stepTimes = useMemo(() => steps.map((s) => s.time), [steps]);
  const sortedNotes = useMemo(() => [...result.notes].sort((a, b) => a.start_time - b.start_time), [result.notes]);
  const stepOfNote = useMemo(() => {
    const map = new Map<Note, number>();
    steps.forEach((step, i) => step.notes.forEach((n) => map.set(n, i)));
    return map;
  }, [steps]);

  const rootRef = useRef<HTMLDivElement>(null);
  const tableRef = useRef<HTMLTableElement>(null);

  // Playhead: the highlighted column is the last one whose start (minus the
  // onset lead) has passed, kept until the next one starts. An
  // animation-frame loop reads audio.currentTime (timeupdate only fires ~4
  // times a second) and, when the column changes, moves the `is-active`
  // class between the cached cells of the two columns - no React re-render,
  // so the change is painted in the same frame it's decided.
  useEffect(() => {
    const root = rootRef.current;
    const table = tableRef.current;
    if (!root || !table) return;

    const columns: Element[][] = steps.map(() => []);
    root.querySelectorAll<HTMLElement>("[data-col],[data-col-time],[data-note-col]").forEach((el) => {
      const index = Number(el.dataset.col ?? el.dataset.colTime ?? el.dataset.noteCol);
      columns[index]?.push(el);
    });

    let active = -1;
    let frame = 0;
    const setActive = (index: number) => {
      if (index === active) return;
      columns[active]?.forEach((el) => el.classList.remove("is-active"));
      columns[index]?.forEach((el) => el.classList.add("is-active"));
      active = index;
      table.dataset.activeStep = String(index);
    };

    const loop = () => {
      frame = requestAnimationFrame(loop);
      const audio = audioRef?.current;
      if (!audio) return;
      setActive(
        audio.currentTime > 0 || !audio.paused ? currentStepIndex(stepTimes, audio.currentTime + ONSET_LEAD_SECONDS) : -1,
      );
    };
    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      setActive(-1);
    };
  }, [audioRef, steps, stepTimes]);

  if (steps.length === 0) {
    return <p className="text-slate-500">No notes detected in this audio.</p>;
  }

  return (
    <div ref={rootRef} className="flex w-full flex-col gap-6">
      <dl className="flex flex-wrap gap-2 text-xs">
        {[
          ["Duration", `${result.duration_seconds.toFixed(2)}s`],
          // Estimated by beat tracking, not measured - shown as approximate.
          ["Tempo (est.)", result.tempo_bpm > 0 ? `~${result.tempo_bpm} bpm` : "unknown"],
          ["Notes", String(result.notes.length)],
        ].map(([label, value]) => (
          <div key={label} className="flex gap-1.5 rounded-md bg-slate-100 px-2.5 py-1">
            <dt className="text-slate-500">{label}</dt>
            <dd className="font-semibold text-slate-800">{value}</dd>
          </div>
        ))}
      </dl>

      {/* Guitar-tab-style grid: one row per string (high e on top, per
          convention), one column per time step. */}
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50">
        <table ref={tableRef} className="border-collapse font-mono text-sm" data-active-step={-1}>
          <tbody>
            {STRING_DISPLAY_ORDER.map((string) => (
              <tr key={string}>
                <td className="sticky left-0 z-10 border-r border-slate-200 bg-slate-100 px-3 py-1 font-semibold text-slate-500">
                  {STRING_LABELS[string]}
                </td>
                {steps.map((step, i) => {
                  const note = step.notes.find((n) => n.string === string);
                  return (
                    <td
                      key={i}
                      data-col={i}
                      data-empty={note ? undefined : ""}
                      className={`min-w-[2.5rem] px-2 py-1 text-center ${
                        note ? "font-semibold text-slate-900" : "text-slate-300"
                      }`}
                    >
                      {note ? note.fret : "-"}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr className="border-t border-slate-200">
              <td className="sticky left-0 z-10 border-r border-slate-200 bg-slate-100 px-3 py-1" />
              {steps.map((step, i) => (
                <td key={i} data-col-time={i} className="px-2 py-1 text-center text-xs text-slate-400">
                  {step.time.toFixed(2)}s
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Exact per-note detail, in time order. */}
      <div className="max-h-96 overflow-y-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-50">
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-2 font-semibold">Start</th>
              <th className="px-4 py-2 font-semibold">End</th>
              <th className="px-4 py-2 font-semibold">String</th>
              <th className="px-4 py-2 font-semibold">Fret</th>
              <th className="px-4 py-2 font-semibold">Pitch</th>
            </tr>
          </thead>
          <tbody className="text-slate-700">
            {sortedNotes.map((note, i) => (
              <tr
                key={i}
                data-note-col={stepOfNote.get(note)}
                className="border-b border-slate-100 last:border-0 hover:bg-slate-50"
              >
                <td className="px-4 py-1.5 tabular-nums">{note.start_time.toFixed(2)}s</td>
                <td className="px-4 py-1.5 tabular-nums">{note.end_time.toFixed(2)}s</td>
                <td className="px-4 py-1.5">{STRING_LABELS[note.string]} ({note.string})</td>
                <td className="px-4 py-1.5 tabular-nums">{note.fret}</td>
                <td className="px-4 py-1.5">{note.pitch}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
