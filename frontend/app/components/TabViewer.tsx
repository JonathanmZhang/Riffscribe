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

interface TabStep {
  time: number;
  notes: Note[];
}

function groupIntoSteps(notes: Note[]): TabStep[] {
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

function isActive(note: Note, currentTime: number | null): boolean {
  return currentTime !== null && note.start_time <= currentTime && currentTime < note.end_time;
}

interface TabViewerProps {
  result: TabResult;
  // Current playback position of the paired <audio> element, in seconds.
  // Notes whose [start_time, end_time) span contains this are highlighted.
  // null/omitted means no audio is playing - nothing is highlighted.
  currentTime?: number | null;
}

export default function TabViewer({ result, currentTime = null }: TabViewerProps) {
  const steps = groupIntoSteps(result.notes);
  const sortedNotes = [...result.notes].sort((a, b) => a.start_time - b.start_time);

  if (steps.length === 0) {
    return <p className="text-slate-500">No notes detected in this audio.</p>;
  }

  return (
    <div className="flex w-full flex-col gap-6">
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
        <table className="border-collapse font-mono text-sm">
          <tbody>
            {STRING_DISPLAY_ORDER.map((string) => (
              <tr key={string}>
                <td className="sticky left-0 border-r border-slate-200 bg-slate-100 px-3 py-1 font-semibold text-slate-500">
                  {STRING_LABELS[string]}
                </td>
                {steps.map((step, i) => {
                  const note = step.notes.find((n) => n.string === string);
                  const active = note ? isActive(note, currentTime) : false;
                  return (
                    <td
                      key={i}
                      className={`min-w-[2.5rem] px-2 py-1 text-center ${
                        active
                          ? "bg-amber-300 font-bold text-slate-900"
                          : note
                            ? "font-semibold text-slate-900"
                            : "text-slate-300"
                      }`}
                    >
                      {note ? note.fret : "-"}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr className="border-t border-slate-200">
              <td className="sticky left-0 border-r border-slate-200 bg-slate-100 px-3 py-1" />
              {steps.map((step, i) => {
                const stepActive = step.notes.some((n) => isActive(n, currentTime));
                return (
                  <td
                    key={i}
                    className={`px-2 py-1 text-center text-xs ${
                      stepActive ? "font-semibold text-amber-700" : "text-slate-400"
                    }`}
                  >
                    {step.time.toFixed(2)}s
                  </td>
                );
              })}
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
                className={`border-b border-slate-100 last:border-0 ${
                  isActive(note, currentTime) ? "bg-amber-200 text-slate-900" : "hover:bg-slate-50"
                }`}
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
