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
// (a chord), mirroring the same tolerance the backend's fretboard mapper
// uses to group simultaneous notes.
const STEP_TOLERANCE_SECONDS = 0.05;

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
    return <p className="text-gray-600">No notes detected in this audio.</p>;
  }

  return (
    <div className="flex w-full max-w-3xl flex-col gap-6">
      <p className="text-sm text-gray-600">
        Duration: {result.duration_seconds.toFixed(2)}s · Tempo: {result.tempo_bpm} bpm ·{" "}
        {result.notes.length} note{result.notes.length === 1 ? "" : "s"}
      </p>

      {/* Guitar-tab-style grid: one row per string (high e on top, per
          convention), one column per time step. */}
      <div className="overflow-x-auto rounded border">
        <table className="border-collapse font-mono text-sm">
          <tbody>
            {STRING_DISPLAY_ORDER.map((string) => (
              <tr key={string}>
                <td className="border-r px-2 py-1 text-gray-500">{STRING_LABELS[string]}|</td>
                {steps.map((step, i) => {
                  const note = step.notes.find((n) => n.string === string);
                  const active = note ? isActive(note, currentTime) : false;
                  return (
                    <td
                      key={i}
                      className={`min-w-[2.5rem] px-2 py-1 text-center ${
                        active ? "bg-yellow-300 font-bold text-black" : ""
                      }`}
                    >
                      {note ? note.fret : "-"}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <td className="border-r px-2 py-1" />
              {steps.map((step, i) => {
                const stepActive = step.notes.some((n) => isActive(n, currentTime));
                return (
                  <td
                    key={i}
                    className={`px-2 py-1 text-center text-xs ${
                      stepActive ? "font-semibold text-yellow-700" : "text-gray-500"
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
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left">
            <th className="py-1 pr-4">Start</th>
            <th className="py-1 pr-4">End</th>
            <th className="py-1 pr-4">String</th>
            <th className="py-1 pr-4">Fret</th>
            <th className="py-1 pr-4">Pitch</th>
          </tr>
        </thead>
        <tbody>
          {sortedNotes.map((note, i) => (
            <tr
              key={i}
              className={`border-b last:border-0 ${isActive(note, currentTime) ? "bg-yellow-300" : ""}`}
            >
              <td className="py-1 pr-4">{note.start_time.toFixed(2)}s</td>
              <td className="py-1 pr-4">{note.end_time.toFixed(2)}s</td>
              <td className="py-1 pr-4">{STRING_LABELS[note.string]} ({note.string})</td>
              <td className="py-1 pr-4">{note.fret}</td>
              <td className="py-1 pr-4">{note.pitch}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
