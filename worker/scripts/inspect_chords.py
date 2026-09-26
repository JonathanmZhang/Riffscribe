"""Chord window inspector: what each pipeline stage does to the notes in one
time window of an audio file.

Usage (inside the worker container):
    python -m scripts.inspect_chords <file> <start_s> <end_s> [--separate] [--seed N]
                                     [--expect NOTES]

Prints, for note onsets in [start_s, end_s):
  1. every Basic Pitch note event, including ones the confidence threshold
     drops (marked), with start, end, pitch, MIDI number and amplitude
  2. how the pipeline's chord grouping groups the kept notes, with each
     group's onset spread
  3. what the fretboard mapper does with each group, including dropped
     notes and why
With --expect (e.g. "G2,B2,D3,G3,B3,G4" for exact notes, or "G,B,D" for
pitch classes in any octave), also says where each expected note was lost,
using the same (a)-(e) categories as scripts/eval_chords.py.

--separate runs the same analysis on the Demucs guitar stem
(separate_guitar_stem), seeded with --seed (default 0) for repeatability.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import tempfile  # noqa: E402

import pretty_midi  # noqa: E402

from scripts.chord_stages import (  # noqa: E402
    CHORD_ONSET_TOLERANCE_SECONDS,
    CONFIDENCE_THRESHOLD,
    classify_chord,
    detect,
    group,
    map_with_trace,
    note_key,
    onset_spread_ms,
    prepare_audio,
    step_index,
)

STAGE_TEXT = {
    "a": "(a) never detected by Basic Pitch",
    "b": "(b) detected but below threshold",
    "c": "(c) grouping",
    "d": "(d) dropped by mapper",
    "ok": "ok, in the tab",
}


def parse_expected(spec: str) -> tuple[list[int], bool]:
    """'G2,B2,D3' -> exact MIDI numbers; 'G,B,D' -> pitch classes."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if all(any(ch.isdigit() for ch in s) for s in items):
        return [pretty_midi.note_name_to_number(s) for s in items], False
    if any(any(ch.isdigit() for ch in s) for s in items):
        raise SystemExit("--expect: use either all exact notes (G2,B2) or all pitch classes (G,B), not a mix")
    return [pretty_midi.note_name_to_number(s + "4") % 12 for s in items], True


def inspect(path: str, start: float, end: float, separate: bool, seed: int, expect: str | None) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        audio = prepare_audio(path, workdir, separate=separate, seed=seed)
        events, activations = detect(audio["normalized_path"])

    kept = [e for e in events if e["kept"]]
    groups = group(kept)
    step_of = step_index(groups)
    mapping = map_with_trace(kept)
    in_window = [e for e in events if start <= e["start_time"] < end]

    source = "separated guitar stem" if separate else "full mix"
    print(f"\n=== {path}  [{start:.2f}s, {end:.2f}s)  on the {source}"
          + (f" (separation {audio['separation_seconds']}s, seed {seed})" if separate else ""))

    print(f"\n1. Basic Pitch note events with onsets in the window ({len(in_window)}; "
          f"threshold {CONFIDENCE_THRESHOLD}):")
    print(f"   {'start':>7} {'end':>7}  {'pitch':<5} {'midi':>4}  {'amp':>5}")
    for e in in_window:
        flag = "" if e["kept"] else f"  <- dropped (< {CONFIDENCE_THRESHOLD})"
        print(f"   {e['start_time']:7.3f} {e['end_time']:7.3f}  {e['pitch']:<5} {e['midi']:>4}  "
              f"{e['amplitude']:5.3f}{flag}")

    window_steps = sorted({step_of[note_key(e)] for e in in_window if e["kept"]})
    print(f"\n2. Grouping of kept notes (onset tolerance {CHORD_ONSET_TOLERANCE_SECONDS * 1000:.0f}ms), "
          f"{len(window_steps)} group(s) touching the window:")
    for s in window_steps:
        g = groups[s]
        pitches = " ".join(n["pitch"] for n in sorted(g, key=lambda n: pretty_midi.note_name_to_number(n["pitch"])))
        print(f"   step {s}: onset {g[0]['start_time']:.3f}s, {len(g)} note(s), spread "
              f"{onset_spread_ms(g):.0f}ms: {pitches}")

    print("\n3. Fretboard mapping of those groups:")
    for s in window_steps:
        print(f"   step {s}:")
        for n in sorted(groups[s], key=lambda n: pretty_midi.note_name_to_number(n["pitch"])):
            key = note_key(n)
            if key in mapping["positions"]:
                string, fret = mapping["positions"][key]
                print(f"     {n['pitch']:<4} -> string {string}, fret {fret}")
            else:
                print(f"     {n['pitch']:<4} -> DROPPED: {mapping['drops'][key]}")

    result = {"file": path, "window": [start, end], "separated": separate, "events": in_window}
    if expect:
        expected, pc_mode = parse_expected(expect)
        chord = classify_chord(expected, pc_mode, (start, end), events, step_of, mapping, activations)
        mode = "pitch classes, any octave" if pc_mode else "exact notes"
        print(f"\n4. Expected {expect} ({mode}): {chord['funnel']['mapped']}/{len(expected)} in the tab, "
              f"complete: {chord['complete']}")
        for n in chord["notes"]:
            detail = ""
            if n["stage"] == "a":
                detail = f"; peak model activation for that pitch in the window: {n['peak_activation']}"
            elif n["stage"] == "b":
                detail = f"; amplitudes {n['amplitudes']}"
            elif n["stage"] in ("c", "d"):
                detail = f"; {n['why']}"
            elif n["stage"] == "ok":
                detail = f"; string {n['position'][0]}, fret {n['position'][1]}"
            print(f"   {n['expected']:<4} {STAGE_TEXT[n['stage']]}{detail}")
        if chord["extras"]:
            print("   (e) extra notes: " + ", ".join(
                f"{x['pitch']} ({x['kind']}, amp {x['amplitude']}{', mapped' if x['mapped'] else ''})"
                for x in chord["extras"]))
        result["chord"] = chord
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file")
    parser.add_argument("start_s", type=float)
    parser.add_argument("end_s", type=float)
    parser.add_argument("--separate", action="store_true", help="analyse the Demucs guitar stem instead")
    parser.add_argument("--seed", type=int, default=0, help="separation seed (default 0)")
    parser.add_argument("--expect", help='expected chord notes, e.g. "G2,B2,D3" or "G,B,D"')
    args = parser.parse_args()
    inspect(args.file, args.start_s, args.end_s, args.separate, args.seed, args.expect)


if __name__ == "__main__":
    main()
