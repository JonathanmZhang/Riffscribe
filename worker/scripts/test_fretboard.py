"""Standalone sanity-check for the fretboard-mapping DP algorithm.

Not part of the Celery pipeline; doesn't touch Redis or require audio.
Feeds hand-built sequences of raw_note_events (the same shape transcribe()
writes to Redis) straight into map_notes_to_positions() and prints the
chosen string/fret for each note, so the mapping can be eyeballed before
running the full pipeline.

Covers two cases:
  - a monophonic ascending scale, exercising the DP over single-note steps
  - chords (simultaneous notes), exercising _voice_chord specifically

Usage (inside the worker container):
    python -m scripts.test_fretboard
"""

from tasks.fretboard import map_notes_to_positions

# Ascending C major scale, one octave, quarter notes at 120bpm (0.5s each).
# A reasonable mapping should walk up the neck smoothly rather than jumping
# wildly between strings/frets for adjacent scale notes.
ASCENDING_SCALE = [
    {"pitch": "C4", "start_time": 0.0, "end_time": 0.5},
    {"pitch": "D4", "start_time": 0.5, "end_time": 1.0},
    {"pitch": "E4", "start_time": 1.0, "end_time": 1.5},
    {"pitch": "F4", "start_time": 1.5, "end_time": 2.0},
    {"pitch": "G4", "start_time": 2.0, "end_time": 2.5},
    {"pitch": "A4", "start_time": 2.5, "end_time": 3.0},
    {"pitch": "B4", "start_time": 3.0, "end_time": 3.5},
    {"pitch": "C5", "start_time": 3.5, "end_time": 4.0},
]

# G major triad, all three notes starting together (identical onsets) - the
# simple/common case for _voice_chord.
G_MAJOR_TRIAD = [
    {"pitch": "G3", "start_time": 0.0, "end_time": 1.0},
    {"pitch": "B3", "start_time": 0.0, "end_time": 1.0},
    {"pitch": "D4", "start_time": 0.0, "end_time": 1.0},
]

# G major with the root doubled an octave up, and onsets a few milliseconds
# apart (realistic ML-inference jitter) - exercises both the 4-note chord
# voicing and the onset-tolerance grouping in _group_into_steps.
G_MAJOR_FOUR_NOTE = [
    {"pitch": "G3", "start_time": 0.000, "end_time": 1.0},
    {"pitch": "B3", "start_time": 0.010, "end_time": 1.0},
    {"pitch": "D4", "start_time": 0.020, "end_time": 1.0},
    {"pitch": "G4", "start_time": 0.030, "end_time": 1.0},
]


def _print_notes(mapped: list[dict]) -> None:
    print(f"{'pitch':<6} {'string':<8} {'fret':<6} {'start':<8} {'end':<8}")
    for note in mapped:
        print(
            f"{note['pitch']:<6} {note['string']:<8} {note['fret']:<6} "
            f"{note['start_time']:<8.3f} {note['end_time']:<8.3f}"
        )


def _check_distinct_strings(mapped: list[dict]) -> None:
    strings = [note["string"] for note in mapped]
    seen = set()
    duplicates = set()
    for s in strings:
        if s in seen:
            duplicates.add(s)
        seen.add(s)

    if duplicates:
        print(f"\n/!\\ UNPLAYABLE: string(s) {sorted(duplicates)} used by more than one note in this chord")
    else:
        print("\nOK: every note in this chord landed on a distinct string")


def run_ascending_scale() -> None:
    print("=" * 60)
    print(f"Test 1: {len(ASCENDING_SCALE)}-note ascending C major scale (C4 -> C5)")
    print("=" * 60)

    mapped = map_notes_to_positions(ASCENDING_SCALE)
    _print_notes(mapped)

    frets = [note["fret"] for note in mapped]
    strings = [note["string"] for note in mapped]
    jumps = [abs(b - a) for a, b in zip(frets, frets[1:])]
    print(f"\nfret sequence:   {frets}")
    print(f"string sequence: {strings}")
    print(f"max fret jump between consecutive notes: {max(jumps) if jumps else 0}")


def run_chord(name: str, notes: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)

    mapped = map_notes_to_positions(notes)
    _print_notes(mapped)
    _check_distinct_strings(mapped)


def main() -> None:
    run_ascending_scale()
    run_chord(f"Test 2: {len(G_MAJOR_TRIAD)}-note G major triad, identical onsets (G3 B3 D4)", G_MAJOR_TRIAD)
    run_chord(
        f"Test 3: {len(G_MAJOR_FOUR_NOTE)}-note G major, near-identical onsets (G3 B3 D4 G4)",
        G_MAJOR_FOUR_NOTE,
    )


if __name__ == "__main__":
    main()
