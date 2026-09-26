"""Synthetic chord test set with known ground truth, for scripts/eval_chords.py.

Renders strummed standard-tuning guitar voicings with FluidSynth and the
FluidR3_GM General MIDI soundfont (MIT license; installed from Debian's
fluid-soundfont-gm package at worker image build time, not committed).

Usage (inside the worker container):
    python -m scripts.make_chord_testset [--out DIR] [--programs 27,29,30] [--seed 0]

Writes, per GM program, chords_prog<N>.wav (44.1kHz) and chords_prog<N>.json
(ground truth: chord name, onset, expected MIDI notes, strum offsets).
Programs are 0-based GM numbers as used in MIDI files: 27 = Electric Guitar
(clean), 29 = Overdriven Guitar, 30 = Distortion Guitar (26 would be
Electric Guitar (jazz)).

Layout: 1s of silence, then each chord below strummed once (notes 10-25ms
apart, low to high), held 1.0s with a 0.25s gap; then a fast progression at
4 chords per second.

Also writes repeats_prog<N>.wav/.json: intentionally repeated notes (eighth
notes at 120 and 160bpm, power-chord chugs, palm-mute-style short notes),
each a separate attack, as a guard against over-merging re-triggers.
"""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import pretty_midi
import soundfile as sf

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SAMPLE_RATE = 44100
DEFAULT_PROGRAMS = [27, 29, 30]
PROGRAM_NAMES = {26: "electric (jazz)", 27: "electric (clean)", 29: "overdriven", 30: "distortion"}

# Standard tuning open strings: E2=40 A2=45 D3=50 G3=55 B3=59 E4=64.
# Each voicing lists its exact MIDI notes, low string to high string.
VOICINGS = {
    # name: (shape, MIDI notes)
    "G": ("320003", [43, 47, 50, 55, 59, 67]),     # G2 B2 D3 G3 B3 G4
    "C": ("x32010", [48, 52, 55, 60, 64]),         # C3 E3 G3 C4 E4
    "D": ("xx0232", [50, 57, 62, 66]),             # D3 A3 D4 F#4
    "Em": ("022000", [40, 47, 52, 55, 59, 64]),    # E2 B2 E3 G3 B3 E4
    "Am": ("x02210", [45, 52, 57, 60, 64]),        # A2 E3 A3 C4 E4
    "E": ("022100", [40, 47, 52, 56, 59, 64]),     # E2 B2 E3 G#3 B3 E4
    "E5 (2-note)": ("02xxxx", [40, 47]),           # E2 B2
    "E5 (3-note)": ("022xxx", [40, 47, 52]),       # E2 B2 E3
    "A5 (2-note)": ("x02xxx", [45, 52]),           # A2 E3
    "A5 (3-note)": ("x022xx", [45, 52, 57]),       # A2 E3 A3
    "F (barre)": ("133211", [41, 48, 53, 57, 60, 65]),   # F2 C3 F3 A3 C4 F4
    "Bm (barre)": ("x24432", [47, 54, 59, 62, 66]),      # B2 F#3 B3 D4 F#4
}
FAST_PROGRESSION = ["G", "C", "D", "Em", "G", "C", "D", "Em"]

LEAD_IN_S = 1.0
HOLD_S = 1.0
GAP_S = 0.25
FAST_CHORD_S = 0.25  # 4 chords per second
STRUM_GAP_MS = (10.0, 25.0)


def build_score(seed: int) -> list[dict]:
    """Chord events with onsets, notes and per-note strum offsets."""
    rng = np.random.default_rng(seed)
    chords = []
    t = LEAD_IN_S

    def add(name: str, onset: float, duration: float, section: str):
        notes = VOICINGS[name][1]
        offsets = np.concatenate([[0.0], np.cumsum(rng.uniform(*STRUM_GAP_MS, size=len(notes) - 1))]) / 1000.0
        velocities = rng.integers(90, 111, size=len(notes))
        chords.append({
            "name": name,
            "section": section,
            "shape": VOICINGS[name][0],
            "onset": round(onset, 4),
            "end": round(onset + duration, 4),
            "notes": notes,
            "note_names": [pretty_midi.note_number_to_name(n) for n in notes],
            "strum_offsets_ms": [round(o * 1000, 1) for o in offsets],
            "velocities": [int(v) for v in velocities],
        })

    for name in VOICINGS:
        add(name, t, HOLD_S, "sustained")
        t += HOLD_S + GAP_S
    t += 0.5
    for i, name in enumerate(FAST_PROGRESSION):
        add(name, t, FAST_CHORD_S, f"fast #{i + 1}")
        t += FAST_CHORD_S
    return chords


# Intentionally repeated notes, as a guard for re-trigger merging: each
# pattern's notes are separate attacks that must stay separate in the tab.
# (name, MIDI notes struck together, bpm, count, note length as a fraction of
# the eighth-note spacing)
REPEAT_PATTERNS = [
    ("E2 eighths @120bpm", [40], 120, 8, 0.9),
    ("A2 eighths @160bpm", [45], 160, 8, 0.9),
    ("B3 eighths @120bpm", [59], 120, 8, 0.9),
    ("E5 chugs @160bpm", [40, 47], 160, 8, 0.9),
    ("E2 palm-mute-style @160bpm", [40], 160, 8, 0.32),  # ~60ms notes
]


def build_repeats(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 1)
    notes = []
    t = LEAD_IN_S
    for name, pitches, bpm, count, length in REPEAT_PATTERNS:
        spacing = 60.0 / bpm / 2  # eighth notes
        for i in range(count):
            onset = t + i * spacing
            for pitch in pitches:
                notes.append({
                    "pattern": name,
                    "midi": pitch,
                    "pitch": pretty_midi.note_number_to_name(pitch),
                    "onset": round(onset, 4),
                    "end": round(onset + spacing * length, 4),
                    "velocity": int(rng.integers(95, 111)),
                })
        t += count * spacing + 0.75
    return notes


def render_notes(notes: list[tuple[int, float, float, int]], program: int, wav_path: str) -> str:
    """notes: (pitch, start, end, velocity)."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    guitar = pretty_midi.Instrument(program=program, name=PROGRAM_NAMES.get(program, f"program {program}"))
    for pitch, start, end, velocity in notes:
        guitar.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    midi.instruments.append(guitar)
    return _render_midi(midi, wav_path)


def render(chords: list[dict], program: int, out_dir: str) -> str:
    notes = [
        (note, chord["onset"] + offset / 1000.0, chord["end"], velocity)
        for chord in chords
        for note, offset, velocity in zip(chord["notes"], chord["strum_offsets_ms"], chord["velocities"])
    ]
    return render_notes(notes, program, os.path.join(out_dir, f"chords_prog{program}.wav"))


def _render_midi(midi: pretty_midi.PrettyMIDI, wav_path: str) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        mid_path = os.path.join(workdir, "chords.mid")
        midi.write(mid_path)
        # Reverb and chorus off so the ground truth isn't smeared; gain chosen
        # to stay clear of clipping on six-note distorted chords.
        subprocess.run(
            ["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5", "-r", str(SAMPLE_RATE),
             "-F", wav_path, SOUNDFONT, mid_path],
            check=True, capture_output=True,
        )
    return wav_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="/app/data/_testaudio/chord_testset")
    parser.add_argument("--programs", default=",".join(map(str, DEFAULT_PROGRAMS)),
                        help="comma-separated 0-based GM programs (default 27,29,30)")
    parser.add_argument("--seed", type=int, default=0, help="strum timing/velocity seed (default 0)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chords = build_score(args.seed)
    for program in [int(p) for p in args.programs.split(",")]:
        wav_path = render(chords, program, args.out)
        audio, sr = sf.read(wav_path)
        truth = {
            "wav": os.path.basename(wav_path),
            "program": program,
            "program_name": PROGRAM_NAMES.get(program, f"program {program}"),
            "soundfont": os.path.basename(SOUNDFONT),
            "sample_rate": sr,
            "seed": args.seed,
            "chords": chords,
        }
        with open(os.path.join(args.out, f"chords_prog{program}.json"), "w") as f:
            json.dump(truth, f, indent=2)
        print(f"program {program} ({truth['program_name']}): {wav_path}, {len(audio) / sr:.1f}s, "
              f"peak {np.abs(audio).max():.2f}, {len(chords)} chords")

        # Repeated-note guard set (separate file, so the chord set is unchanged).
        repeats = build_repeats(args.seed)
        wav_path = render_notes([(n["midi"], n["onset"], n["end"], n["velocity"]) for n in repeats],
                                program, os.path.join(args.out, f"repeats_prog{program}.wav"))
        audio, sr = sf.read(wav_path)
        with open(os.path.join(args.out, f"repeats_prog{program}.json"), "w") as f:
            json.dump({"wav": os.path.basename(wav_path), "program": program,
                       "program_name": truth["program_name"], "sample_rate": sr, "seed": args.seed,
                       "patterns": [p[0] for p in REPEAT_PATTERNS], "notes": repeats}, f, indent=2)
        print(f"  repeats: {wav_path}, {len(audio) / sr:.1f}s, peak {np.abs(audio).max():.2f}, "
              f"{len(repeats)} notes in {len(REPEAT_PATTERNS)} patterns")


if __name__ == "__main__":
    main()
