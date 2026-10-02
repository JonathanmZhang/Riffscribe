"""Synthetic bend / vibrato test set with exact ground truth, because
EGSet12 has no bends to score against (gp_techniques.py). Single notes
rendered with FluidSynth + FluidR3_GM (MIT), pitch moved with the MIDI
pitch wheel. Measurement only. Runs in the worker image:

    python /x/make_bend_testset.py        # -> /app/data/_expression/bend_testset/

Per General MIDI program (clean 27, overdriven 29, distortion 30): 8 pitches
x 9 note types = 72 notes, each 1.2s long with 0.5s of silence after it.

  plain            no pitch movement (the control: a detector should stay quiet)
  bend_half        up 1 semitone over 150ms, starting 120ms after the onset, held
  bend_whole       up 2 semitones the same way
  bend_slow        up 2 semitones over 450ms
  bend_release     up 2 semitones, held, back down to the fretted pitch
  prebend_release  starts 2 semitones up, released to the fretted pitch
  vibrato_narrow   +/-25 cents at 5.5 Hz from 200ms after the onset
  vibrato_wide     +/-50 cents at 6 Hz
  bend_vibrato     up 2 semitones, then +/-30 cents of vibrato on the bent pitch

A wheel-bent sample is cleaner than a real bend (no change of timbre or
level, a perfectly smooth glide), so this is the easy case.
"""

import json
import math
import os
import sys

import pretty_midi
import soundfile as sf

sys.path.insert(0, "/app")
from scripts.make_chord_testset import PROGRAM_NAMES, _render_midi  # noqa: E402

OUT_DIR = "/app/data/_expression/bend_testset"
PROGRAMS = [27, 29, 30]
# Fretted pitches a bend is likely on: G string and above, frets 5-15.
PITCHES = [57, 60, 62, 64, 67, 69, 72, 76]  # A3 C4 D4 E4 G4 A4 C5 E5
NOTE_S, GAP_S, LEAD_IN_S = 1.2, 0.5, 1.0
WHEEL_RANGE_SEMITONES = 2.0  # FluidSynth's default pitch-wheel range
WHEEL_STEP_S = 0.005
TYPES = ["plain", "bend_half", "bend_whole", "bend_slow", "bend_release", "prebend_release",
         "vibrato_narrow", "vibrato_wide", "bend_vibrato"]


def ramp(t: float, start: float, length: float, amount: float) -> float:
    return amount * min(1.0, max(0.0, (t - start) / length))


def offset_semitones(kind: str, t: float) -> float:
    """Pitch offset from the fretted pitch, t seconds after the onset."""
    if kind == "plain":
        return 0.0
    if kind == "bend_half":
        return ramp(t, 0.12, 0.15, 1.0)
    if kind == "bend_whole":
        return ramp(t, 0.12, 0.15, 2.0)
    if kind == "bend_slow":
        return ramp(t, 0.12, 0.45, 2.0)
    if kind == "bend_release":
        return ramp(t, 0.12, 0.15, 2.0) - ramp(t, 0.65, 0.15, 2.0)
    if kind == "prebend_release":
        return 2.0 - ramp(t, 0.30, 0.15, 2.0)
    if kind == "vibrato_narrow":
        return 0.25 * math.sin(2 * math.pi * 5.5 * (t - 0.2)) if t >= 0.2 else 0.0
    if kind == "vibrato_wide":
        return 0.50 * math.sin(2 * math.pi * 6.0 * (t - 0.2)) if t >= 0.2 else 0.0
    if kind == "bend_vibrato":
        return ramp(t, 0.12, 0.15, 2.0) + (0.30 * math.sin(2 * math.pi * 5.5 * (t - 0.4)) if t >= 0.4 else 0.0)
    raise ValueError(kind)


TRUTH = {  # kind -> (is a bend, is vibrato, largest offset in semitones)
    "plain": (False, False, 0.0), "bend_half": (True, False, 1.0), "bend_whole": (True, False, 2.0),
    "bend_slow": (True, False, 2.0), "bend_release": (True, False, 2.0), "prebend_release": (True, False, 2.0),
    "vibrato_narrow": (False, True, 0.25), "vibrato_wide": (False, True, 0.5), "bend_vibrato": (True, True, 2.3),
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for program in PROGRAMS:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        guitar = pretty_midi.Instrument(program=program)
        notes, t = [], LEAD_IN_S
        # Types interleaved, so every pitch gets every type and neighbours differ.
        for i, pitch in enumerate(PITCHES):
            for kind in TYPES[i % len(TYPES):] + TYPES[:i % len(TYPES)]:
                guitar.notes.append(pretty_midi.Note(velocity=96, pitch=pitch, start=t, end=t + NOTE_S))
                steps = int(NOTE_S / WHEEL_STEP_S)
                for k in range(steps + 1):
                    value = offset_semitones(kind, k * WHEEL_STEP_S) / WHEEL_RANGE_SEMITONES
                    guitar.pitch_bends.append(pretty_midi.PitchBend(int(round(min(1.0, value) * 8191)), t + k * WHEEL_STEP_S))
                # Wheel back to centre during the silence, before the next onset.
                guitar.pitch_bends.append(pretty_midi.PitchBend(0, t + NOTE_S + GAP_S / 2))
                bend, vibrato, depth = TRUTH[kind]
                notes.append({"onset": round(t, 4), "end": round(t + NOTE_S, 4), "midi": pitch, "type": kind,
                              "bend": bend, "vibrato": vibrato, "max_offset_semitones": depth})
                t += NOTE_S + GAP_S
        midi.instruments.append(guitar)
        wav = _render_midi(midi, os.path.join(OUT_DIR, f"bends_prog{program}.wav"))
        audio, sr = sf.read(wav)
        with open(os.path.join(OUT_DIR, f"bends_prog{program}.json"), "w") as f:
            json.dump({"wav": os.path.basename(wav), "program": program, "program_name": PROGRAM_NAMES[program],
                       "notes": notes}, f, indent=1)
        print(f"program {program} ({PROGRAM_NAMES[program]}): {len(notes)} notes, {len(audio) / sr:.1f}s, "
              f"peak {abs(audio).max():.2f}")


if __name__ == "__main__":
    main()
