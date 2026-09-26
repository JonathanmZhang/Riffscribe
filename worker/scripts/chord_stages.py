"""Pipeline-faithful stage functions shared by scripts/inspect_chords.py and
scripts/eval_chords.py. Every stage calls the real pipeline code
(ensure_decodable_audio / normalize_to_wav / separate_guitar_stem /
detect_note_events / _group_into_steps / map_notes_to_positions) - nothing is
re-implemented - with no Celery or Redis. Measurement only: nothing here
changes pipeline behavior.

Loss categories used when checking a chord's expected notes:
  (a) never detected by Basic Pitch at any amplitude
  (b) detected, but every matching event is below the confidence threshold
  (c) kept, but grouped into a different step than the rest of its chord, or
      the chord's step also absorbed notes from a neighbouring chord
  (d) kept and grouped, but dropped by the fretboard mapper
  (e) extra notes that aren't in the chord (octave errors of a chord note,
      repeats of a chord note, or other phantoms)
"""

import logging
import os
import random
import time

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from basic_pitch.note_creation import model_frames_to_time

from tasks.audio_io import ensure_decodable_audio, normalize_to_wav
from tasks.fretboard import (
    CHORD_ONSET_TOLERANCE_SECONDS,
    MAX_FRET,
    _candidates_for_pitch,
    _group_into_steps,
    map_notes_to_positions,
)
from tasks.separate import separate_guitar_stem
from tasks.transcribe import CONFIDENCE_THRESHOLD, detect_note_events

# Basic Pitch's pitch axis starts at A0 (MIDI 21), 88 bins.
BASIC_PITCH_MIDI_OFFSET = 21
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------- stage 0: audio


def prepare_audio(input_path: str, workdir: str, separate: bool = False, seed: int = 0) -> dict:
    """Decodes the input the way ingest_audio does (ffmpeg extraction for
    video/m4a), optionally separates the guitar the way separate_guitar does,
    and normalizes to the 22.05kHz mono WAV Basic Pitch is fed. Separation is
    seeded so repeated runs are identical (Demucs' shifts are random)."""
    source = ensure_decodable_audio(input_path, workdir)
    info = {"separated": separate, "separation_seconds": None}
    if separate:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        start = time.perf_counter()
        stem, sample_rate = separate_guitar_stem(source)
        info["separation_seconds"] = round(time.perf_counter() - start, 2)
        source = os.path.join(workdir, "guitar_stem.wav")
        sf.write(source, stem.T, sample_rate, subtype="PCM_16")
    info["normalized_path"] = normalize_to_wav(source, os.path.join(workdir, "normalized.wav"))
    return info


# ------------------------------------------------------------- stage 1: detect


class Activations:
    """Basic Pitch's per-frame note activations, for asking "did the model
    hear this pitch at all?" about notes it never emitted as events."""

    def __init__(self, model_output: dict):
        self.note = np.asarray(model_output["note"])  # (frames, 88)
        self.times = np.asarray(model_frames_to_time(self.note.shape[0]))

    def peak(self, midi: int, start: float, end: float) -> float:
        index = midi - BASIC_PITCH_MIDI_OFFSET
        if not 0 <= index < self.note.shape[1]:
            return 0.0
        frames = (self.times >= start) & (self.times < end)
        return float(self.note[frames, index].max()) if frames.any() else 0.0


def detect(normalized_path: str) -> tuple[list[dict], Activations]:
    """Every Basic Pitch event, each flagged kept/dropped by the same
    confidence threshold transcribe uses."""
    events, model_output = detect_note_events(normalized_path)
    for event in events:
        event["kept"] = event["amplitude"] >= CONFIDENCE_THRESHOLD
    events.sort(key=lambda e: (e["start_time"], e["midi"]))
    return events, Activations(model_output)


# ---------------------------------------------------- stage 2 + 3: group, map


def group(kept: list[dict]) -> list[list[dict]]:
    """Exactly the pipeline's chord grouping (onset tolerance included)."""
    return _group_into_steps(kept)


def note_key(note: dict) -> tuple:
    return (note["start_time"], note["pitch"])


def map_with_trace(kept: list[dict]) -> dict:
    """Runs the real map_notes_to_positions on every kept note, then works
    out which notes it dropped and why. Returns positions (by note key), drop
    reasons (by note key) and the mapper's own warning messages."""
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    fretboard_logger = logging.getLogger("tasks.fretboard")
    handler = _Capture(level=logging.WARNING)
    previous_propagate = fretboard_logger.propagate
    fretboard_logger.addHandler(handler)
    fretboard_logger.propagate = False
    try:
        mapped = map_notes_to_positions(kept)
    finally:
        fretboard_logger.removeHandler(handler)
        fretboard_logger.propagate = previous_propagate

    positions = {note_key(m): (m["string"], m["fret"]) for m in mapped}
    drops = {}
    for note in kept:
        if note_key(note) in positions:
            continue
        if not _candidates_for_pitch(pretty_midi.note_name_to_number(note["pitch"])):
            drops[note_key(note)] = f"outside the playable range (frets 0-{MAX_FRET}, standard tuning)"
        else:
            drops[note_key(note)] = (
                "chord conflict: no voicing on distinct strings with the rest of its step; "
                "dropped as part of the lowest-amplitude conflicting set"
            )
    return {"positions": positions, "drops": drops, "messages": messages}


def onset_spread_ms(step: list[dict]) -> float:
    return (max(n["start_time"] for n in step) - min(n["start_time"] for n in step)) * 1000.0


# --------------------------------------------------------- per-chord analysis


def _matches(event: dict, expected_midi: int | None, expected_pc: int | None) -> bool:
    return event["midi"] == expected_midi if expected_midi is not None else event["midi"] % 12 == expected_pc


def classify_chord(
    expected: list[int],
    pitch_class_mode: bool,
    window: tuple[float, float],
    events: list[dict],
    step_of: dict,
    mapping: dict,
    activations: Activations,
    owner=None,
) -> dict:
    """Follows each expected note of one chord through the stages.

    expected: MIDI numbers, or pitch classes (0-11) when pitch_class_mode
      (for real recordings where the voicing/octave isn't known).
    window: (start, end) - events with onsets in it are attributed to this
      chord.
    step_of: note key -> index of its group from group().
    owner: optional callable(note) -> "this" | "other" | None, used to detect
      a step that merged this chord with a neighbouring chord ("other"); None
      means the note belongs to no chord. By default any kept note outside
      the window counts as a neighbour's.
    """
    start, end = window
    in_window = [e for e in events if start <= e["start_time"] < end]
    kept_in_window = [e for e in in_window if e["kept"]]
    owner = owner or (lambda n: "this" if start <= n["start_time"] < end else "other")

    notes = []
    chosen_keys = set()
    for target in expected:
        midi = None if pitch_class_mode else target
        pc = target if pitch_class_mode else None
        label = PITCH_CLASS_NAMES[pc] if pitch_class_mode else pretty_midi.note_number_to_name(midi)
        matching = [e for e in in_window if _matches(e, midi, pc)]
        kept = [e for e in matching if e["kept"]]
        entry = {"expected": label, "stage": None}
        if not matching:
            if pitch_class_mode:
                peak = max(activations.peak(m, start, end) for m in range(40, 89) if m % 12 == pc)
            else:
                peak = activations.peak(midi, start, end)
            entry.update(stage="a", peak_activation=round(peak, 3))
        elif not kept:
            entry.update(stage="b", amplitudes=[round(e["amplitude"], 3) for e in matching])
        else:
            # The earliest kept event is the one at the strum; later events of
            # the same pitch (re-triggers during the hold, common with
            # distortion) are repeats, reported under (e).
            best = min(kept, key=lambda e: (e["start_time"], -e["amplitude"]))
            chosen_keys.add(note_key(best))
            entry.update(event=best, amplitude=round(best["amplitude"], 3))
        notes.append(entry)

    # Grouping: the chord's step is the one holding most of its kept notes.
    kept_entries = [n for n in notes if "event" in n]
    step_counts: dict[int, int] = {}
    for n in kept_entries:
        s = step_of[note_key(n["event"])]
        step_counts[s] = step_counts.get(s, 0) + 1
    main_step = min(step_counts, key=lambda s: (-step_counts[s], s)) if step_counts else None
    merged_with_neighbour = main_step is not None and any(
        e["kept"] and owner(e) == "other" and step_of.get(note_key(e)) == main_step for e in events
    )

    for n in kept_entries:
        key = note_key(n["event"])
        n["step"] = step_of[key]
        # In the tab at all (possibly in the wrong column), for precision.
        n["in_tab"] = key in mapping["positions"]
        if n["step"] != main_step:
            n["stage"] = "c"
            n["why"] = f"split into step {n['step']} (chord is in step {main_step})"
        elif merged_with_neighbour:
            n["stage"] = "c"
            n["why"] = f"chord's step {main_step} also contains a neighbouring chord's notes"
        elif key in mapping["drops"]:
            n["stage"] = "d"
            n["why"] = mapping["drops"][key]
        else:
            n["stage"] = "ok"
            n["position"] = mapping["positions"][key]

    # Extras: kept notes attributed to this chord that aren't the chosen
    # event for an expected note.
    extras = []
    expected_pcs = {t % 12 for t in expected} if not pitch_class_mode else set(expected)
    for e in kept_in_window:
        if note_key(e) in chosen_keys:
            continue
        if pitch_class_mode:
            kind = "chord tone (other octave or repeat)" if e["midi"] % 12 in expected_pcs else "phantom"
        elif e["midi"] in expected:
            kind = "repeat of a chord note"
        elif e["midi"] % 12 in expected_pcs:
            kind = "octave error"
        else:
            kind = "phantom"
        extras.append(
            {
                "pitch": e["pitch"],
                "midi": e["midi"],
                "amplitude": round(e["amplitude"], 3),
                "kind": kind,
                "mapped": note_key(e) in mapping["positions"],
            }
        )

    for n in notes:
        if "event" in n:
            ev = n.pop("event")
            n["onset"] = round(ev["start_time"], 3)
    ok = sum(1 for n in notes if n["stage"] == "ok")
    return {
        "notes": notes,
        "extras": extras,
        "main_step": main_step,
        "merged_with_neighbour": merged_with_neighbour,
        "funnel": {
            "expected": len(expected),
            "detected": sum(1 for n in notes if n["stage"] != "a"),
            "kept": sum(1 for n in notes if n["stage"] not in ("a", "b")),
            "grouped": sum(1 for n in notes if n["stage"] in ("d", "ok")),
            "mapped": ok,
        },
        "complete": ok == len(expected),
        "clean": ok == len(expected) and not extras,
    }


def step_index(groups: list[list[dict]]) -> dict:
    return {note_key(n): i for i, g in enumerate(groups) for n in g}


__all__ = [
    "CHORD_ONSET_TOLERANCE_SECONDS",
    "CONFIDENCE_THRESHOLD",
    "Activations",
    "classify_chord",
    "detect",
    "group",
    "map_with_trace",
    "note_key",
    "onset_spread_ms",
    "prepare_audio",
    "step_index",
]
