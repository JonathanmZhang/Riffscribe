"""Pipeline-faithful stage functions shared by the chord measurement scripts
(inspect_chords, eval_chords, regression_check). Every stage calls the real
pipeline code (ensure_decodable_audio / normalize_to_wav /
separate_guitar_stem / detect_note_events / select_notes /
map_notes_with_steps) - nothing is re-implemented - with no Celery or Redis.

Post-detection behavior (note selection, grouping) is parameterised by a
StageConfig whose defaults are the pipeline's, so experiments can sweep
settings while the defaults always measure what the pipeline does.

Basic Pitch detection (and separation) results are cached per input file,
since nothing downstream of detection changes them; the cache key covers the
file contents, separation settings and library versions.

Loss categories used when checking a chord's expected notes:
  (a) never detected by Basic Pitch at any amplitude
  (b) detected, but every matching event was filtered out as low-confidence
  (c) kept, but grouped into a different step than the rest of its chord, or
      the chord's step also absorbed notes from a neighbouring chord
  (d) kept and grouped, but dropped by the fretboard mapper
  (e) extra notes that aren't in the chord (octave errors of a chord note,
      repeats of a chord note, or other phantoms)
"""

import dataclasses
import hashlib
import importlib.metadata
import logging
import os
import pickle
import random
import time

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from basic_pitch.note_creation import model_frames_to_time

from tasks import fretboard, transcribe
from tasks.audio_io import ensure_decodable_audio, normalize_to_wav
from tasks.fretboard import CHORD_ONSET_TOLERANCE_SECONDS, MAX_FRET, _candidates_for_pitch
from tasks.separate import DEMUCS_MODEL, DEMUCS_SHIFTS, separate_guitar_stem
from tasks.transcribe import CONFIDENCE_THRESHOLD, detect_note_events

# Basic Pitch's pitch axis starts at A0 (MIDI 21), 88 bins.
BASIC_PITCH_MIDI_OFFSET = 21
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CACHE_DIR = os.environ.get("STAGE_CACHE_DIR", "/app/data/_testaudio/.stage_cache")


@dataclasses.dataclass(frozen=True)
class StageConfig:
    """Post-detection settings. Defaults = what the pipeline does."""

    threshold: float = CONFIDENCE_THRESHOLD
    chord_floor: float = transcribe.CHORD_TONE_CONFIDENCE_FLOOR
    # Basic Pitch's own note-creation settings (transcribe.notes_from_model_output).
    bp_onset: float = transcribe.BASIC_PITCH_ONSET_THRESHOLD
    bp_frame: float = transcribe.BASIC_PITCH_FRAME_THRESHOLD
    bp_min_note_ms: float = transcribe.BASIC_PITCH_MIN_NOTE_LENGTH_MS
    # Technique merges after selection (transcribe.clean_notes), 0 or 1.
    vibrato_merge: int = int(transcribe.VIBRATO_MERGE)
    glide_merge: int = int(transcribe.GLIDE_MERGE)

    def describe(self) -> str:
        return ", ".join(f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self))


PIPELINE = StageConfig()


def add_config_args(parser) -> None:
    """Adds one --flag per StageConfig field (default: the pipeline's)."""
    for field in dataclasses.fields(StageConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default) if field.default is not None
                            else float, default=field.default,
                            help=f"post-detection setting (pipeline default {field.default})")


def config_from_args(args) -> StageConfig:
    return StageConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(StageConfig)})


# ------------------------------------------------------- stage 0 + 1: detect


class Activations:
    """Basic Pitch's raw model output for one file. Its per-frame note
    activations answer "did the model hear this pitch at all?" about notes it
    never emitted, and the full output lets run_stages derive note events for
    any Basic Pitch note-creation settings without re-running the network."""

    def __init__(self, model_output: dict):
        self.model_output = model_output
        self.note = np.asarray(model_output["note"])  # (frames, 88)
        self.times = np.asarray(model_frames_to_time(self.note.shape[0]))

    def peak(self, midi: int, start: float, end: float) -> float:
        index = midi - BASIC_PITCH_MIDI_OFFSET
        if not 0 <= index < self.note.shape[1]:
            return 0.0
        frames = (self.times >= start) & (self.times < end)
        return float(self.note[frames, index].max()) if frames.any() else 0.0


def _cache_key(input_path: str, separate: bool, seed: int) -> str:
    digest = hashlib.sha1()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    # "mo": the cache holds Basic Pitch's model output, not finished events.
    digest.update(f"|mo|sep={separate}|seed={seed}|bp={importlib.metadata.version('basic-pitch')}".encode())
    if separate:
        digest.update(f"|{DEMUCS_MODEL}|shifts={DEMUCS_SHIFTS}|torch={torch.__version__}".encode())
    return digest.hexdigest()[:20]


def detect_file(input_path: str, workdir: str, separate: bool = False, seed: int = 0,
                use_cache: bool = True) -> tuple[list[dict], Activations, dict]:
    """Decodes the input the way ingest_audio does (ffmpeg extraction for
    video/m4a), optionally separates the guitar the way separate_guitar does
    (seeded, since Demucs' shifts are random), normalizes to the 22.05kHz
    mono WAV Basic Pitch is fed, and returns (every note event at the
    pipeline's Basic Pitch settings, activations, info). The model output is
    cached per file + separation settings; pass the activations to
    run_stages to get events for other Basic Pitch settings."""
    cache_path = None
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, _cache_key(input_path, separate, seed) + ".pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                model_output, info = pickle.load(f)
            return transcribe.notes_from_model_output(model_output), Activations(model_output), {**info, "cached": True}

    source = ensure_decodable_audio(input_path, workdir)
    info = {"separated": separate, "separation_seconds": None, "seed": seed}
    if separate:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        start = time.perf_counter()
        stem, sample_rate = separate_guitar_stem(source)
        info["separation_seconds"] = round(time.perf_counter() - start, 2)
        source = os.path.join(workdir, "guitar_stem.wav")
        sf.write(source, stem.T, sample_rate, subtype="PCM_16")
    normalized = normalize_to_wav(source, os.path.join(workdir, "normalized.wav"))
    events, model_output = detect_note_events(normalized)
    model_output = {k: np.asarray(v, dtype=np.float32) for k, v in model_output.items()}

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump((model_output, info), f)
    return events, Activations(model_output), {**info, "cached": False}


# ------------------------------------------- stages 2-4: select, group, map


def note_key(note: dict) -> tuple:
    return (note["start_time"], note["pitch"])


def run_stages(events: list[dict], config: StageConfig = PIPELINE, activations: "Activations | None" = None) -> dict:
    """Runs the pipeline's post-detection stages on raw events: note
    selection (transcribe.select_notes), then grouping and fretboard
    mapping (fretboard.map_notes_with_steps). Returns the selected events
    (with "kept" flags), the kept notes, the groups the mapper used, each
    kept note's group index, positions and drop reasons.

    With activations (the cached model output), the events are first
    re-derived with the config's Basic Pitch settings (bp_*); otherwise the
    given events are used as they are."""
    if activations is not None:
        events = transcribe.notes_from_model_output(
            activations.model_output, onset_threshold=config.bp_onset, frame_threshold=config.bp_frame,
            min_note_length_ms=config.bp_min_note_ms,
        )
    selected = sorted(transcribe.select_notes(events, threshold=config.threshold, chord_floor=config.chord_floor),
                      key=lambda e: (e["start_time"], e["midi"]))
    kept = transcribe.clean_notes([e for e in selected if e["kept"]], bool(config.vibrato_merge),
                                  bool(config.glide_merge))
    # A note a merge absorbed is no longer in the tab: unkept in "events"
    # too, so the per-chord analysis counts it as lost after detection.
    remaining = {note_key(n) for n in kept}
    selected = [e if not e["kept"] or note_key(e) in remaining else {**e, "kept": False, "merged_away": True}
                for e in selected]

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
        mapped, groups = fretboard.map_notes_with_steps(kept)
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
    return {
        "events": selected,
        "kept": kept,
        "groups": groups,
        "step_of": {note_key(n): i for i, g in enumerate(groups) for n in g},
        "mapping": {"positions": positions, "drops": drops, "messages": messages},
    }


def onset_spread_ms(step: list[dict]) -> float:
    return (max(n["start_time"] for n in step) - min(n["start_time"] for n in step)) * 1000.0


# --------------------------------------------------------- per-chord analysis


def _matches(event: dict, expected_midi: int | None, expected_pc: int | None) -> bool:
    return event["midi"] == expected_midi if expected_midi is not None else event["midi"] % 12 == expected_pc


def classify_chord(
    expected: list[int],
    pitch_class_mode: bool,
    window: tuple[float, float],
    stages: dict,
    activations: Activations,
    owner=None,
) -> dict:
    """Follows each expected note of one chord through the stages.

    expected: MIDI numbers, or pitch classes (0-11) when pitch_class_mode
      (for real recordings where the voicing/octave isn't known).
    window: (start, end) - events with onsets in it are attributed to this
      chord.
    stages: the result of run_stages().
    owner: optional callable(note) -> "this" | "other" | None, used to detect
      a step that merged this chord with a neighbouring chord ("other"); None
      means the note belongs to no chord. By default any kept note outside
      the window counts as a neighbour's.
    """
    events, step_of, mapping = stages["events"], stages["step_of"], stages["mapping"]
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


def window_metrics(stages: dict, start: float, end: float) -> dict:
    """Ground-truth-free summary of what the stages do inside one window of
    a real recording: how many events are kept vs filtered, how the kept
    notes are grouped into columns, and what the mapper drops."""
    in_window = [e for e in stages["events"] if start <= e["start_time"] < end]
    kept = [e for e in in_window if e["kept"]]
    step_of, mapping = stages["step_of"], stages["mapping"]
    columns = sorted({step_of[note_key(e)] for e in kept})
    sizes = [sum(1 for e in kept if step_of[note_key(e)] == s) for s in columns]
    tab_sizes = [sum(1 for e in kept if step_of[note_key(e)] == s and note_key(e) in mapping["positions"])
                 for s in columns]
    chord_cols = [n for n in tab_sizes if n >= 3]
    # Same-pitch events where the next one starts while (or within 50ms
    # after) the previous is sounding: re-trigger candidates.
    retriggers = 0
    by_pitch: dict[int, list[dict]] = {}
    for e in kept:
        by_pitch.setdefault(e["midi"], []).append(e)
    for evs in by_pitch.values():
        evs.sort(key=lambda e: e["start_time"])
        retriggers += sum(1 for a, b in zip(evs, evs[1:]) if b["start_time"] <= a["end_time"] + 0.05)
    return {
        "events": len(in_window),
        "kept": len(kept),
        "filtered": len(in_window) - len(kept),
        "filtered_amp_ge_0_35": sum(1 for e in in_window if not e["kept"] and e["amplitude"] >= 0.35),
        "columns": len(columns),
        "chord_columns_ge3": len(chord_cols),
        "mean_notes_per_chord_column": round(sum(chord_cols) / len(chord_cols), 2) if chord_cols else 0.0,
        "tab_notes": sum(tab_sizes),
        "mapper_drops": sum(1 for e in kept if note_key(e) in mapping["drops"]),
        "retrigger_candidates": retriggers,
        "below_e2": sum(1 for e in kept if e["midi"] < 40),
    }
