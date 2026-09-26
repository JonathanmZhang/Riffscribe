"""A/B comparison: transcription of the full mix vs. of the Demucs-separated
guitar stem, plus separation timing.

Runs directly in the worker container, no Celery or Redis. Both runs use the
exact pipeline code: normalize_to_wav -> extract_notes (Basic Pitch +
confidence filter) -> map_notes_to_positions; run (b) first separates the
guitar with separate_guitar_stem, as the separate_guitar task does.

Usage (inside the worker container):
    python -m scripts.ab_separation <audio file> [--out results.json] [--seed N]

The separated stem is saved as <input name>_guitar_stem.wav next to --out
(or next to the input if --out isn't given) so it can be listened to.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from tasks.audio_io import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    ensure_decodable_audio,
    normalize_to_wav,
    probe_duration_seconds,
)
from tasks.fretboard import map_notes_to_positions  # noqa: E402
from tasks.separate import DEMUCS_MODEL, DEMUCS_SHIFTS, _load_model, separate_guitar_stem  # noqa: E402
from tasks.transcribe import CONFIDENCE_THRESHOLD, extract_notes  # noqa: E402

# Lowest note of a standard-tuned guitar. Detected notes below it are either
# bleed/phantoms (bass, kick, subharmonics) or genuinely lower-tuned guitar.
E2_MIDI = 40
# If the unseparated mix already has this share of sub-E2 notes, the metric
# can't separate "bleed" from "the song is tuned below E2" (e.g. drop D), so
# it's flagged as unreliable for judging separation.
SUB_E2_UNRELIABLE_SHARE = 0.05


def _metrics(raw_count: int, kept: list[dict], mapped: list[dict], duration: float, seconds: float) -> dict:
    midi = [pretty_midi.note_name_to_number(n["pitch"]) for n in kept]
    return {
        "raw_notes": raw_count,
        "kept_notes": len(kept),
        "mapped_notes": len(mapped),
        "dropped_unplayable": len(kept) - len(mapped),
        "notes_per_second": round(len(kept) / duration, 2) if duration else 0.0,
        "pitch_range": (
            [pretty_midi.note_number_to_name(min(midi)), pretty_midi.note_number_to_name(max(midi))] if midi else None
        ),
        "below_e2": sum(1 for m in midi if m < E2_MIDI),
        "total_seconds": round(seconds, 2),
    }


def _transcribe_and_map(normalized_path: str) -> tuple[int, list[dict], list[dict]]:
    raw_count, kept = extract_notes(normalized_path)
    return raw_count, kept, map_notes_to_positions(kept)


def _warm_up(workdir: str) -> float:
    """One-time Basic Pitch model load + librosa JIT, so neither run's timing
    includes it (the pipeline pays it once per worker process)."""
    start = time.perf_counter()
    # A quiet 1s tone rather than silence: Basic Pitch warns (divide by zero)
    # on all-zero input.
    t = np.arange(TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE
    tone = os.path.join(workdir, "warmup.wav")
    sf.write(tone, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")
    extract_notes(normalize_to_wav(tone, os.path.join(workdir, "warmup_normalized.wav")))
    return time.perf_counter() - start


def _seed_everything(seed: int) -> None:
    """Demucs' shifts pick random time offsets (Python's random), so without
    this the stem - and every downstream count - varies slightly between
    runs of the same file. numpy and torch are seeded too for completeness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(input_path: str, out_path: str | None, seed: int = 0) -> dict:
    duration = probe_duration_seconds(input_path)
    stem_dir = os.path.dirname(os.path.abspath(out_path or input_path))
    stem_path = os.path.join(stem_dir, os.path.splitext(os.path.basename(input_path))[0] + "_guitar_stem.wav")

    with tempfile.TemporaryDirectory() as workdir:
        # Same decode path as ingest_audio (ffmpeg extraction for video/m4a).
        decodable = ensure_decodable_audio(input_path, workdir)
        warmup_seconds = _warm_up(workdir)
        load_start = time.perf_counter()
        _load_model(DEMUCS_MODEL)
        model_load_seconds = time.perf_counter() - load_start

        # (a) full mix: exactly the pipeline without isolate_guitar.
        start = time.perf_counter()
        normalized = normalize_to_wav(decodable, os.path.join(workdir, "mix_normalized.wav"))
        a = _metrics(*_transcribe_and_map(normalized), duration, time.perf_counter() - start)

        # (b) separated: the pipeline with isolate_guitar.
        _seed_everything(seed)
        start = time.perf_counter()
        stem, sample_rate = separate_guitar_stem(decodable)
        separation_seconds = time.perf_counter() - start
        sf.write(stem_path, stem.T, sample_rate, subtype="PCM_16")
        normalized = normalize_to_wav(stem_path, os.path.join(workdir, "stem_normalized.wav"))
        b = _metrics(*_transcribe_and_map(normalized), duration, time.perf_counter() - start)

    b["separation_seconds"] = round(separation_seconds, 2)
    b["separation_rtf"] = round(separation_seconds / duration, 2) if duration else None

    a_share = a["below_e2"] / a["kept_notes"] if a["kept_notes"] else 0.0
    return {
        "input": input_path,
        "duration_seconds": round(duration, 2),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "demucs_model": DEMUCS_MODEL,
        "demucs_shifts": DEMUCS_SHIFTS,
        "seed": seed,
        "warmup_seconds": round(warmup_seconds, 2),
        "model_load_seconds": round(model_load_seconds, 2),
        "stem_path": stem_path,
        "unseparated": a,
        "separated": b,
        "below_e2_reliable": a_share < SUB_E2_UNRELIABLE_SHARE,
        "unseparated_below_e2_share": round(a_share, 3),
    }


def _print_table(r: dict) -> None:
    a, b = r["unseparated"], r["separated"]

    def delta(key: str) -> str:
        d = b[key] - a[key]
        return f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"

    def pitch_range(m: dict) -> str:
        return f"{m['pitch_range'][0]}–{m['pitch_range'][1]}" if m["pitch_range"] else "–"

    rows = [
        ("raw notes detected", a["raw_notes"], b["raw_notes"], delta("raw_notes")),
        (f"kept (conf >= {r['confidence_threshold']})", a["kept_notes"], b["kept_notes"], delta("kept_notes")),
        ("mapped to fretboard", a["mapped_notes"], b["mapped_notes"], delta("mapped_notes")),
        ("dropped as unplayable", a["dropped_unplayable"], b["dropped_unplayable"], delta("dropped_unplayable")),
        ("kept notes / second", a["notes_per_second"], b["notes_per_second"], delta("notes_per_second")),
        ("pitch range (kept)", pitch_range(a), pitch_range(b), ""),
        ("notes below E2" + ("" if r["below_e2_reliable"] else " (!)"), a["below_e2"], b["below_e2"], delta("below_e2")),
        ("total time (s)", a["total_seconds"], b["total_seconds"], delta("total_seconds")),
    ]
    widths = [max(len(str(row[i])) for row in rows + [("", "full mix", "separated guitar", "delta")]) for i in range(4)]

    print(
        f"\nA/B: {r['input']}  ({r['duration_seconds']}s, {r['demucs_model']}, "
        f"shifts={r['demucs_shifts']}, seed={r['seed']})"
    )
    header = ("", "full mix", "separated guitar", "delta")
    for row in [header, tuple("-" * w for w in widths)] + rows:
        print(f"  {str(row[0]):<{widths[0]}}  {str(row[1]):>{widths[1]}}  {str(row[2]):>{widths[2]}}  {str(row[3]):>{widths[3]}}")

    print(
        f"\n  separation: {b['separation_seconds']}s for {r['duration_seconds']}s of audio "
        f"(real-time factor {b['separation_rtf']}; wall time / audio duration, lower is faster)"
    )
    print(f"  one-time costs excluded from the totals: Basic Pitch/librosa warm-up {r['warmup_seconds']}s, "
          f"Demucs model load {r['model_load_seconds']}s")
    if not r["below_e2_reliable"]:
        print(
            f"\n  (!) 'notes below E2' is NOT a reliable bleed signal for this clip: the unseparated mix already "
            f"has {a['below_e2']} sub-E2 notes ({r['unseparated_below_e2_share']:.0%} of kept notes, threshold "
            f"{SUB_E2_UNRELIABLE_SHARE:.0%}), which points to a tuning below E2 (e.g. drop D) or bass-register "
            "content, not separation bleed."
        )
    print(f"\n  stem saved to {r['stem_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio_file")
    parser.add_argument("--out", help="write results as JSON here; the stem WAV is saved next to it")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for Python random, numpy and torch before separation, so repeat runs match (default 0)",
    )
    args = parser.parse_args()

    # map_notes_to_positions logs a warning per dropped note; the counts are
    # in the table, so keep the output readable.
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)

    results = run(args.audio_file, args.out, args.seed)
    _print_table(results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  results written to {args.out}")


if __name__ == "__main__":
    main()
