"""Standalone exploration script for Spotify's basic-pitch inference.

Not part of the Celery pipeline. Run manually to see exactly what
basic_pitch.inference.predict() returns before wiring it into transcribe().

Usage (inside the worker container):
    python scripts/test_basic_pitch.py /app/data/<job_id>/normalized.wav
"""

import sys

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"Running basic-pitch inference on: {wav_path}\n")

    model_output, midi_data, note_events = predict(wav_path, ICASSP_2022_MODEL_PATH)

    print("=== model_output (raw per-frame model arrays) ===")
    print(f"type: {type(model_output)}")
    for key, value in model_output.items():
        shape = getattr(value, "shape", "n/a")
        dtype = getattr(value, "dtype", "n/a")
        print(f"  '{key}': type={type(value).__name__}, shape={shape}, dtype={dtype}")

    print("\n=== midi_data (pretty_midi.PrettyMIDI) ===")
    print(f"type: {type(midi_data)}")
    print(f"instrument count: {len(midi_data.instruments)}")
    for i, instrument in enumerate(midi_data.instruments):
        print(
            f"  instrument[{i}]: program={instrument.program}, "
            f"is_drum={instrument.is_drum}, note_count={len(instrument.notes)}"
        )
    try:
        print(f"estimated tempo: {midi_data.estimate_tempo():.2f} bpm")
    except Exception as exc:
        print(f"estimated tempo: could not estimate ({exc})")

    print(f"\n=== note_events ({len(note_events)} detected) ===")
    print("each entry: (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends)")
    if not note_events:
        print("  (no notes detected)")
    for i, (start_time, end_time, pitch_midi, amplitude, pitch_bends) in enumerate(note_events):
        bend_info = f"{len(pitch_bends)} frames" if pitch_bends is not None else "None"
        print(
            f"  [{i}] start={start_time:.3f}s end={end_time:.3f}s "
            f"pitch_midi={pitch_midi} amplitude={amplitude:.3f} pitch_bends={bend_info}"
        )


if __name__ == "__main__":
    main()
