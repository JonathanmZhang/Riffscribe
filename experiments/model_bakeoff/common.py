"""Shared bits for the model bake-off (feasibility only, not pipeline code).

Every candidate runner reads the same clean-tone EGSet12 segment clips and
writes the same notes JSON:
    {"model": ..., "segment": ..., "audio_seconds": ..., "load_seconds": ...,
     "infer_seconds": ..., "peak_rss_mb": ..., "notes": [{"onset", "end", "midi"}]}
Note times are relative to the clip start; score.py shifts them back.

Runners are pure numpy/soundfile here so they work in any container.
"""

import json
import os
import resource

DATA = os.environ.get("BAKEOFF_DATA", "/app/data")
CLIPS = os.path.join(DATA, "_bakeoff", "clips")
OUT = os.path.join(DATA, "_bakeoff", "notes")

# One segment per type, clean tone (ids/bounds from data/egset12/benchmark.json).
SEGMENTS = {
    "chords": "02-2",       # 106 notes, chord share 0.96
    "single-note": "05-2",  # 42 notes, no chords
    "fast": "04-2",         # 52 notes, 9 fast single-note gaps
}


def clip_path(segment_id: str) -> str:
    return os.path.join(CLIPS, f"{segment_id}.wav")


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (Linux: ru_maxrss in KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def write_notes(model: str, segment_id: str, audio_seconds: float, load_seconds: float,
                infer_seconds: float, notes: list[dict], **extra) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{model}__{segment_id}.json")
    with open(path, "w") as f:
        json.dump({"model": model, "segment": segment_id, "audio_seconds": round(audio_seconds, 3),
                   "load_seconds": round(load_seconds, 2), "infer_seconds": round(infer_seconds, 2),
                   "peak_rss_mb": peak_rss_mb(), **extra,
                   "notes": sorted(notes, key=lambda n: (n["onset"], n["midi"]))}, f, indent=1)
    return path
