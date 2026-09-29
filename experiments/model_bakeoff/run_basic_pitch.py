"""Cuts the bake-off clips and runs the Basic Pitch baseline on them.

Runs in the worker image (python run_basic_pitch.py cut | <segment id>), one
process per segment so load time and peak RSS are per run. Writes two notes
files per segment:
  basic-pitch-raw   every Basic Pitch event at the pipeline's note-creation
                    settings, before any confidence filter (comparable to the
                    candidates' raw model output)
  basic-pitch-tab   the pipeline's final tab notes (select_notes + fretboard
                    mapping), i.e. what the product ships today
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")  # worker code (tasks/, scripts/)
sys.path.insert(0, os.path.dirname(__file__))

import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402

from common import CLIPS, SEGMENTS, clip_path, write_notes  # noqa: E402

EGSET = "/app/data/egset12"


def cut() -> None:
    bench = json.load(open(os.path.join(EGSET, "benchmark.json")))
    bounds = {s["id"]: (s["start"], s["end"]) for p in bench["performances"] for s in p["segments"]}
    os.makedirs(CLIPS, exist_ok=True)
    for kind, seg in SEGMENTS.items():
        start, end = bounds[seg]
        audio, sr = sf.read(os.path.join(EGSET, f"{seg[:2]}.wav"), dtype="float32")
        sf.write(clip_path(seg), audio[int(start * sr):int(end * sr)], sr, subtype="PCM_16")
        print(f"{kind:<12} {seg}  {start:.3f}-{end:.3f}s  sr {sr}  ch {audio.ndim}")


def run(seg: str) -> None:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    from scripts.chord_stages import PIPELINE, detect_file, run_stages

    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    t0 = time.perf_counter()
    Model(ICASSP_2022_MODEL_PATH)  # load cost alone; run_inference loads it again per call
    load = time.perf_counter() - t0

    # Warm-up pass: librosa's numba JIT and TF graph setup are one-time
    # per-container costs; time the steady-state second pass.
    with tempfile.TemporaryDirectory() as workdir:
        detect_file(clip_path(seg), workdir, use_cache=False)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        events, activations, _ = detect_file(clip_path(seg), workdir, use_cache=False)
    stages = run_stages(events, PIPELINE, activations)
    infer = time.perf_counter() - t0
    duration = sf.info(clip_path(seg)).duration

    raw = [{"onset": e["start_time"], "end": e["end_time"], "midi": e["midi"]} for e in events]
    tab = [{"onset": k[0], "end": None, "midi": pretty_midi.note_name_to_number(k[1])}
           for k in stages["mapping"]["positions"]]
    # infer includes one more model load inside run_inference (how the pipeline runs it).
    write_notes("basic-pitch-raw", seg, duration, load, infer, raw)
    write_notes("basic-pitch-tab", seg, duration, load, infer, tab)
    print(f"{seg}: {len(raw)} raw events, {len(tab)} tab notes, {infer:.1f}s for {duration:.1f}s audio")


if __name__ == "__main__":
    cut() if sys.argv[1] == "cut" else run(sys.argv[1])
