"""Runs beat trackers on the EGSet12 benchmark audio (all three tones) and
saves beat and downbeat times for scripts/rhythm_benchmark.py to score.
Runs in the rhythm-bench image:
    python /rhythm/track.py

Trackers (data/_rhythm/<tracker>/<tone>_<NN>.json):
  librosa       librosa.beat.beat_track on the audio as transcribe loads it
                (22.05kHz mono) - what the app runs today, but keeping the
                beat times instead of only the BPM. librosa has no downbeat
                tracker, so downbeats are a heuristic: assume 4/4 and take the
                phase (every 4th beat) with the highest mean onset strength.
  beat_this     beat_this final0 (CPJKU, ISMIR 2024), beats and downbeats,
                no DBN postprocessing (that needs madmom, whose models are
                non-commercial).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.egset12_benchmark import PERFORMANCES, TONES, audio_path  # noqa: E402

OUT = "/app/data/_rhythm"
SR = 22050


def track_librosa(path: str) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beats = librosa.frames_to_time(frames, sr=sr)
    strength = onset_env[np.minimum(frames, len(onset_env) - 1)]
    phase = max(range(4), key=lambda k: strength[k::4].mean() if len(strength[k::4]) else -1) if len(beats) else 0
    return {"beats": beats.tolist(), "downbeats": beats[phase::4].tolist(),
            "tempo_bpm": float(np.atleast_1d(tempo)[0])}  # the number the app shows today


_beat_this = {}


def track_beat_this(path: str) -> dict:
    from beat_this.inference import File2Beats

    if not _beat_this:
        _beat_this["f2b"] = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = _beat_this["f2b"](path)
    return {"beats": [float(b) for b in beats], "downbeats": [float(d) for d in downbeats]}


TRACKERS = {"librosa": track_librosa, "beat_this": track_beat_this}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    for name, fn in TRACKERS.items():
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        fn(audio_path(PERFORMANCES[0], "clean"))  # warm-up (model load, JIT)
        for tone in TONES:
            for p in PERFORMANCES:
                t0 = time.perf_counter()
                out = fn(audio_path(p, tone))
                seconds = time.perf_counter() - t0
                out.update(seconds=round(seconds, 3), audio_seconds=round(librosa.get_duration(path=audio_path(p, tone)), 2))
                json.dump(out, open(os.path.join(OUT, name, f"{tone}_{p}.json"), "w"))
                print(f"{name:<10} {tone:<9} {p}: {len(out['beats'])} beats, {len(out['downbeats'])} downbeats, "
                      f"{seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
