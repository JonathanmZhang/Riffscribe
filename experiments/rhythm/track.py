"""Runs beat trackers on the EGSet12 benchmark audio (all three tones) and
saves beat and downbeat times for scripts/rhythm_benchmark.py to score.
Runs in the worker image (beat_this is part of it since the pipeline uses
it):
    python /rhythm/track.py

Both trackers get the audio the pipeline gives them: the recording
normalized to 22.05kHz mono (tasks/audio_io.normalize_to_wav).

Trackers (data/_rhythm/<tracker>/<tone>_<NN>.json):
  librosa       librosa.beat.beat_track - what the app ran before beat_this,
                but keeping the beat times instead of only the BPM. librosa
                has no downbeat tracker, so downbeats are a heuristic: assume
                4/4 and take the phase (every 4th beat) with the highest mean
                onset strength.
  beat_this     tasks/beats.track_beats, as the pipeline runs it.
"""

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "/app")

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.egset12_benchmark import PERFORMANCES, TONES, audio_path  # noqa: E402
from tasks.audio_io import normalize_to_wav  # noqa: E402
from tasks.beats import track_beats  # noqa: E402

OUT = "/app/data/_rhythm"


def track_librosa(path: str) -> dict:
    y, sr = librosa.load(path, sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beats = librosa.frames_to_time(frames, sr=sr)
    strength = onset_env[np.minimum(frames, len(onset_env) - 1)]
    phase = max(range(4), key=lambda k: strength[k::4].mean() if len(strength[k::4]) else -1) if len(beats) else 0
    return {"beats": beats.tolist(), "downbeats": beats[phase::4].tolist(),
            "tempo_bpm": float(np.atleast_1d(tempo)[0])}  # the number the app showed before beat_this


def track_beat_this(path: str) -> dict:
    beats, downbeats = track_beats(path)
    return {"beats": beats, "downbeats": downbeats}


TRACKERS = {"librosa": track_librosa, "beat_this": track_beat_this}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    with tempfile.TemporaryDirectory() as workdir:
        normalized = {}
        for tone in TONES:
            for p in PERFORMANCES:
                normalized[tone, p] = normalize_to_wav(audio_path(p, tone), os.path.join(workdir, f"{tone}_{p}.wav"))
        for name, fn in TRACKERS.items():
            os.makedirs(os.path.join(OUT, name), exist_ok=True)
            fn(normalized["clean", PERFORMANCES[0]])  # warm-up (model load, JIT)
            for (tone, p), path in normalized.items():
                t0 = time.perf_counter()
                out = fn(path)
                seconds = time.perf_counter() - t0
                out.update(seconds=round(seconds, 3), audio_seconds=round(librosa.get_duration(path=path), 2))
                json.dump(out, open(os.path.join(OUT, name, f"{tone}_{p}.json"), "w"))
                print(f"{name:<10} {tone:<9} {p}: {len(out['beats'])} beats, {len(out['downbeats'])} downbeats, "
                      f"{seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
