"""Firefire (102s, full mix) prep and Basic Pitch per-song timing. Runs in the
worker image: python /bakeoff/prep_firefire.py

Writes data/_bakeoff/firefire/{mix,stem}.wav for MT3:
  mix   the pipeline's decode (ensure_decodable_audio: ffmpeg, original rate)
  stem  Demucs guitar stem of the mix via separate_guitar_stem, seeded like
        scripts/regression_check.py (detect_file(separate=True, seed=0)), so
        it's the stem Basic Pitch's firefire numbers were measured on
and data/_bakeoff/firefire/timing.json with steady-state (warmed) times for
Basic Pitch's detection on mix and stem and for the separation itself.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from scripts.chord_stages import PIPELINE, run_stages  # noqa: E402
from scripts.regression_check import FIREFIRE  # noqa: E402
from tasks.audio_io import ensure_decodable_audio, normalize_to_wav  # noqa: E402
from tasks.separate import separate_guitar_stem  # noqa: E402
from tasks.transcribe import detect_note_events  # noqa: E402

OUT = "/app/data/_bakeoff/firefire"


def bp_seconds(path: str, workdir: str) -> float:
    """Pipeline-equivalent Basic Pitch time for one file: normalize, detect,
    select + map (run_stages)."""
    t0 = time.perf_counter()
    normalized = normalize_to_wav(path, os.path.join(workdir, "normalized.wav"))
    events, model_output = detect_note_events(normalized)
    run_stages(events, PIPELINE)
    return time.perf_counter() - t0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    timing = {}
    with tempfile.TemporaryDirectory() as workdir:
        source = ensure_decodable_audio(FIREFIRE, workdir)
        shutil.copy(source, os.path.join(OUT, "mix.wav"))
        timing["audio_seconds"] = round(sf.info(source).duration, 2)

        # Separation, seeded as in chord_stages.detect_file.
        for attempt in ("warmup", "timed"):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            t0 = time.perf_counter()
            stem, sr = separate_guitar_stem(source)
            timing[f"separation_seconds_{attempt}"] = round(time.perf_counter() - t0, 1)
            print(f"separation ({attempt}): {timing[f'separation_seconds_{attempt}']}s", flush=True)
        sf.write(os.path.join(OUT, "stem.wav"), stem.T, sr, subtype="PCM_16")

        bp_seconds(source, workdir)  # warm-up (TF model, numba JIT)
        timing["basic_pitch_mix_seconds"] = round(bp_seconds(source, workdir), 1)
        timing["basic_pitch_stem_seconds"] = round(bp_seconds(os.path.join(OUT, "stem.wav"), workdir), 1)
    timing["separation_threads"] = torch.get_num_threads()
    json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
    print(timing)


if __name__ == "__main__":
    main()
