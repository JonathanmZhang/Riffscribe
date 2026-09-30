"""Chord names for the tab: BTC (Park et al., ISMIR 2019, "A Bi-Directional
Transformer for Musical Chord Recognition"; MIT code and weights), large
vocabulary model (12 roots x 14 qualities, plus no-chord). The repo is
cloned into the image at /opt/btc (worker/Dockerfile).

Pure: no Redis or Celery, so scripts can call recognize_chords directly.
Measured on EGSet12's chord columns (branch experiment/chord-recognition):
~68-73% of roots right across clean and processed-distortion tones,
~0.01s per second of audio on CPU.
"""

import logging
import os
import sys

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

BTC_DIR = "/opt/btc"
BTC_CHECKPOINT = os.path.join(BTC_DIR, "test", "btc_model_large_voca.pt")

# BTC's qualities -> the suffix guitarists write after the root.
QUALITY_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug", "min6": "m6", "maj6": "6", "min7": "m7",
    "minmaj7": "m(maj7)", "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5", "sus2": "sus2", "sus4": "sus4",
}

_model: dict = {}


def _load() -> dict:
    """Loads BTC once per process (a few MB; ~1s)."""
    if not _model:
        if BTC_DIR not in sys.path:
            sys.path.insert(0, BTC_DIR)
        from btc_model import BTC_model
        from utils.hparams import HParams
        from utils.mir_eval_modules import idx2voca_chord

        # BTC's HParams.load uses the pre-PyYAML-6 yaml.load(f) signature.
        with open(os.path.join(BTC_DIR, "run_config.yaml")) as f:
            config = HParams(**yaml.safe_load(f))
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170
        model = BTC_model(config=config.model)
        checkpoint = torch.load(BTC_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        _model.update(model=model, config=config, mean=checkpoint["mean"], std=checkpoint["std"],
                      labels=idx2voca_chord())
    return _model


def chord_display_name(label: str) -> str | None:
    """BTC label (e.g. 'A:min7', 'E', 'N') -> display name ('Am7', 'E'), or
    None for no chord / unknown ('N', 'X')."""
    if label in ("N", "X"):
        return None
    root, _, quality = label.partition(":")
    return root + QUALITY_SUFFIX[quality or "maj"]


def recognize_chords(audio_path: str) -> list[dict]:
    """Timed chord segments for audio_path: [{"start", "end", "name"}], times
    in seconds, no-chord stretches left out and consecutive segments with
    the same name merged. Inference is BTC's own (its test.py): CQT
    features in 10s blocks, frames of 10/108 s."""
    m = _load()  # also puts BTC on sys.path
    from utils.mir_eval_modules import audio_file_to_features

    config = m["config"]
    feature, seconds_per_frame, duration = audio_file_to_features(audio_path, config)
    feature = (feature.T - m["mean"]) / m["std"]
    block = config.model["timestep"]
    feature = np.pad(feature, ((0, block - feature.shape[0] % block), (0, 0)))
    predictions: list[int] = []
    with torch.no_grad():
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        for b in range(feature.shape[0] // block):
            hidden, _ = m["model"].self_attn_layers(x[:, block * b:block * (b + 1), :])
            prediction, _ = m["model"].output_layer(hidden)
            predictions += [int(p) for p in prediction.squeeze()]

    segments: list[dict] = []
    previous = None
    for i, p in enumerate(predictions):
        start = i * seconds_per_frame
        if start >= duration:
            break
        end = min(start + seconds_per_frame, duration)
        name = chord_display_name(m["labels"][p])
        if name is not None and name == previous:  # same chord as the frame just before
            segments[-1]["end"] = end
        elif name is not None:
            segments.append({"start": start, "end": end, "name": name})
        previous = name
    return [{"start": round(s["start"], 3), "end": round(s["end"], 3), "name": s["name"]} for s in segments]
