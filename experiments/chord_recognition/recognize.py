"""Runs the chord recognizers on a list of audio files and saves their
segments. Runs in the chords-bench image:
    python /chords/recognize.py <manifest.json>
Manifest: list of {"audio": path, "name": output name}. Writes
data/_chords/<model>/<name>.json = {"seconds": ..., "audio_seconds": ...,
"segments": [{"start", "end", "label"}]}. Existing outputs are skipped.

Models:
  chordino   NNLS-Chroma's Chordino Vamp plugin (GPL-2.0), default settings
  crema      crema 0.2.0 chord model (ISC)
  btc        BTC large-vocabulary model (170 classes; MIT)
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/app/data/_chords"
BTC_DIR = "/opt/btc"


def chordino(path: str) -> list[dict]:
    import vamp

    y, sr = librosa.load(path, sr=44100, mono=True)
    events = vamp.collect(y, sr, "nnls-chroma:chordino", output="simplechord")["list"]
    duration = len(y) / sr
    times = [float(e["timestamp"]) for e in events]
    return [{"start": t, "end": times[i + 1] if i + 1 < len(times) else duration, "label": e["label"]}
            for i, (t, e) in enumerate(zip(times, events))]


def crema_segments(path: str) -> list[dict]:
    from crema.analyze import analyze

    ann = analyze(filename=path).annotations["chord", 0]
    return [{"start": float(o.time), "end": float(o.time + o.duration), "label": o.value} for o in ann.data]


_btc = {}


def btc(path: str) -> list[dict]:
    import torch

    sys.path.insert(0, BTC_DIR)
    from btc_model import BTC_model
    from utils.hparams import HParams
    from utils.mir_eval_modules import audio_file_to_features, idx2voca_chord

    if not _btc:
        import yaml

        # HParams.load uses the pre-PyYAML-6 yaml.load(f) signature.
        with open(os.path.join(BTC_DIR, "run_config.yaml")) as f:
            config = HParams(**yaml.safe_load(f))
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170
        model = BTC_model(config=config.model)
        ckpt = torch.load(os.path.join(BTC_DIR, "test", "btc_model_large_voca.pt"), map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _btc.update(model=model, config=config, mean=ckpt["mean"], std=ckpt["std"], idx=idx2voca_chord())
    model, config = _btc["model"], _btc["config"]
    # As in BTC's test.py.
    feature, per_second, length = audio_file_to_features(path, config)
    feature = (feature.T - _btc["mean"]) / _btc["std"]
    steps = config.model["timestep"]
    feature = np.pad(feature, ((0, steps - feature.shape[0] % steps), (0, 0)))
    preds = []
    with torch.no_grad():
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        for t in range(feature.shape[0] // steps):
            out, _ = model.self_attn_layers(x[:, steps * t:steps * (t + 1), :])
            p, _ = model.output_layer(out)
            preds += [int(v) for v in p.squeeze()]
    segments = []
    for i, p in enumerate(preds):
        start = i * per_second
        if start >= length:
            break
        if segments and segments[-1]["label"] == _btc["idx"][p]:
            segments[-1]["end"] = min(start + per_second, length)
        else:
            segments.append({"start": start, "end": min(start + per_second, length), "label": _btc["idx"][p]})
    return segments


MODELS = {"chordino": chordino, "crema": crema_segments, "btc": btc}


def main(manifest: str) -> None:
    jobs = json.load(open(manifest))
    for model, fn in MODELS.items():
        os.makedirs(os.path.join(OUT, model), exist_ok=True)
        fn(jobs[0]["audio"])  # warm-up: model loads, JIT
        for job in jobs:
            path = os.path.join(OUT, model, job["name"] + ".json")
            if os.path.exists(path):
                continue
            t0 = time.perf_counter()
            segments = fn(job["audio"])
            seconds = time.perf_counter() - t0
            audio_seconds = librosa.get_duration(path=job["audio"])
            json.dump({"seconds": round(seconds, 2), "audio_seconds": round(audio_seconds, 2), "segments": segments},
                      open(path, "w"))
            print(f"{model:<9} {job['name']:<22} {len(segments):>4} segments  {seconds:6.1f}s "
                  f"({seconds / audio_seconds:.3f} s/s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
