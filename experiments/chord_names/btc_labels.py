"""BTC large-vocabulary chord labels for the EGSet12 benchmark audio (all
three tones). Runs in the chord-names image:
    python /chord_names/btc_labels.py
Writes data/_chord_names/btc/egset12_<tone>_<NN>.json =
{"seconds", "audio_seconds", "segments": [{"start", "end", "label"}]}.
Inference mirrors BTC's own test.py.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.egset12_benchmark import PERFORMANCES, TONES, audio_path  # noqa: E402

BTC_DIR = "/opt/btc"
OUT = "/app/data/_chord_names/btc"
sys.path.insert(0, BTC_DIR)

from btc_model import BTC_model  # noqa: E402
from utils.hparams import HParams  # noqa: E402
from utils.mir_eval_modules import audio_file_to_features, idx2voca_chord  # noqa: E402


def load():
    # HParams.load uses the pre-PyYAML-6 yaml.load(f) signature.
    with open(os.path.join(BTC_DIR, "run_config.yaml")) as f:
        config = HParams(**yaml.safe_load(f))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170
    model = BTC_model(config=config.model)
    ckpt = torch.load(os.path.join(BTC_DIR, "test", "btc_model_large_voca.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt["mean"], ckpt["std"], idx2voca_chord()


def segments(path: str, model, config, mean, std, idx) -> tuple[list[dict], float]:
    feature, per_second, length = audio_file_to_features(path, config)
    feature = (feature.T - mean) / std
    steps = config.model["timestep"]
    feature = np.pad(feature, ((0, steps - feature.shape[0] % steps), (0, 0)))
    preds = []
    with torch.no_grad():
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        for t in range(feature.shape[0] // steps):
            out, _ = model.self_attn_layers(x[:, steps * t:steps * (t + 1), :])
            p, _ = model.output_layer(out)
            preds += [int(v) for v in p.squeeze()]
    segs: list[dict] = []
    for i, p in enumerate(preds):
        start = i * per_second
        if start >= length:
            break
        end = min(start + per_second, length)
        if segs and segs[-1]["label"] == idx[p]:
            segs[-1]["end"] = end
        else:
            segs.append({"start": start, "end": end, "label": idx[p]})
    return segs, length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    os.makedirs(OUT, exist_ok=True)
    loaded = load()
    for tone in TONES:
        for p in PERFORMANCES:
            t0 = time.perf_counter()
            segs, length = segments(audio_path(p, tone), *loaded)
            seconds = time.perf_counter() - t0
            json.dump({"seconds": round(seconds, 2), "audio_seconds": round(length, 2), "segments": segs},
                      open(os.path.join(OUT, f"egset12_{tone}_{p}.json"), "w"))
            print(f"{tone} {p}: {len(segs)} segments, {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
