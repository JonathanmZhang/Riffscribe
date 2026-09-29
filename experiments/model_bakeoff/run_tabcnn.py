"""TabCNN (GuitarProFX-trained, Pedroza et al. DAFx 2024) on one bake-off
clip, on CPU. Runs in the bakeoff-tabcnn image:
    python /bakeoff/run_tabcnn.py <segment id>

Uses amt-tools' own front end and decoding (the model was trained with it):
CQT (22.05kHz, hop 512, 192 bins, 24/octave, dB-scaled), then per-string
tablature -> notes via StackedNoteTranscriber. Every note comes with a
string, so fret positions come for free (not scored here: pitch only).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import time  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from common import clip_path, write_notes  # noqa: E402


def transcribe_notes(model, path: str) -> list[dict]:
    import amt_tools.tools as tools
    from amt_tools.features import CQT
    from amt_tools.inference import run_offline
    from amt_tools.transcribe import ComboEstimator, StackedNoteTranscriber, TablatureWrapper

    proc = CQT(sample_rate=22050, hop_length=512, n_bins=192, bins_per_octave=24)
    audio, _ = tools.load_normalize_audio(path, fs=22050)
    feats = proc.process_audio(audio)
    features = {tools.KEY_FEATS: feats, tools.KEY_TIMES: proc.get_times(audio)}
    estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                StackedNoteTranscriber(profile=model.profile)])
    with torch.no_grad():
        predictions = run_offline(features, model, estimator)
    notes = []
    for string_index, (pitches, intervals) in predictions[tools.KEY_NOTES].items():
        for pitch, (onset, end) in zip(pitches, intervals):
            notes.append({"onset": float(onset), "end": float(end), "midi": int(round(float(pitch))),
                          "string": 6 - int(string_index)})  # 1 = high e ... 6 = low E
    return notes


def main(seg: str) -> None:
    import amt_tools  # noqa: F401  (classes for the pickled model)

    t0 = time.perf_counter()
    model = torch.load("/opt/tabcnn/model.pt", map_location="cpu", weights_only=False)
    model.change_device("cpu")  # it was pickled with device cuda:0
    model.eval()
    load = time.perf_counter() - t0

    path = clip_path(seg)
    transcribe_notes(model, path)  # warm-up, as for every model
    t0 = time.perf_counter()
    notes = transcribe_notes(model, path)
    infer = time.perf_counter() - t0
    duration = sf.info(path).duration
    write_notes("tabcnn-gpfx", seg, duration, load, infer, notes)
    midis = [n["midi"] for n in notes]
    print(f"{seg}: {len(notes)} notes (midi {min(midis, default=0)}-{max(midis, default=0)}), "
          f"{infer:.2f}s for {duration:.1f}s audio, load {load:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1])
