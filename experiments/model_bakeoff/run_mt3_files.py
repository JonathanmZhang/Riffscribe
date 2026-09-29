"""MT3 on a batch of audio files, model loaded once. Runs in the bakeoff-mt3
image:
    python /bakeoff/run_mt3_files.py <manifest.json>
The manifest is a list of {"audio": path, "name": output name}. Writes
data/_bakeoff/mt3/<name>.json with EVERY note MT3 emits (program and is_drum
kept), so dedup / instrument filtering / scoring happen later, in the worker
image. Existing outputs are skipped, so an interrupted run can resume.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import librosa  # noqa: E402

from common import DATA, peak_rss_mb  # noqa: E402
from run_mt3 import CHECKPOINT, SAMPLE_RATE, InferenceModel  # noqa: E402

OUT = os.path.join(DATA, "_bakeoff", "mt3")


def main(manifest_path: str) -> None:
    jobs = json.load(open(manifest_path))
    os.makedirs(OUT, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j["name"] + ".json"))]
    print(f"{len(todo)} of {len(jobs)} files to run", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    model = InferenceModel(CHECKPOINT)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    # Warm-up (JAX compilation) on 10s of the first file.
    warm, _ = librosa.load(todo[0]["audio"], sr=SAMPLE_RATE, mono=True, duration=10.0)
    model(warm)

    for job in todo:
        audio, _ = librosa.load(job["audio"], sr=SAMPLE_RATE, mono=True)
        t0 = time.perf_counter()
        ns = model(audio)
        infer = time.perf_counter() - t0
        notes = [{"onset": round(n.start_time, 4), "end": round(n.end_time, 4), "midi": int(n.pitch),
                  "program": int(n.program), "is_drum": bool(n.is_drum)} for n in ns.notes]
        with open(os.path.join(OUT, job["name"] + ".json"), "w") as f:
            json.dump({"name": job["name"], "audio": job["audio"], "audio_seconds": round(len(audio) / SAMPLE_RATE, 3),
                       "infer_seconds": round(infer, 2), "peak_rss_mb": peak_rss_mb(),
                       "notes": sorted(notes, key=lambda n: (n["onset"], n["midi"]))}, f)
        print(f"{job['name']}: {len(notes)} notes, {infer:.1f}s for {len(audio) / SAMPLE_RATE:.1f}s audio "
              f"({infer / (len(audio) / SAMPLE_RATE):.2f} s/s), peak {peak_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
