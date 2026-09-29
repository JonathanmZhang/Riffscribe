"""Scores every notes file in data/_bakeoff/notes with the EGSet12 benchmark's
pitch matching (scripts/egset12_benchmark._match: same MIDI pitch, onset
within 50ms, one-to-one) against the segment's annotated notes.

Runs in the worker image: python score.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(__file__))

from common import OUT, SEGMENTS  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, _match, load_truth  # noqa: E402


def main() -> None:
    bench = json.load(open(BENCHMARK_JSON))
    bounds = {s["id"]: (s["start"], s["end"]) for p in bench["performances"] for s in p["segments"]}
    kind_of = {v: k for k, v in SEGMENTS.items()}
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(path))
        seg = r["segment"]
        start, end = bounds[seg]
        truth = [n for n in load_truth(seg[:2])[0] if start <= n["onset"] < end]
        notes = [{"onset": n["onset"] + start, "midi": n["midi"]} for n in r["notes"]]
        matched = len(_match(truth, notes))
        rows.append({
            "model": r["model"], "segment": seg, "kind": kind_of.get(seg, "?"),
            "truth": len(truth), "notes": len(notes), "matched": matched,
            "recall": matched / len(truth), "precision": matched / len(notes) if notes else 0.0,
            "rtf": r["infer_seconds"] / r["audio_seconds"], "load_s": r["load_seconds"],
            "peak_rss_mb": r["peak_rss_mb"],
        })

    print(f"{'model':<18} {'segment':<17} {'recall':>7} {'prec':>6} {'tab/truth':>10} {'s per s':>8} "
          f"{'load s':>7} {'peak MB':>8}")
    for r in sorted(rows, key=lambda r: (r["model"], list(SEGMENTS).index(r["kind"]))):
        print(f"{r['model']:<18} {r['kind'] + ' ' + r['segment']:<17} {r['recall']:>7.1%} {r['precision']:>6.1%} "
              f"{str(r['notes']) + '/' + str(r['truth']):>10} {r['rtf']:>8.2f} {r['load_s']:>7.1f} "
              f"{r['peak_rss_mb']:>8.0f}")
    print("\npooled over the 3 segments:")
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        t, n, m = (sum(r[k] for r in rs) for k in ("truth", "notes", "matched"))
        rtf = sum(r["rtf"] for r in rs) / len(rs)
        print(f"  {model:<18} recall {m / t:.1%}  precision {m / n if n else 0:.1%}  "
              f"mean s/s {rtf:.2f}  max peak {max(r['peak_rss_mb'] for r in rs):.0f} MB  ({len(rs)} segments)")
    with open(os.path.join(os.path.dirname(OUT), "scores.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
