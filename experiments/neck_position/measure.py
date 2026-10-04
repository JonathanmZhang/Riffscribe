"""Neck position, measured on EGSet12: how much closer does the tab get to
where the player plays when the user picks the neck position?

For each performance the window is set from the ground truth: the median
fret of all its annotated notes, +- 3 frets (open strings allowed when the
window reaches fret 0). That is the best case for a user who watched the
video and picked "around fret N". The notes are the pipeline's (cached Basic
Pitch output, select_notes, clean_notes - scripts.chord_stages.run_stages);
only the mapping changes (tasks/fretmap.map_notes_to_positions with the
window vs "auto"), so pitch recall/precision must be identical - checked per
run, and the script stops if any note differs.

Scoring as in scripts/egset12_benchmark.py: a tab note matches a truth note
of the same pitch starting within 50ms, one to one; position agreement =
same string AND fret, among matched notes.

Context rows (not the asked-for number):
  median +- 2        a narrower window from the same median
  best UI setting    per performance and tone, whichever of the UI's
                     choices (Open position, around fret 1-17) agrees most
                     with the truth: an oracle upper bound for the control

Runs in the worker image (repo at /repo for the stale-image guard, this
folder at /x):

    python /x/measure.py [--json /x/results.json]
"""

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")

from scripts import build_info  # noqa: E402
from scripts import egset12_benchmark as eb  # noqa: E402
from scripts.chord_stages import detect_file, run_stages  # noqa: E402
from tasks import fretmap  # noqa: E402

UI_SETTINGS = ["open"] + list(fretmap.NECK_CENTRES)


def median_window(truth: list[dict], half: int) -> tuple[int, int, int]:
    median = statistics.median_low(n["fret"] for n in truth)
    return median, max(0, median - half), min(fretmap.MAX_FRET, median + half)


def score(truth: list[dict], mapped: list[dict], seg_type) -> dict:
    tab = [{"onset": n["start_time"], "midi": fretmap.note_name_to_number(n["pitch"]),
            "string": n["string"], "fret": n["fret"]} for n in mapped]
    matches = eb._match(truth, tab)
    counts: dict[str, dict] = {}
    for i, t in enumerate(truth):
        for kind in (seg_type(t["onset"]), "all"):
            b = counts.setdefault(kind, {"truth": 0, "tab": 0, "matched": 0, "position": 0})
            b["truth"] += 1
            if i in matches:
                b["matched"] += 1
                n = tab[matches[i]]
                b["position"] += (n["string"], n["fret"]) == (t["string"], t["fret"])
    for n in tab:
        for kind in (seg_type(n["onset"]), "all"):
            counts.setdefault(kind, {"truth": 0, "tab": 0, "matched": 0, "position": 0})["tab"] += 1
    return counts


def add(total: dict, counts: dict) -> None:
    for kind, b in counts.items():
        t = total.setdefault(kind, {"truth": 0, "tab": 0, "matched": 0, "position": 0})
        for k, v in b.items():
            t[k] += v


def pct(b: dict, key: str, of: str) -> float | None:
    return round(100 * b[key] / b[of], 1) if b[of] else None


def same_notes(a: list[dict], b: list[dict]) -> bool:
    key = lambda n: (n["start_time"], n["pitch"], n["end_time"])  # noqa: E731
    return sorted(map(key, a)) == sorted(map(key, b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="also write the numbers here")
    build_info.add_args(parser)
    args = parser.parse_args()
    build_info.guard(args.allow_stale)
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)

    benchmark = json.load(open(eb.BENCHMARK_JSON))
    configs = ["auto", "median +- 3", "median +- 2", "best UI setting"]
    totals = {tone: {c: {} for c in configs} for tone in eb.TONES}
    per_perf = []
    for perf in benchmark["performances"]:
        p = perf["performance"]
        truth, _ = eb.load_truth(p)
        segments = perf["segments"]

        def seg_type(t):
            for s in segments:
                if s["start"] <= t < s["end"]:
                    return s["type"]
            return segments[-1]["type"]

        median, lo, hi = median_window(truth, 3)
        _, lo2, hi2 = median_window(truth, 2)
        in_window = sum(lo <= n["fret"] <= hi for n in truth) / len(truth)
        row = {"performance": p, "median_fret": median, "window": [lo, hi],
               "truth_in_window": round(100 * in_window, 1), "tones": {}}
        for tone in eb.TONES:
            with tempfile.TemporaryDirectory() as workdir:
                events, activations, _ = detect_file(eb.audio_path(p, tone), workdir)
            kept = run_stages(events, activations=activations)["kept"]
            mapped = {
                "auto": fretmap.map_notes_to_positions(kept),
                "median +- 3": fretmap.map_notes_to_positions(kept, (lo, hi)),
                "median +- 2": fretmap.map_notes_to_positions(kept, (lo2, hi2)),
            }
            ui = {s: fretmap.map_notes_to_positions(kept, fretmap.neck_window(s)) for s in UI_SETTINGS}
            for name, notes in [*mapped.items(), *ui.items()]:
                if not same_notes(notes, mapped["auto"]):
                    sys.exit(f"{p} {tone} {name}: the notes differ from auto - only positions may change")
            ui_scores = {s: score(truth, notes, seg_type) for s, notes in ui.items()}
            best = max(UI_SETTINGS, key=lambda s: (ui_scores[s]["all"]["position"], s == "open"))
            scores = {name: score(truth, notes, seg_type) for name, notes in mapped.items()}
            scores["best UI setting"] = ui_scores[best]
            for name, counts in scores.items():
                add(totals[tone][name], counts)
            row["tones"][tone] = {
                **{name: pct(counts["all"], "position", "matched") for name, counts in scores.items()},
                "best_ui": best,
                "matched": scores["auto"]["all"]["matched"],
            }
        per_perf.append(row)

    print("\nPosition agreement per performance (window = truth median fret +- 3):")
    print(f"  {'perf':<5}{'median':>7}{'window':>9}{'truth in':>10}   " +
          "   ".join(f"{tone:>16}" for tone in eb.TONES))
    print(f"  {'':<5}{'':>7}{'':>9}{'window':>10}   " + "   ".join(f"{'auto -> window':>16}" for _ in eb.TONES))
    for row in per_perf:
        cells = []
        for tone in eb.TONES:
            t = row["tones"][tone]
            cells.append(f"{t['auto']:5.1f} -> {t['median +- 3']:5.1f}".rjust(16))
        window = "{}-{}".format(*row["window"])
        print(f"  {row['performance']:<5}{row['median_fret']:>7}{window:>9}"
              f"{row['truth_in_window']:>9.1f}%   " + "   ".join(cells))

    print("\nPer tone (all performances): pitch recall | precision | position agreement [matched notes]")
    for tone in eb.TONES:
        for kind in ["all"] + eb.SEGMENT_TYPES:
            for name in configs:
                b = totals[tone][name].get(kind)
                if not b:
                    continue
                print(f"  {tone:<9}{kind:<12}{name:<16}{pct(b, 'matched', 'truth'):5.1f} | "
                      f"{pct(b, 'matched', 'tab'):5.1f} | {pct(b, 'position', 'matched'):5.1f}  [{b['matched']}]")
        print()
    print("best UI setting per performance:", {r["performance"]: {t: r["tones"][t]["best_ui"] for t in eb.TONES}
                                              for r in per_perf})

    if args.json:
        summary = {tone: {name: {kind: {**b, "pitch_recall": pct(b, "matched", "truth"),
                                         "pitch_precision": pct(b, "matched", "tab"),
                                         "position_agreement": pct(b, "position", "matched")}
                                  for kind, b in totals[tone][name].items()}
                           for name in configs} for tone in eb.TONES}
        with open(args.json, "w") as f:
            json.dump({"per_tone": summary, "per_performance": per_perf}, f, indent=2)


if __name__ == "__main__":
    main()
