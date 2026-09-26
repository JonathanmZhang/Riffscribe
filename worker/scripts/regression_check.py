"""Regression check for chord-accuracy changes: runs the synthetic chord eval
(all programs), the fixed firefire windows (full mix and separated) and the
8s solo clip, for one or more post-detection settings.

Usage (inside the worker container):
    python -m scripts.regression_check [--save PATH] [--compare PATH] [settings flags]
    python -m scripts.regression_check --configs "threshold=0.4" "chord_floor=0.35" ...

With --configs, each "field=value[,field=value]" spec (StageConfig fields,
unspecified fields at pipeline defaults) is evaluated and printed as one row
of a comparison table. Detection is cached (scripts/chord_stages.py), so
sweeps are fast after the first run.

The firefire windows are fixed: they were picked (from raw Basic Pitch
event clusters in both the full mix and the separated stem) as dense
strumming, and serve as the real-world check across fixes. There is no
ground truth for them, so only ground-truth-free metrics are reported.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402

import pretty_midi  # noqa: E402

from scripts.chord_stages import (  # noqa: E402
    PIPELINE,
    StageConfig,
    add_config_args,
    config_from_args,
    detect_file,
    run_stages,
    window_metrics,
)
from scripts.eval_chords import evaluate_program, evaluate_repeats  # noqa: E402

TESTSET = "/app/data/_testaudio/chord_testset"
FIREFIRE = "/app/data/_testaudio/firefire.webm"
FIREFIRE_WINDOWS = [(28.0, 31.0), (39.5, 42.5), (60.0, 63.0)]
SOLO_CLIP = "/app/data/samples/tab_sample.ogg"
# Ground truth for SOLO_CLIP, read from the tab it illustrates on Wikimedia
# Commons (File:GuitarTabulatureSample1.svg): 4 bars of 4/4 in A major,
# 14 columns / 30 notes, as (beat, MIDI notes). It's mostly 2-3 note chords,
# not a single-note line. Beat times are aligned to the audio by fitting
# tempo and offset against the raw Basic Pitch events (see _fit_solo_truth).
SOLO_TRUTH = [
    (0.0, [45, 57]), (1.5, [61]), (2.0, [45, 61, 64]), (3.0, [69]),
    (4.0, [50, 62, 66]), (5.0, [69]), (5.5, [66]), (6.0, [45, 61, 64]),
    (8.0, [44, 59, 62]), (9.0, [40]), (9.5, [64]), (10.0, [45, 52, 61]), (11.0, [49, 57]),
    (12.0, [52, 56, 59]), (14.0, [45, 57]),
]
SOLO_MATCH_S = 0.10


def _synthetic(config: StageConfig) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(TESTSET, "chords_prog*.json"))):
        truth = json.load(open(path))
        result = evaluate_program(truth, os.path.join(TESTSET, truth["wav"]), config=config)
        s = result["summary"]
        out[str(truth["program"])] = {
            "recall_mapped": s["recall"]["mapped"],
            "recall_kept": s["recall"]["kept"],
            "precision": s["precision"],
            "complete_pct": s["complete_pct"],
            "complete_pct_fast": s["complete_pct_fast"],
            "clean_pct": s["clean_pct"],
            "extras_in_tab": s["extras_in_tab"],
            "losses_by_stage": s["losses_by_stage"],
        }
        repeats_json = os.path.join(TESTSET, f"repeats_prog{truth['program']}.json")
        if os.path.exists(repeats_json):
            rt = json.load(open(repeats_json))
            r = evaluate_repeats(rt, os.path.join(TESTSET, rt["wav"]), config)
            out[str(truth["program"])]["repeats"] = {
                "recall": r["recall"], "extras": r["extras"],
                "by_pattern": {k: f"{v['separate_in_tab']}/{v['expected']} (+{v['extras']})"
                               for k, v in r["patterns"].items()},
            }
    return out


def _firefire(config: StageConfig) -> dict:
    out = {}
    for separate in (False, True):
        with tempfile.TemporaryDirectory() as workdir:
            events, _, _ = detect_file(FIREFIRE, workdir, separate=separate, seed=0)
        stages = run_stages(events, config)
        for start, end in FIREFIRE_WINDOWS:
            out[f"{start:g}-{end:g}{' sep' if separate else ''}"] = window_metrics(stages, start, end)
    return out


def _fit_solo_truth(events: list[dict]) -> tuple[float, float]:
    """(seconds per beat, offset) that line up the most tab notes with raw
    events of the same pitch. Uses raw events, so the alignment doesn't
    depend on the settings under test."""
    by_pitch: dict[int, list[float]] = {}
    for e in events:
        by_pitch.setdefault(e["midi"], []).append(e["start_time"])
    best = (-1, 0.5, 0.0)
    for spb in [0.40 + 0.002 * i for i in range(101)]:
        for offset in [-0.2 + 0.01 * i for i in range(171)]:
            hits = sum(
                any(abs(t - (offset + beat * spb)) <= SOLO_MATCH_S for t in by_pitch.get(m, ()))
                for beat, notes in SOLO_TRUTH for m in notes
            )
            if hits > best[0]:
                best = (hits, spb, offset)
    return best[1], best[2]


def _solo(config: StageConfig) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        events, _, _ = detect_file(SOLO_CLIP, workdir)
    stages = run_stages(events, config)
    positions = stages["mapping"]["positions"]
    notes = sorted((round(k[0], 3), k[1], *v) for k, v in positions.items())

    # Score the tab against the known tab: a truth note counts when a mapped
    # note of the same pitch starts within SOLO_MATCH_S of it.
    spb, offset = _fit_solo_truth(events)
    mapped = [(k[0], pretty_midi.note_name_to_number(k[1])) for k in positions]
    used, hits, complete = set(), 0, 0
    for beat, truth_notes in SOLO_TRUTH:
        t = offset + beat * spb
        column_hits = 0
        for m in truth_notes:
            match = next((i for i, (s, p) in enumerate(mapped)
                          if i not in used and p == m and abs(s - t) <= SOLO_MATCH_S), None)
            if match is not None:
                used.add(match)
                column_hits += 1
        hits += column_hits
        complete += column_hits == len(truth_notes)
    total = sum(len(n) for _, n in SOLO_TRUTH)
    return {
        "tab_notes": len(notes),
        "notes": notes,
        "recall": round(hits / total, 3),
        "precision": round(len(used) / len(mapped), 3) if mapped else None,
        "complete_columns": complete,
        "columns": len(SOLO_TRUTH),
        "alignment": {"seconds_per_beat": round(spb, 3), "offset": round(offset, 2)},
    }


def run_all(config: StageConfig) -> dict:
    return {"config": config.describe(), "synthetic": _synthetic(config), "firefire": _firefire(config),
            "solo": _solo(config)}


def _solo_diff(base: dict | None, cur: dict) -> str:
    text = (f"{cur['tab_notes']} tab notes; vs the known tab: recall {cur['recall']:.0%}, precision "
            f"{cur['precision']:.0%}, complete columns {cur['complete_columns']}/{cur['columns']}")
    if base and "recall" in base:
        text += (f"  [baseline {base['recall']:.0%} / {base['precision']:.0%} / "
                 f"{base['complete_columns']}/{base['columns']}]")
    return text


def print_report(cur: dict, base: dict | None) -> None:
    def d(val, ref, pct=False, fmt="{:.1%}"):
        s = fmt.format(val) if pct else f"{val}"
        if ref is None:
            return s
        delta = val - ref
        return s + (f" ({delta:+.1%})" if pct else f" ({delta:+g})") if delta else s + " (=)"

    print(f"\nsettings: {cur['config']}" + (f"   [baseline: {base['config']}]" if base else ""))
    print("  synthetic (per program): recall mapped | precision | complete % (fast) | extras in tab | lost a/b/c/d")
    for prog, s in cur["synthetic"].items():
        r = base["synthetic"][prog] if base else None
        lost = "/".join(str(s["losses_by_stage"][k]) for k in "abcd")
        print(f"    prog {prog}: {d(s['recall_mapped'], r and r['recall_mapped'], True)} | "
              f"{d(s['precision'], r and r['precision'], True)} | "
              f"{d(s['complete_pct'], r and r['complete_pct'])} ({d(s['complete_pct_fast'], r and r['complete_pct_fast'])}) | "
              f"{d(s['extras_in_tab'], r and r['extras_in_tab'])} | {lost}")
        if "repeats" in s:
            rr = r.get("repeats") if r else None
            print(f"      repeated notes kept separate: {d(s['repeats']['recall'], rr and rr['recall'], True)}, "
                  f"extras {s['repeats']['extras']} | " + ", ".join(f"{k}: {v}" for k, v in s["repeats"]["by_pattern"].items()))
    print("  firefire windows: kept/events | tab notes | columns (>=3-note) | mean notes per >=3 column | "
          "mapper drops | retrigger candidates")
    for w, m in cur["firefire"].items():
        r = base["firefire"][w] if base else None
        print(f"    {w:<13} {m['kept']}/{m['events']} | {d(m['tab_notes'], r and r['tab_notes'])} | "
              f"{m['columns']} ({d(m['chord_columns_ge3'], r and r['chord_columns_ge3'])}) | "
              f"{m['mean_notes_per_chord_column']} | {m['mapper_drops']} | {m['retrigger_candidates']}")
    print(f"  solo clip: {_solo_diff(base['solo'] if base else None, cur['solo'])}")


def _parse_spec(spec: str) -> StageConfig:
    fields = {f.name: f for f in dataclasses.fields(StageConfig)}
    kwargs = {}
    for part in filter(None, spec.split(",")):
        name, value = part.split("=")
        name = name.strip().replace("-", "_")
        default = fields[name].default
        kwargs[name] = (value.strip() if isinstance(default, str)
                        else None if value.strip().lower() == "none" else float(value))
    return StageConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", help="write results JSON here")
    parser.add_argument("--compare", help="baseline results JSON to compare against")
    parser.add_argument("--configs", nargs="+", help='sweep: e.g. "threshold=0.4" "chord_floor=0.35"')
    add_config_args(parser)
    args = parser.parse_args()
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)

    base = json.load(open(args.compare)) if args.compare else None
    configs = [_parse_spec(s) for s in args.configs] if args.configs else [config_from_args(args)]
    results = []
    for config in configs:
        cur = run_all(config)
        print_report(cur, base)
        results.append(cur)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results[0] if len(results) == 1 else results, f, indent=2)
        print(f"\nresults written to {args.save}")


if __name__ == "__main__":
    main()


__all__ = ["PIPELINE", "run_all"]
