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

from scripts.chord_stages import (  # noqa: E402
    PIPELINE,
    StageConfig,
    add_config_args,
    config_from_args,
    detect_file,
    run_stages,
    window_metrics,
)
from scripts.eval_chords import evaluate_program  # noqa: E402

TESTSET = "/app/data/_testaudio/chord_testset"
FIREFIRE = "/app/data/_testaudio/firefire.webm"
FIREFIRE_WINDOWS = [(28.0, 31.0), (39.5, 42.5), (60.0, 63.0)]
SOLO_CLIP = "/app/data/samples/tab_sample.ogg"


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


def _solo(config: StageConfig) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        events, _, _ = detect_file(SOLO_CLIP, workdir)
    stages = run_stages(events, config)
    positions = stages["mapping"]["positions"]
    notes = sorted((round(k[0], 3), k[1], *v) for k, v in positions.items())
    return {"tab_notes": len(notes), "notes": notes}


def run_all(config: StageConfig) -> dict:
    return {"config": config.describe(), "synthetic": _synthetic(config), "firefire": _firefire(config),
            "solo": _solo(config)}


def _solo_diff(base: dict, cur: dict) -> str:
    b, c = {tuple(n) for n in base["notes"]}, {tuple(n) for n in cur["notes"]}
    return f"{cur['tab_notes']} notes (+{len(c - b)} / -{len(b - c)} vs baseline)"


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
    print("  firefire windows: kept/events | tab notes | columns (>=3-note) | mean notes per >=3 column | "
          "mapper drops | retrigger candidates")
    for w, m in cur["firefire"].items():
        r = base["firefire"][w] if base else None
        print(f"    {w:<13} {m['kept']}/{m['events']} | {d(m['tab_notes'], r and r['tab_notes'])} | "
              f"{m['columns']} ({d(m['chord_columns_ge3'], r and r['chord_columns_ge3'])}) | "
              f"{m['mean_notes_per_chord_column']} | {m['mapper_drops']} | {m['retrigger_candidates']}")
    print(f"  solo clip: {_solo_diff(base['solo'], cur['solo']) if base else str(cur['solo']['tab_notes']) + ' notes'}")


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
