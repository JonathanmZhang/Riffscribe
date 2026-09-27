"""Tunes the hand-position mapper's cost constants (fretboard.HandCosts) on
EGSet12 performances by coordinate descent: vary one constant over its grid,
keep the value that scores best, repeat until a full pass finds no
improvement. The objective is mean position agreement over the three tones.
The same search (same grid, same starting point) is used everywhere.

Usage (inside the worker container, after scripts.egset12_benchmark build):
    python -m scripts.tune_hand_mapper crossval   # 2-fold CV by performance
    python -m scripts.tune_hand_mapper final      # tune on all 12 performances

crossval tunes on half A and scores held-out half B, then the reverse, and
pools the two held-out results, so every performance is scored with
constants tuned without it. It prints them next to the old mapper's numbers
on the same performances (from the saved baseline).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

from scripts.chord_stages import StageConfig  # noqa: E402
from scripts.egset12_benchmark import (  # noqa: E402
    HELDOUT_PERFORMANCES,
    PERFORMANCES,
    TONES,
    TUNING_PERFORMANCES,
    evaluate,
    pool,
)
from tasks.fretboard import HandCosts  # noqa: E402

# Every search starts here (the first round's defaults, no open-position bonus).
START = HandCosts(span=4, stretch_cost=0.25, out_of_span_cost=4.0, shift_cost=4.0, shift_per_fret=0.5,
                  open_string_cost=0.0, fret_height_cost=0.02, max_voicing_span=5,
                  open_position_max_hand=3, open_position_bonus=0.0)
GRID = {
    "open_position_bonus": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
    "shift_cost": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "shift_per_fret": [0.0, 0.25, 0.5, 1.0],
    "fret_height_cost": [0.0, 0.02, 0.05, 0.1, 0.2],
    "out_of_span_cost": [2.0, 4.0, 6.0, 10.0],
    "stretch_cost": [0.25, 0.5, 1.0, 2.0],
    "open_string_cost": [-0.5, 0.0, 0.5],
    "max_voicing_span": [4, 5],
    "span": [4, 5],
}
BASELINE_JSON = "/app/data/_testaudio/hand_baseline.json"


def score(costs: HandCosts, performances: list[str]) -> float:
    result = evaluate(StageConfig(hand_costs=costs), performances=performances)
    return sum(result[t]["all"]["position_agreement"] for t in TONES) / len(TONES)


def tune(performances: list[str], label: str) -> HandCosts:
    best, best_score = START, score(START, performances)
    improved, rounds = True, 0
    while improved and rounds < 4:
        improved, rounds = False, rounds + 1
        for name, values in GRID.items():
            for value in values:
                if getattr(best, name) == value:
                    continue
                candidate = dataclasses.replace(best, **{name: value})
                s = score(candidate, performances)
                if s > best_score + 1e-4:
                    best, best_score, improved = candidate, s, True
    print(f"[{label}] tuned on {performances}: {best_score * 100:.2f}% -> {dataclasses.asdict(best)}", flush=True)
    return best


def _row(label: str, r: dict) -> str:
    cells = []
    for tone in TONES:
        cells.append("/".join(f"{r[tone][k]['position_agreement'] * 100:4.1f}" for k in ("all", "chords", "single-note")))
    shifts = "/".join(f"{r['playability'][t]['tab']['shifts_per_minute']:.1f}" for t in TONES)
    return f"  {label:<22} " + " | ".join(cells) + f" | shifts/min {shifts}"


def crossval(out_path: str) -> None:
    folds = [("A->B", TUNING_PERFORMANCES, HELDOUT_PERFORMANCES), ("B->A", HELDOUT_PERFORMANCES, TUNING_PERFORMANCES)]
    held_out, fold_costs = [], {}
    for label, train, test in folds:
        costs = tune(train, label)
        fold_costs[label] = dataclasses.asdict(costs)
        held_out.append(evaluate(StageConfig(hand_costs=costs), performances=test))
    pooled = pool(held_out)
    old = json.load(open(BASELINE_JSON))["all"]  # untuned, so its "held-out" is all 12
    print("\nCross-validated position agreement on held-out performances (pooled over both folds):")
    print("  columns: all/chords/single-note per tone (clean | moderate | heavy), then tab shifts per minute")
    print(_row("old mapper", old))
    print(_row("hand mapper (CV)", pooled))
    print("  " + " " * 22 + " player shifts/min: " + "/".join(f"{old['playability'][t]['truth']['shifts_per_minute']:.1f}" for t in TONES))
    with open(out_path, "w") as f:
        json.dump({"fold_costs": fold_costs, "pooled_heldout": pooled, "old": old}, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["crossval", "final"])
    parser.add_argument("--out", default="/app/data/_testaudio/hand_crossval.json")
    args = parser.parse_args()
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    if args.command == "crossval":
        crossval(args.out)
    else:
        costs = tune(list(PERFORMANCES), "all 12")
        with open("/app/data/_testaudio/hand_final.json", "w") as f:
            json.dump(dataclasses.asdict(costs), f, indent=1)


if __name__ == "__main__":
    main()
