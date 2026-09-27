"""Tunes the hand-position mapper's cost constants (fretboard.HandCosts) on
the EGSet12 TUNING half only, by coordinate descent: vary one constant over
its grid, keep the value that scores best, repeat until a full pass finds no
improvement. The objective is mean position agreement over the three tones.
The held-out half is not touched here.

Usage (inside the worker container, after scripts.egset12_benchmark build):
    python -m scripts.tune_hand_mapper [--out results.json]
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

from scripts.chord_stages import StageConfig  # noqa: E402
from scripts.egset12_benchmark import TONES, TUNING_PERFORMANCES, evaluate  # noqa: E402
from tasks.fretboard import HAND_COSTS, HandCosts  # noqa: E402

GRID = {
    "fret_height_cost": [0.0, 0.02, 0.05, 0.1, 0.2, 0.4],
    "out_of_span_cost": [2.0, 4.0, 6.0, 10.0],
    "stretch_cost": [0.25, 0.5, 1.0, 2.0],
    "shift_cost": [0.5, 1.0, 2.0, 4.0, 8.0],
    "shift_per_fret": [0.0, 0.25, 0.5, 1.0],
    "open_string_cost": [-1.0, -0.5, 0.0, 0.5],
    "max_voicing_span": [4, 5],
    "span": [4, 5],
}


def score(costs: HandCosts) -> tuple[float, dict]:
    result = evaluate(StageConfig(hand_costs=costs), performances=TUNING_PERFORMANCES)
    return sum(result[t]["all"]["position_agreement"] for t in TONES) / len(TONES), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="/app/data/_testaudio/hand_tuning.json")
    args = parser.parse_args()
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)

    best = HAND_COSTS
    best_score, _ = score(best)
    history = [{"costs": dataclasses.asdict(best), "score": best_score}]
    print(f"start {dataclasses.asdict(best)}: {best_score * 100:.2f}%", flush=True)
    improved, rounds = True, 0
    while improved and rounds < 4:
        improved, rounds = False, rounds + 1
        for name, values in GRID.items():
            for value in values:
                if getattr(best, name) == value:
                    continue
                candidate = dataclasses.replace(best, **{name: value})
                s, _ = score(candidate)
                history.append({"costs": dataclasses.asdict(candidate), "score": s})
                if s > best_score + 1e-4:
                    best, best_score, improved = candidate, s, True
                    print(f"round {rounds}: {name}={value} -> {s * 100:.2f}%", flush=True)
        print(f"after round {rounds}: {dataclasses.asdict(best)} = {best_score * 100:.2f}%", flush=True)
    with open(args.out, "w") as f:
        json.dump({"best": dataclasses.asdict(best), "best_score": best_score, "history": history}, f, indent=1)
    print(f"best: {dataclasses.asdict(best)} = {best_score * 100:.2f}% (tuning half, mean over tones)")


if __name__ == "__main__":
    main()
