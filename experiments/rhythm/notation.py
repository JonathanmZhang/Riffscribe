"""Notation measurements for the MusicXML export, on the pipeline's own
code (tasks/rhythm.py), against EGSet12's Guitar Pro rhythms. Runs in the
worker image after `rhythm_benchmark truth` and track.py:
    python /rhythm/notation.py

1. Note lengths: length_rule "per_step" (chords and fast single notes: Basic
   Pitch's length; other single notes: gap to the next step) vs "sounding"
   and "next_onset" alone, for both groupings ("slot": notes on the same
   16th form a chord; "strum": the tab's 150ms groups at their first note's
   16th), on the true beats and on beat_this's beats. A note's length is
   scored as notated: snapped, then cut at the next step (one voice).
   Start and length are right as in quantize.py (start within half a 16th
   of the true position; length within 10% of the notated length, in true
   quarters). Segment types (chords / single-note / fast) are EGSet12's
   labels, used only for reporting; the rule itself decides per step from
   the notes.

2. Bars: downbeat F (+-70ms) of rhythm.bar_starts (4/4, beat_this's beats
   in 4s, phase from its downbeats) vs beat_this's own downbeats. Also the
   ceiling the job overrides allow: the best tempo_factor x
   bar_offset_beats per performance.

Every beat_this row is run with rhythm.REGULARIZE_BEATS off and on ("reg":
beats kept at one metrical level), plus beat F of both grids.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, "/app")

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.chord_stages import PIPELINE, detect_file, run_stages  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, PERFORMANCES, SEGMENT_TYPES, TONES, audio_path  # noqa: E402
from scripts.rhythm_benchmark import TOLERANCE_S, TRACKS_DIR, TRUTH_JSON  # noqa: E402
from tasks import rhythm  # noqa: E402

MATCH_S = 0.05
GROUPINGS = ("slot", "strum")


def tab_notes(tone: str, p: str) -> list[dict]:
    """The pipeline's tab notes, in TabResult shape (+ midi)."""
    with tempfile.TemporaryDirectory() as workdir:
        events, activations, _ = detect_file(audio_path(p, tone), workdir)
    stages = run_stages(events, PIPELINE, activations)
    positions = stages["mapping"]["positions"]
    return sorted(({"start_time": n["start_time"], "end_time": n["end_time"], "pitch": n["pitch"], "midi": n["midi"],
                    "string": positions[(n["start_time"], n["pitch"])][0],
                    "fret": positions[(n["start_time"], n["pitch"])][1]}
                   for n in stages["kept"] if (n["start_time"], n["pitch"]) in positions),
                  key=lambda n: n["start_time"])


def match(tab: list[dict], truth: list[dict]) -> list[tuple[int, int]]:
    pairs = sorted((abs(n["start_time"] - t["onset"]), i, j) for i, n in enumerate(tab) for j, t in enumerate(truth)
                   if n["midi"] == t["midi"] and abs(n["start_time"] - t["onset"]) <= MATCH_S)
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def score(tab, truth, pairs, grid, true_grid, rule, grouping) -> tuple[list[dict], int]:
    steps = rhythm.quantize_notes(tab, grid, rule, grouping)
    step_of = {id(n): s for s in steps for n in s["notes"]}
    rows = []
    for i, j in pairs:
        n, t = tab[i], truth[j]
        s = step_of.get(id(n))
        if s is None:  # dropped: same slot and string as another note
            rows.append({"kind": t["kind"], "start": False, "length": False, "dropped": True})
            continue
        q_est = (rhythm.slot_time(s["slot"], grid) - true_grid["offset"]) / true_grid["spq"]
        length_q = s["length"] * rhythm._slot_seconds(s["slot"], grid) / true_grid["spq"]
        rows.append({"kind": t["kind"], "start": bool(abs(q_est - t["q"]) < 0.125),
                     "length": bool(abs(length_q / t["dur_q"] - 1) < 0.1), "dropped": False, "step": s["kind"]})
    return rows, len(tab) - sum(len(s["notes"]) for s in steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    truth_all = json.load(open(TRUTH_JSON))
    segments = {perf["performance"]: perf["segments"] for perf in json.load(open(BENCHMARK_JSON))["performances"]}

    rows: dict[tuple, list[dict]] = {}
    dropped: Counter = Counter()
    bars: dict[tuple, dict] = {}
    for p in PERFORMANCES:
        tr = truth_all[p]
        kind = lambda t: next((s["type"] for s in segments[p] if s["start"] <= t < s["end"]),  # noqa: E731
                              segments[p][-1]["type"])
        truth = [{**n, "kind": kind(n["onset"])} for n in tr["notes"] if "onset" in n]
        n_q = int(np.ceil(max(n["end"] for n in truth) / tr["grid"]["spq"])) + 8
        true_beats = [tr["grid"]["offset"] + k * tr["grid"]["spq"] for k in range(-8, n_q)]
        ref_down = np.array(tr["downbeats"])
        for tone in TONES:
            tab = tab_notes(tone, p)
            pairs = match(tab, truth)
            bt = json.load(open(os.path.join(TRACKS_DIR, "beat_this", f"{tone}_{p}.json")))
            start, end = tab[0]["start_time"], max(n["end_time"] for n in tab)
            grids = {"true beats": rhythm.bar_grid(true_beats, tr["downbeats"], start, end)}
            for reg in (False, True):
                rhythm.REGULARIZE_BEATS = reg
                grids["beat_this" + (" reg" if reg else "")] = rhythm.bar_grid(bt["beats"], bt["downbeats"], start, end)
            for beats_name, grid in grids.items():
                for grouping in GROUPINGS:
                    for rule in rhythm.LENGTH_RULES:
                        r, d = score(tab, truth, pairs, grid, tr["grid"], rule, grouping)
                        rows.setdefault((beats_name, grouping, rule, tone), []).extend(r)
                        if rule == "per_step":  # the same for every rule
                            dropped[beats_name, grouping, tone] += d

            f = lambda est: mir_eval.beat.f_measure(ref_down, np.array(est), TOLERANCE_S)  # noqa: E731
            ref_beats = np.array(tr["beats"])
            bars[tone, p] = {"beat_this": f(bt["downbeats"]),
                             "beat F: beat_this": mir_eval.beat.f_measure(ref_beats, np.array(bt["beats"]), TOLERANCE_S)}
            for reg in (False, True):
                rhythm.REGULARIZE_BEATS = reg
                tag = " reg" if reg else ""
                combos = {(tf, off): f(rhythm.bar_starts(bt["beats"], bt["downbeats"], tf, off))
                          for tf in rhythm.TEMPO_FACTORS for off in rhythm.BAR_OFFSETS}
                bars[tone, p].update({
                    f"4/4 default{tag}": combos[1.0, 0],
                    f"best offset (factor 1){tag}": max(combos[1.0, o] for o in rhythm.BAR_OFFSETS),
                    f"best factor + offset{tag}": max(combos.values()),
                    f"best{tag}": max(combos, key=combos.get),
                    f"beat F: grid{tag}": mir_eval.beat.f_measure(
                        ref_beats, np.array(rhythm.beat_grid(bt["beats"], bt["downbeats"])), TOLERANCE_S),
                })
            rhythm.REGULARIZE_BEATS = True

    pct = lambda rs, k: f"{np.mean([r[k] for r in rs]):5.1%}" if rs else "   - "  # noqa: E731
    print("1. % of matched notes right, start / length as notated  [notes]")
    print(f"{'beats':<11}{'grouping':<9}{'rule':<11}{'tone':<9}" + "".join(f"{k:<22}" for k in ["all"] + SEGMENT_TYPES))
    for (beats_name, grouping, rule, tone), rs in rows.items():
        cells = [f"{pct([r for r in rs if k == 'all' or r['kind'] == k], 'start')} / "
                 f"{pct([r for r in rs if k == 'all' or r['kind'] == k], 'length')} "
                 f"[{sum(k == 'all' or r['kind'] == k for r in rs)}]" for k in ["all"] + SEGMENT_TYPES]
        print(f"{beats_name:<11}{grouping:<9}{rule:<11}{tone:<9}" + "".join(c.ljust(22) for c in cells))
    print("\nper-step kind the rule saw (true beats, slot grouping, clean), by EGSet12 segment type:")
    rs = rows["true beats", "slot", "per_step", "clean"]
    for k in SEGMENT_TYPES:
        c = Counter(r.get("step") for r in rs if r["kind"] == k)
        print(f"  {k:<12} " + ", ".join(f"{s}: {v}" for s, v in c.most_common()))
    print("\ntab notes dropped (same 16th and string as another): " + ", ".join(
        f"{b}/{g}/{t}: {v}" for (b, g, t), v in dropped.items() if v))

    print("\n2. downbeat F (+-70ms), mean over 12 performances")
    variants = ["beat_this"] + [v + tag for tag in ("", " reg") for v in
                                ("4/4 default", "best offset (factor 1)", "best factor + offset")] + \
               ["beat F: beat_this", "beat F: grid", "beat F: grid reg"]
    print(f"{'':<30}" + "".join(f"{t:>10}" for t in TONES))
    for v in variants:
        print(f"{v:<30}" + "".join(f"{np.mean([bars[t, p][v] for p in PERFORMANCES]):>10.1%}" for t in TONES))
    for tone in TONES:
        print(f"  {tone} (beat_this / 4/4 reg / best reg): " + "  ".join(
            f"{p}: {bars[tone, p]['beat_this']:.2f}/{bars[tone, p]['4/4 default reg']:.2f}"
            f"/{bars[tone, p]['best factor + offset reg']:.2f}{bars[tone, p]['best reg']}" for p in PERFORMANCES))
    json.dump({"notes": {"|".join(k): v for k, v in rows.items()},
               "bars": {f"{t}|{p}": {k: list(x) if isinstance(x, tuple) else x for k, x in v.items()}
                        for (t, p), v in bars.items()}},
              open(os.path.join(TRACKS_DIR, "notation.json"), "w"))


if __name__ == "__main__":
    main()
