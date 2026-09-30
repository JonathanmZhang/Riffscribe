"""Quantization, measurement only (nothing in the pipeline uses it): snap
each note's start to a 16th-note position within its beat, and its length
to the nearest note value, then score against the Guitar Pro rhythms.
Runs in the worker image after `rhythm_benchmark truth` and track.py:
    python /rhythm/quantize.py

Quantizer, for a beat grid b_0 < b_1 < ... (extrapolated past both ends):
  start   j = the beat interval the onset falls in, s = round(4 * position
          within it); the note is at beat j + s/4 (s = 4 is the next beat).
  length  (end - start) in beats of that interval, snapped to the nearest of
          16th, 8th, dotted 8th, quarter, dotted quarter, half, whole (1/4 ...
          4 beats), nearest by ratio (log scale).

Scoring, per matched note (a tab note of the same pitch within 50ms of a
score note's JAMS onset; the score note gives the true position q in
quarters and the notated length, tied notes as one):
  start right   the quantized time, read on the true grid, is q (within
                half a 16th; with the true beats it's exact).
  length right  the snapped value, in true quarters (value x the beat's
                length on the true grid), is the notated length within 10%.
                With half-tempo beats a "quarter" is a true half note.

Rows:
  truth notes / true beats      the JAMS onsets and ends themselves: how
                                well a 16th grid + 7 note values can represent
                                this playing at all (no detection, no beat
                                tracking).
  tab notes / true beats        the pipeline's notes: quantizer + detection.
  tab notes / beat_this beats   + beat-tracking errors.
Lengths are also scored from the time to the next onset in the tab (next
tab note starting 50ms+ later), a diagnostic: how notation usually reads.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.chord_stages import PIPELINE, detect_file, run_stages  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, PERFORMANCES, SEGMENT_TYPES, TONES, audio_path  # noqa: E402
from scripts.rhythm_benchmark import TRACKS_DIR, TRUTH_JSON, tempo_octave  # noqa: E402

VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]  # in beats
MATCH_S = 0.05
NEXT_ONSET_S = 0.05


def quantize(t: float, beats: np.ndarray) -> tuple[float, float]:
    """(quantized time, length of the beat it's in, in seconds)."""
    j = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
    ibi = beats[j + 1] - beats[j]
    return beats[j] + round((t - beats[j]) / ibi * 4) / 4 * ibi, ibi


def snap_length(seconds: float, ibi: float) -> float:
    beats = max(seconds, 1e-3) / ibi
    return min(VALUES, key=lambda v: abs(np.log(beats / v)))


def match(tab: list[dict], truth: list[dict]) -> list[tuple[int, int]]:
    pairs = sorted((abs(n["start"] - t["onset"]), i, j) for i, n in enumerate(tab) for j, t in enumerate(truth)
                   if n["midi"] == t["midi"] and abs(n["start"] - t["onset"]) <= MATCH_S)
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def tab_notes(tone: str, p: str) -> list[dict]:
    """The pipeline's tab notes (kept and mapped), sorted by start."""
    with tempfile.TemporaryDirectory() as workdir:
        events, activations, _ = detect_file(audio_path(p, tone), workdir)
    stages = run_stages(events, PIPELINE, activations)
    return sorted(({"start": n["start_time"], "end": n["end_time"], "midi": n["midi"]} for n in stages["kept"]
                   if (n["start_time"], n["pitch"]) in stages["mapping"]["positions"]), key=lambda n: n["start"])


def with_next_onset(notes: list[dict]) -> list[dict]:
    starts = np.array(sorted(n["start"] for n in notes))
    for n in notes:
        later = starts[starts >= n["start"] + NEXT_ONSET_S]
        n["next"] = float(later[0]) if len(later) else n["end"]
    return notes


def score(notes: list[dict], truth: list[dict], pairs, beats: np.ndarray, grid: dict) -> list[dict]:
    """One row per matched note: kind (segment), start/length right."""
    out = []
    for i, j in pairs:
        n, t = notes[i], truth[j]
        tau, ibi = quantize(n["start"], beats)
        q_est = (tau - grid["offset"]) / grid["spq"]
        beat_q = ibi / grid["spq"]  # this beat's length in true quarters
        ratio = lambda secs: snap_length(secs, ibi) * beat_q / t["dur_q"]  # noqa: E731
        out.append({"kind": t["kind"], "start": bool(abs(q_est - t["q"]) < 0.125), "q_err": float(q_est - t["q"]),
                    "length_ratio": float(ratio(n["end"] - n["start"])),
                    "length": bool(abs(ratio(n["end"] - n["start"]) - 1) < 0.1),
                    "length_next": bool(abs(ratio(n["next"] - n["start"]) - 1) < 0.1)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    truth_all = json.load(open(TRUTH_JSON))
    segments = {perf["performance"]: perf["segments"] for perf in json.load(open(BENCHMARK_JSON))["performances"]}

    rows: dict[tuple, list[dict]] = {}
    coverage: Counter = Counter()
    octave_of: dict = {}
    for p in PERFORMANCES:
        tr = truth_all[p]
        grid = tr["grid"]
        kind = lambda t: next((s["type"] for s in segments[p] if s["start"] <= t < s["end"]),  # noqa: E731
                              segments[p][-1]["type"])
        truth = [{**n, "kind": kind(n["onset"])} for n in tr["notes"] if "onset" in n]
        n_q = int(np.ceil(max(n["end"] for n in truth) / grid["spq"])) + 8
        true_beats = grid["offset"] + np.arange(-8, n_q) * grid["spq"]

        jams = with_next_onset([{"start": n["onset"], "end": n["end"], "midi": n["midi"]} for n in truth])
        rows["truth notes / true beats", "clean"] = rows.get(("truth notes / true beats", "clean"), []) + score(
            jams, truth, [(k, k) for k in range(len(truth))], true_beats, grid)
        for tone in TONES:
            notes = with_next_onset(tab_notes(tone, p))
            pairs = match(notes, truth)
            coverage[tone, "matched"] += len(pairs)
            coverage[tone, "truth"] += len(truth)
            bt = np.array(json.load(open(os.path.join(TRACKS_DIR, "beat_this", f"{tone}_{p}.json")))["beats"])
            octave = tempo_octave(60 / float(np.median(np.diff(bt))), tr["fitted_bpm"])
            octave_of[tone, p] = octave
            for name, beats in (("tab notes / true beats", true_beats), ("tab notes / beat_this beats", bt)):
                scored = score(notes, truth, pairs, beats, grid)
                for r in scored:
                    r["octave"] = octave
                rows[name, tone] = rows.get((name, tone), []) + scored

    def pct(rs, key):
        return f"{np.mean([r[key] for r in rs]):6.1%}" if rs else "    - "

    print("% of matched notes right: start (16th position) | length from note end | length to next onset  [notes]")
    print(f"{'':<30}{'tone':<10}" + "".join(f"{k:<28}" for k in ["all"] + SEGMENT_TYPES))
    for (name, tone), rs in rows.items():
        cells = []
        for k in ["all"] + SEGMENT_TYPES:
            sub = [r for r in rs if k == "all" or r["kind"] == k]
            cells.append(f"{pct(sub, 'start')} {pct(sub, 'length')} {pct(sub, 'length_next')} [{len(sub)}]".ljust(28))
        print(f"{name:<30}{tone:<10}" + "".join(cells))

    print("\nbeat_this beats, by the tempo its beats come out at (right / half):")
    for tone in TONES:
        rs = rows["tab notes / beat_this beats", tone]
        print(f"  {tone:<9} " + "  ".join(
            f"{o}: start {pct([r for r in rs if r['octave'] == o], 'start')}, "
            f"length {pct([r for r in rs if r['octave'] == o], 'length')} "
            f"[{sum(r['octave'] == o for r in rs)} notes, "
            f"{sum(octave_of[tone, p] == o for p in PERFORMANCES)} perf]"
            for o in ("right", "half", "double", "other") if any(r["octave"] == o for r in rs)))

    print("\nstart errors, in 16ths (quantized - true), share of all matched notes:")
    for (name, tone), rs in rows.items():
        errs = Counter(int(round(r["q_err"] * 4)) for r in rs)
        top = ", ".join(f"{k:+d}: {v / len(rs):.0%}" for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:6])
        print(f"  {name:<30}{tone:<10}{top}")

    print("\nlength from note end, snapped vs notated: too short / right / too long")
    for (name, tone), rs in rows.items():
        for k in ["all"] + SEGMENT_TYPES:
            sub = [r["length_ratio"] for r in rs if k == "all" or r["kind"] == k]
            if sub:
                print(f"  {name:<30}{tone:<10}{k:<12} {np.mean([x < 0.9 for x in sub]):4.0%} / "
                      f"{np.mean([abs(x - 1) < 0.1 for x in sub]):4.0%} / {np.mean([x > 1.1 for x in sub]):4.0%}")

    print("\ncoverage (tab notes matched to a score note with a JAMS onset): " + ", ".join(
        f"{t} {coverage[t, 'matched']}/{coverage[t, 'truth']}" for t in TONES))

    # How the scores are written: are 16ths enough?
    notes = [n for p in PERFORMANCES for n in truth_all[p]["notes"]]
    off_grid = [n for n in notes if abs(n["q"] * 4 - round(n["q"] * 4)) > 1e-6]
    unrepresentable = Counter(n["dur_q"] for n in notes if not any(abs(n["dur_q"] - v) < 1e-6 for v in VALUES))
    per_perf = {p: sum(n["tuplet"] for n in truth_all[p]["notes"]) for p in PERFORMANCES}
    print(f"\nGP scores: {len(notes)} note onsets; in tuplets (triplets) {sum(n['tuplet'] for n in notes)} "
          f"({sum(n['tuplet'] for n in notes) / len(notes):.1%}); per performance "
          + ", ".join(f"{p}: {c}" for p, c in per_perf.items() if c))
    print(f"  onsets off the 16th grid: {len(off_grid)} ({len(off_grid) / len(notes):.1%})")
    print(f"  notated lengths outside the 7 values: {sum(unrepresentable.values())} "
          f"({sum(unrepresentable.values()) / len(notes):.1%}): "
          + ", ".join(f"{k:g}q x{v}" for k, v in unrepresentable.most_common()))
    print("  notated lengths: " + ", ".join(f"{k:g}q x{v}" for k, v in sorted(Counter(n["dur_q"] for n in notes).items())))
    json.dump({f"{n}|{t}": rs for (n, t), rs in rows.items()}, open(os.path.join(TRACKS_DIR, "quantize.json"), "w"))


if __name__ == "__main__":
    main()
