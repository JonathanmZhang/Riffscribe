"""Reposition-only measurement: move each chord step's DETECTED notes into a
playable voicing of the chord BTC names there, without adding or removing
any note. Measurement only - the pipeline is untouched. Runs in the
chord-names image after btc_labels.py:
    python /chord_names/reposition.py

For every step of today's tab (Basic Pitch -> select_notes -> mapper) with
>= 2 notes:
  1. BTC's label covering most of the 0.3s after the step's first onset.
  2. If there is no chord, or any detected pitch class isn't in the chord,
     the step is left unchanged.
  3. Otherwise, among the chord's playable voicings (voicings() below, the
     generator from experiment/chord-recognition) that contain EVERY
     detected pitch, pick the one whose frets for those notes are nearest
     their current frets, and move the notes to its strings/frets. None
     found: unchanged.
"oracle" does the same with EGSet12's ground-truth chord name (when a
ground-truth column starts within 80ms), as a ceiling for the naming.

Scores (scripts/egset12_benchmark matching, whole performance, attributed
to segments): pitch recall/precision - asserted identical to today's -
position agreement among pitch-matched notes, and tab F1, where a note
only counts as right with the right pitch AND string/fret.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(__file__))

import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from functools import lru_cache  # noqa: E402
from itertools import product  # noqa: E402

from chordlib import NO_CHORD, Chord, from_harte, from_midis, label_at  # noqa: E402
from scripts.chord_stages import PIPELINE, detect_file, run_stages  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, SEGMENT_TYPES, TONES, _match, audio_path, load_truth  # noqa: E402
from scripts.egset12_benchmark import evaluate as evaluate_bp  # noqa: E402

LABELS = "/app/data/_chord_names/btc"
OUT = "/app/data/_chord_names/reposition.json"
OPEN_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
MAX_FRET = 15
STEP_MATCH_S = 0.08


@lru_cache(maxsize=None)
def voicings(root: int, pcs: frozenset, bass: int) -> tuple:
    """Playable voicings of a chord: contiguous strings from a low string up
    to high e (4-6 strings; 2-3 from the root for power chords), bass on the
    lowest string, all chord tones present (the fifth may be left out of
    4+ tone chords), fretted notes within 3 frets, at most 4 fingers with
    the lowest fret barred, open strings only in open position (frets <= 4).
    Each voicing is a tuple of (string, fret)."""
    fifth = (root + 7) % 12
    required = pcs - {fifth} if len(pcs) >= 4 else pcs
    power = {(p - root) % 12 for p in pcs} <= {0, 7}
    layouts = [[low, low - 1, low - 2][:n] for low in (6, 5, 4) for n in (2, 3)] if power else \
        [list(range(low, 0, -1)) for low in (6, 5, 4)]
    found = set()
    for strings in layouts:
        if len(strings) < (2 if power else 4):
            continue
        for base in range(0, MAX_FRET - 2):
            options = []
            for s in strings:
                frets = sorted({f for f in [0, *range(max(base, 1), base + 4)] if f <= MAX_FRET
                                and (OPEN_MIDI[s] + f) % 12 in pcs})
                options.append(frets)
            for combo in product(*options):
                if (OPEN_MIDI[strings[0]] + combo[0]) % 12 != bass:
                    continue
                if not required <= {(OPEN_MIDI[s] + f) % 12 for s, f in zip(strings, combo)}:
                    continue
                fretted = [f for f in combo if f > 0]
                if fretted:
                    low, high = min(fretted), max(fretted)
                    if high - low > 3 or (0 in combo and high > 4):
                        continue
                    at_low = fretted.count(low)
                    if (len(fretted) - at_low) + 1 > 4:
                        continue
                found.add(tuple(zip(strings, combo)))
    return tuple(sorted(found))


def tab_steps(stages: dict) -> list[dict]:
    positions = stages["mapping"]["positions"]
    steps = []
    for group in stages["groups"]:
        notes = [{"onset": n["start_time"], "midi": n["midi"], "string": positions[(n["start_time"], n["pitch"])][0],
                  "fret": positions[(n["start_time"], n["pitch"])][1]}
                 for n in group if (n["start_time"], n["pitch"]) in positions]
        if notes:
            steps.append({"t": min(n["onset"] for n in notes), "notes": notes})
    return steps


def reposition(step: dict, chord: Chord) -> tuple[list[dict], str]:
    """Returns (notes, outcome); outcome is one of: single, no_chord,
    not_in_chord, no_voicing, same, moved."""
    notes = step["notes"]
    if len(notes) < 2:
        return notes, "single"
    if chord.root is None:
        return notes, "no_chord"
    if any(n["midi"] % 12 not in chord.pcs for n in notes):
        return notes, "not_in_chord"
    detected = [n["midi"] for n in notes]
    best, best_key = None, None
    for v in voicings(chord.root, chord.pcs, chord.bass if chord.bass is not None else chord.root):
        by_midi = {OPEN_MIDI[s] + f: (s, f) for s, f in v}
        if not all(m in by_midi for m in detected) or len(set(detected)) < len(detected):
            continue
        key = (sum(abs(by_midi[n["midi"]][1] - n["fret"]) for n in notes), max(f for _, f in v))
        if best_key is None or key < best_key:
            best, best_key = by_midi, key
    if best is None:
        return notes, "no_voicing"
    moved = [{**n, "string": best[n["midi"]][0], "fret": best[n["midi"]][1]} for n in notes]
    same = all((a["string"], a["fret"]) == (b["string"], b["fret"]) for a, b in zip(notes, moved))
    return moved, "same" if same else "moved"


def main() -> None:
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    benchmark = json.load(open(BENCHMARK_JSON))
    counts: dict = {}
    outcomes: dict = {}
    flips: dict = {}

    def bucket(system, tone, kind):
        return counts.setdefault((system, tone, kind), {"truth": 0, "tab": 0, "matched": 0, "position": 0})

    for perf in benchmark["performances"]:
        p = perf["performance"]
        truth, _ = load_truth(p)
        gt = [(c["time"], from_midis(c["notes"])) for s in perf["segments"] for c in s["ground_truth_chords"]]

        def seg_type(t):
            for s in perf["segments"]:
                if s["start"] <= t < s["end"]:
                    return s["type"]
            return perf["segments"][-1]["type"]

        for tone in TONES:
            with tempfile.TemporaryDirectory() as workdir:
                events, activations, _ = detect_file(audio_path(p, tone), workdir)
            steps = tab_steps(run_stages(events, PIPELINE, activations))
            labels = json.load(open(os.path.join(LABELS, f"egset12_{tone}_{p}.json")))["segments"]

            def oracle(t):
                near = [(abs(gt_t - t), ch) for gt_t, ch in gt if abs(gt_t - t) <= STEP_MATCH_S]
                return min(near, key=lambda x: x[0])[1] if near else NO_CHORD

            systems = {"today": [n for s in steps for n in s["notes"]]}
            for name, namer in (("btc", lambda t: from_harte(label_at(labels, t))), ("oracle", oracle)):
                out = []
                for step in steps:
                    notes, outcome = reposition(step, namer(step["t"]))
                    out += notes
                    o = outcomes.setdefault(f"{name}|{tone}", {})
                    o[outcome] = o.get(outcome, 0) + 1
                    if outcome == "moved":
                        o["notes_moved"] = o.get("notes_moved", 0) + sum(
                            (a["string"], a["fret"]) != (b["string"], b["fret"]) for a, b in zip(step["notes"], notes))
                systems[name] = out

            # Per matched note: did repositioning fix or break its position?
            # (Same notes in the same order in every system, so the matches agree.)
            base_matches = _match(truth, systems["today"])
            for system in ("btc", "oracle"):
                f = flips.setdefault(f"{system}|{tone}", {"fixed": 0, "broken": 0, "wrong_to_wrong": 0})
                for i, j in base_matches.items():
                    t, a, b = truth[i], systems["today"][j], systems[system][j]
                    if (a["string"], a["fret"]) == (b["string"], b["fret"]):
                        continue
                    ok_a = (a["string"], a["fret"]) == (t["string"], t["fret"])
                    ok_b = (b["string"], b["fret"]) == (t["string"], t["fret"])
                    f["fixed" if ok_b else "broken" if ok_a else "wrong_to_wrong"] += 1

            for system, notes in systems.items():
                matches = _match(truth, notes)
                for i, t in enumerate(truth):
                    for kind in (seg_type(t["onset"]), "all"):
                        b = bucket(system, tone, kind)
                        b["truth"] += 1
                        if i in matches:
                            b["matched"] += 1
                            n = notes[matches[i]]
                            b["position"] += (n["string"], n["fret"]) == (t["string"], t["fret"])
                for n in notes:
                    for kind in (seg_type(n["onset"]), "all"):
                        bucket(system, tone, kind)["tab"] += 1

    reference = evaluate_bp(PIPELINE, benchmark)
    result = {"scores": {}, "outcomes": outcomes, "position_changes": flips}
    for (system, tone, kind), b in counts.items():
        if system == "today":
            r = reference[tone][kind]
            assert (b["matched"], b["tab"], b["truth"], b["position"]) == (r["matched"], r["tab"], r["truth"], r["position"])
        else:  # no notes added or removed: pitch scores must equal today's
            t = counts[("today", tone, kind)]
            assert (b["matched"], b["tab"], b["truth"]) == (t["matched"], t["tab"], t["truth"]), (system, tone, kind)
        recall, precision = b["matched"] / b["truth"], b["matched"] / b["tab"]
        pos_r, pos_p = b["position"] / b["truth"], b["position"] / b["tab"]
        result["scores"][f"{system}|{tone}|{kind}"] = {
            **b, "recall": recall, "precision": precision, "pitch_f1": 2 * recall * precision / (recall + precision),
            "position": b["position"] / b["matched"], "tab_f1": 2 * pos_r * pos_p / (pos_r + pos_p) if b["position"] else 0.0,
        }
    json.dump(result, open(OUT, "w"), indent=1)

    print("tone      type          pitch R/P (all systems)   pitch F1 | position agreement: today  btc  oracle | "
          "tab F1 (pitch+position): today  btc  oracle")
    for tone in TONES:
        for kind in ["all"] + SEGMENT_TYPES:
            s = {k: result["scores"][f"{k}|{tone}|{kind}"] for k in ("today", "btc", "oracle")}
            print(f"{tone:<9} {kind:<12} {s['today']['recall']:6.1%} / {s['today']['precision']:6.1%}   "
                  f"{s['today']['pitch_f1'] * 100:5.1f}   | " + "  ".join(f"{s[k]['position']:6.1%}" for k in s) +
                  "  | " + "  ".join(f"{s[k]['tab_f1'] * 100:5.1f}" for k in s) + f"   n={s['today']['truth']}")
    print("\nStep outcomes:")
    for k, v in sorted(outcomes.items()):
        print(f"  {k:<16} {v}")
    print("\nRepositioned notes that match a true note: fixed / broken / wrong before and after")
    for k, v in sorted(flips.items()):
        print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
