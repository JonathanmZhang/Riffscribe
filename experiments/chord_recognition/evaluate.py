"""Chord-recognition experiment: chord-name accuracy (part 2) and chord-shape
completion (part 3). Measurement only - the pipeline is untouched.
Runs in the chords-bench image after recognize.py:
    python /chords/evaluate.py

Part 2 scores each namer against EGSet12's 203 ground-truth chord columns
(benchmark.json, named by egset12_benchmark.chord_name from the annotated
notes). 76 are clusters no template names; root/majmin/exact are scored
on the 127 named ones. A model's label for a column is the one covering
most of the 0.3s after the column onset (chordlib.label_at).

Namers: chordino, crema, btc (recognize.py output), and "bp-notes": the
same template matcher that named the ground truth, applied to the notes of
the pipeline tab step (Basic Pitch -> select_notes -> mapper) that starts
nearest the column (within 80ms).

Part 3 completes chord steps of today's tab (steps with >= 2 mapped notes)
with a generated voicing of the namer's chord: the playable voicing that
contains the most detected pitches (>= 2), nearest the current position.
Detected notes in the voicing move to its strings; missing voicing notes
are added as inferred at the step's onset; a detected note outside the
voicing keeps its place and wins its string. Guards: "any" (complete
whenever >= 2 notes fit) and "agree" (also require every detected note's
pitch class to be in the chord). "oracle" uses the ground-truth chord name
(when a ground-truth column starts within 80ms), as an upper bound.
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

import pretty_midi  # noqa: E402

from chordlib import NO_CHORD, Chord, from_chordino, from_harte, from_midis, label_at, score  # noqa: E402
from scripts.chord_stages import PIPELINE, detect_file, run_stages  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, SEGMENT_TYPES, TONES, _match, audio_path, load_truth  # noqa: E402
from scripts.regression_check import FIREFIRE, FIREFIRE_WINDOWS  # noqa: E402

CHORDS = "/app/data/_chords"
MODELS = {"chordino": from_chordino, "crema": from_harte, "btc": from_harte}
NAMERS = list(MODELS) + ["bp-notes"]
STEP_MATCH_S = 0.08
OPEN_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
MAX_FRET = 15


# ------------------------------------------------------------------ helpers


def load_segments(model: str, name: str) -> dict:
    return json.load(open(os.path.join(CHORDS, model, name + ".json")))


def model_chord(model: str, segments: list[dict], t: float) -> Chord:
    return MODELS[model](label_at(segments, t))


def tab_steps(stages: dict) -> list[dict]:
    """The pipeline's steps as they end up in the tab: anchor time and the
    mapped notes (onset, midi, string, fret)."""
    positions = stages["mapping"]["positions"]
    steps = []
    for group in stages["groups"]:
        notes = [{"onset": n["start_time"], "midi": n["midi"], "string": positions[(n["start_time"], n["pitch"])][0],
                  "fret": positions[(n["start_time"], n["pitch"])][1], "inferred": False}
                 for n in group if (n["start_time"], n["pitch"]) in positions]
        if notes:
            steps.append({"t": min(n["onset"] for n in notes), "notes": notes})
    return steps


def nearest_step(steps: list[dict], t: float) -> dict | None:
    best = min(steps, key=lambda s: abs(s["t"] - t), default=None)
    return best if best is not None and abs(best["t"] - t) <= STEP_MATCH_S else None


# ----------------------------------------------------------------- voicings


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


def complete_step(step: dict, chord: Chord, guard: str) -> tuple[list[dict], bool]:
    """Returns (notes, completed?)."""
    notes = step["notes"]
    if chord.root is None or len(notes) < 2:
        return notes, False
    if guard == "agree" and any(n["midi"] % 12 not in chord.pcs for n in notes):
        return notes, False
    bass = chord.bass if chord.bass is not None else chord.root
    detected = {n["midi"] for n in notes}
    ref = [n["fret"] for n in notes if n["fret"] > 0]
    ref_fret = sum(ref) / len(ref) if ref else 0.0
    best, best_key = None, None
    for v in voicings(chord.root, chord.pcs, bass):
        midis = {OPEN_MIDI[s] + f for s, f in v}
        hits = len(detected & midis)
        if hits < 2:
            continue
        frets = [f for _, f in v if f > 0]
        key = (-hits, abs((sum(frets) / len(frets) if frets else 0.0) - ref_fret), len(v))
        if best_key is None or key < best_key:
            best, best_key = v, key
    if best is None:
        return notes, False
    by_midi = {OPEN_MIDI[s] + f: (s, f) for s, f in best}
    out, used_strings = [], set()
    for n in notes:  # detected notes in the voicing take its positions
        if n["midi"] in by_midi:
            s, f = by_midi[n["midi"]]
            out.append({**n, "string": s, "fret": f})
            used_strings.add(s)
    for n in notes:  # detected notes outside it keep theirs, and their string
        if n["midi"] not in by_midi and n["string"] not in used_strings:
            out.append(n)
            used_strings.add(n["string"])
    for midi, (s, f) in by_midi.items():
        if midi not in detected and s not in used_strings:
            out.append({"onset": step["t"], "midi": midi, "string": s, "fret": f, "inferred": True})
            used_strings.add(s)
    return out, True


# ------------------------------------------------------------------ scoring


def main() -> None:
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    benchmark = json.load(open(BENCHMARK_JSON))
    names: dict = {}      # (namer, tone) -> list of per-column scores
    examples: list = []   # a few wrong names, for the report
    counts: dict = {}     # (system, tone, kind) -> truth/tab/matched/position/inferred...
    completion_stats: dict = {}

    def bucket(system, tone, kind):
        return counts.setdefault((system, tone, kind), {"truth": 0, "tab": 0, "matched": 0, "position": 0,
                                                        "inferred": 0, "inferred_matched": 0})

    for perf in benchmark["performances"]:
        p = perf["performance"]
        truth, _ = load_truth(p)
        columns = [c for s in perf["segments"] for c in s["ground_truth_chords"]]
        # Ground truth chords, rebuilt from the column notes (must equal the stored names).
        gt = []
        for c in columns:
            chord = from_midis(c["notes"])
            assert chord.label == c["name"] or chord.root is None, (chord.label, c["name"])
            gt.append((c["time"], chord))

        def seg_type(t):
            for s in perf["segments"]:
                if s["start"] <= t < s["end"]:
                    return s["type"]
            return perf["segments"][-1]["type"]

        for tone in TONES:
            name = f"egset12_{tone}_{p}"
            with tempfile.TemporaryDirectory() as workdir:
                events, activations, _ = detect_file(audio_path(p, tone), workdir)
            steps = tab_steps(run_stages(events, PIPELINE, activations))
            segs = {m: load_segments(m, name)["segments"] for m in MODELS}

            def namer_chord(namer, t):
                if namer == "bp-notes":
                    step = nearest_step(steps, t)
                    return from_midis([n["midi"] for n in step["notes"]]) if step else NO_CHORD
                if namer == "oracle":
                    near = [(abs(gt_t - t), ch) for gt_t, ch in gt if abs(gt_t - t) <= STEP_MATCH_S]
                    return min(near, key=lambda x: x[0])[1] if near else NO_CHORD
                return model_chord(namer, segs[namer], t)

            # Part 2: names at the ground-truth columns.
            for t, chord in gt:
                for namer in NAMERS:
                    est = namer_chord(namer, t)
                    sc = score(chord, est)
                    names.setdefault((namer, tone), []).append({**sc, "named": chord.root is not None,
                                                               "est_is_chord": est.root is not None})
                    if tone == "clean" and chord.root is not None and not sc["root"] and len(examples) < 400:
                        examples.append((namer, p, round(t, 2), chord.label, est.label))

            # Part 3: completion of today's tab.
            systems = {"today": [n for s in steps for n in s["notes"]]}
            for namer in ("btc", "crema", "chordino", "oracle"):
                for guard in ("any", "agree"):
                    out, done, wrong = [], 0, 0
                    for step in steps:
                        chord = namer_chord(namer, step["t"])
                        notes, completed = complete_step(step, chord, guard)
                        out += notes
                        if completed:
                            done += 1
                            near = [(abs(gt_t - step["t"]), ch) for gt_t, ch in gt
                                    if abs(gt_t - step["t"]) <= STEP_MATCH_S and ch.root is not None]
                            if near and min(near, key=lambda x: x[0])[1].pcs != chord.pcs:
                                wrong += 1
                    systems[f"{namer}/{guard}"] = out
                    st = completion_stats.setdefault((f"{namer}/{guard}", tone), {"completed": 0, "wrong_name": 0})
                    st["completed"] += done
                    st["wrong_name"] += wrong

            for system, notes in systems.items():
                matches = _match(truth, notes)
                matched_tab = set(matches.values())
                for i, t in enumerate(truth):
                    for kind in (seg_type(t["onset"]), "all"):
                        b = bucket(system, tone, kind)
                        b["truth"] += 1
                        if i in matches:
                            b["matched"] += 1
                            n = notes[matches[i]]
                            b["position"] += (n["string"], n["fret"]) == (t["string"], t["fret"])
                for j, n in enumerate(notes):
                    for kind in (seg_type(n["onset"]), "all"):
                        b = bucket(system, tone, kind)
                        b["tab"] += 1
                        if n["inferred"]:
                            b["inferred"] += 1
                            b["inferred_matched"] += j in matched_tab

    # Sanity: "today" must reproduce the benchmark's own evaluate().
    from scripts.egset12_benchmark import evaluate as evaluate_bp
    reference = evaluate_bp(PIPELINE, benchmark)
    for tone in TONES:
        for kind, r in reference[tone].items():
            mine = counts[("today", tone, kind)]
            assert (mine["matched"], mine["tab"], mine["truth"], mine["position"]) == \
                (r["matched"], r["tab"], r["truth"], r["position"]), (tone, kind)

    result = {"names": {}, "completion": {}, "completion_stats": {}, "examples": examples}
    for (namer, tone), rows in names.items():
        named = [r for r in rows if r["named"]]
        mm = [r for r in named if r["majmin"] is not None]
        result["names"][f"{namer}|{tone}"] = {
            "columns": len(rows), "named": len(named), "majmin_eligible": len(mm),
            "root": sum(r["root"] for r in named) / len(named),
            "majmin": sum(r["majmin"] for r in mm) / len(mm),
            "exact": sum(r["exact"] for r in named) / len(named),
            "exact_bass": sum(r["exact_bass"] for r in named) / len(named),
            "clusters_called_chord": sum(r["est_is_chord"] for r in rows if not r["named"]),
        }
    for (system, tone, kind), b in counts.items():
        result["completion"][f"{system}|{tone}|{kind}"] = {
            **b, "recall": b["matched"] / b["truth"], "precision": b["matched"] / b["tab"] if b["tab"] else None,
            "position": b["position"] / b["matched"] if b["matched"] else None,
            "inferred_precision": b["inferred_matched"] / b["inferred"] if b["inferred"] else None,
        }
    result["completion_stats"] = {f"{k[0]}|{k[1]}": v for k, v in completion_stats.items()}
    result["firefire"] = firefire()
    result["timing"] = timing()
    json.dump(result, open(os.path.join(CHORDS, "results.json"), "w"), indent=1)
    report(result)


def firefire() -> dict:
    out = {}
    for source, separate in (("mix", False), ("stem", True)):
        with tempfile.TemporaryDirectory() as workdir:
            events, activations, _ = detect_file(FIREFIRE, workdir, separate=separate, seed=0)
        steps = tab_steps(run_stages(events, PIPELINE, activations))
        for start, end in FIREFIRE_WINDOWS:
            w = {}
            for m in MODELS:
                segs = load_segments(m, f"firefire_{source}")["segments"]
                w[m] = [(round(max(s["start"], start), 2), s["label"]) for s in segs if s["end"] > start and s["start"] < end]
            w["bp-notes"] = [(round(s["t"], 2), from_midis([n["midi"] for n in s["notes"]]).label)
                             for s in steps if start <= s["t"] < end and len(s["notes"]) >= 3]
            out[f"{start:g}-{end:g} {source}"] = w
    return out


def timing() -> dict:
    t = {}
    for m in MODELS:
        runs = [json.load(open(os.path.join(CHORDS, m, f))) for f in os.listdir(os.path.join(CHORDS, m))]
        t[m] = round(sum(r["seconds"] for r in runs) / sum(r["audio_seconds"] for r in runs), 4)
    return t


def report(r: dict) -> None:
    print("Part 2 - chord names at EGSet12's ground-truth columns (named columns only)")
    print(f"  {'namer':<10}{'tone':<10}{'root':>7}{'majmin':>8}{'exact':>7}{'+bass':>7}   n (majmin n)  clusters called a chord")
    for namer in NAMERS:
        for tone in TONES:
            x = r["names"][f"{namer}|{tone}"]
            print(f"  {namer:<10}{tone:<10}{x['root']:>7.1%}{x['majmin']:>8.1%}{x['exact']:>7.1%}{x['exact_bass']:>7.1%}"
                  f"   {x['named']} ({x['majmin_eligible']})  {x['clusters_called_chord']}/{x['columns'] - x['named']}")
    print("\nPart 3 - tab after completion: recall / precision [position] | inferred notes: count, precision")
    systems = ["today"] + [f"{n}/{g}" for n in ("btc", "crema", "chordino", "oracle") for g in ("any", "agree")]
    for tone in TONES:
        for kind in ["all", "chords"]:
            print(f"  {tone} {kind}")
            for s in systems:
                x = r["completion"][f"{s}|{tone}|{kind}"]
                st = r["completion_stats"].get(f"{s}|{tone}", {})
                inf = f"{x['inferred']:>4}, {x['inferred_precision']:.1%}" if x["inferred"] else "   0"
                extra = f"  steps completed {st['completed']}, wrong name {st['wrong_name']}" if st and kind == "all" else ""
                print(f"    {s:<16}{x['recall']:6.1%} / {x['precision']:6.1%} [{x['position']:5.1%}] | {inf}{extra}")
    print("\nFirefire windows")
    for w, models in r["firefire"].items():
        print(f"  {w}")
        for m, labels in models.items():
            print(f"    {m:<9} " + "  ".join(f"{t}:{lab}" for t, lab in labels))
    print("\nSeconds per second of audio:", r["timing"])


if __name__ == "__main__":
    main()
