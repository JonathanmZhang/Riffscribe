"""Technique cleanup, measured: do the two merges in tasks/techniques.py
remove wrong notes where vibrato and bends/slides happen (IDMT-SMT-Guitar,
real and labelled), without removing right notes where they don't (EGSet12,
which has no vibrato or bends at all)?

  vibrato merge   a run of joined same-pitch notes with vibrato -> one note
                  ("runs": the whole run; "wobble": only across the joins
                  that lie inside the wobble, so a note struck again before
                  the vibrato starts stays separate)
  glide merge     a note the pitch glides into -> part of the note it left

Runs in the worker image (repo at /repo for the stale-image guard, this
folder at /x, experiments/expression at /expression for its IDMT loader
and model-output cache):

    python /x/cleanup.py idmt
    python /x/cleanup.py egset12

The notes are the pipeline's: Basic Pitch note creation, select_notes, then
transcribe.clean_notes with the merges switched per configuration. Two
sources for the pitch offsets are compared on IDMT: Basic Pitch's integer
pitch_bends (what the pipeline has) and the finer contour of
experiments/expression/bends.py.

Scoring, as in egset12_benchmark: a tab note is right if a truth note of the
same pitch starts within 50ms of it, one to one. "Wrong notes" are the tab
notes that match no truth note. For every note a merge removes, the audit
says whether it was a right note before the merge (a real note lost) or a
wrong one (an extra note removed). On IDMT the truth pitch of a bend or
slide is the fretted (starting) pitch, and a note re-struck by vibrato
matches nothing, so both merges remove wrong notes when they work.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")
sys.path.insert(0, "/expression")


import bends as expression  # noqa: E402  (experiments/expression/bends.py)
from scripts import build_info  # noqa: E402
from scripts.chord_stages import StageConfig  # noqa: E402
from scripts.egset12_benchmark import BENCHMARK_JSON, TONES, _match, audio_path, load_truth  # noqa: E402
from scripts.egset12_benchmark import evaluate as evaluate_egset12  # noqa: E402
from tasks import techniques, transcribe  # noqa: E402

# name -> the cleanup applied to the kept notes. "runs" merges a whole run
# with vibrato; "wobble" only across joins that lie inside the wobble (at the
# detector's 12 cents, and at lower thresholds to show the sensitivity).
CONFIGS = {
    "off": lambda kept: kept,
    "vibrato, runs": lambda kept: techniques.cleanup(kept, 1, False),
    "vibrato, wobble": lambda kept: techniques.cleanup(kept, 2, False),
    "  wobble >= 9": lambda kept: techniques.merge_vibrato(kept, "wobble", wobble_threshold=9.0),
    "  wobble >= 6": lambda kept: techniques.merge_vibrato(kept, "wobble", wobble_threshold=6.0),
    "glide merge": lambda kept: techniques.cleanup(kept, 0, True),
}
# The same, as StageConfig settings, for the benchmark through the mapper.
STAGES = {"off": (0, 0), "vibrato, runs": (1, 0), "vibrato, wobble": (2, 0), "glide merge": (0, 1)}
VIBRATO_CONFIGS = [name for name in CONFIGS if "vibrato" in name or "wobble" in name]


def kept_notes(output: dict, fine: bool) -> list[dict]:
    """The pipeline's kept notes for a model output, with their pitch
    offsets. fine=False: only Basic Pitch's integer pitch_bends, as in the
    pipeline. fine=True: also the finer contour, which techniques prefers."""
    events = [e for e in expression.tab_notes(output) if e["kept"]]
    pipeline = [e for e in transcribe.select_notes(transcribe.notes_from_model_output(output)) if e["kept"]]
    assert [(e["midi"], e["start_time"], e["bends"]) for e in sorted(events, key=lambda e: (e["start_time"], e["midi"]))] \
        == [(e["midi"], e["start_time"], e["bends"]) for e in sorted(pipeline, key=lambda e: (e["start_time"], e["midi"]))]
    if not fine:
        events = [{k: v for k, v in e.items() if k != "cents"} for e in events]
    return events


def key(note: dict) -> tuple:
    return (note["start_time"], note["midi"])


def score(truth: list[dict], notes: list[dict]) -> tuple[dict, set]:
    """({truth, tab, right}, keys of the right tab notes)."""
    tab = [{"onset": n["start_time"], "midi": n["midi"]} for n in notes]
    matches = _match(truth, tab)
    return {"truth": len(truth), "tab": len(tab), "right": len(matches)}, {key(notes[j]) for j in matches.values()}


def style_at(truths: list[dict], note: dict) -> str:
    """The labelled style of the truth note a tab note lies on (most
    overlap, within the scoring pitch window), or "no truth note"."""
    best, label = 0.0, "no truth note"
    for t in truths:
        overlap = min(note["end_time"], t["end"]) - max(note["start_time"], t["onset"])
        if overlap > best and expression.PITCH_WINDOW[0] <= note["midi"] - t["midi"] <= expression.PITCH_WINDOW[1]:
            best, label = overlap, t.get("style", "note")
    return label


def line(name: str, c: dict, base: dict | None = None) -> str:
    wrong = c["tab"] - c["right"]
    text = (f"  {name:<16} tab notes {c['tab']:>5}  right {c['right']:>5}  wrong {wrong:>5}  "
            f"recall {100 * c['right'] / c['truth']:5.1f}%  precision {100 * c['right'] / c['tab']:5.1f}%")
    if base is not None:
        text += f"   | wrong notes {wrong - (base['tab'] - base['right']):+d}, right notes {c['right'] - base['right']:+d}"
    return text


def run_idmt(args) -> dict:
    files = [(wav, notes) for wav, notes in expression.idmt_files(os.path.join(expression.DATA, "idmt"))]
    results = {}
    for fine in (False, True):
        source = "fine contour" if fine else "pitch_bends (the pipeline's source)"
        totals = {name: Counter() for name in CONFIGS}
        audits = {name: defaultdict(Counter) for name in CONFIGS}
        shown = {name: defaultdict(Counter) for name in CONFIGS}
        marked = {name: Counter() for name in CONFIGS}
        by_dataset = {name: defaultdict(Counter) for name in CONFIGS}
        for wav, truths in files:
            dataset = truths[0]["dataset"]
            kept = kept_notes(expression.model_output_for(wav), fine)
            truth = [{"onset": t["onset"], "midi": t["midi"]} for t in truths]
            _, right_before = score(truth, kept)
            for name, clean in CONFIGS.items():
                cleaned = clean(kept)
                counts, _ = score(truth, cleaned)
                totals[name].update(counts)
                by_dataset[name][dataset].update(counts)
                remaining = {key(n) for n in cleaned}
                for note in kept:
                    if key(note) not in remaining:
                        outcome = "real note lost" if key(note) in right_before else "extra note removed"
                        audits[name][style_at(truths, note)][outcome] += 1
                for note in cleaned:
                    if note.get("vibrato"):
                        marked[name][style_at(truths, note)] += 1
                rows = [{**n, "kept": True} for n in cleaned]
                for t in truths:
                    if dataset == "dataset2":
                        own = expression.own_notes(t, rows)
                        shows = " ".join(f"{e['midi'] - t['midi']:+d}".replace("+0", "0") for e in own)
                        shown[name][t["style"]][expression.shown(shows)] += 1
        print(f"\n== IDMT-SMT-Guitar, datasets 1 + 2 ({len(files)} files), pitch offsets from {source}")
        for name in CONFIGS:
            print(line(name, totals[name], None if name == "off" else totals["off"]))
        for dataset in ("dataset1", "dataset2"):
            print(f"  {dataset} only:")
            for name in CONFIGS:
                print("  " + line(name, by_dataset[name][dataset], None if name == "off" else by_dataset["off"][dataset]))
        print("  notes removed by a merge, by the labelled style of the truth note they lie on:")
        for name in list(CONFIGS)[1:]:
            removed = Counter()
            for style, c in sorted(audits[name].items()):
                removed.update(c)
            print(f"    {name}: {removed['extra note removed']} extra notes removed, {removed['real note lost']} real notes lost  "
                  + ", ".join(f"{style}: -{c['extra note removed']} extra / -{c['real note lost']} real"
                              for style, c in sorted(audits[name].items())))
        print("  notes marked vibrato, by the style of the truth note they lie on:")
        for name in VIBRATO_CONFIGS:
            print(f"    {name}: " + ", ".join(f"{style}: {n}" for style, n in sorted(marked[name].items())))
        print("  dataset 2, what the tab shows per truth note: " + " | ".join(expression.SHOWN))
        for style in ("vibrato", "bend", "slide", "normal", "harmonic", "dead note"):
            for name in CONFIGS:
                c = shown[name][style]
                n = sum(c.values())
                print(f"    {style:<10} {name:<16} {n:>5} | " + " | ".join(
                    f"{c[label]:>4} ({100 * c[label] / n:3.0f}%)" for label in expression.SHOWN))
        results[source] = {name: dict(totals[name]) for name in CONFIGS}
        results[source + " audit"] = {name: {s: dict(c) for s, c in audits[name].items()} for name in CONFIGS}
    return results


def run_egset12(args) -> dict:
    """EGSet12 has no vibrato, bends or slides, so every merge and every
    vibrato mark is a false alarm. The benchmark's own evaluation (through
    the fretboard mapper), then the same audit as on IDMT."""
    results = {}
    print("\n== EGSet12 benchmark (run_stages + mapper): pitch recall / precision / position, per tone")
    base = None
    for name, (vibrato_merge, glide_merge) in STAGES.items():
        result = evaluate_egset12(StageConfig(vibrato_merge=vibrato_merge, glide_merge=glide_merge))
        results[name] = result
        base = base or result
        cells = []
        for tone in TONES:
            r, b = result[tone]["all"], base[tone]["all"]
            cells.append(f"{tone} {100 * r['pitch_recall']:.1f} / {100 * r['pitch_precision']:.1f} / "
                         f"{100 * r['position_agreement']:.1f} (right {r['matched'] - b['matched']:+d}, "
                         f"tab notes {r['tab'] - b['tab']:+d})")
        print(f"  {name:<16} " + "   ".join(cells))
        for kind in ("chords", "single-note", "fast"):
            print(f"  {'':<16} {kind:<12} " + "   ".join(
                f"{tone} {100 * result[tone][kind]['pitch_recall']:.1f} / {100 * result[tone][kind]['pitch_precision']:.1f}"
                for tone in TONES))

    benchmark = json.load(open(BENCHMARK_JSON))
    print("\n== EGSet12, notes removed by a merge and notes marked vibrato (all are false alarms), per tone")
    audit = {}
    for tone in TONES:
        counts = {name: Counter() for name in CONFIGS}
        for perf in benchmark["performances"]:
            truth, _ = load_truth(perf["performance"])
            truth = [{"onset": t["onset"], "midi": t["midi"]} for t in truth]
            kept = kept_notes(expression.model_output_for(audio_path(perf["performance"], tone)), False)
            _, right_before = score(truth, kept)
            for name, clean in CONFIGS.items():
                cleaned = clean(kept)
                remaining = {key(n) for n in cleaned}
                for note in kept:
                    if key(note) not in remaining:
                        counts[name]["real note lost" if key(note) in right_before else "extra note removed"] += 1
                counts[name]["marked vibrato"] += sum(1 for n in cleaned if n.get("vibrato"))
                counts[name]["notes"] += len(kept)
        audit[tone] = {name: dict(c) for name, c in counts.items()}
        for name in list(CONFIGS)[1:]:
            c = counts[name]
            print(f"  {tone:<9} {name:<16} of {c['notes']} kept notes: {c['real note lost']} real notes lost, "
                  f"{c['extra note removed']} wrong notes removed, {c['marked vibrato']} marked vibrato")
    results["audit"] = audit
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["idmt", "egset12"])
    parser.add_argument("--save", help="write the numbers as JSON here")
    build_info.add_args(parser)
    args = parser.parse_args()
    build = build_info.guard(args.allow_stale)
    print(f"settings: glide >= {techniques.GLIDE_CENTS:.0f} cents over {techniques.GLIDE_FRAMES} frames; vibrato >= "
          f"{techniques.VIBRATO_CENTS:.0f} cents at {techniques.VIBRATO_BAND_HZ[0]:.0f}-{techniques.VIBRATO_BAND_HZ[1]:.0f} Hz; "
          f"joined = next note starts {techniques.JOIN_GAP_S[0] * 1000:.0f}..{techniques.JOIN_GAP_S[1] * 1000:.0f}ms from the end")
    results = {"idmt": run_idmt, "egset12": run_egset12}[args.mode](args)
    if args.save:
        with open(args.save, "w") as f:
            json.dump({"build": build, "mode": args.mode, "results": results}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
