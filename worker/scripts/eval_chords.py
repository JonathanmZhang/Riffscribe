"""Chord accuracy evaluation against the synthetic test set from
scripts/make_chord_testset.py.

Usage (inside the worker container):
    python -m scripts.eval_chords [--testset DIR] [--programs 27,29,30] [--separate]
                                  [--seed 0] [--out results.json] [--worst 3]

Runs each test WAV through the pipeline stages directly (no Celery), using
the same stage functions as scripts/inspect_chords.py, and follows every
ground-truth chord note through them. A lost note is attributed to the
first stage that lost it:
  (a) never detected by Basic Pitch at any amplitude
  (b) detected, but below the confidence threshold
  (c) kept, but split into a different step than its chord, or its chord's
      step merged with a neighbouring chord
  (d) dropped by the fretboard mapper
and extra notes are reported as (e): octave errors of a chord note, repeats
of a chord note, and other phantoms.

Per program it reports cumulative chord-note recall (detected -> kept ->
grouped correctly -> mapped), precision of the final tab, and the share of
chords that come out complete. Results are saved as JSON for before/after
comparisons.

A detected note is attributed to the chord whose window contains its onset:
from 60ms before the chord's first note to 60ms before the next chord's.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pretty_midi  # noqa: E402

from scripts.chord_stages import (  # noqa: E402
    CHORD_ONSET_TOLERANCE_SECONDS,
    PIPELINE,
    StageConfig,
    add_config_args,
    classify_chord,
    config_from_args,
    detect_file,
    run_stages,
)

PRE_ONSET_S = 0.06
TAIL_S = 0.5
EXTRA_KINDS = ("octave error", "repeat of a chord note", "phantom")


def _windows(chords: list[dict]) -> list[tuple[float, float]]:
    starts = [c["onset"] - PRE_ONSET_S for c in chords]
    ends = starts[1:] + [chords[-1]["end"] + TAIL_S]
    return list(zip(starts, ends))


def evaluate_program(truth: dict, wav_path: str, separate: bool = False, seed: int = 0,
                     config: StageConfig = PIPELINE) -> dict:
    start_time = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        events, activations, audio = detect_file(wav_path, workdir, separate=separate, seed=seed)

    stages = run_stages(events, config)
    kept, mapping = stages["kept"], stages["mapping"]
    chords = truth["chords"]
    windows = _windows(chords)

    def chord_of(note: dict) -> int | None:
        for i, (lo, hi) in enumerate(windows):
            if lo <= note["start_time"] < hi:
                return i
        return None

    results = []
    for i, (chord, window) in enumerate(zip(chords, windows)):
        def owner(note, i=i):
            j = chord_of(note)
            return None if j is None else ("this" if j == i else "other")

        analysis = classify_chord(chord["notes"], False, window, stages, activations, owner)
        results.append({"name": chord["name"], "section": chord["section"], "onset": chord["onset"],
                        "expected": chord["note_names"], **analysis})

    # Summary.
    expected_total = sum(r["funnel"]["expected"] for r in results)
    funnel = {k: sum(r["funnel"][k] for r in results) for k in ("detected", "kept", "grouped", "mapped")}
    in_tab_correct = sum(1 for r in results for n in r["notes"] if n.get("in_tab"))
    mapped_total = len(mapping["positions"])
    extras = {k: sum(1 for r in results for x in r["extras"] if x["kind"] == k) for k in EXTRA_KINDS}
    extras_in_tab = sum(1 for r in results for x in r["extras"] if x["mapped"])
    unattributed = sum(1 for e in kept if chord_of(e) is None)
    losses = {s: sum(1 for r in results for n in r["notes"] if n["stage"] == s) for s in "abcd"}

    def complete_share(section_prefix: str | None) -> float:
        rs = [r for r in results if section_prefix is None or r["section"].startswith(section_prefix)]
        return round(100 * sum(r["complete"] for r in rs) / len(rs), 1)

    def section_stats(prefix: str) -> dict:
        """Recall/precision/completeness for one section. Precision here is
        over tab notes attributed to that section's chords (correct chord
        notes vs mapped extras), so unattributed notes don't count."""
        rs = [r for r in results if r["section"].startswith(prefix)]
        expected = sum(r["funnel"]["expected"] for r in rs)
        correct = sum(1 for r in rs for n in r["notes"] if n.get("in_tab"))
        extra = sum(1 for r in rs for x in r["extras"] if x["mapped"])
        return {
            "recall": round(sum(r["funnel"]["mapped"] for r in rs) / expected, 3),
            "precision": round(correct / (correct + extra), 3) if correct + extra else None,
            "complete_pct": round(100 * sum(r["complete"] for r in rs) / len(rs), 1),
        }

    summary = {
        "sections": {"sustained": section_stats("sustained"), "fast": section_stats("fast")},
        "program": truth["program"],
        "program_name": truth["program_name"],
        "chords": len(results),
        "expected_notes": expected_total,
        "recall": {k: round(v / expected_total, 3) for k, v in funnel.items()},
        "losses_by_stage": losses,
        "precision": round(in_tab_correct / mapped_total, 3) if mapped_total else None,
        "tab_notes": mapped_total,
        "extras_kept": extras,
        "extras_in_tab": extras_in_tab,
        "kept_notes_outside_any_chord": unattributed,
        "complete_pct": complete_share(None),
        "complete_pct_sustained": complete_share("sustained"),
        "complete_pct_fast": complete_share("fast"),
        "clean_pct": round(100 * sum(r["clean"] for r in results) / len(results), 1),
        "separation_seconds": audio["separation_seconds"],
        "total_seconds": round(time.perf_counter() - start_time, 1),
    }
    return {"summary": summary, "chords": results}


REPEAT_MATCH_S = 0.08  # < half the 160bpm eighth-note spacing (0.1875s)


def evaluate_repeats(truth: dict, wav_path: str, config: StageConfig = PIPELINE) -> dict:
    """Guard for re-trigger merging: every intentionally repeated note in the
    repeats test set should appear as its own note in the tab. A truth note
    counts when a mapped note of the same pitch starts within REPEAT_MATCH_S
    of it; mapped notes of a pattern's pitches that match nothing are extras."""
    with tempfile.TemporaryDirectory() as workdir:
        events, _, _ = detect_file(wav_path, workdir)
    stages = run_stages(events, config)
    mapped = sorted((k[0], pretty_midi.note_name_to_number(k[1])) for k in stages["mapping"]["positions"])
    used: set[int] = set()
    patterns: dict[str, dict] = {}
    for note in truth["notes"]:
        p = patterns.setdefault(note["pattern"], {"expected": 0, "separate_in_tab": 0, "extras": 0,
                                                  "pitches": set(), "span": [note["onset"], note["end"]]})
        p["expected"] += 1
        p["pitches"].add(note["midi"])
        p["span"][1] = note["end"]
        match = next((i for i, (s, m) in enumerate(mapped)
                      if i not in used and m == note["midi"] and abs(s - note["onset"]) <= REPEAT_MATCH_S), None)
        if match is not None:
            used.add(match)
            p["separate_in_tab"] += 1
    for p in patterns.values():
        lo, hi = p["span"][0] - REPEAT_MATCH_S, p["span"][1] + 0.3
        p["extras"] = sum(1 for i, (s, m) in enumerate(mapped) if i not in used and m in p["pitches"] and lo <= s < hi)
        p["pitches"] = sorted(p["pitches"])
    expected = sum(p["expected"] for p in patterns.values())
    kept = sum(p["separate_in_tab"] for p in patterns.values())
    return {"recall": round(kept / expected, 3), "expected": expected, "separate_in_tab": kept,
            "extras": sum(p["extras"] for p in patterns.values()), "patterns": patterns}


def _print_summary(programs: dict, separate: bool, config: StageConfig = PIPELINE) -> None:
    rows = [p["summary"] for p in programs.values()]
    print(f"\nChord eval ({'separated guitar stem' if separate else 'full signal'}; {config.describe()}; "
          f"onset tolerance {CHORD_ONSET_TOLERANCE_SECONDS * 1000:.0f}ms)")
    head = ["", *[f"prog {r['program']} ({r['program_name']})" for r in rows]]
    lines = [
        ("chords / expected notes", [f"{r['chords']} / {r['expected_notes']}" for r in rows]),
        ("recall: detected (any amp)", [f"{r['recall']['detected']:.1%}" for r in rows]),
        ("recall: kept (>= threshold)", [f"{r['recall']['kept']:.1%}" for r in rows]),
        ("recall: grouped correctly", [f"{r['recall']['grouped']:.1%}" for r in rows]),
        ("recall: mapped (in tab, right column)", [f"{r['recall']['mapped']:.1%}" for r in rows]),
        ("lost at a / b / c / d", ["{a} / {b} / {c} / {d}".format(**r["losses_by_stage"]) for r in rows]),
        ("precision (tab notes that are chord notes)", [f"{r['precision']:.1%}" for r in rows]),
        ("extras kept: octave / repeat / phantom",
         ["{} / {} / {}".format(*[r["extras_kept"][k] for k in EXTRA_KINDS]) for r in rows]),
        ("extras that reach the tab", [str(r["extras_in_tab"]) for r in rows]),
        ("chords complete (all / sustained / fast)",
         [f"{r['complete_pct']:.0f}% / {r['complete_pct_sustained']:.0f}% / {r['complete_pct_fast']:.0f}%" for r in rows]),
        ("chords complete with no extras", [f"{r['clean_pct']:.0f}%" for r in rows]),
    ]
    widths = [max(len(head[0]), *(len(label) for label, _ in lines))]
    widths += [max(len(head[i + 1]), *(len(vals[i]) for _, vals in lines)) for i in range(len(rows))]
    print("  " + "  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(head, widths))))
    for label, vals in lines:
        print("  " + label.ljust(widths[0]) + "  " + "  ".join(v.rjust(w) for v, w in zip(vals, widths[1:])))


def _print_worst(programs: dict, n: int) -> list[str]:
    by_chord: dict[str, list[tuple[int, dict]]] = {}
    for prog, p in programs.items():
        for r in p["chords"]:
            by_chord.setdefault(f"{r['name']} [{r['section']}]", []).append((prog, r))

    def badness(item):
        rows = [r for _, r in item[1]]
        return (sum(r["complete"] for r in rows),
                sum(r["funnel"]["mapped"] for r in rows) / sum(r["funnel"]["expected"] for r in rows),
                -sum(len(r["extras"]) for r in rows))

    worst = sorted(by_chord.items(), key=badness)[:n]
    print(f"\nWorst {n} chords across programs (fewest complete, then lowest recall, then most extras):")
    for label, rows in worst:
        print(f"\n  {label}  expected {' '.join(rows[0][1]['expected'])}")
        for prog, r in rows:
            f = r["funnel"]
            print(f"    prog {prog}: {f['mapped']}/{f['expected']} in tab"
                  f"{', merged with neighbour' if r['merged_with_neighbour'] else ''}")
            for note in r["notes"]:
                if note["stage"] == "ok":
                    continue
                detail = {"a": lambda x: f"(a) not detected, peak activation {x['peak_activation']}",
                          "b": lambda x: f"(b) below threshold, amplitudes {x['amplitudes']}",
                          "c": lambda x: f"(c) {x['why']}",
                          "d": lambda x: f"(d) mapper: {x['why']}"}[note["stage"]](note)
                print(f"      {note['expected']:<4} {detail}")
            if r["extras"]:
                print("      (e) " + ", ".join(f"{x['pitch']} {x['kind']} (amp {x['amplitude']}"
                                               f"{', in tab' if x['mapped'] else ''})" for x in r["extras"]))
    return [label for label, _ in worst]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--testset", default="/app/data/_testaudio/chord_testset")
    parser.add_argument("--programs", help="comma-separated programs to evaluate (default: all in the test set)")
    parser.add_argument("--separate", action="store_true", help="evaluate on the Demucs guitar stem")
    parser.add_argument("--seed", type=int, default=0, help="separation seed (default 0)")
    parser.add_argument("--out", help="JSON results path (default: <testset>/eval[_separated].json)")
    parser.add_argument("--worst", type=int, default=3, help="how many worst chords to detail (default 3)")
    add_config_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    truth_paths = sorted(glob.glob(os.path.join(args.testset, "chords_prog*.json")))
    if args.programs:
        wanted = {int(p) for p in args.programs.split(",")}
        truth_paths = [p for p in truth_paths if json.load(open(p))["program"] in wanted]
    if not truth_paths:
        raise SystemExit(f"no chords_prog*.json in {args.testset}; run scripts.make_chord_testset first")

    programs = {}
    for path in truth_paths:
        truth = json.load(open(path))
        programs[truth["program"]] = evaluate_program(
            truth, os.path.join(args.testset, truth["wav"]), args.separate, args.seed, config)

    _print_summary(programs, args.separate, config)
    worst = _print_worst(programs, args.worst)

    out = args.out or os.path.join(args.testset, "eval_separated.json" if args.separate else "eval.json")
    with open(out, "w") as f:
        json.dump({"separate": args.separate, "seed": args.seed, "config": config.describe(),
                   "onset_tolerance_s": CHORD_ONSET_TOLERANCE_SECONDS, "worst": worst,
                   "programs": {str(k): v for k, v in programs.items()}}, f, indent=2, default=list)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
