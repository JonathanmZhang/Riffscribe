"""Full MT3 benchmark vs Basic Pitch, scored in the worker image from MT3's
saved output (run_mt3_files.py). Nothing here changes the pipeline.
    python /bakeoff/bench_mt3.py

MT3 mode throughout: all instruments, drums excluded (no pitch), same-pitch
notes within 50ms merged across programs ("dedup"). For the tab, MT3's
notes go through the pipeline's own post-detection code (run_stages:
select_notes, then grouping + fretboard mapping) with amplitude 1.0 - MT3
has no per-note confidence (its checkpoint has one velocity bin), so every
note passes select_notes and the mapper's lowest-amplitude conflict drops
become ties.

Sections:
  1/2  EGSet12, 12 performances x 3 tones: pitch recall/precision by segment
       type for raw notes and final tab (plus position agreement), scored
       exactly like scripts/egset12_benchmark.evaluate (whole performance
       matched, then attributed to segments by onset).
  4    Octave errors among MT3's false notes, and candidate fixes.
  3    Firefire windows: MT3 on the full mix and on the Demucs stem, per
       instrument family, and window metrics next to Basic Pitch's.
  5    Processing time.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(__file__))

import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402

import pretty_midi  # noqa: E402

from scripts.chord_stages import PIPELINE, detect_file, run_stages, window_metrics  # noqa: E402
from scripts.egset12_benchmark import (  # noqa: E402
    BENCHMARK_JSON, PERFORMANCES, SEGMENT_TYPES, TONES, _match, audio_path, load_truth,
)
from scripts.egset12_benchmark import evaluate as evaluate_bp  # noqa: E402
from scripts.regression_check import FIREFIRE, FIREFIRE_WINDOWS  # noqa: E402

BAKEOFF = "/app/data/_bakeoff"
MT3_DIR = os.path.join(BAKEOFF, "mt3")
DEDUP_S = 0.05
OCTAVE_S = 0.05
LOWEST_GUITAR_MIDI = 40  # E2, open low E in standard tuning

FAMILIES = [(0, "piano"), (8, "chromatic perc"), (16, "organ"), (24, "guitar"), (32, "bass"), (40, "strings"),
            (48, "ensemble/choir"), (56, "brass"), (64, "reed"), (72, "pipe"), (80, "synth lead"),
            (88, "synth pad"), (96, "synth fx"), (104, "ethnic"), (112, "percussive"), (120, "sound fx")]


def family(note: dict) -> str:
    if note.get("is_drum"):
        return "drums"
    return [name for low, name in FAMILIES if note["program"] >= low][-1]


# ------------------------------------------------------------------ MT3 notes


def load_mt3(name: str) -> dict:
    return json.load(open(os.path.join(MT3_DIR, name + ".json")))


def dedup(notes: list[dict]) -> list[dict]:
    """Pitched notes only; a note is dropped when a kept note of the same
    pitch starts within DEDUP_S of it (MT3 often emits one note under two
    programs). The first-starting copy is kept."""
    kept: list[dict] = []
    for n in sorted((n for n in notes if not n["is_drum"]), key=lambda n: (n["onset"], n["midi"], n["program"])):
        if not any(k["midi"] == n["midi"] and abs(k["onset"] - n["onset"]) <= DEDUP_S for k in kept[-24:]):
            kept.append(n)
    return kept


def octave_fix(notes: list[dict], mode: str) -> list[dict]:
    """mode "none"; "drop_upper": drop a note when a note exactly an octave
    below starts within OCTAVE_S (a harmonic ghost); "fold_low": raise notes
    below E2 by an octave (guitar can't play them in standard tuning), unless
    that pitch is already there; "both"."""
    out = list(notes)
    if mode in ("fold_low", "both"):
        folded = []
        for n in out:
            if n["midi"] < LOWEST_GUITAR_MIDI:
                up = n["midi"] + 12
                if any(o["midi"] == up and abs(o["onset"] - n["onset"]) <= OCTAVE_S for o in out):
                    continue
                n = {**n, "midi": up}
            folded.append(n)
        out = folded
    if mode in ("drop_upper", "both"):
        out = [n for n in out if not any(o["midi"] == n["midi"] - 12 and abs(o["onset"] - n["onset"]) <= OCTAVE_S
                                         for o in out)]
    return out


def to_events(notes: list[dict]) -> list[dict]:
    return [{"pitch": pretty_midi.note_number_to_name(n["midi"]), "midi": n["midi"], "start_time": n["onset"],
             "end_time": max(n["end"], n["onset"] + 0.01), "amplitude": 1.0, "program": n.get("program")}
            for n in notes]


def tab_of(stages: dict) -> list[dict]:
    return [{"onset": k[0], "midi": pretty_midi.note_name_to_number(k[1]), "string": v[0], "fret": v[1]}
            for k, v in stages["mapping"]["positions"].items()]


# --------------------------------------------------------------- EGSet12 eval

OCTAVE_MODES = ["none", "drop_upper", "fold_low", "both"]


def egset12() -> dict:
    benchmark = json.load(open(BENCHMARK_JSON))
    counts: dict = {}
    fp_kinds: dict = {}
    truth_octave_pairs = [0, 0]  # truth notes with another truth note an octave below within 50ms, all truth

    def bucket(system, tone, kind):
        return counts.setdefault((system, tone, kind), {"truth": 0, "tab": 0, "matched": 0, "position": 0})

    for perf in benchmark["performances"]:
        p = perf["performance"]
        truth, _ = load_truth(p)
        segments = perf["segments"]

        def seg_type(t):
            for s in segments:
                if s["start"] <= t < s["end"]:
                    return s["type"]
            return segments[-1]["type"]

        truth_octave_pairs[0] += sum(1 for t in truth if any(
            u["midi"] == t["midi"] - 12 and abs(u["onset"] - t["onset"]) <= OCTAVE_S for u in truth))
        truth_octave_pairs[1] += len(truth)

        for tone in TONES:
            with tempfile.TemporaryDirectory() as workdir:
                events, activations, _ = detect_file(audio_path(p, tone), workdir)
            bp_stages = run_stages(events, PIPELINE, activations)
            systems = {
                "bp_raw": [{"onset": e["start_time"], "midi": e["midi"]} for e in bp_stages["events"]],
                "bp_tab": tab_of(bp_stages),
            }
            mt3 = dedup(load_mt3(f"egset12_{tone}_{p}")["notes"])
            for mode in OCTAVE_MODES:
                fixed = octave_fix(mt3, mode)
                suffix = "" if mode == "none" else f"+{mode}"
                systems["mt3_raw" + suffix] = fixed
                systems["mt3_tab" + suffix] = tab_of(run_stages(to_events(fixed), PIPELINE))

            for system, notes in systems.items():
                matches = _match(truth, notes)
                for i, t in enumerate(truth):
                    for kind in (seg_type(t["onset"]), "all"):
                        b = bucket(system, tone, kind)
                        b["truth"] += 1
                        if i in matches:
                            b["matched"] += 1
                            n = notes[matches[i]]
                            if "string" in n:
                                b["position"] += (n["string"], n["fret"]) == (t["string"], t["fret"])
                for n in notes:
                    for kind in (seg_type(n["onset"]), "all"):
                        bucket(system, tone, kind)["tab"] += 1
                if system in ("mt3_raw", "mt3_tab", "bp_raw", "bp_tab"):
                    classify_fps(truth, notes, matches, fp_kinds.setdefault((system, tone), {}))

    result = {}
    for (system, tone, kind), b in counts.items():
        result.setdefault(system, {}).setdefault(tone, {})[kind] = {
            **b,
            "recall": round(b["matched"] / b["truth"], 4) if b["truth"] else None,
            "precision": round(b["matched"] / b["tab"], 4) if b["tab"] else None,
            "position": round(b["position"] / b["matched"], 4) if b["matched"] and "_tab" in system else None,
        }
    # Sanity: bp_tab must reproduce the benchmark's own evaluate().
    reference = evaluate_bp(PIPELINE, benchmark)
    for tone in TONES:
        for kind, r in reference[tone].items():
            mine = result["bp_tab"][tone][kind]
            assert (mine["matched"], mine["tab"], mine["truth"]) == (r["matched"], r["tab"], r["truth"]), (tone, kind)
    return {"scores": result, "fp_kinds": {f"{s}|{t}": v for (s, t), v in fp_kinds.items()},
            "truth_octave_doublings": {"notes": truth_octave_pairs[0], "of": truth_octave_pairs[1]}}


def classify_fps(truth: list[dict], notes: list[dict], matches: dict, out: dict) -> None:
    """Each unmatched note, first category that applies:
      octave_ghost_up/down  exactly 12 above/below a truth note (onset within
                            50ms) that IS matched by another note
      octave_sub_up/down    ... whose truth note is NOT matched (the octave
                            note replaced the right one)
      two_octaves           exactly 24 off a truth note
      fragment              same pitch as a truth note sounding at its onset
      other"""
    matched_truth = set(matches)
    used = set(matches.values())
    for j, n in enumerate(notes):
        if j in used:
            continue
        kind = "other"
        for i, t in enumerate(truth):
            if abs(t["onset"] - n["onset"]) > OCTAVE_S:
                continue
            d = n["midi"] - t["midi"]
            if abs(d) == 12:
                kind = f"octave_{'ghost' if i in matched_truth else 'sub'}_{'up' if d > 0 else 'down'}"
                break
            if abs(d) == 24:
                kind = "two_octaves"
        if kind == "other" and any(t["midi"] == n["midi"] and t["onset"] + 0.05 < n["onset"] < t["end"] for t in truth):
            kind = "fragment"
        out[kind] = out.get(kind, 0) + 1
        out["fp"] = out.get("fp", 0) + 1


# ------------------------------------------------------------------- firefire


def firefire() -> dict:
    out = {}
    for source, separate in (("mix", False), ("stem", True)):
        mt3_all = load_mt3(f"firefire_{source}")
        mt3 = dedup(mt3_all["notes"])
        stages = run_stages(to_events(mt3), PIPELINE)
        program_of = {(n["onset"], pretty_midi.note_number_to_name(n["midi"])): family(n) for n in mt3}
        with tempfile.TemporaryDirectory() as workdir:
            events, activations, _ = detect_file(FIREFIRE, workdir, separate=separate, seed=0)
        bp = run_stages(events, PIPELINE, activations)
        for start, end in FIREFIRE_WINDOWS:
            in_win = [n for n in mt3_all["notes"] if start <= n["onset"] < end]
            by_family: dict = {}
            for n in in_win:
                by_family[family(n)] = by_family.get(family(n), 0) + 1
            tab_family: dict = {}
            for key in stages["mapping"]["positions"]:
                if start <= key[0] < end:
                    tab_family[program_of[key]] = tab_family.get(program_of[key], 0) + 1
            out[f"{start:g}-{end:g} {source}"] = {
                "mt3_emitted_by_family": by_family,
                "mt3_tab_by_family": tab_family,
                "mt3": window_metrics(stages, start, end),
                "bp": window_metrics(bp, start, end),
            }
    return out


# --------------------------------------------------------------------- timing


def timing() -> dict:
    t = {}
    for tone in TONES:
        runs = [load_mt3(f"egset12_{tone}_{p}") for p in PERFORMANCES]
        audio = sum(r["audio_seconds"] for r in runs)
        infer = sum(r["infer_seconds"] for r in runs)
        t[f"egset12_{tone}"] = {"audio_s": round(audio, 1), "mt3_s": round(infer, 1),
                                "mt3_s_per_s": round(infer / audio, 3),
                                "peak_rss_mb": max(r["peak_rss_mb"] for r in runs)}
    for source in ("mix", "stem"):
        r = load_mt3(f"firefire_{source}")
        t[f"firefire_{source}"] = {"audio_s": r["audio_seconds"], "mt3_s": r["infer_seconds"],
                                   "mt3_s_per_s": round(r["infer_seconds"] / r["audio_seconds"], 3)}
    path = os.path.join(BAKEOFF, "firefire", "timing.json")
    if os.path.exists(path):
        t["firefire_basic_pitch_and_separation"] = json.load(open(path))
    return t


def main() -> None:
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    result = {"egset12": egset12(), "firefire": firefire(), "timing": timing()}
    with open(os.path.join(BAKEOFF, "bench_mt3.json"), "w") as f:
        json.dump(result, f, indent=1)
    report(result)


def report(result: dict) -> None:
    s = result["egset12"]["scores"]

    def cell(system, tone, kind, key="recall"):
        v = s[system][tone][kind][key]
        return "  -  " if v is None else f"{v * 100:5.1f}"

    print("EGSet12 (36 segments x 3 tones): recall / precision [position agreement] by segment type")
    print(f"  {'tone':<9}{'type':<12}" + "".join(f"{x:>22}" for x in ("bp_raw", "mt3_raw", "bp_tab", "mt3_tab")))
    for tone in TONES:
        for kind in ["all"] + SEGMENT_TYPES:
            row = f"  {tone:<9}{kind:<12}"
            for system in ("bp_raw", "mt3_raw", "bp_tab", "mt3_tab"):
                pos = s[system][tone][kind]["position"]
                row += f"{cell(system, tone, kind)} / {cell(system, tone, kind, 'precision')}" + (
                    f" [{pos * 100:4.1f}]" if pos is not None else "       ")
            print(row + f"   n={s['bp_raw'][tone][kind]['truth']}")
    print("\nOctave fixes on MT3 (all segments): raw recall/precision | tab recall/precision")
    for mode in OCTAVE_MODES:
        suffix = "" if mode == "none" else f"+{mode}"
        print(f"  {mode:<11}" + "  ".join(
            f"{tone}: {cell('mt3_raw' + suffix, tone, 'all')}/{cell('mt3_raw' + suffix, tone, 'all', 'precision')} | "
            f"{cell('mt3_tab' + suffix, tone, 'all')}/{cell('mt3_tab' + suffix, tone, 'all', 'precision')}"
            for tone in TONES))
    print("\nFalse-note categories:")
    for key, kinds in sorted(result["egset12"]["fp_kinds"].items()):
        print(f"  {key:<18} {kinds}")
    d = result["egset12"]["truth_octave_doublings"]
    print(f"  ground truth: {d['notes']} of {d['of']} notes have a true note an octave below within 50ms")
    print("\nFirefire windows")
    for w, r in result["firefire"].items():
        m, b = r["mt3"], r["bp"]
        print(f"  {w:<15} MT3 emitted {r['mt3_emitted_by_family']}")
        print(f"  {'':<15} MT3 tab by family {r['mt3_tab_by_family']}")
        print(f"  {'':<15} tab notes MT3 {m['tab_notes']} vs BP {b['tab_notes']} | >=3 columns {m['chord_columns_ge3']} vs "
              f"{b['chord_columns_ge3']} | mean notes/chord col {m['mean_notes_per_chord_column']} vs "
              f"{b['mean_notes_per_chord_column']} | mapper drops {m['mapper_drops']} vs {b['mapper_drops']} | "
              f"below E2 {m['below_e2']} vs {b['below_e2']}")
    print("\nTiming:", json.dumps(result["timing"], indent=1))


if __name__ == "__main__":
    main()
