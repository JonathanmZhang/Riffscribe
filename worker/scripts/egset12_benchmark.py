"""Real-guitar note-recall benchmark built on EGSet12.

EGSet12 (H. Pedroza, W. Abreu, R. Corey, I. R. Roman, DAFx 2024;
https://zenodo.org/records/11406378, CC BY 4.0): twelve ~30s solo electric
guitar performances (Telecaster through a clean amp, mic'd), each annotated
per note with string and fret (JAMS). Download it first with
`python -m scripts.download_egset12`.

Usage (inside the worker container):
    python -m scripts.egset12_benchmark build     # render tones, write benchmark.json
    python -m scripts.egset12_benchmark eval [settings flags]

build (deterministic):
  - Cuts each performance into ~3 segments at quiet points and labels each
    from the annotations: "chords" (most notes in 3+ note chords), "fast"
    (runs of single notes at 16th notes of ~110bpm or faster), otherwise
    "single-note".
  - Names every ground-truth chord column from its notes (for chord labels
    later).
  - Renders two PROCESSED-DISTORTION versions of every performance with
    pedalboard's Distortion plugin (moderate and heavy drive, simple cab-like
    filtering) - a stand-in for driven tones, not real amplifier recordings.

eval scores the pipeline's tab against the annotations, per tone and segment
type: PITCH recall/precision (a tab note matches a ground-truth note of the
same MIDI pitch starting within 50ms, any string, one-to-one) and POSITION
agreement (same string AND fret, among pitch-matched notes). Notes are
matched over the whole performance, then attributed to segments by onset, so
segment boundaries don't create false misses.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before TensorFlow is imported

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402

from scripts.chord_stages import (  # noqa: E402
    PIPELINE,
    StageConfig,
    add_config_args,
    config_from_args,
    detect_file,
    run_stages,
)

DATA_DIR = "/app/data/egset12"
BENCHMARK_JSON = os.path.join(DATA_DIR, "benchmark.json")
PERFORMANCES = [f"{i:02d}" for i in range(1, 13)]
TONES = ["clean", "moderate", "heavy"]  # clean = the original recordings
SEGMENT_TYPES = ["chords", "single-note", "fast"]
MATCH_S = 0.05

OPEN_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}  # string number -> open pitch
# Labelling: notes within this of a group's first note form one column.
COLUMN_S = 0.08
# 16th notes at 110bpm; single-note gaps at or below this count as "fast".
# EGSet12 has little such material (21 gaps in total, 19 in performance 04).
FAST_IOI_S = 60 / 110 / 4
MODERATE_DRIVE_DB = 12
HEAVY_DRIVE_DB = 20

# Split by performance for tuning the fretboard mapper without overfitting:
# constants are tuned on TUNING only, HELDOUT is the real result. Pairs of
# similar performances (open-position 06/10, up-neck chords 04/08,
# single-note lines 05/07, slow mixed 01/03, fast chordal 02/12, high-neck
# mixed 09/11) are split across the halves, balancing note counts (766 / 801).
TUNING_PERFORMANCES = ["02", "03", "04", "05", "10", "11"]
HELDOUT_PERFORMANCES = ["01", "06", "07", "08", "09", "12"]

# Playability proxy: the fretting hand covers HAND_WINDOW frets from the
# index finger (one more with a stretch).
HAND_WINDOW = 4


def hand_stats(columns: list[list[int]], duration: float) -> dict:
    """Follows a hand through a tab's columns (the fretted frets of each
    column, in time order): it stays put while each column fits within the
    window from the index finger (+1 stretch), otherwise shifts the least
    distance that covers the column. Open strings never force a shift."""
    hand, shifts, distance = None, 0, 0
    for frets in columns:
        fretted = [f for f in frets if f > 0]
        if not fretted:
            continue
        lo, hi = min(fretted), max(fretted)
        if hand is None:
            hand = lo
            continue
        if hand <= lo and hi <= hand + HAND_WINDOW:
            continue
        options = range(max(hi - HAND_WINDOW, 1), max(lo, hi - HAND_WINDOW, 1) + 1)
        new_hand = min(options, key=lambda h: abs(h - hand))
        shifts += 1
        distance += abs(new_hand - hand)
        hand = new_hand
    minutes = duration / 60 if duration else 1
    return {"shifts": shifts, "shift_frets": distance, "minutes": minutes}


# ------------------------------------------------------------ ground truth


def load_truth(performance: str) -> tuple[list[dict], float | None]:
    """EGSet12 JAMS (GuitarSet layout): one note_midi annotation per string,
    data_source 0 = low E ... 5 = high e. Returns notes with onset, end,
    midi, string (1 = high e ... 6 = low E, as in the pipeline) and fret."""
    with open(os.path.join(DATA_DIR, f"{performance}.jams")) as f:
        jams = json.load(f)
    notes, tempo = [], None
    for ann in jams["annotations"]:
        if ann["namespace"] == "tempo":
            tempo = ann["data"][0]["value"]
        if ann["namespace"] != "note_midi":
            continue
        string = 6 - int(ann["annotation_metadata"]["data_source"])
        for x in ann["data"]:
            midi = int(round(x["value"]))
            notes.append({"onset": x["time"], "end": x["time"] + x["duration"], "midi": midi,
                          "string": string, "fret": midi - OPEN_MIDI[string]})
    return sorted(notes, key=lambda n: (n["onset"], n["midi"])), tempo


def _columns(notes: list[dict]) -> list[list[dict]]:
    columns: list[list[dict]] = []
    for note in notes:
        if columns and note["onset"] - columns[-1][0]["onset"] <= COLUMN_S:
            columns[-1].append(note)
        else:
            columns.append([note])
    return columns


# Chord templates: intervals above the root.
CHORD_TEMPLATES = {
    "": (0, 4, 7), "m": (0, 3, 7), "5": (0, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9), "add9": (0, 2, 4, 7), "madd9": (0, 2, 3, 7), "7sus4": (0, 5, 7, 10),
    "9": (0, 2, 4, 7, 10), "m9": (0, 2, 3, 7, 10), "maj9": (0, 2, 4, 7, 11),
    "7#9": (0, 3, 4, 7, 10), "13": (0, 4, 7, 9, 10),
}
PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def chord_name(midis: list[int]) -> str:
    """Names a chord from its notes: an exact template match over the pitch
    classes, preferring the bass note as root, with a slash for inversions
    (e.g. "C/E"). Unmatched sets are returned as their pitch classes."""
    pcs = {m % 12 for m in midis}
    bass = min(midis) % 12
    matches = []
    for root in sorted(pcs, key=lambda pc: pc != bass):
        intervals = tuple(sorted((pc - root) % 12 for pc in pcs))
        for suffix, template in CHORD_TEMPLATES.items():
            if intervals == tuple(sorted(template)):
                matches.append((root, suffix))
    if not matches:
        return "(" + " ".join(PC_NAMES[pc] for pc in sorted(pcs)) + ")"
    root, suffix = matches[0]
    name = PC_NAMES[root] + suffix
    return name if root == bass else f"{name}/{PC_NAMES[bass]}"


def _segment_bounds(notes: list[dict], duration: float, parts: int = 3) -> list[float]:
    """Cuts near duration * k/parts, at the time within +-2.5s where the
    fewest annotated notes are sounding (ties: farthest from any onset)."""
    cuts = [0.0]
    onsets = np.array([n["onset"] for n in notes])
    for k in range(1, parts):
        best = None
        for t in np.arange(duration * k / parts - 2.5, duration * k / parts + 2.5, 0.01):
            sounding = sum(1 for n in notes if n["onset"] < t < n["end"])
            clearance = float(np.min(np.abs(onsets - t))) if len(onsets) else 1.0
            key = (sounding, -clearance)
            if best is None or key < best[0]:
                best = (key, float(t))
        cuts.append(round(best[1], 3))
    cuts.append(round(duration, 3))
    return cuts


def _label(notes: list[dict]) -> tuple[str, dict]:
    columns = _columns(notes)
    in_chords = sum(len(c) for c in columns if len({n["midi"] for n in c}) >= 3)
    singles = [c[0]["onset"] for c in columns if len(c) == 1]
    iois = np.diff(singles) if len(singles) > 1 else np.array([])
    fast_iois = int(np.sum(iois <= FAST_IOI_S + 1e-3))
    stats = {"notes": len(notes), "chord_note_share": round(in_chords / len(notes), 2) if notes else 0.0,
             "fast_single_iois": fast_iois}
    if notes and in_chords / len(notes) >= 0.5:
        return "chords", stats
    if fast_iois >= 6:
        return "fast", stats
    return "single-note", stats


# ------------------------------------------------------------------- build


def _render_tones(performance: str) -> None:
    from pedalboard import Distortion, HighpassFilter, LowpassFilter, Pedalboard

    audio, sr = sf.read(os.path.join(DATA_DIR, f"{performance}.wav"), dtype="float32")
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio  # the two channels are copies
    # Normalize first so a drive setting means the same for every recording.
    mono = mono * (10 ** (-1 / 20) / max(float(np.abs(mono).max()), 1e-9))
    chains = {
        # Processed distortion, not real amp recordings: pedalboard's Distortion
        # (tanh waveshaper) between simple cab-like high/low-pass filters.
        # Drives picked by crest factor: RMS/peak ~0.40 (moderate) and ~0.63
        # (heavy), vs ~0.15 for the clean recordings.
        "moderate": Pedalboard([HighpassFilter(90), Distortion(drive_db=MODERATE_DRIVE_DB), LowpassFilter(6500)]),
        "heavy": Pedalboard([HighpassFilter(110), Distortion(drive_db=HEAVY_DRIVE_DB), LowpassFilter(4800)]),
    }
    for tone, board in chains.items():
        out = board(mono[np.newaxis, :], sr)[0]
        out *= 10 ** (-1 / 20) / max(float(np.abs(out).max()), 1e-9)  # peak -1 dBFS
        path = os.path.join(DATA_DIR, "tones", tone, f"{performance}.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, out, sr, subtype="PCM_16")


def audio_path(performance: str, tone: str) -> str:
    return os.path.join(DATA_DIR, f"{performance}.wav") if tone == "clean" else \
        os.path.join(DATA_DIR, "tones", tone, f"{performance}.wav")


def build() -> dict:
    performances = []
    for p in PERFORMANCES:
        notes, tempo = load_truth(p)
        duration = sf.info(os.path.join(DATA_DIR, f"{p}.wav")).duration
        bounds = _segment_bounds(notes, duration)
        segments = []
        for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
            seg_notes = [n for n in notes if start <= n["onset"] < end]
            label, stats = _label(seg_notes)
            chords = [{"time": round(c[0]["onset"], 3), "notes": sorted({n["midi"] for n in c}),
                       "name": chord_name([n["midi"] for n in c])}
                      for c in _columns(seg_notes) if len({n["midi"] for n in c}) >= 3]
            segments.append({"id": f"{p}-{i + 1}", "start": start, "end": end, "type": label, **stats,
                             "ground_truth_chords": chords})
        performances.append({"performance": p, "tempo": tempo, "duration": round(duration, 3), "segments": segments})
        _render_tones(p)
    benchmark = {
        "source": "EGSet12, https://zenodo.org/records/11406378 (CC BY 4.0)",
        "tones": {"clean": "original recordings (Telecaster, clean amp, mic'd)",
                  "moderate": f"processed distortion: pedalboard Distortion drive {MODERATE_DRIVE_DB}dB, "
                              "cab-like filtering (not a real amp)",
                  "heavy": f"processed distortion: pedalboard Distortion drive {HEAVY_DRIVE_DB}dB, "
                           "cab-like filtering (not a real amp)"},
        "performances": performances,
    }
    with open(BENCHMARK_JSON, "w") as f:
        json.dump(benchmark, f, indent=2)
    return benchmark


# -------------------------------------------------------------------- eval


def _match(truth: list[dict], tab: list[dict]) -> dict[int, int]:
    """One-to-one pitch matches within MATCH_S, closest pairs first:
    truth index -> tab index."""
    pairs = sorted(
        (abs(t["onset"] - n["onset"]), i, j)
        for i, t in enumerate(truth) for j, n in enumerate(tab)
        if t["midi"] == n["midi"] and abs(t["onset"] - n["onset"]) <= MATCH_S
    )
    used_t, used_n, matches = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_t and j not in used_n:
            matches[i] = j
            used_t.add(i)
            used_n.add(j)
    return matches


def evaluate(config: StageConfig = PIPELINE, benchmark: dict | None = None,
             performances: list[str] | None = None) -> dict:
    """Scores every performance (or only `performances`). Besides the
    per-tone/segment-type metrics, result["playability"][tone] has hand
    shifts per minute and average shift distance for the tab, and for the
    player's own tab ("truth") as the reference."""
    benchmark = benchmark or json.load(open(BENCHMARK_JSON))
    counts: dict[tuple[str, str], dict] = {}
    play: dict[str, dict] = {}

    def bucket(tone, kind):
        return counts.setdefault((tone, kind), {"truth": 0, "tab": 0, "matched": 0, "position": 0})

    def add_play(tone, who, stats):
        p = play.setdefault(tone, {}).setdefault(who, {"shifts": 0, "shift_frets": 0, "minutes": 0.0})
        for k in p:
            p[k] += stats[k]

    for perf in benchmark["performances"]:
        if performances is not None and perf["performance"] not in performances:
            continue
        truth, _ = load_truth(perf["performance"])
        segments = perf["segments"]

        def seg_type(t):
            for s in segments:
                if s["start"] <= t < s["end"]:
                    return s["type"]
            return segments[-1]["type"]

        for tone in TONES:
            with tempfile.TemporaryDirectory() as workdir:
                events, activations, _ = detect_file(audio_path(perf["performance"], tone), workdir)
            stages = run_stages(events, config, activations)
            tab = [{"onset": k[0], "midi": pretty_midi.note_name_to_number(k[1]), "string": v[0], "fret": v[1]}
                   for k, v in stages["mapping"]["positions"].items()]
            matches = _match(truth, tab)
            for i, t in enumerate(truth):
                for kind in (seg_type(t["onset"]), "all"):
                    b = bucket(tone, kind)
                    b["truth"] += 1
                    if i in matches:
                        b["matched"] += 1
                        n = tab[matches[i]]
                        b["position"] += (n["string"], n["fret"]) == (t["string"], t["fret"])
            for n in tab:
                for kind in (seg_type(n["onset"]), "all"):
                    bucket(tone, kind)["tab"] += 1

            tab_columns = [[stages["mapping"]["positions"][k][1] for k in (
                (note["start_time"], note["pitch"]) for note in group) if k in stages["mapping"]["positions"]]
                for group in stages["groups"]]
            add_play(tone, "tab", hand_stats(tab_columns, perf["duration"]))
            add_play(tone, "truth", hand_stats([[n["fret"] for n in c] for c in _columns(truth)], perf["duration"]))

    result = {}
    for (tone, kind), b in counts.items():
        result.setdefault(tone, {})[kind] = {
            **b,
            "pitch_recall": round(b["matched"] / b["truth"], 3) if b["truth"] else None,
            "pitch_precision": round(b["matched"] / b["tab"], 3) if b["tab"] else None,
            "position_agreement": round(b["position"] / b["matched"], 3) if b["matched"] else None,
        }
    result["playability"] = {
        tone: {who: {"shifts_per_minute": round(p["shifts"] / p["minutes"], 1),
                     "avg_shift_frets": round(p["shift_frets"] / p["shifts"], 2) if p["shifts"] else 0.0}
               for who, p in by_who.items()}
        for tone, by_who in play.items()
    }
    return result


def print_eval(result: dict, base: dict | None = None) -> None:
    def fmt(v, ref):
        if v is None:
            return "  -  "
        s = f"{v * 100:5.1f}%"
        return s + (f" ({(v - ref) * 100:+.1f})" if ref is not None and v != ref else "")

    print("  EGSet12 (real guitar): pitch recall | pitch precision | position agreement  [truth notes]")
    for tone in TONES:
        for kind in ["all"] + SEGMENT_TYPES:
            r = result.get(tone, {}).get(kind)
            if not r:
                continue
            ref = base.get(tone, {}).get(kind) if base else None
            print(f"    {tone:<9} {kind:<12} {fmt(r['pitch_recall'], ref and ref['pitch_recall'])} | "
                  f"{fmt(r['pitch_precision'], ref and ref['pitch_precision'])} | "
                  f"{fmt(r['position_agreement'], ref and ref['position_agreement'])}  [{r['truth']}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["build", "eval"])
    add_config_args(parser)
    args = parser.parse_args()
    logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)
    if args.command == "build":
        b = build()
        segs = [s for p in b["performances"] for s in p["segments"]]
        print(f"{len(segs)} segments from {len(b['performances'])} performances; types: "
              + ", ".join(f"{t} {sum(s['type'] == t for s in segs)}" for t in SEGMENT_TYPES)
              + f"; ground-truth chord columns: {sum(len(s['ground_truth_chords']) for s in segs)}")
        for s in segs:
            names = sorted({c["name"] for c in s["ground_truth_chords"]})
            print(f"  {s['id']:<6} {s['start']:6.2f}-{s['end']:6.2f}s  {s['type']:<12} notes {s['notes']:>3}  "
                  f"chord share {s['chord_note_share']:.2f}  fast IOIs {s['fast_single_iois']:>3}  chords: {', '.join(names[:8])}")
    else:
        print_eval(evaluate(config_from_args(args)))


if __name__ == "__main__":
    main()
