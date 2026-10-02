"""Bend and vibrato detection from Basic Pitch's pitch_bends output, which
the pipeline computes and discards. Measurement only: nothing in the
pipeline uses it. Runs in the worker image (repo at /repo for the
stale-image guard, this folder at /x):

    python /x/bends.py synthetic    # make_bend_testset.py's notes, exact truth
    python /x/bends.py idmt         # IDMT-SMT-Guitar dataset 1: real, labelled
    python /x/bends.py egset12      # no bends annotated: false positives only

The notes are the pipeline's own: Basic Pitch's note creation with the
pipeline's settings (transcribe.notes_from_model_output's arguments), then
transcribe.select_notes. Basic Pitch's get_pitch_bends gives each note one
integer per frame (11.6ms): the offset, in thirds of a semitone, of the
strongest contour bin within +/-25 bins of the note's pitch.

Basic Pitch does not keep a bent note as one note. Its per-note offsets
only cover about -1..+3 thirds of a semitone before it ends the note and
starts another on the next semitone (see README.md), so a bend arrives as
two or three notes on neighbouring pitches with no gap between them. The
bend detector therefore works on the joins between notes:

  glide    two kept notes A then B, 1 or 2 semitones apart, B starting
           within -30..+60ms of A's end. The join is a glide if the pitch
           is moving across it: over A's last 3 frames (35ms) its offset is
           GLIDE_CENTS (25) or more towards B, or over B's first 3 frames
           its offset is that far back towards A. Offsets are measured
           from the file's resting offset (the median over all frames; Basic
           Pitch's bins sit one third of a semitone off the note's pitch).
           Two sources: the finer contour (centroid of the contour bins
           +/-3 around each frame's strongest bin, in cents), and the
           integer pitch_bends alone.
  vibrato  per run of same-pitch notes that follow each other without a
           gap (wide vibrato splits a note into several): remove the slow
           movement (median over 21 frames), take the spectrum of the rest
           from 8 frames after the onset. Vibrato if the strongest
           component between 4 and 8 Hz has an amplitude of at least
           VIBRATO_CENTS (12), holds at least half of the moving energy,
           and the run lasts at least 4 of its cycles.

The settings were fixed after looking at one example of each synthetic note
type and before any scoring; `--sweep` shows how the numbers move with them.
One variant was added later and is labelled as such, "joins + drift": a
glide, or a single note whose pitch holds two levels DRIFT_CENTS (40) apart
for 116ms each and has no vibrato. It was added after the first IDMT run
showed real bends that stay one note, so it is not an advance prediction.

Scoring, per ground-truth note: the kept tab notes that overlap it by 50ms
or more and lie from 1 semitone below to 3 above its pitch are "its"
notes. A bend counts as detected if any join between two of them is a
glide; vibrato if any same-pitch run of them triggers. Recall is over the
truth notes that have the technique; precision is over all truth notes the
detector fired on. The "tab shows" column is what the tab holds today for
each truth note: its notes' pitches relative to the fretted pitch, in time
order ("0" = one right note; "0 +2" = the bend drawn as a second, wrong
note; "0 0" = one note split in two).
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/app")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import scipy.signal  # noqa: E402
from basic_pitch import note_creation as nc  # noqa: E402
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP  # noqa: E402

from scripts import build_info  # noqa: E402
from tasks import transcribe  # noqa: E402
from tasks.audio_io import ensure_decodable_audio, normalize_to_wav  # noqa: E402

DATA = "/app/data/_expression"
CACHE_DIR = os.path.join(DATA, "cache")
FPS = AUDIO_SAMPLE_RATE / FFT_HOP  # 86.1 contour frames per second
CENTS_PER_BIN = 100.0 / 3

GLIDE_CENTS = 25.0
GLIDE_FRAMES = 3
JOIN_GAP_S = (-0.03, 0.06)
DRIFT_CENTS = 40.0
VIBRATO_CENTS = 12.0
VIBRATO_BAND_HZ = (4.0, 8.0)
VIBRATO_ENERGY_SHARE = 0.5
OVERLAP_S = 0.05
PITCH_WINDOW = (-1, 3)  # semitones around the truth pitch
SOURCES = {"cents": "fine contour", "bend_cents": "pitch_bends"}


# ------------------------------------------------------------- detection


def model_output_for(path: str) -> dict:
    """Basic Pitch's raw output for a file, prepared as ingest_audio does
    (22.05kHz mono). Cached by the file's bytes."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    digest = hashlib.sha1(open(path, "rb").read()).hexdigest()[:20]
    cache = os.path.join(CACHE_DIR, digest + ".pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    with tempfile.TemporaryDirectory() as workdir:
        normalized = normalize_to_wav(ensure_decodable_audio(path, workdir), os.path.join(workdir, "n.wav"))
        _, output = transcribe.detect_note_events(normalized)
    output = {k: np.asarray(v, dtype=np.float32) for k, v in output.items()}
    pickle.dump(output, open(cache, "wb"))
    return output


def refined_cents(contours: np.ndarray, start: int, end: int, midi: int, bends: list[int]) -> np.ndarray:
    """Per frame: cents from the note's pitch, from the centroid of the
    contour bins +/-3 around the frame's strongest bin."""
    centre = int(np.round(nc.midi_pitch_to_contour_bin(midi)))
    out = np.zeros(end - start)
    for i, frame in enumerate(range(start, end)):
        peak = centre + bends[i]
        lo, hi = max(0, peak - 3), min(contours.shape[1], peak + 4)
        weights = contours[frame, lo:hi]
        total = float(weights.sum())
        position = float((weights * np.arange(lo, hi)).sum() / total) if total > 0 else float(peak)
        out[i] = (position - centre) * CENTS_PER_BIN
    return out


def tab_notes(output: dict) -> list[dict]:
    """The pipeline's note events (same call as transcribe
    .notes_from_model_output), each with its pitch_bends, the same in cents,
    the finer contour, and select_notes' "kept" flag."""
    min_frames = int(np.round(transcribe.BASIC_PITCH_MIN_NOTE_LENGTH_MS / 1000 * FPS))
    raw = nc.output_to_notes_polyphonic(
        output["note"], output["onset"], onset_thresh=transcribe.BASIC_PITCH_ONSET_THRESHOLD,
        frame_thresh=transcribe.BASIC_PITCH_FRAME_THRESHOLD, infer_onsets=True, min_note_len=min_frames,
        min_freq=None, max_freq=None, melodia_trick=True)
    times = nc.model_frames_to_time(output["contour"].shape[0])
    events = []
    for start, end, midi, amplitude, bends in nc.get_pitch_bends(output["contour"], raw):
        bends = [int(b) for b in bends]
        events.append({"midi": int(midi), "pitch": librosa.midi_to_note(int(midi)), "start_time": float(times[start]),
                       "end_time": float(times[end]), "amplitude": float(amplitude), "bends": bends,
                       "bend_cents": np.asarray(bends, dtype=float) * CENTS_PER_BIN,
                       "cents": refined_cents(output["contour"], start, end, int(midi), bends)})
    return transcribe.select_notes(events)


def resting_offset(events: list[dict], key: str) -> float:
    """The file's offset for an unbent note (median over all frames)."""
    values = np.concatenate([np.asarray(e[key], dtype=float) for e in events]) if events else np.zeros(1)
    return float(np.median(values))


def glide(a: dict, b: dict, rest: float, key: str = "cents", threshold: float = GLIDE_CENTS) -> tuple[bool, bool]:
    """Is the join from note a to note b a glide? Returns (either side moves
    towards the other note, both sides do). key: "cents" (finer contour) or
    "bend_cents" (the integer pitch_bends, in cents)."""
    step = b["midi"] - a["midi"]
    gap = b["start_time"] - a["end_time"]
    if abs(step) not in (1, 2) or not JOIN_GAP_S[0] <= gap <= JOIN_GAP_S[1]:
        return False, False
    direction = 1 if step > 0 else -1
    leaving = direction * (float(np.mean(a[key][-GLIDE_FRAMES:])) - rest) >= threshold
    arriving = -direction * (float(np.mean(b[key][:GLIDE_FRAMES])) - rest) >= threshold
    return leaving or arriving, leaving and arriving


def held_drift_cents(series_cents, sustain: int = 10) -> float:
    """Within one note: the highest level its pitch holds for `sustain`
    frames (116ms) minus the lowest such level, in cents."""
    if len(series_cents) < sustain:
        return 0.0
    smooth = scipy.signal.medfilt(np.asarray(series_cents, dtype=float), 5)
    windows = np.lib.stride_tricks.sliding_window_view(smooth, sustain)
    return float(windows.min(axis=1).max() - windows.max(axis=1).min())


def same_pitch_runs(events: list[dict]) -> list[list[dict]]:
    """events (time order) grouped into runs of one pitch with no gap."""
    runs: list[list[dict]] = []
    for e in events:
        for run in runs:
            if run[-1]["midi"] == e["midi"] and JOIN_GAP_S[0] <= e["start_time"] - run[-1]["end_time"] <= JOIN_GAP_S[1]:
                run.append(e)
                break
        else:
            runs.append([e])
    return runs


def vibrato(series_cents: np.ndarray) -> tuple[float, float]:
    """(amplitude in cents, rate in Hz) of the strongest 4-8 Hz component of
    the pitch after removing slow movement; (0, 0) if none qualifies."""
    x = np.asarray(series_cents, dtype=float)[8:]
    if len(x) < 24:
        return 0.0, 0.0
    x = x - scipy.signal.medfilt(x, 21)
    window = np.hanning(len(x))
    n = 4 * len(x)
    spectrum = np.abs(np.fft.rfft(x * window, n))
    freqs = np.fft.rfftfreq(n, 1 / FPS)
    band = (freqs >= VIBRATO_BAND_HZ[0]) & (freqs <= VIBRATO_BAND_HZ[1])
    if not band.any():
        return 0.0, 0.0
    k = int(np.argmax(np.where(band, spectrum, 0)))
    amplitude = 2 * spectrum[k] / window.sum()
    rate = float(freqs[k])
    # Energy near the peak (+/-1.5 Hz) against all moving energy above 1.5 Hz.
    power = spectrum**2
    near = power[np.abs(freqs - rate) <= 1.5].sum()
    moving = power[freqs >= 1.5].sum()
    share = near / moving if moving > 0 else 0.0
    if share < VIBRATO_ENERGY_SHARE or len(x) / FPS < 4 / rate:
        return 0.0, 0.0
    return float(amplitude), rate


def run_vibrato(run: list[dict], key: str) -> float:
    return vibrato(np.concatenate([np.asarray(e[key], dtype=float) for e in run]))[0]


# ---------------------------------------------------------------- scoring


def own_notes(truth: dict, events: list[dict]) -> list[dict]:
    return sorted((e for e in events if e["kept"]
                   and PITCH_WINDOW[0] <= e["midi"] - truth["midi"] <= PITCH_WINDOW[1]
                   and min(e["end_time"], truth["end"]) - max(e["start_time"], truth["onset"]) >= OVERLAP_S),
                  key=lambda e: e["start_time"])


def score_note(truth: dict, events: list[dict], rests: dict) -> dict:
    """What the tab holds for one truth note ({onset, end, midi}), and what
    the detectors say about it."""
    own = own_notes(truth, events)
    row = {"tab_notes": len(own), "shows": " ".join(f"{e['midi'] - truth['midi']:+d}".replace("+0", "0") for e in own),
           "glide": {}, "vib": {}}
    for key in SOURCES:
        for threshold in (15, 25, 40):
            joins = [glide(a, b, rests[key], key, threshold) for a in own for b in own if a is not b]
            row["glide"][f"{key}/{threshold}/either"] = any(j[0] for j in joins)
            row["glide"][f"{key}/{threshold}/both"] = any(j[1] for j in joins)
        row["vib"][key] = max((run_vibrato(run, key) for run in same_pitch_runs(own)), default=0.0)
    row["drift"] = max((held_drift_cents(e["cents"]) for e in own), default=0.0)
    return row


def prf(rows: list[dict], truth_key: str, fired) -> tuple[int, int, int, int]:
    """(truth positives, detected of those, detector fired in total, notes)."""
    positives = [r for r in rows if r[truth_key]]
    return len(positives), sum(1 for r in positives if fired(r)), sum(1 for r in rows if fired(r)), len(rows)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "    - (0/0)"


DETECTORS = {
    "bend, fine contour": ("bend", lambda r: r["glide"][f"cents/{GLIDE_CENTS:.0f}/either"]),
    "bend, pitch_bends only": ("bend", lambda r: r["glide"][f"bend_cents/{GLIDE_CENTS:.0f}/either"]),
    # Added after the first IDMT run showed real bends that stay one note
    # (so not fixed in advance like the others): a glide, or one note whose
    # pitch holds two levels DRIFT_CENTS apart and has no vibrato.
    "bend, joins + drift": ("bend", lambda r: r["glide"][f"cents/{GLIDE_CENTS:.0f}/either"]
                            or (r["drift"] >= DRIFT_CENTS and r["vib"]["cents"] < VIBRATO_CENTS)),
    "vibrato, fine contour": ("vibrato", lambda r: r["vib"]["cents"] >= VIBRATO_CENTS),
    "vibrato, pitch_bends only": ("vibrato", lambda r: r["vib"]["bend_cents"] >= VIBRATO_CENTS),
}


SHOWN = ["one right note", "right pitch, split in 2+", "right + another pitch", "another pitch only", "nothing"]


def shown(shows: str) -> str:
    """score_note's "shows" string as one of SHOWN."""
    offsets = shows.split()
    if not offsets:
        return "nothing"
    if all(o == "0" for o in offsets):
        return SHOWN[0] if len(offsets) == 1 else SHOWN[1]
    return SHOWN[2] if "0" in offsets else SHOWN[3]


def report(rows: list[dict], label_key: str, sweep: bool) -> dict:
    """Prints the tables for rows (one per truth note, with "bend",
    "vibrato" and score_note's fields) and returns the headline numbers."""
    fired = {name: f for name, (_, f) in DETECTORS.items()}
    print(f"\n  by {label_key}: notes | bend fired (fine / pitch_bends / joins + drift) | vibrato fired (fine / pitch_bends) | tab shows")
    groups = defaultdict(list)
    for r in rows:
        groups[r[label_key]].append(r)
    for label, group in sorted(groups.items()):
        shows = Counter(r["shows"] or "nothing" for r in group).most_common(5)
        print(f"    {label:<16} {len(group):>4} | {sum(map(fired['bend, fine contour'], group)):>4} / "
              f"{sum(map(fired['bend, pitch_bends only'], group)):>4} / "
              f"{sum(map(fired['bend, joins + drift'], group)):>4} | "
              f"{sum(map(fired['vibrato, fine contour'], group)):>4} / "
              f"{sum(map(fired['vibrato, pitch_bends only'], group)):>4} | "
              + ", ".join(f"[{k}] x{v}" for k, v in shows))

    print(f"  what the tab shows today, by {label_key}: " + " | ".join(SHOWN))
    for label, group in sorted(groups.items()):
        counts = Counter(shown(r["shows"]) for r in group)
        print(f"    {label:<16} {len(group):>4} | " + " | ".join(
            f"{counts[name]:>4} ({100 * counts[name] / len(group):3.0f}%)" for name in SHOWN))

    out = {}
    for name, (key, f) in DETECTORS.items():
        positives, hit, fired_total, n = prf(rows, key, f)
        print(f"  {name:<26} recall {pct(hit, positives)}   precision {pct(hit, fired_total)}   "
              f"fired on {pct(fired_total - hit, n - positives)} of the notes without it")
        out[name] = {"positives": positives, "detected": hit, "fired": fired_total, "notes": n}

    if sweep:
        print("  sweep, bend: source / sides / cents -> recall / precision")
        for key, source in SOURCES.items():
            for sides in ("either", "both"):
                cells = []
                for threshold in (15, 25, 40):
                    p, h, f, _ = prf(rows, "bend", lambda r, k=f"{key}/{threshold}/{sides}": r["glide"][k])
                    cells.append(f">={threshold}: {100 * h / p if p else 0:.0f}/{100 * h / f if f else 0:.0f}")
                print(f"    {source:<12} {sides:<6} " + "   ".join(cells))
        print("  sweep, vibrato (fine contour): amplitude threshold in cents -> recall / precision")
        cells = []
        for cents in (6, 9, 12, 18, 25):
            p, h, f, _ = prf(rows, "vibrato", lambda r, c=cents: r["vib"]["cents"] >= c)
            cells.append(f">={cents}: {100 * h / p if p else 0:.0f}/{100 * h / f if f else 0:.0f}")
        print("    " + "   ".join(cells))
    return out


def score_file(path: str, truths: list[dict]) -> list[dict]:
    events = tab_notes(model_output_for(path))
    kept = [e for e in events if e["kept"]]
    rests = {key: resting_offset(kept, key) for key in SOURCES}
    return [{**truth, **score_note(truth, events, rests)} for truth in truths]


# ------------------------------------------------------------------ modes


def pyin_between_ms(path: str, start: float, end: float, midi_a: int, midi_b: int) -> float | None:
    """How long the played pitch spends between two pitches in [start, end]:
    milliseconds of pYIN f0 more than 30 cents away from both. A glide
    passes through the pitches in between; a picked or hammered step
    doesn't. None if mostly unvoiced. Only meaningful where one note sounds
    at a time."""
    low, high = min(midi_a, midi_b), max(midi_a, midi_b)
    audio, sr = librosa.load(path, sr=22050, mono=True, offset=max(0.0, start), duration=max(0.1, end - start))
    f0, voiced, _ = librosa.pyin(audio, fmin=librosa.midi_to_hz(low - 3), fmax=librosa.midi_to_hz(high + 3),
                                 sr=sr, frame_length=2048, hop_length=256)
    f0 = f0[voiced & ~np.isnan(f0)]
    if len(f0) < 8:
        return None
    cents = 1200 * np.log2(f0 / librosa.midi_to_hz(low))
    between = (cents > 30) & (cents < (high - low) * 100 - 30)
    return float(between.sum() * 256 / sr * 1000)


def run_synthetic(args) -> dict:
    results = {}
    all_rows = []
    for truth_path in sorted(glob.glob(os.path.join(DATA, "bend_testset", "bends_prog*.json"))):
        truth = json.load(open(truth_path))
        wav = os.path.join(os.path.dirname(truth_path), truth["wav"])
        rows = score_file(wav, truth["notes"])
        # pYIN's time between the two pitches where the truth is known: the
        # whole-step bends (150ms ramps) against the plain notes.
        medians = {}
        for kind in ("bend_whole", "plain"):
            values = [pyin_between_ms(wav, n["onset"], n["end"], n["midi"], n["midi"] + 2)
                      for n in truth["notes"] if n["type"] == kind]
            medians[kind] = np.median([v for v in values if v is not None])
        print(f"\n== synthetic, program {truth['program']} ({truth['program_name']}), {len(rows)} notes "
              f"(pYIN time between the two pitches: whole-step bends median {medians['bend_whole']:.0f}ms, "
              f"plain notes {medians['plain']:.0f}ms)")
        results[f"prog{truth['program']}"] = report(rows, "type", args.sweep)
        all_rows += rows
    print(f"\n== synthetic, all programs, {len(all_rows)} notes")
    results["all"] = report(all_rows, "type", args.sweep)
    return results


IDMT_STYLES = {"NO": "normal", "BE": "bend", "SL": "slide", "VI": "vibrato", "HA": "harmonic", "DN": "dead note"}


def idmt_files(root: str) -> list[tuple[str, list[dict]]]:
    """(wav, truth notes) for every annotated file of datasets 1 and 2.
    Dataset 1: single plain notes and chords on 4 guitar settings (all
    labelled normal). Dataset 2: note sequences and licks on 3 guitars with
    the expression styles labelled (bend, slide, vibrato, harmonic, dead
    note) and the plucking style (picked, finger-style, muted)."""
    files = []
    for xml_path in sorted(glob.glob(os.path.join(root, "**", "annotation", "*.xml"), recursive=True)):
        parts = xml_path.replace(os.sep, "/").split("/")
        dataset = next((p for p in parts if p in ("dataset1", "dataset2")), None)
        if dataset is None:
            continue
        tree = ET.parse(xml_path).getroot()
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(xml_path)), "audio")
        # Dataset 2's audioFileName has a leading backslash and sometimes an
        # older name; its wav is named like the annotation file.
        wav = os.path.join(audio_dir, tree.findtext(".//audioFileName").lstrip("\\/"))
        base = os.path.basename(xml_path)[:-4]
        if not os.path.exists(wav):
            wav = os.path.join(audio_dir, base + ".wav")
        if not os.path.exists(wav):
            # The Lick11 annotations (..._FNVSBHD.xml) belong to ..._FN.wav:
            # same events and length. Longest wav name the annotation's starts with.
            prefixes = sorted((w for w in os.listdir(audio_dir) if base.startswith(w[:-4])), key=len)
            if not prefixes:
                continue
            wav = os.path.join(audio_dir, prefixes[-1])
        group = f"dataset 1, {parts[-3]}" if dataset == "dataset1" else f"dataset 2, guitar {base.split('_')[0]}"
        notes = []
        for event in tree.iter("event"):
            style = event.findtext("expressionStyle")
            notes.append({"onset": float(event.findtext("onsetSec")), "end": float(event.findtext("offsetSec")),
                          "midi": int(event.findtext("pitch")), "style": IDMT_STYLES.get(style, style),
                          "excitation": event.findtext("excitationStyle"), "guitar": group, "dataset": dataset,
                          "bend": style == "BE", "vibrato": style == "VI", "file": os.path.basename(wav)})
        files.append((wav, notes))
    return files


def run_idmt(args) -> dict:
    files = idmt_files(os.path.join(DATA, "idmt"))
    rows = []
    for i, (wav, notes) in enumerate(files):
        rows += score_file(wav, notes)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(files)} files", flush=True)
    results = {}
    for dataset, title in (("dataset1", "dataset 1 (plain single notes and chords)"),
                           ("dataset2", "dataset 2 (labelled techniques)")):
        group = [r for r in rows if r["dataset"] == dataset]
        print(f"\n== IDMT-SMT-Guitar {title}: {len({r['file'] for r in group})} files, {len(group)} annotated notes")
        results[dataset] = report(group, "style", args.sweep and dataset == "dataset2")
        if dataset == "dataset2":
            # A slide is a glide too: the detector can't tell it from a bend.
            for row in group:
                row["bend_or_slide"] = row["style"] in ("bend", "slide")
            for name in ("bend, fine contour", "bend, pitch_bends only", "bend, joins + drift"):
                positives, hit, fired_total, n = prf(group, "bend_or_slide", DETECTORS[name][1])
                print(f"  as 'bend or slide': {name:<24} recall {pct(hit, positives)}   precision {pct(hit, fired_total)}")
                results[dataset][f"bend or slide / {name}"] = {"positives": positives, "detected": hit, "fired": fired_total}
    for guitar in sorted({r["guitar"] for r in rows if r["dataset"] == "dataset2"}):
        group = [r for r in rows if r["guitar"] == guitar]
        print(f"\n== {guitar}: {len(group)} notes")
        results[guitar] = report(group, "style", False)
    group = [r for r in rows if r["dataset"] == "dataset2"]
    print("\n== dataset 2 by plucking style (PK picked, FS finger-style, MU muted): notes | tab shows")
    for excitation in sorted({r["excitation"] for r in group}):
        sub = [r for r in group if r["excitation"] == excitation and r["style"] == "normal"]
        shows = Counter(r["shows"] or "nothing" for r in sub).most_common(4)
        print(f"    {excitation} normal {len(sub):>5} | " + ", ".join(f"[{k}] x{v}" for k, v in shows))
    return results


def run_egset12(args) -> dict:
    """EGSet12 marks no bends or vibrato, so every detection is a false
    positive against its Guitar Pro files. Where the notes involved sound
    alone, pYIN says whether the pitch really glides between them."""
    totals = Counter()
    found = []
    for wav in sorted(glob.glob("/app/data/egset12/[0-9][0-9].wav")):
        events = sorted((e for e in tab_notes(model_output_for(wav)) if e["kept"]), key=lambda e: e["start_time"])
        rest = resting_offset(events, "cents")

        def alone(start: float, end: float, members: list[dict]) -> bool:
            return not any(o["start_time"] < end and o["end_time"] > start
                           for o in events if not any(o is m for m in members))

        counts = Counter(notes=len(events))
        for a in events:
            for b in events:
                if a is b or abs(b["midi"] - a["midi"]) not in (1, 2) \
                        or not JOIN_GAP_S[0] <= b["start_time"] - a["end_time"] <= JOIN_GAP_S[1]:
                    continue
                counts["adjacent joins"] += 1
                if not glide(a, b, rest)[0]:
                    continue
                counts["joins"] += 1
                lone = alone(a["start_time"], b["end_time"], [a, b])
                between = None
                if lone:
                    between = pyin_between_ms(wav, a["end_time"] - 0.15, b["start_time"] + 0.15, a["midi"], b["midi"])
                found.append({"file": os.path.basename(wav), "kind": "bend", "start": round(a["start_time"], 2),
                              "notes": f"{a['pitch']}>{b['pitch']}", "alone": lone,
                              "pyin_between_ms": None if between is None else round(between)})
        for run in same_pitch_runs(events):
            amplitude = run_vibrato(run, "cents")
            if amplitude < VIBRATO_CENTS:
                continue
            counts["vibrato"] += 1
            found.append({"file": os.path.basename(wav), "kind": "vibrato", "start": round(run[0]["start_time"], 2),
                          "notes": run[0]["pitch"], "alone": alone(run[0]["start_time"], run[-1]["end_time"], run),
                          "amplitude_cents": round(amplitude, 1)})
        totals.update(counts)
        print(f"  {os.path.basename(wav)}: {counts['notes']} tab notes, {counts['adjacent joins']} joins 1-2 semitones "
              f"apart, bend fired on {counts['joins']}, vibrato fired on {counts['vibrato']}", flush=True)
    print(f"\n== EGSet12 clean, all 12: {totals['notes']} tab notes, 0 bends and 0 vibrato in the Guitar Pro files")
    print(f"  bend fired on    {totals['joins']} joins ({pct(totals['joins'], totals['adjacent joins'])} of the joins "
          f"1-2 semitones apart; {100 * totals['joins'] / totals['notes']:.1f} per 100 tab notes)")
    print(f"  vibrato fired on {totals['vibrato']} notes ({100 * totals['vibrato'] / totals['notes']:.1f} per 100 tab notes)")
    print("  every detection (for a join whose two notes sound alone: pYIN's time between the two pitches "
          "within 150ms of the join):")
    for m in found:
        print(f"    {m}")
    checked = [m for m in found if m.get("pyin_between_ms") is not None]
    gliding = [m for m in checked if m["pyin_between_ms"] >= 35]
    print(f"  of {totals['joins']} bend detections, {len(checked)} could be checked with pYIN; "
          f"the pitch spends 35ms or more between the two notes in {len(gliding)}")
    return {"totals": dict(totals), "detections": found}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["synthetic", "idmt", "egset12"])
    parser.add_argument("--sweep", action="store_true", help="also print the threshold sweeps")
    parser.add_argument("--save", help="write the headline numbers as JSON here")
    build_info.add_args(parser)
    args = parser.parse_args()
    build = build_info.guard(args.allow_stale)
    print(f"settings: glide >= {GLIDE_CENTS:.0f} cents over {GLIDE_FRAMES} frames on either side of a join; vibrato >= "
          f"{VIBRATO_CENTS} cents at {VIBRATO_BAND_HZ[0]}-{VIBRATO_BAND_HZ[1]} Hz; Basic Pitch onset "
          f"{transcribe.BASIC_PITCH_ONSET_THRESHOLD} / frame {transcribe.BASIC_PITCH_FRAME_THRESHOLD} / "
          f"{transcribe.BASIC_PITCH_MIN_NOTE_LENGTH_MS:.0f}ms")
    results = {"synthetic": run_synthetic, "idmt": run_idmt, "egset12": run_egset12}[args.mode](args)
    if args.save:
        with open(args.save, "w") as f:
            json.dump({"build": build, "mode": args.mode, "results": results}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
