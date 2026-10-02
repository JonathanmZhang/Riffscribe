"""Note cleanup from Basic Pitch's per-note pitch offsets: repairs what
vibrato and bends/slides do to the note list. Pure (numpy/scipy only), used
by transcribe.extract_notes and by the measurement scripts.

Basic Pitch gives each note one integer per frame (11.6ms), "bends": the
offset of the strongest contour bin from the note's pitch, in thirds of a
semitone. An unbent note reads about +1, not 0, so offsets are measured
from the file's resting offset (the median over all frames).

Two things go wrong with pitch movement (experiments/expression/README.md):
  - vibrato splits one note into several same-pitch notes with no gap;
  - a bend or slide ends the note and starts another 1-2 semitones away,
    with no gap, as the pitch glides across.

merge_vibrato joins a run of same-pitch notes that follow each other
without a gap when the run's pitch wobbles at 4-8 Hz (and marks the note
"vibrato"); its "wobble" variant only joins across a join that lies inside
the wobble. merge_glides joins a note into the one before it when the pitch
is moving across the join; the result keeps the first note's pitch. Neither
looks at how the note was plucked, so a glide can be a bend or a slide.
"""

import numpy as np
import scipy.signal

FRAMES_PER_SECOND = 22050 / 256  # Basic Pitch's contour frame rate
CENTS_PER_BIN = 100.0 / 3

# Two notes are joined ("no gap") when the second starts this close to the
# first one's end (seconds: overlap allowed, gap allowed).
JOIN_GAP_S = (-0.03, 0.06)
# Glide: the pitch offset over the last / first GLIDE_FRAMES of the two notes
# is this many cents towards the other note, on either side of the join.
GLIDE_CENTS = 25.0
GLIDE_FRAMES = 3
# Vibrato: the strongest 4-8 Hz component of the pitch, after removing slow
# movement, is at least this large and holds at least this share of the
# moving energy, over at least 4 of its cycles.
VIBRATO_CENTS = 12.0
VIBRATO_BAND_HZ = (4.0, 8.0)
VIBRATO_ENERGY_SHARE = 0.5
# "wobble" joins: within a run that has vibrato, two pieces are only merged
# if the wobble is there on both sides of their join, looking this many
# frames each way (0.5s). A side shorter than the minimum (0.3s, about two
# cycles) can't show one, so the join is left alone.
WOBBLE_FRAMES = 43
WOBBLE_MIN_FRAMES = 26


def offsets_cents(note: dict) -> np.ndarray:
    """A note's per-frame pitch offsets in cents ("cents" if the caller
    supplied a finer contour, else Basic Pitch's integer "bends")."""
    if "cents" in note:
        return np.asarray(note["cents"], dtype=float)
    return np.asarray(note["bends"], dtype=float) * CENTS_PER_BIN


def resting_offset(notes: list[dict]) -> float:
    """The offset of an unbent note in this file: the median over all frames."""
    if not notes:
        return 0.0
    return float(np.median(np.concatenate([offsets_cents(n) for n in notes])))


def joined(a: dict, b: dict) -> bool:
    return JOIN_GAP_S[0] <= b["start_time"] - a["end_time"] <= JOIN_GAP_S[1]


def is_glide(a: dict, b: dict, rest: float, threshold: float = GLIDE_CENTS) -> bool:
    """Does the pitch glide from note a into note b (1-2 semitones apart,
    joined)? True if a ends moving towards b or b starts coming from a."""
    step = b["midi"] - a["midi"]
    if abs(step) not in (1, 2) or not joined(a, b):
        return False
    direction = 1 if step > 0 else -1
    leaving = direction * (float(np.mean(offsets_cents(a)[-GLIDE_FRAMES:])) - rest) >= threshold
    arriving = -direction * (float(np.mean(offsets_cents(b)[:GLIDE_FRAMES])) - rest) >= threshold
    return leaving or arriving


def vibrato_cents(series: np.ndarray) -> float:
    """Amplitude in cents of the strongest 4-8 Hz component of a pitch
    series (one value per frame), or 0 if it doesn't qualify as vibrato."""
    x = np.asarray(series, dtype=float)[8:]  # skip the attack
    if len(x) < 24:
        return 0.0
    x = x - scipy.signal.medfilt(x, 21)
    window = np.hanning(len(x))
    n = 4 * len(x)
    spectrum = np.abs(np.fft.rfft(x * window, n))
    freqs = np.fft.rfftfreq(n, 1 / FRAMES_PER_SECOND)
    band = (freqs >= VIBRATO_BAND_HZ[0]) & (freqs <= VIBRATO_BAND_HZ[1])
    if not band.any():
        return 0.0
    k = int(np.argmax(np.where(band, spectrum, 0)))
    rate = float(freqs[k])
    power = spectrum**2
    near = power[np.abs(freqs - rate) <= 1.5].sum()
    moving = power[freqs >= 1.5].sum()
    if moving <= 0 or near / moving < VIBRATO_ENERGY_SHARE or len(x) / FRAMES_PER_SECOND < 4 / rate:
        return 0.0
    return float(2 * spectrum[k] / window.sum())


def same_pitch_runs(notes: list[dict]) -> list[list[dict]]:
    """notes (in time order) grouped into runs of one pitch with no gap."""
    runs: list[list[dict]] = []
    for note in notes:
        for run in runs:
            if run[-1]["midi"] == note["midi"] and joined(run[-1], note):
                run.append(note)
                break
        else:
            runs.append([note])
    return runs


def _absorb(first: dict, rest: list[dict], **extra) -> dict:
    """first extended over rest: its pitch and start, the latest end, the
    highest amplitude, the offsets end to end."""
    series = [first, *rest]
    if any(n.get("vibrato") for n in series):
        extra["vibrato"] = True
    merged = {**first, **extra,
              "end_time": max(n["end_time"] for n in series),
              "amplitude": max(n["amplitude"] for n in series),
              "merged": first.get("merged", 0) + sum(1 + n.get("merged", 0) for n in rest)}
    for key in ("bends", "cents"):
        if key in first:
            merged[key] = [v for n in series for v in n[key]]
    return merged


def wobble_cents(segment: np.ndarray) -> float:
    """Amplitude in cents of the strongest 4-8 Hz component of a short,
    already detrended stretch of pitch; 0 if it is too short to tell."""
    x = np.asarray(segment, dtype=float)
    if len(x) < WOBBLE_MIN_FRAMES:
        return 0.0
    window = np.hanning(len(x))
    n = 8 * len(x)
    spectrum = np.abs(np.fft.rfft(x * window, n))
    freqs = np.fft.rfftfreq(n, 1 / FRAMES_PER_SECOND)
    band = (freqs >= VIBRATO_BAND_HZ[0]) & (freqs <= VIBRATO_BAND_HZ[1])
    return float(2 * spectrum[band].max() / window.sum())


def _wobble_groups(run: list[dict], threshold: float) -> list[list[dict]]:
    """The pieces of a run with vibrato, grouped across the joins that lie
    inside the wobble. A piece struck again before the vibrato starts, or
    after it has stopped, stays its own note."""
    series = np.concatenate([offsets_cents(n) for n in run])
    series = series - scipy.signal.medfilt(series, 21)
    groups, frame = [[run[0]]], 0
    for previous, note in zip(run, run[1:]):
        frame += len(offsets_cents(previous))
        before = series[max(0, frame - WOBBLE_FRAMES):frame]
        after = series[frame:frame + WOBBLE_FRAMES]
        if wobble_cents(before) >= threshold and wobble_cents(after) >= threshold:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def merge_vibrato(notes: list[dict], joins: str = "run", threshold: float = VIBRATO_CENTS,
                  wobble_threshold: float = VIBRATO_CENTS) -> list[dict]:
    """A run of joined same-pitch notes with vibrato becomes one note,
    marked "vibrato". joins="run": the whole run. joins="wobble": only
    across the joins that lie inside the wobble (_wobble_groups). A single
    note with vibrato is only marked."""
    out = []
    for run in same_pitch_runs(sorted(notes, key=lambda n: n["start_time"])):
        if vibrato_cents(np.concatenate([offsets_cents(n) for n in run])) < threshold:
            out.extend(run)
            continue
        for group in ([run] if joins == "run" else _wobble_groups(run, wobble_threshold)):
            if len(group) > 1 or len(run) == 1 or vibrato_cents(offsets_cents(group[0])) >= threshold:
                out.append(_absorb(group[0], group[1:], vibrato=True))
            else:
                out.append(group[0])
    return sorted(out, key=lambda n: (n["start_time"], n["midi"]))


def merge_glides(notes: list[dict], threshold: float = GLIDE_CENTS) -> list[dict]:
    """A note the pitch glides into is absorbed by the note it glides from;
    chains (fret, +1, +2, or up and back) collapse into their first note."""
    ordered = sorted(notes, key=lambda n: n["start_time"])
    rest = resting_offset(ordered)
    root = list(range(len(ordered)))  # index of the note each one is absorbed into
    for j, b in enumerate(ordered):
        for i in range(j - 1, -1, -1):
            a = ordered[i]
            if b["start_time"] - a["start_time"] > 30:  # no note is this long
                break
            if is_glide(a, b, rest, threshold):
                root[j] = root[i]
                break
    out = []
    for i, note in enumerate(ordered):
        if root[i] != i:
            continue
        absorbed = [ordered[j] for j in range(len(ordered)) if root[j] == i and j != i]
        out.append(_absorb(note, absorbed, glide=True) if absorbed else note)
    return sorted(out, key=lambda n: (n["start_time"], n["midi"]))


VIBRATO_JOINS = {1: "run", 2: "wobble"}


def cleanup(notes: list[dict], vibrato_merge: int, glide_merge: bool) -> list[dict]:
    """The enabled merges, vibrato first (so a bend held with vibrato is one
    note before its glide is looked at). vibrato_merge: 0 off, 1 whole runs,
    2 only across joins inside the wobble. Notes need "midi", times,
    "amplitude" and "bends"."""
    if vibrato_merge:
        notes = merge_vibrato(notes, VIBRATO_JOINS[int(vibrato_merge)])
    if glide_merge:
        notes = merge_glides(notes)
    return notes
