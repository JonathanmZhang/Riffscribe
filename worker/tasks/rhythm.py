"""Beat grid, bar lines and note quantization for notation (the MusicXML
export in tasks/musicxml.py).

Pure and standard-library only: the backend image copies this file (and
musicxml.py) in through the "worker" build context in docker-compose.yml
and builds the export on request, so nothing here may import numpy,
Celery or other worker-only packages. Changing it needs a rebuild of the
backend as well as the worker.

Measured on EGSet12 against its Guitar Pro rhythms
(experiments/rhythm/README.md, experiments/rhythm/notation.py).

Bars: 4/4 by default. The effective beat grid is beat_this's beats kept at
one metrical level (regular_beats: it can switch to half or double tempo
for a stretch of a song), then halved (every other beat) or doubled
(midpoints added) by the job's tempo_factor.
The beats are grouped in 4s. The phase (which beat is the "1") is the one
most of beat_this's downbeats land on, moved later by the job's
bar_offset_beats (0-3). Both overrides are job-level and change only this
grouping and the quantization, never the transcription.

Quantization: every note start snaps to the nearest 16th of the beat it
falls in. Notes at the same 16th form one chord (one step). Each step's
length is snapped to the nearest of 16th, 8th, dotted 8th, quarter,
dotted quarter, half and whole, then cut at the next step, since the
notation is one voice. The length comes from length_rule:
  "per_step" (default)  chords (2+ notes) and fast single notes: Basic
                        Pitch's length (the longest note of the step);
                        other single notes: the gap to the next step.
  "sounding"            Basic Pitch's length everywhere.
  "next_onset"          the gap to the next step everywhere.
"""

import bisect
import math
import statistics

BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4  # quantize to 16th notes
SLOTS_PER_BAR = BEATS_PER_BAR * STEPS_PER_BEAT
TEMPO_FACTORS = (0.5, 1.0, 2.0)
BAR_OFFSETS = (0, 1, 2, 3)
# Note values, in 16ths: 16th, 8th, dotted 8th, quarter, dotted quarter, half, whole.
NOTE_VALUES = (1, 2, 3, 4, 6, 8, 16)
LENGTH_RULES = ("per_step", "sounding", "next_onset")
# A single note is "fast" when its step starts within this of a neighbour's:
# 16th notes at 110bpm, the same bound as EGSet12's "fast" segment label.
FAST_GAP_SECONDS = 60 / 110 / 4
# The tab's chord grouping window (fretboard.CHORD_ONSET_TOLERANCE_SECONDS;
# not imported, so this file stays free of worker dependencies).
STRUM_SECONDS = 0.15
# A downbeat counts for a beat when it's within this share of a beat of it.
DOWNBEAT_MATCH_BEATS = 0.25
# beat_this can switch metrical level mid-song (firefire: a half-tempo
# intro). regular_beats keeps one level: drops beats closer than SHORT_GAP
# spacings, fills gaps within FILL_TOLERANCE of 2 or 3 spacings.
REGULARIZE_BEATS = True
SHORT_GAP = 0.6
FILL_TOLERANCE = 0.15
# Grid for jobs without beats (older jobs, or beat tracking failed).
DEFAULT_TEMPO_BPM = 120


def check_overrides(tempo_factor: float, bar_offset_beats: int) -> None:
    if tempo_factor not in TEMPO_FACTORS:
        raise ValueError(f"tempo_factor must be one of {TEMPO_FACTORS}, got {tempo_factor!r}")
    if bar_offset_beats not in BAR_OFFSETS:
        raise ValueError(f"bar_offset_beats must be one of {BAR_OFFSETS}, got {bar_offset_beats!r}")


def _nearest(grid: list[float], t: float) -> int:
    i = bisect.bisect_left(grid, t)
    if i == 0:
        return 0
    if i == len(grid):
        return len(grid) - 1
    return i if grid[i] - t < t - grid[i - 1] else i - 1


def _matched_downbeats(grid: list[float], downbeats: list[float]) -> list[int]:
    """Grid index of every downbeat that lands on a beat of the grid."""
    out = []
    for d in downbeats:
        i = _nearest(grid, d)
        spacing = grid[min(i + 1, len(grid) - 1)] - grid[max(i - 1, 0)]
        spacing /= max(min(i + 1, len(grid) - 1) - max(i - 1, 0), 1)
        if abs(grid[i] - d) <= DOWNBEAT_MATCH_BEATS * spacing:
            out.append(i)
    return out


def regular_beats(beats: list[float]) -> list[float]:
    """beat_this's beats kept at one metrical level: a beat closer than
    SHORT_GAP of the median spacing to the last kept one is dropped (a
    stretch tracked at double tempo), and a gap of ~2 or ~3 spacings is
    filled evenly (a stretch tracked at half tempo, or missed beats)."""
    if len(beats) < 4:
        return beats
    spacing = _spacing(beats)
    kept = [beats[0]]
    for t in beats[1:]:
        if t - kept[-1] >= SHORT_GAP * spacing:
            kept.append(t)
    out = [kept[0]]
    for a, b in zip(kept, kept[1:]):
        n = round((b - a) / spacing)
        if n >= 2 and abs((b - a) / n / spacing - 1) <= FILL_TOLERANCE:
            out += [a + k * (b - a) / n for k in range(1, n)]
        out.append(b)
    return out


def beat_grid(beats: list[float], downbeats: list[float], tempo_factor: float = 1.0) -> list[float]:
    """The effective beats: beat_this's (kept at one metrical level if
    REGULARIZE_BEATS), doubled (midpoints added) or halved (every other
    beat, the half that carries more downbeats)."""
    beats = sorted(beats)
    if REGULARIZE_BEATS:
        beats = regular_beats(beats)
    if tempo_factor == 2.0 and len(beats) >= 2:
        doubled = [t for a, b in zip(beats, beats[1:]) for t in (a, (a + b) / 2)]
        return doubled + [beats[-1]]
    if tempo_factor == 0.5 and len(beats) >= 4:
        return max((beats[0::2], beats[1::2]), key=lambda g: len(_matched_downbeats(g, downbeats)))
    return beats


def bar_phase(grid: list[float], downbeats: list[float], bar_offset_beats: int = 0) -> int:
    """Index (mod 4) of the beats that start bars: the one most downbeats
    land on (ties: the earliest), moved later by bar_offset_beats."""
    votes = [0] * BEATS_PER_BAR
    for i in _matched_downbeats(grid, downbeats):
        votes[i % BEATS_PER_BAR] += 1
    return (votes.index(max(votes)) + bar_offset_beats) % BEATS_PER_BAR


def _spacing(values: list[float]) -> float:
    return statistics.median(b - a for a, b in zip(values, values[1:]))


def _extrapolator(grid: list[float]):
    """g(i) for any integer i: the grid, continued before and after at the
    median spacing of its first / last 8 beats."""
    before, after = _spacing(grid[:9]), _spacing(grid[-9:])

    def g(i: int) -> float:
        if i < 0:
            return grid[0] + i * before
        if i >= len(grid):
            return grid[-1] + (i - len(grid) + 1) * after
        return grid[i]

    return g


def effective_grid(beats: list[float], downbeats: list[float], tempo_factor: float = 1.0,
                   tempo_bpm: float = 0, start: float = 0.0) -> tuple[list[float], list[float]]:
    """(beats, downbeats) after tempo_factor. Without at least two beats,
    a constant grid from `start` at tempo_bpm (x tempo_factor; 120 if
    unknown), with a downbeat on its first beat."""
    grid = beat_grid(beats, downbeats, tempo_factor)
    if len(grid) >= 2:
        return grid, downbeats
    seconds_per_beat = 60.0 / ((tempo_bpm or DEFAULT_TEMPO_BPM) * tempo_factor)
    return [start + k * seconds_per_beat for k in range(2)], [start]


def bar_starts(beats: list[float], downbeats: list[float], tempo_factor: float = 1.0,
               bar_offset_beats: int = 0) -> list[float]:
    """Bar start times (seconds) within the beats: every 4th beat of the
    effective grid, from the bar phase. Empty without beats."""
    check_overrides(tempo_factor, bar_offset_beats)
    grid = beat_grid(beats, downbeats, tempo_factor)
    if len(grid) < 2:
        return []
    phase = bar_phase(grid, downbeats, bar_offset_beats)
    return [round(t, 3) for t in grid[phase::BEATS_PER_BAR]]


def bar_grid(beats: list[float], downbeats: list[float], start: float, end: float, tempo_factor: float = 1.0,
             bar_offset_beats: int = 0, tempo_bpm: float = 0) -> list[float]:
    """The beat grid the notation is written on: the effective beats,
    extended past both ends to cover [start, end], starting on a bar line
    (bars start at grid[0], grid[4], ...) and ending on one."""
    check_overrides(tempo_factor, bar_offset_beats)
    grid, downbeats = effective_grid(beats, downbeats, tempo_factor, tempo_bpm, start)
    phase = bar_phase(grid, downbeats, bar_offset_beats)
    g = _extrapolator(grid)
    # First bar: the last bar line at or before `start` (plus half a 16th, so
    # a note played just ahead of a bar line still lands in that bar).
    i = bisect.bisect_right(grid, start) - 1 if start >= grid[0] else \
        math.floor((start - grid[0]) / _spacing(grid[:9]))
    while g(i + 1) <= start + (g(i + 2) - g(i + 1)) / STEPS_PER_BEAT / 2:
        i += 1
    while g(i) > start + (g(i + 1) - g(i)) / STEPS_PER_BEAT / 2:
        i -= 1
    first = i - (i - phase) % BEATS_PER_BAR
    last = first + BEATS_PER_BAR
    while g(last) < end:
        last += BEATS_PER_BAR
    return [g(k) for k in range(first, last + 1)]


def quantize_time(t: float, grid: list[float]) -> int:
    """16th-note slot of time t on the grid (slot 0 = grid[0])."""
    j = min(max(bisect.bisect_right(grid, t) - 1, 0), len(grid) - 2)
    frac = (t - grid[j]) / (grid[j + 1] - grid[j])
    return max(j * STEPS_PER_BEAT + round(frac * STEPS_PER_BEAT), 0)


def slot_time(slot: int, grid: list[float]) -> float:
    """Time (seconds) of a 16th-note slot on the grid."""
    j = min(slot // STEPS_PER_BEAT, len(grid) - 2)
    return grid[j] + (slot - j * STEPS_PER_BEAT) / STEPS_PER_BEAT * (grid[j + 1] - grid[j])


def snap_length(sixteenths: float) -> int:
    """Nearest note value (in 16ths) by ratio."""
    sixteenths = max(sixteenths, 0.25)
    return min(NOTE_VALUES, key=lambda v: abs(math.log(sixteenths / v)))


def _slot_seconds(slot: int, grid: list[float]) -> float:
    j = min(slot // STEPS_PER_BEAT, len(grid) - 2)
    return (grid[j + 1] - grid[j]) / STEPS_PER_BEAT


def _strums(notes: list[dict]) -> list[list[dict]]:
    """The tab's steps: notes within STRUM_SECONDS of a group's first note."""
    groups: list[list[dict]] = []
    for note in notes:
        if groups and note["start_time"] - groups[-1][0]["start_time"] <= STRUM_SECONDS:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def quantize_notes(notes: list[dict], grid: list[float], length_rule: str = "per_step",
                   grouping: str = "slot") -> list[dict]:
    """Steps of the notation, in time order: {"slot", "length" (16ths,
    as notated), "kind" ("chord" | "fast" | "single"), "notes"}. Notes
    are the input dicts (start_time, end_time and, for the tab, string /
    fret / pitch). Notes that land on the same slot and string as an
    earlier note are dropped (a string plays one note at a time).

    grouping: "slot" - each note at its own nearest 16th, notes on the same
    16th form a chord; "strum" - the tab's 150ms groups, each placed at
    the 16th of its first note."""
    if length_rule not in LENGTH_RULES:
        raise ValueError(f"length_rule must be one of {LENGTH_RULES}")
    ordered = sorted(notes, key=lambda n: n["start_time"])
    if grouping == "slot":
        placed = [(quantize_time(n["start_time"], grid), n) for n in ordered]
    else:
        placed = [(quantize_time(g[0]["start_time"], grid), n) for g in _strums(ordered) for n in g]
    by_slot: dict[int, list[dict]] = {}
    for slot, note in placed:
        step = by_slot.setdefault(slot, [])
        if "string" in note and any(n["string"] == note["string"] for n in step):
            continue
        step.append(note)
    slots = sorted(by_slot)
    onsets = [by_slot[s][0]["start_time"] for s in slots]
    steps = []
    for k, slot in enumerate(slots):
        group = by_slot[slot]
        gaps = [onsets[k] - onsets[k - 1]] if k > 0 else []
        gaps += [onsets[k + 1] - onsets[k]] if k + 1 < len(slots) else []
        kind = "chord" if len(group) > 1 else "fast" if gaps and min(gaps) <= FAST_GAP_SECONDS else "single"
        sounding = max(n["end_time"] - n["start_time"] for n in group) / _slot_seconds(slot, grid)
        room = slots[k + 1] - slot if k + 1 < len(slots) else None
        use_gap = room is not None and (length_rule == "next_onset" or (length_rule == "per_step" and kind == "single"))
        length = snap_length(room if use_gap else sounding)
        steps.append({"slot": slot, "length": min(length, room) if room is not None else length, "kind": kind,
                      "notes": group})
    return steps


def quantize_chords(chords: list[dict], grid: list[float]) -> list[dict]:
    """Chord symbols to show: [{"slot", "name"}]. Each segment start snaps
    to a 16th; a symbol is kept when it lasts at least a beat and differs
    from the last one kept, so BTC's short flickers don't clutter the staff."""
    out: list[dict] = []
    for seg in chords:
        slot = quantize_time(seg["start"], grid)
        if quantize_time(seg["end"], grid) - slot < STEPS_PER_BEAT:
            continue
        if out and out[-1]["name"] == seg["name"]:
            continue
        if out and out[-1]["slot"] == slot:
            out[-1] = {"slot": slot, "name": seg["name"]}
        else:
            out.append({"slot": slot, "name": seg["name"]})
    return out


def measure_starts(result: dict, tempo_factor: float = 1.0, bar_offset_beats: int = 0) -> list[float]:
    """Start time (seconds) of every measure of the MusicXML export, in
    order: the export's bar lines on the audio's time axis (the first can be
    slightly negative, before the audio starts). The frontend uses them as
    sync points so the sheet-music cursor follows the recording."""
    score = notation(result, tempo_factor, bar_offset_beats)
    return [round(slot_time(bar * SLOTS_PER_BAR, score["grid"]), 3) for bar in range(score["bars"])]


def notation(result: dict, tempo_factor: float = 1.0, bar_offset_beats: int = 0,
             length_rule: str = "per_step", grouping: str = "slot") -> dict:
    """Everything the MusicXML export needs from a TabResult dict: the grid,
    tempo, number of bars, steps and chord symbols."""
    notes = result.get("notes") or []
    start = min((n["start_time"] for n in notes), default=0.0)
    end = max((n["end_time"] for n in notes), default=start)
    grid = bar_grid(result.get("beats") or [], result.get("downbeats") or [], start, end, tempo_factor,
                    bar_offset_beats, result.get("tempo_bpm") or 0)
    steps = quantize_notes(notes, grid, length_rule, grouping)
    bars = max(math.ceil(max((s["slot"] + s["length"] for s in steps), default=1) / SLOTS_PER_BAR), 1)
    return {
        "grid": grid,
        "tempo_bpm": round(60.0 / _spacing(grid)),
        "bars": bars,
        "steps": steps,
        "chords": [c for c in quantize_chords(result.get("chords") or [], grid) if c["slot"] < bars * SLOTS_PER_BAR],
    }
