"""Chord label parsing and comparison for the chord-recognition experiment.

Every chord - a model's label, a ground-truth column, a baseline guess - is
reduced to a Chord: root pitch class, the set of pitch classes, and the
bass pitch class. Root None means "no chord" (N/X, or a pitch set that
matches no template).
"""

import re
from typing import NamedTuple

import mir_eval

from scripts.egset12_benchmark import CHORD_TEMPLATES, PC_NAMES

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class Chord(NamedTuple):
    root: int | None
    pcs: frozenset
    bass: int | None
    label: str

    @property
    def intervals(self) -> frozenset:
        return frozenset((p - self.root) % 12 for p in self.pcs) if self.root is not None else frozenset()

    @property
    def majmin(self) -> str | None:
        """'maj' / 'min' for chords with a perfect fifth and one clear third
        (both thirds, as in 7#9, count as major); None for sus, 5, dim, aug,
        m7b5 etc., which majmin scoring leaves out (as mir_eval's does)."""
        iv = self.intervals
        if self.root is None or 7 not in iv:
            return None
        if 4 in iv:
            return "maj"
        if 3 in iv:
            return "min"
        return None


NO_CHORD = Chord(None, frozenset(), None, "N")


def pc_of(name: str) -> int:
    pc = NOTE_PC[name[0].upper()]
    for accidental in name[1:]:
        pc += {"#": 1, "b": -1}[accidental]
    return pc % 12


def from_template(root: int, suffix: str, bass: int | None = None) -> Chord:
    pcs = frozenset((root + i) % 12 for i in CHORD_TEMPLATES[suffix])
    name = PC_NAMES[root] + suffix + (f"/{PC_NAMES[bass]}" if bass is not None and bass != root else "")
    return Chord(root, pcs, root if bass is None else bass, name)


def from_midis(midis: list[int]) -> Chord:
    """Same rule as egset12_benchmark.chord_name (which named the ground
    truth): exact template match over the pitch classes, bass note preferred
    as root. Unmatched sets get root None but keep their pitch classes."""
    pcs = frozenset(m % 12 for m in midis)
    bass = min(midis) % 12
    for root in sorted(pcs, key=lambda pc: pc != bass):
        intervals = tuple(sorted((pc - root) % 12 for pc in pcs))
        for suffix, template in CHORD_TEMPLATES.items():
            if intervals == tuple(sorted(template)):
                return from_template(root, suffix, bass)
    return Chord(None, pcs, bass, "(" + " ".join(PC_NAMES[p] for p in sorted(pcs)) + ")")


def from_harte(label: str) -> Chord:
    """BTC / crema labels (Harte syntax, e.g. 'C:min7/b7', 'D', 'N')."""
    if label in ("N", "X", ""):
        return Chord(None, frozenset(), None, label)
    root, bitmap, bass = mir_eval.chord.encode(label)
    if root < 0:
        return Chord(None, frozenset(), None, label)
    pcs = frozenset((root + i) % 12 for i in range(12) if bitmap[i])
    return Chord(int(root), pcs, int((root + bass) % 12), label)


# Chordino's dictionary suffixes -> Harte qualities.
CHORDINO_QUALITY = {
    "": "maj", "m": "min", "dim": "dim", "aug": "aug", "7": "7", "maj7": "maj7", "m7": "min7", "6": "maj6",
    "m6": "min6", "dim7": "dim7", "m7b5": "hdim7", "sus4": "sus4", "sus2": "sus2", "9": "9", "maj9": "maj9",
    "m9": "min9", "7sus4": "sus4(b7)", "add9": "maj(9)", "madd9": "min(9)", "mmaj7": "minmaj7",
}


def from_chordino(label: str) -> Chord:
    """Chordino labels, e.g. 'Cmaj7/E', 'F#m', 'Bbm7b5', 'N'."""
    if label in ("N", ""):
        return Chord(None, frozenset(), None, label)
    m = re.fullmatch(r"([A-G][#b]?)([^/]*)(?:/([A-G][#b]?))?", label)
    if not m or m.group(2) not in CHORDINO_QUALITY:
        raise ValueError(f"unparsed Chordino label {label!r}")
    root = pc_of(m.group(1))
    chord = from_harte(f"{PC_NAMES[root]}:{CHORDINO_QUALITY[m.group(2)]}")
    bass = pc_of(m.group(3)) if m.group(3) else root
    return Chord(chord.root, chord.pcs, bass, label)


def score(truth: Chord, est: Chord) -> dict:
    """Per-column scores; None where the metric doesn't apply to this truth."""
    named = truth.root is not None
    return {
        "root": (est.root == truth.root) if named else None,
        "majmin": (est.root == truth.root and est.majmin == truth.majmin) if named and truth.majmin else None,
        "exact": (est.pcs == truth.pcs) if named else None,
        "exact_bass": (est.pcs == truth.pcs and est.bass == truth.bass) if named else None,
    }


def label_at(segments: list[dict], t: float, window: float = 0.3) -> str:
    """The label covering most of [t, t + window] (segments: start, end, label)."""
    best, best_overlap = "N", 0.0
    for s in segments:
        overlap = min(s["end"], t + window) - max(s["start"], t)
        if overlap > best_overlap:
            best, best_overlap = s["label"], overlap
    return best
