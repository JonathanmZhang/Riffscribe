"""MusicXML export of a finished tab: one guitar part with two staves, standard
notation (treble clef, sounding an octave lower) and a 6-line TAB staff with
each note's string and fret, in 4/4. Note starts and lengths are quantized
by tasks/rhythm.py; lengths crossing a bar line are tied, gaps are rests,
and BTC's chord names are chord symbols above the notation staff.

Written directly with ElementTree (MusicXML 4.0, partwise) rather than via
music21, whose tablature export can't put notation and TAB for the same
notes in one part. Pure and standard-library only, like tasks/rhythm.py:
the backend imports it to serve GET /jobs/{id}/musicxml.
"""

import datetime
import xml.etree.ElementTree as ET

from tasks import rhythm

DIVISIONS = rhythm.STEPS_PER_BEAT  # <duration> units per quarter note: 16ths
BAR = rhythm.SLOTS_PER_BAR
# Written note values, in 16ths: (type, dots). Longer lengths are tied.
NOTE_TYPES = {16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1), 4: ("quarter", 0),
              3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0)}
# Standard tuning, TAB staff line 1 (bottom) = string 6 (low E).
TUNING = [("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)]
# BTC's chord suffixes (tasks/chords.QUALITY_SUFFIX) -> MusicXML <kind>.
CHORD_KINDS = {"": "major", "m": "minor", "dim": "diminished", "aug": "augmented", "6": "major-sixth",
               "m6": "minor-sixth", "7": "dominant", "m7": "minor-seventh", "maj7": "major-seventh",
               "m(maj7)": "major-minor", "dim7": "diminished-seventh", "m7b5": "half-diminished",
               "sus2": "suspended-second", "sus4": "suspended-fourth"}
NOTATION_VOICE, TAB_VOICE = "1", "5"  # MuseScore's convention: voices 1-4 per staff
# Playback tones: the General MIDI program (1-128, as MusicXML's midi-program
# counts; alphaTab and MIDI use this minus 1) and instrument name written
# into the part. Only what a synth plays changes, never the notes.
PLAYBACK_TONES = {
    "clean": (28, "Electric Guitar"),
    "overdriven": (30, "Overdriven Guitar"),
    "distorted": (31, "Distortion Guitar"),
    "acoustic": (26, "Acoustic Guitar (steel)"),
}
DEFAULT_TONE = "clean"
DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">')


def _split_name(name: str) -> tuple[str, int, str]:
    """'C#4' -> ('C', 1, '4'); 'Ebm7' -> ('E', -1, 'm7'). Letter, alter, rest.
    No pitch octave or BTC chord suffix starts with '#' or 'b', so every
    '#'/'b' right after the letter is an accidental."""
    alter, rest = 0, name[1:]
    while rest[:1] in ("#", "b") and rest:
        alter += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return name[0].upper(), alter, rest


def _pitch_parts(pitch: str) -> tuple[str, int, int]:
    step, alter, octave = _split_name(pitch)
    return step, alter, int(octave)


def _midi(pitch: str) -> int:
    step, alter, octave = _pitch_parts(pitch)
    return 12 * (octave + 1) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + alter


def _pieces(length: int) -> list[int]:
    """A length in 16ths as written note values, longest first."""
    out = []
    while length > 0:
        piece = next(v for v in NOTE_TYPES if v <= length)
        out.append(piece)
        length -= piece
    return out


def _timeline(steps: list[dict], bars: int, cuts: set[int]) -> list[dict]:
    """Notes and rests covering every bar: [{"start", "length", "step" (None
    for a rest), "tie_stop", "tie_start"}], each a single written value
    within one bar. Notes split at bar lines are tied; rests are also split
    at `cuts` (chord symbol positions)."""
    spans, cursor = [], 0
    for step in steps:
        if step["slot"] > cursor:
            spans.append((cursor, step["slot"], None))
        spans.append((step["slot"], step["slot"] + step["length"], step))
        cursor = step["slot"] + step["length"]
    if cursor < bars * BAR:
        spans.append((cursor, bars * BAR, None))

    events = []
    for start, end, step in spans:
        bounds = {b for b in range(start - start % BAR + BAR, end, BAR)}
        if step is None:
            bounds |= {c for c in cuts if start < c < end}
        edges = [start] + sorted(bounds) + [end]
        written = []
        for a, b in zip(edges, edges[1:]):
            for piece in _pieces(b - a):
                written.append({"start": a, "length": piece, "step": step})
                a += piece
        for k, ev in enumerate(written):
            ev["tie_stop"] = step is not None and k > 0
            ev["tie_start"] = step is not None and k < len(written) - 1
        events += written
    return events


def _sub(parent: ET.Element, tag: str, value=None, **attrib) -> ET.Element:
    """Child element with text `value` (if given) and attributes (a_b -> a-b)."""
    el = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrib.items()})
    if value is not None:
        el.text = str(value)
    return el


def _note(measure: ET.Element, ev: dict, voice: str, staff: int, note: dict | None, chord: bool) -> None:
    el = _sub(measure, "note")
    if chord:
        _sub(el, "chord")
    if note is None:
        _sub(el, "rest")
    else:
        step, alter, octave = _pitch_parts(note["pitch"])
        pitch = _sub(el, "pitch")
        _sub(pitch, "step", step)
        if alter:
            _sub(pitch, "alter", alter)
        _sub(pitch, "octave", octave)
    _sub(el, "duration", ev["length"])
    if ev["tie_stop"]:
        _sub(el, "tie", type="stop")
    if ev["tie_start"]:
        _sub(el, "tie", type="start")
    _sub(el, "voice", voice)
    note_type, dots = NOTE_TYPES[ev["length"]]
    _sub(el, "type", note_type)
    for _ in range(dots):
        _sub(el, "dot")
    _sub(el, "staff", staff)
    tab = staff == 2 and note is not None
    if ev["tie_stop"] or ev["tie_start"] or tab:
        notations = _sub(el, "notations")
        if ev["tie_stop"]:
            _sub(notations, "tied", type="stop")
        if ev["tie_start"]:
            _sub(notations, "tied", type="start")
        if tab:
            technical = _sub(notations, "technical")
            _sub(technical, "string", note["string"])
            _sub(technical, "fret", note["fret"])
    # Vibrato (the note's "vibrato" flag): a wavy line on both staves, on the
    # note's first piece when it is tied over a bar line.
    if note is not None and note.get("vibrato") and not ev["tie_stop"]:
        notations = el.find("notations")
        if notations is None:
            notations = _sub(el, "notations")
        ornaments = _sub(notations, "ornaments")
        _sub(ornaments, "wavy-line", type="start", number=1)
        _sub(ornaments, "wavy-line", type="stop", number=1)


def _harmony(measure: ET.Element, name: str, offset: int) -> None:
    step, alter, suffix = _split_name(name)
    el = _sub(measure, "harmony")
    root = _sub(el, "root")
    _sub(root, "root-step", step)
    if alter:
        _sub(root, "root-alter", alter)
    _sub(el, "kind", CHORD_KINDS.get(suffix, "other"), text=suffix)
    if offset:
        _sub(el, "offset", offset)
    _sub(el, "staff", 1)


def _attributes(measure: ET.Element) -> None:
    attributes = _sub(measure, "attributes")
    _sub(attributes, "divisions", DIVISIONS)
    key = _sub(attributes, "key")
    _sub(key, "fifths", 0)
    time = _sub(attributes, "time")
    _sub(time, "beats", rhythm.BEATS_PER_BAR)
    _sub(time, "beat-type", 4)
    _sub(attributes, "staves", 2)
    clef = _sub(attributes, "clef", number=1)
    _sub(clef, "sign", "G")
    _sub(clef, "line", 2)
    _sub(clef, "clef-octave-change", -1)  # guitar sounds an octave below written
    tab_clef = _sub(attributes, "clef", number=2)
    _sub(tab_clef, "sign", "TAB")
    _sub(tab_clef, "line", 5)
    details = _sub(attributes, "staff-details", number=2)
    _sub(details, "staff-lines", len(TUNING))
    for line, (step, octave) in enumerate(TUNING, start=1):
        tuning = _sub(details, "staff-tuning", line=line)
        _sub(tuning, "tuning-step", step)
        _sub(tuning, "tuning-octave", octave)


def build(score: dict, title: str, tone: str = DEFAULT_TONE) -> str:
    """MusicXML text for a rhythm.notation() score. tone: a PLAYBACK_TONES key."""
    program, instrument_name = PLAYBACK_TONES[tone]
    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", title)
    encoding = _sub(_sub(root, "identification"), "encoding")
    _sub(encoding, "software", "Riffscribe")
    _sub(encoding, "encoding-date", datetime.date.today().isoformat())
    part_list = _sub(root, "part-list")
    score_part = _sub(part_list, "score-part", id="P1")
    _sub(score_part, "part-name", "Guitar")
    instrument = _sub(score_part, "score-instrument", id="P1-I1")
    _sub(instrument, "instrument-name", instrument_name)
    midi = _sub(score_part, "midi-instrument", id="P1-I1")
    _sub(midi, "midi-channel", 1)
    _sub(midi, "midi-program", program)
    part = _sub(root, "part", id="P1")

    chords = {c["slot"]: c["name"] for c in score["chords"]}
    events = _timeline(score["steps"], score["bars"], set(chords))
    for bar in range(score["bars"]):
        measure = _sub(part, "measure", number=bar + 1)
        if bar == 0:
            _attributes(measure)
            direction = _sub(measure, "direction", placement="above")
            metronome = _sub(_sub(direction, "direction-type"), "metronome")
            _sub(metronome, "beat-unit", "quarter")
            _sub(metronome, "per-minute", score["tempo_bpm"])
            _sub(direction, "staff", 1)
            _sub(direction, "sound", tempo=score["tempo_bpm"])
        in_bar = [ev for ev in events if bar * BAR <= ev["start"] < (bar + 1) * BAR]
        for staff, voice in ((1, NOTATION_VOICE), (2, TAB_VOICE)):
            if staff == 2:
                _sub(_sub(measure, "backup"), "duration", BAR)
            for ev in in_bar:
                if staff == 1:
                    for slot in sorted(s for s in chords if ev["start"] <= s < ev["start"] + ev["length"]):
                        _harmony(measure, chords[slot], slot - ev["start"])
                notes = sorted(ev["step"]["notes"], key=lambda n: _midi(n["pitch"])) if ev["step"] else [None]
                for k, note in enumerate(notes):
                    _note(measure, ev, voice, staff, note, chord=k > 0)
        if bar == score["bars"] - 1:
            _sub(_sub(measure, "barline", location="right"), "bar-style", "light-heavy")

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + DOCTYPE + "\n" + ET.tostring(root, encoding="unicode") + "\n"


def to_musicxml(result: dict, tempo_factor: float = 1.0, bar_offset_beats: int = 0,
                title: str = "Riffscribe transcription", tone: str = DEFAULT_TONE) -> str:
    """MusicXML text for a TabResult dict, with the job's overrides and the
    playback tone (a PLAYBACK_TONES key)."""
    return build(rhythm.notation(result, tempo_factor, bar_offset_beats), title, tone)
