"""Validates Riffscribe MusicXML exports: the official MusicXML 4.0 XSD
(W3C), then a music21 parse with structural checks. Runs in a throwaway
container, so the worker and backend images don't carry music21:

    docker run --rm -v "$(pwd -W)/experiments/musicxml:/v" python:3.11-slim \
        sh -c "pip install -q music21==9.3.0 xmlschema==3.4.3 && python /v/validate.py /v/out/*.musicxml"

(Git Bash: prefix MSYS_NO_PATHCONV=1.) The XSD files are downloaded once
into /v/xsd from the W3C musicxml repository (tag v4.0), with their imports
pointed at the local copies.

music21 checks, per file: one part; every bar is 4/4 and filled to 4
quarters on both staves; every note on the TAB staff has a string and a
fret; tied notes pair up; chord symbols are counted. Exits 1 on any failure.
"""

import os
import sys
import urllib.request

XSD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xsd")
XSD_URL = "https://raw.githubusercontent.com/w3c/musicxml/v4.0/schema/{}"


def schema():
    import xmlschema

    os.makedirs(XSD_DIR, exist_ok=True)
    for name in ("musicxml.xsd", "xlink.xsd", "xml.xsd"):
        path = os.path.join(XSD_DIR, name)
        if not os.path.exists(path):
            text = urllib.request.urlopen(XSD_URL.format(name)).read().decode("utf-8")
            text = text.replace("http://www.musicxml.org/xsd/xml.xsd", "xml.xsd") \
                       .replace("http://www.musicxml.org/xsd/xlink.xsd", "xlink.xsd")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    return xmlschema.XMLSchema(os.path.join(XSD_DIR, "musicxml.xsd"))


def music21_checks(path: str) -> list[str]:
    from music21 import converter, harmony, meter, note, stream  # noqa: F401

    problems = []
    score = converter.parse(path)
    parts = score.parts
    if len(parts) != 1:
        # music21 imports a two-staff part as one PartStaff per staff.
        staves = [p for p in parts if isinstance(p, stream.PartStaff)]
        if len(staves) != 2:
            problems.append(f"expected one part with two staves, got {len(parts)} parts")
    tab_notes, tab_missing, ties_open = 0, 0, 0
    for staff_index, staff in enumerate(parts):
        for m in staff.getElementsByClass(stream.Measure):
            ts = m.timeSignature or (staff.recurse().getElementsByClass(meter.TimeSignature).first())
            if ts is None or ts.ratioString != "4/4":
                problems.append(f"staff {staff_index + 1} bar {m.number}: time signature {ts}")
            if abs(m.duration.quarterLength - 4.0) > 1e-6:
                problems.append(f"staff {staff_index + 1} bar {m.number}: {m.duration.quarterLength} quarters")
        for n in staff.recurse().notes:
            if n.tie is not None and n.tie.type == "start":
                ties_open += 1
            if n.tie is not None and n.tie.type == "stop":
                ties_open -= 1
    if ties_open:
        problems.append(f"{ties_open} unmatched tie start(s)")
    # String/fret straight from the XML: music21 moves a chord's technical
    # indications onto the chord, so they can't be read per note there.
    import xml.etree.ElementTree as ET

    for el in ET.parse(path).getroot().iter("note"):
        if el.findtext("staff") == "2" and el.find("rest") is None:
            tab_notes += 1
            if el.find("notations/technical/string") is None or el.find("notations/technical/fret") is None:
                tab_missing += 1
    if tab_missing:
        problems.append(f"{tab_missing}/{tab_notes} TAB notes without string/fret")
    chord_symbols = len(score.recurse().getElementsByClass(harmony.ChordSymbol))
    notes = sum(len(n.pitches) for n in parts[0].recurse().notes if not isinstance(n, harmony.ChordSymbol))
    rests = len(parts[0].recurse().getElementsByClass(note.Rest))
    bars = len(parts[0].getElementsByClass(stream.Measure))
    print(f"    music21: {len(parts)} staves, {bars} bars, {notes} notation notes, {tab_notes} TAB notes, "
          f"{rests} rests, {chord_symbols} chord symbols, tempo "
          f"{[t.number for t in score.recurse().getElementsByClass('MetronomeMark')][:1]}")
    return problems


def main() -> None:
    xsd = schema()
    failed = False
    for path in sys.argv[1:]:
        print(path)
        errors = list(xsd.iter_errors(path))
        print(f"    XSD (MusicXML 4.0): {'valid' if not errors else f'{len(errors)} error(s)'}")
        for e in errors[:5]:
            print(f"      {e.reason} at {e.path}")
        problems = music21_checks(path)
        for p in problems[:10]:
            print(f"    PROBLEM: {p}")
        failed |= bool(errors or problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
