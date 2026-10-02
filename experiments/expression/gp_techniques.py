"""Counts playing-technique markings in EGSet12's Guitar Pro files (GP7/8:
Content/score.gpif inside the .gp zip), and checks whether the JAMS
pitch_contour annotations hold any real pitch movement. Standard library
only; measurement only.

    python experiments/expression/gp_techniques.py [data/egset12]

In GPIF a technique is a <Property> of a <Note> (Bended, Slide, HopoOrigin,
HopoDestination, PalmMuted, Muted, Harmonic, Tapped, LeftHandTapped) or a
child element of it (<Vibrato>, <LetRing/>, <Tie>, <Accent>, <Trill>), or a
child of a <Beat> (<Tremolo>, <Whammy>, <Arpeggio>, ...). A <Note> is stored
once and referenced by every beat that plays it, so the script also counts
the played occurrences.
"""

import collections
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/egset12"
# Properties every note has: pitch and position, not techniques.
PLAIN_NOTE_PROPERTIES = {"ConcertPitch", "Fret", "Midi", "String", "TransposedPitch"}
PLAIN_NOTE_CHILDREN = {"Properties", "InstrumentArticulation"}
# Beat children and properties that are layout, rhythm or pickup settings.
PLAIN_BEAT_CHILDREN = {"Rhythm", "Notes", "Dynamic", "Properties", "Legato", "TransposedPitchStemOrientation",
                       "ConcertPitchStemOrientation", "UserTransposedPitchStemOrientation"}
PLAIN_BEAT_PROPERTIES = {"PrimaryPickupVolume", "PrimaryPickupTone"}
TECHNIQUES = {
    "bends": ["note property Bended"],
    "slides": ["note property Slide"],
    "hammer-ons / pull-offs": ["note property HopoOrigin", "note property HopoDestination"],
    "vibrato": ["note <Vibrato>", "beat <Whammy>"],
    "palm mutes": ["note property PalmMuted"],
}


def count(path: str) -> tuple[collections.Counter, int, int]:
    root = ET.fromstring(zipfile.ZipFile(path).read("Content/score.gpif"))
    found = collections.Counter()
    beats = list(root.find("Beats"))
    played = collections.Counter(note_id for b in beats for note_id in (b.findtext("Notes") or "").split())
    notes = list(root.find("Notes"))
    for note in notes:
        times = played.get(note.get("id"), 0)
        for prop in note.iter("Property"):
            if prop.get("name") not in PLAIN_NOTE_PROPERTIES:
                found[f"note property {prop.get('name')}"] += times
        for child in note:
            if child.tag not in PLAIN_NOTE_CHILDREN:
                found[f"note <{child.tag}>"] += times
    for beat in beats:
        for child in beat:
            if child.tag not in PLAIN_BEAT_CHILDREN:
                found[f"beat <{child.tag}>"] += 1
        for prop in beat.iter("Property"):
            if prop.get("name") not in PLAIN_BEAT_PROPERTIES:
                found[f"beat property {prop.get('name')}"] += 1
    return found, len(notes), sum(played.values())


def contour_deviation_cents(path: str) -> tuple[int, float]:
    """(points, largest distance in cents of a pitch_contour frequency from
    the nearest equal-tempered pitch) over a JAMS file."""
    points, worst = 0, 0.0
    for annotation in json.load(open(path))["annotations"]:
        if annotation["namespace"] != "pitch_contour":
            continue
        values = annotation["data"]["value"] if isinstance(annotation["data"], dict) else \
            [d["value"] for d in annotation["data"]]
        for value in values:
            if value.get("voiced") and value.get("frequency", 0) > 0:
                midi = 69 + 12 * math.log2(value["frequency"] / 440.0)
                worst = max(worst, abs(midi - round(midi)) * 100)
                points += 1
    return points, worst


def main() -> None:
    total = collections.Counter()
    total_notes = total_played = 0
    print("file  distinct notes  played notes  markings other than pitch/position")
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.gp"))):
        found, notes, played = count(path)
        total.update(found)
        total_notes += notes
        total_played += played
        print(f"{os.path.basename(path)[:-3]:>4}  {notes:>14}  {played:>12}  {dict(sorted(found.items())) or '-'}")
    print(f"\nall   {total_notes:>14}  {total_played:>12}  {dict(sorted(total.items()))}")
    print("\ntechnique              marked notes/beats")
    for name, keys in TECHNIQUES.items():
        print(f"{name:<22} {sum(total[k] for k in keys)}")

    print("\nJAMS pitch_contour: points, largest deviation from an equal-tempered pitch")
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jams"))):
        points, worst = contour_deviation_cents(path)
        print(f"{os.path.basename(path)[:-5]:>4}  {points:>6} points  {worst:.3f} cents")


if __name__ == "__main__":
    main()
