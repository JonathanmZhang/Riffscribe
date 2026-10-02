"""Does alphaTab draw technique markings that arrive in MusicXML? Builds a
3-bar test score with the app's own exporter (tasks/musicxml.py), adds
standard MusicXML technique elements to its notes on both staves, loads it
into the app's alphaTab build in headless Chromium, and reports what
alphaTab's score model holds for each note, plus a screenshot. Measurement
only: the exporter doesn't write any of these elements today.

    docker run --rm --network stratotab_default -v <repo>/experiments/expression:/x \
        -v <repo>/worker:/w:ro -v <repo>/data/_expression:/d \
        mcr.microsoft.com/playwright/python:v1.63.0-noble sh -c \
        "pip install -q playwright==1.63.0 && python /x/alphatab_techniques.py"

Output: /d/alphatab_techniques/{techniques.musicxml, techniques.png, report.json}.
"""

import asyncio
import json
import os
import sys
import xml.etree.ElementTree as ET

from playwright.async_api import async_playwright

sys.path.insert(0, "/w")
from tasks import musicxml  # noqa: E402

OUT_DIR = "/d/alphatab_techniques"
# One quarter note per beat at 120 bpm, on the B string. (label, fret).
NOTES = [("plain", 8), ("bend up a whole step", 8), ("bend and release", 8), ("pre-bend and release", 8),
         ("hammer-on origin", 8), ("hammer-on target", 10), ("slide origin", 8), ("slide target", 10),
         ("vibrato", 8), ("palm mute (play/mute)", 8), ("palm mute (P.M. + dashes)", 8), ("dead note", 8)]
PITCH = {8: "G4", 10: "A4"}


def bend(alter: float, *flags: str) -> ET.Element:
    el = ET.Element("bend")
    ET.SubElement(el, "bend-alter").text = str(alter)
    for flag in flags:
        ET.SubElement(el, flag)
    return el


def technical_of(note: ET.Element) -> ET.Element:
    notations = note.find("notations")
    if notations is None:
        notations = ET.SubElement(note, "notations")
    technical = notations.find("technical")
    return technical if technical is not None else ET.SubElement(notations, "technical")


def mark(measure: ET.Element, note: ET.Element, label: str) -> None:
    """Adds the MusicXML elements for label to one note."""
    notations = note.find("notations")
    if notations is None:
        notations = ET.SubElement(note, "notations")
    if label == "bend up a whole step":
        technical_of(note).append(bend(2))
    elif label == "bend and release":
        technical_of(note).extend([bend(2), bend(-2, "release")])
    elif label == "pre-bend and release":
        technical_of(note).extend([bend(2, "pre-bend"), bend(-2, "release")])
    elif label in ("hammer-on origin", "hammer-on target"):
        kind = "start" if label.endswith("origin") else "stop"
        ET.SubElement(technical_of(note), "hammer-on", type=kind).text = "H" if kind == "start" else None
        ET.SubElement(notations, "slur", type=kind, number="1")
    elif label in ("slide origin", "slide target"):
        ET.SubElement(notations, "slide", {"type": "start" if label.endswith("origin") else "stop",
                                           "line-type": "solid", "number": "1"})
    elif label == "vibrato":
        ornaments = ET.SubElement(notations, "ornaments")
        ET.SubElement(ornaments, "wavy-line", type="start", number="1")
        ET.SubElement(ornaments, "wavy-line", type="stop", number="1")
    elif label == "palm mute (play/mute)":
        ET.SubElement(ET.SubElement(note, "play"), "mute").text = "palm"
    elif label == "palm mute (P.M. + dashes)":
        staff = note.findtext("staff")
        index = list(measure).index(note)
        start = ET.Element("direction", placement="above")
        ET.SubElement(ET.SubElement(start, "direction-type"), "words").text = "P.M."
        ET.SubElement(ET.SubElement(start, "direction-type"), "dashes", type="start", number="1")
        ET.SubElement(start, "staff").text = staff
        stop = ET.Element("direction", placement="above")
        ET.SubElement(ET.SubElement(stop, "direction-type"), "dashes", type="stop", number="1")
        ET.SubElement(stop, "staff").text = staff
        measure.insert(index + 1, stop)
        measure.insert(index, start)
    elif label == "dead note":
        head = ET.Element("notehead")
        head.text = "x"
        note.insert(list(note).index(note.find("staff")), head)
    if len(notations) == 0:
        note.remove(notations)


def build() -> str:
    result = {"job_id": "techniques", "duration_seconds": len(NOTES) * 0.5, "tempo_bpm": 120, "chords": [],
              "notes": [{"string": 2, "fret": fret, "start_time": i * 0.5, "end_time": i * 0.5 + 0.5, "pitch": PITCH[fret]}
                        for i, (_, fret) in enumerate(NOTES)],
              "beats": [i * 0.5 for i in range(len(NOTES) + 1)], "downbeats": [i * 2.0 for i in range(len(NOTES) // 4 + 1)]}
    text = musicxml.to_musicxml(result, title="Technique markings test")
    header, body = text.split("<score-partwise", 1)
    root = ET.fromstring("<score-partwise" + body)
    counters = {"1": 0, "2": 0}
    for measure in root.iter("measure"):
        for note in [n for n in measure if n.tag == "note" and n.find("pitch") is not None]:
            staff = note.findtext("staff")
            mark(measure, note, NOTES[counters[staff]][0])
            counters[staff] += 1
    assert counters == {"1": len(NOTES), "2": len(NOTES)}, counters
    ET.indent(root, space="  ")
    return header + ET.tostring(root, encoding="unicode") + "\n"


INSPECT_JS = """
async (xml) => {
  if (!window.alphaTab) await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/alphatab/alphaTab.min.js"; s.onload = resolve; s.onerror = reject; document.head.appendChild(s);
  });
  const at = window.alphaTab;
  document.body.innerHTML = '<div id="sheet" style="width:1100px;background:white"></div>';
  const api = new at.AlphaTabApi(document.getElementById("sheet"), {
    core: {scriptFile: location.origin + "/alphatab/alphaTab.min.js", fontDirectory: location.origin + "/alphatab/font/"},
    display: {layoutMode: at.LayoutMode.Page, scale: 1.0},
    player: {playerMode: at.PlayerMode.Disabled},
  });
  const errors = [];
  api.error.on((e) => errors.push(String(e.message ?? e)));
  const rendered = new Promise((resolve) => api.renderFinished.on(resolve));
  api.load(new TextEncoder().encode(xml));
  await rendered;
  const name = (enumeration, value) => enumeration[value];
  const notes = [];
  api.score.tracks[0].staves.forEach((staff, staffIndex) => staff.bars.forEach((bar) => bar.voices.forEach((voice) =>
    voice.beats.forEach((beat) => beat.notes.forEach((note) => notes.push({
      staff: staffIndex + 1, bar: bar.index + 1, fret: note.fret, string: note.string,
      bend: note.hasBend ? name(at.model.BendType, note.bendType) : null,
      bendPoints: note.hasBend ? note.bendPoints.map((p) => [p.offset, p.value]) : null,
      hammerPullOrigin: note.isHammerPullOrigin, hammerPullDestination: note.isHammerPullDestination,
      slideOut: name(at.model.SlideOutType, note.slideOutType), slur: note.isSlurOrigin || note.isSlurDestination,
      vibrato: name(at.model.VibratoType, note.vibrato), palmMute: note.isPalmMute, beatPalmMute: beat.isPalmMute,
      dead: note.isDead,
    }))))));
  return {version: at.meta.version, errors, staves: api.score.tracks[0].staves.length, notes};
}
"""


async def forward(port, host):
    async def pipe(reader, writer):
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    async def handle(reader, writer):
        try:
            up_reader, up_writer = await asyncio.open_connection(host, port)
        except Exception:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    await asyncio.start_server(handle, "127.0.0.1", port)


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xml = build()
    with open(os.path.join(OUT_DIR, "techniques.musicxml"), "w", encoding="utf-8") as f:
        f.write(xml)
    await forward(3000, "frontend")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 900})
        await page.goto("http://localhost:3000/")
        report = await page.evaluate(INSPECT_JS, xml)
        await asyncio.sleep(2)
        await page.locator("#sheet").screenshot(path=os.path.join(OUT_DIR, "techniques.png"))
        await browser.close()
    print(f"alphaTab {report['version']}, {report['staves']} staves, errors: {report['errors'] or 'none'}")
    for staff in (1, 2):
        print(f"staff {staff} ({'notation' if staff == 1 else 'TAB'}):")
        for (label, _), note in zip(NOTES, [n for n in report["notes"] if n["staff"] == staff]):
            held = {k: v for k, v in note.items()
                    if k not in ("staff", "bar", "fret", "string") and v not in (None, False, "None")}
            print(f"  {label:<28} fret {note['fret']:>2} -> {held or 'nothing'}")
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


asyncio.run(main())
