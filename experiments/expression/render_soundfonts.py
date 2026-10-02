"""Renders one job's sheet music to WAV files with alphaTab's own synth, once
per playback tone and SoundFont, for listening. It drives the real page (the
tone selector included) in headless Chromium and calls alphaTab's
exportAudio, so each file is what the app's Synth button would play with
that SoundFont. Measurement only: the app keeps its default SoundFont.

Runs in the Playwright image on the compose network (the stack must be up):

    docker run --rm --network stratotab_default \
        -v <repo>/experiments/expression:/x -v <repo>/data/_expression:/d \
        mcr.microsoft.com/playwright/python:v1.63.0-noble sh -c \
        "pip install -q playwright==1.63.0 numpy && python /x/render_soundfonts.py <job id>"

SoundFonts are read from /d/soundfonts (not committed; README.md says where
each comes from). Output: /d/soundfont_compare/<tone>__<soundfont>.wav,
44.1kHz stereo, every file scaled to the same RMS level (-20 dBFS, peak kept
under -1 dBFS) so loudness doesn't decide the comparison.
"""

import asyncio
import base64
import functools
import http.server
import json
import os
import sys
import threading
import wave

import numpy as np
from playwright.async_api import async_playwright

JOB = sys.argv[1]
FONT_DIR = "/d/soundfonts"
OUT_DIR = "/d/soundfont_compare"
# Tone -> the 0-based MIDI program alphaTab should hold after loading the
# export (MusicXML's midi-program minus 1).
TONES = {"clean": 27, "overdriven": 29, "distorted": 30, "acoustic": 25}
# (label, file or None for alphaTab's bundled font, tones it is rendered for,
# program override). The FreePats fonts hold one preset at program 0.
FONTS = [
    ("sonivox-default", None, list(TONES), None),
    ("FluidR3_GM", "FluidR3_GM.sf2", list(TONES), None),
    ("GeneralUser-GS", "GeneralUser-GS.sf2", list(TONES), None),
    ("FreePats-FSBS-clean-small", "FSBS-clean-small.sf2", ["clean"], 0),
    ("FreePats-FSBS-dist2", "FSBS-dist2.sf2", ["distorted"], 0),
]

RENDER_JS = """
async ({url, program}) => {
  const api = window.riffscribeSheet, at = window.alphaTab;
  const track = api.score.tracks[0];
  const original = track.playbackInfo.program;
  if (program !== null) track.playbackInfo.program = program;
  const font = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const options = new at.synth.AudioExportOptions();
  options.soundFonts = [font];
  options.sampleRate = 44100;
  options.useSyncPoints = false;
  const exporter = await api.exportAudio(options);
  const chunks = [];
  let total = 0, chunk;
  while ((chunk = await exporter.render(2000))) { chunks.push(chunk.samples); total += chunk.samples.length; }
  exporter.destroy();
  track.playbackInfo.program = original;
  const pcm = new Float32Array(total);
  let at_ = 0;
  for (const c of chunks) { pcm.set(c, at_); at_ += c.length; }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
  return {samples: total, used_program: program !== null ? program : original, data: btoa(binary)};
}
"""


async def forward(port, host):
    """localhost:<port> in this container -> the compose service, so the page
    reaches the API at the URL it was built with."""
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


class FontHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the SoundFont files to the page (another origin, so with CORS)."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass


def serve_fonts(port: int = 8765) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(FontHandler, directory=FONT_DIR))
    threading.Thread(target=server.serve_forever, daemon=True).start()


def write_wav(path: str, samples: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(samples**2))) or 1e-9
    peak = float(np.abs(samples).max()) or 1e-9
    gain = min(10 ** (-20 / 20) / rms, 10 ** (-1 / 20) / peak)
    pcm = np.clip(samples * gain, -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())
    return {"seconds": round(len(samples) / 2 / 44100, 2), "raw_rms_dbfs": round(20 * np.log10(rms), 1),
            "raw_peak_dbfs": round(20 * np.log10(peak), 1), "gain_db": round(20 * np.log10(gain), 1)}


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    await forward(3000, "frontend")
    await forward(8000, "backend")
    serve_fonts()
    report = {"job": JOB, "files": []}
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 1400})
        page.on("crash", lambda _: print("PAGE CRASHED", flush=True))
        page.on("console", lambda m: print("console:", m.text[:300], flush=True) if m.type in ("error", "warning") else None)
        await page.goto(f"http://localhost:3000/?debug=1&job={JOB}")
        await page.get_by_role("button", name="Sheet music").click()
        await page.get_by_role("button", name="Synth").click()
        await page.get_by_role("button", name="Play", exact=True).wait_for(timeout=120000)
        info = await page.evaluate("""() => { const t = window.riffscribeSheet.score.tracks[0];
            return {name: t.name, program: t.playbackInfo.program, bars: window.riffscribeSheet.score.masterBars.length,
                    alphaTab: window.alphaTab.meta.version}; }""")
        print("loaded:", info, flush=True)
        report["loaded"] = info
        for tone, program in TONES.items():
            await page.get_by_label("Tone").select_option(tone)
            await page.wait_for_function(
                f"() => window.riffscribeSheet.score && window.riffscribeSheet.score.tracks[0].playbackInfo.program === {program}",
                timeout=60000)
            await asyncio.sleep(1)
            for label, file, tones, override in FONTS:
                if tone not in tones:
                    continue
                if file and not os.path.exists(os.path.join(FONT_DIR, file)):
                    print(f"{tone} {label}: font file missing, skipped", flush=True)
                    continue
                url = f"http://127.0.0.1:8765/{file}" if file else "/alphatab/soundfont/sonivox.sf2"
                try:
                    result = await page.evaluate(RENDER_JS, {"url": url, "program": override})
                except Exception as exc:
                    print(f"{tone} {label}: FAILED {str(exc)[:300]}", flush=True)
                    report["files"].append({"tone": tone, "soundfont": label, "error": str(exc)[:300]})
                    continue
                samples = np.frombuffer(base64.b64decode(result["data"]), dtype="<f4")
                name = f"{tone}__{label}.wav"
                stats = write_wav(os.path.join(OUT_DIR, name), samples)
                entry = {"tone": tone, "soundfont": label, "file": name, "program": result["used_program"], **stats}
                report["files"].append(entry)
                print(entry, flush=True)
        await browser.close()
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


asyncio.run(main())
