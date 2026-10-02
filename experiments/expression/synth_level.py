"""Peak level of alphaTab's synth output with the app's default SoundFont,
per job and playback tone, at one or more master volumes. Used to choose the
synth's master volume (SheetMusicView's SYNTH_MASTER_VOLUME) so the output
stays under full scale: anything above 0 dBFS clips at the sound card.

Same method as render_soundfonts.py: the real page in headless Chromium,
alphaTab's exportAudio. Runs in the Playwright image on the compose network:

    docker run --rm --network stratotab_default -v <repo>/experiments/expression:/x \
        mcr.microsoft.com/playwright/python:v1.63.0-noble sh -c \
        "pip install -q playwright==1.63.0 numpy && python /x/synth_level.py 1.0,0.5 <job id> [<job id> ...]"
"""

import asyncio
import base64
import sys

import numpy as np
from playwright.async_api import async_playwright

VOLUMES = [float(v) for v in sys.argv[1].split(",")]
JOBS = sys.argv[2:]
TONES = {"clean": 27, "overdriven": 29, "distorted": 30, "acoustic": 25}

PEAK_JS = """
async (volume) => {
  const api = window.riffscribeSheet, at = window.alphaTab;
  const options = new at.synth.AudioExportOptions();
  options.sampleRate = 44100;
  options.useSyncPoints = false;
  options.masterVolume = volume;
  const exporter = await api.exportAudio(options);
  let peak = 0, sum = 0, count = 0, chunk;
  while ((chunk = await exporter.render(2000))) {
    for (const s of chunk.samples) { const a = Math.abs(s); if (a > peak) peak = a; sum += s * s; count++; }
  }
  exporter.destroy();
  return {peak, rms: Math.sqrt(sum / Math.max(1, count)), seconds: count / 2 / 44100, live: api.masterVolume};
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


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


async def main():
    await forward(3000, "frontend")
    await forward(8000, "backend")
    worst = {v: -99.0 for v in VOLUMES}
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for job in JOBS:
            page = await browser.new_page(viewport={"width": 1280, "height": 1400})
            await page.goto(f"http://localhost:3000/?debug=1&job={job}")
            await page.get_by_role("button", name="Sheet music").click()
            await page.get_by_role("button", name="Synth").click()
            await page.get_by_role("button", name="Play", exact=True).wait_for(timeout=180000)
            for tone, program in TONES.items():
                await page.get_by_label("Tone").select_option(tone)
                await page.wait_for_function(
                    f"() => window.riffscribeSheet.score && window.riffscribeSheet.score.tracks[0].playbackInfo.program === {program}",
                    timeout=120000)
                await asyncio.sleep(1)
                cells = []
                for volume in VOLUMES:
                    r = await page.evaluate(PEAK_JS, volume)
                    worst[volume] = max(worst[volume], db(r["peak"]))
                    cells.append(f"volume {volume}: peak {db(r['peak']):+5.1f} dBFS, rms {db(r['rms']):6.1f}")
                print(f"{job[:8]} {tone:<10} {r['seconds']:6.1f}s  " + "   ".join(cells)
                      + f"   (live player's masterVolume: {r['live']})", flush=True)
            await page.close()
        await browser.close()
    for volume in VOLUMES:
        print(f"highest peak at volume {volume}: {worst[volume]:+.1f} dBFS")


asyncio.run(main())
