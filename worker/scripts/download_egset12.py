"""Downloads EGSet12 into data/egset12/ (gitignored) for the note-recall
benchmark (scripts/egset12_benchmark.py).

EGSet12: twelve real solo electric guitar performances with per-note string/
fret annotations (JAMS) and Guitar Pro tabs, by H. Pedroza, W. Abreu, R.
Corey and I. R. Roman - https://zenodo.org/records/11406378 - licensed CC BY
4.0. Introduced in "Leveraging real electric guitar tones and effects to
improve robustness in guitar tablature transcription modeling", DAFx 2024.

Usage (inside the worker container):
    python -m scripts.download_egset12

Fetches the .wav/.jams/.gp files (not the bundled model weights), checks
each against Zenodo's MD5, and skips files already present and valid.
"""

import hashlib
import json
import os
import urllib.request

RECORD_API = "https://zenodo.org/api/records/11406378"
OUT_DIR = "/app/data/egset12"
WANTED = (".wav", ".jams", ".gp")


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with urllib.request.urlopen(RECORD_API, timeout=60) as response:
        record = json.load(response)
    files = [f for f in record["files"] if f["key"].endswith(WANTED)]
    for f in sorted(files, key=lambda f: f["key"]):
        path = os.path.join(OUT_DIR, f["key"])
        expected = f["checksum"].split(":", 1)[1]
        if os.path.exists(path) and _md5(path) == expected:
            continue
        urllib.request.urlretrieve(f["links"]["self"], path)
        if _md5(path) != expected:
            raise SystemExit(f"checksum mismatch for {f['key']}")
        print(f"downloaded {f['key']} ({f['size'] / 1e6:.1f} MB)")
    print(f"EGSet12: {len(files)} files in {OUT_DIR} (CC BY 4.0, https://zenodo.org/records/11406378)")


if __name__ == "__main__":
    main()
