# Chord names: reposition-only measurement (2026-09-29)

This is a measurement only; the pipeline is untouched. It follows up
`experiment/chord-recognition`, where adding "inferred" chord notes lowered
F1 but moving the detected notes into the named chord's voicing improved
fret positions.

**Setup:** BTC (large vocabulary, MIT code and weights) runs in the
`chord-names` image (`Dockerfile`), a layer on the unchanged worker image.
It needs no new Python packages, only a sed fix for numpy aliases that were
removed, and a PyYAML 6 workaround in `btc_labels.py`. Its labels match the
chord-recognition branch's run exactly (36/36 files). Speed is about 0.01
seconds per second of audio.

**Method (`reposition.py`):** for every step of today's tab with at least 2
notes:

1. Take BTC's chord label at the step's onset.
2. If there's no chord, or any detected pitch class isn't in the chord,
   leave the step unchanged.
3. Otherwise, among the chord's playable voicings that contain every
   detected pitch, choose the one closest to the notes' current frets.
   Move the notes onto it. No note is added or removed.

The voicing generator is the same one as on the chord-recognition branch.
**oracle** does the same with the ground-truth chord name, as a ceiling on
naming.

**Scoring** uses the EGSet12 benchmark's matching. The script asserts that
pitch recall and precision are identical to today's in every cell, and
that "today" reproduces `egset12_benchmark.evaluate` exactly. **Tab F1**
counts a note as correct only with the right pitch and the right
string/fret.

| tone | segment type | pitch R / P (unchanged) | pitch F1 | position: today → BTC (oracle) | tab F1: today → BTC (oracle) |
|---|---|---|---|---|---|
| clean | all | 71.9 / 85.2 | 78.0 | 57.0 → **59.8** (59.4) | 44.4 → **46.7** (46.4) |
| clean | chords | 61.9 / 83.5 | 71.1 | 56.1 → 58.5 (61.1) | 39.9 → 41.6 (43.4) |
| clean | single-note | 89.5 / 86.1 | 87.8 | 57.9 → 61.7 (57.7) | 50.9 → 54.2 (50.7) |
| clean | fast | 73.5 / 96.8 | 83.6 | 57.4 → 57.4 (57.4) | 47.9 → 47.9 (47.9) |
| moderate | all | 55.8 / 80.7 | 66.0 | 54.5 → 55.1 (55.9) | 35.9 → 36.4 (36.9) |
| moderate | chords | 39.3 / 78.4 | 52.3 | 54.2 → 56.3 (57.6) | 28.3 → 29.5 (30.2) |
| moderate | single-note | 82.8 / 80.8 | 81.8 | 54.3 → 53.8 (54.3) | 44.4 → 44.0 (44.4) |
| moderate | fast | 71.1 / 98.3 | 82.5 | 57.6 → 57.6 (57.6) | 47.6 → 47.6 (47.6) |
| heavy | all | 38.5 / 71.7 | 50.1 | 54.1 → 54.6 (54.6) | 27.1 → 27.4 (27.4) |
| heavy | chords | 17.1 / 60.4 | 26.6 | 53.7 → 57.4 (55.6) | 14.3 → 15.3 (14.8) |
| heavy | single-note | 74.7 / 75.4 | 75.1 | 54.4 → 53.6 (54.4) | 40.8 → 40.3 (40.8) |
| heavy | fast | 51.8 / 95.6 | 67.2 | 53.5 → 53.5 (53.5) | 35.9 → 35.9 (35.9) |

Repositioned notes that match a true note, counted per tone:

| | clean | moderate | heavy |
|---|---|---|---|
| BTC: fixed / broken / wrong both times | **37 / 5** / 4 | 11 / 5 / 3 | 9 / 6 / 2 |
| oracle | 39 / 11 / 2 | 19 / 6 / 0 | 3 / 0 / 0 |

Step outcomes, clean (BTC): 469 single-note steps (never touched), 42
moved, 74 already in place, 93 with a detected note outside the named
chord, 49 with no chord, and 32 with no voicing that fits every detected
note. `data/_chord_names/reposition.json` has every tone.

**Findings**

- On clean audio, reposition-only gives a small, clearly positive gain:
  +2.8 points of position agreement and +2.3 tab F1, with 37 notes fixed
  for 5 broken.
- On the processed-distortion tones it's roughly neutral (+0.3 to +0.6).
  It helps chord segments but slightly hurts the chord-like steps inside
  single-note segments.
- Fast segments never change, because their steps are single notes.
- BTC's names do about as well as ground-truth names. On clean they do
  slightly better, because the oracle only names steps within 80ms of a
  ground-truth chord column and breaks more notes (11).
- Pitch results are unchanged by construction.
