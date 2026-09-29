# Chord recognition and chord-shape completion (2026-09-29)

This is a measurement only. Nothing is wired into the pipeline, and
nothing under `worker/` changed. It asks two questions: can a dedicated
chord-recognition model name EGSet12's chords, and can those names
complete the chord shapes in our tab?

## Candidates and licenses

| candidate | code | weights | status |
|---|---|---|---|
| Chordino / NNLS-Chroma (Mauch) | GPL-2.0 | none (hand-built chord dictionary + HMM) | measured. **GPL**: fine in a public repo, but shipping it in Riffscribe puts GPL obligations on the combined work. The repo has no LICENSE file yet |
| BTC, large vocabulary (Park et al., ISMIR 2019) | MIT | MIT (in repo) | measured |
| crema 0.2.0 (McFee) | ISC/BSD | ISC (ships in the package) | measured |
| madmom CNN/CRF chord recognizer | BSD | CC BY-NC-SA 4.0 | dropped: non-commercial, share-alike |
| Essentia ChordsDetection | AGPL-3.0 | none | dropped: stricter than GPL, nothing Chordino lacks |
| bp-notes (ours) | - | - | baseline: our chord template matcher on the notes of the nearest pipeline tab step |

All three models run in `chords-bench` (the `Dockerfile` here), a layer on
the unchanged worker image. Integration costs if this ever ships:

- crema's `pumpp` breaks on scikit-learn >= 1.6, so the image pins 1.5.2.
  The worker has 1.9.1.
- `protobuf<5` has to stay pinned for TF 2.15.
- BTC needs a sed fix for numpy aliases that were removed, plus a PyYAML
  6 workaround.
- Chordino is built from source (Vamp SDK + Boost).

Speed, in seconds per second of audio: BTC 0.011, crema 0.019, Chordino
0.023. That's negligible next to Basic Pitch.

## Part 2: chord names at EGSet12's 203 ground-truth columns

The ground-truth names come from our own template matcher
(`egset12_benchmark.chord_name`) applied to the annotated notes. They are
not human labels. 76 of the 203 columns are clusters that match no
template, so root, majmin and exact are scored on the other 127; majmin
covers the 100 of those that are major or minor. A model's label for a
column is the one covering most of the 0.3s after its onset.

- **root:** the root matches.
- **majmin:** the root and the major/minor quality match.
- **exact:** the pitch-class set matches.
- **+bass:** exact, plus the same bass note.

| namer | tone | root | majmin | exact | +bass | clusters called a chord |
|---|---|---|---|---|---|---|
| **Chordino** | clean | **71.7%** | **74.0%** | 42.5% | 17.3% | 76/76 |
| | moderate | 72.4% | 73.0% | 41.7% | 17.3% | 76/76 |
| | heavy | 70.1% | 72.0% | 42.5% | 17.3% | 76/76 |
| BTC | clean | 67.7% | 70.0% | 25.2% | 15.7% | 58/76 |
| | moderate | 73.2% | 70.0% | 13.4% | 5.5% | 52/76 |
| | heavy | 67.7% | 65.0% | 6.3% | 0.8% | 50/76 |
| crema | clean | 55.1% | 59.0% | 10.2% | 3.1% | 76/76 |
| | moderate | 65.4% | 69.0% | 12.6% | 3.1% | 69/76 |
| | heavy | 63.8% | 67.0% | 7.1% | 2.4% | 68/76 |
| bp-notes | clean | 59.8% | 45.0% | **43.3%** | **37.0%** | 20/76 |
| | moderate | 22.8% | 15.0% | 13.4% | 10.2% | 10/76 |
| | heavy | 1.6% | 1.0% | 0.0% | 0.0% | 0/76 |

- **Chordino** is the best namer, and it's barely affected by distortion
  (72% root in every tone). It gets inversions wrong (17% with bass) and
  always names something, even for clusters.
- **bp-notes** is only competitive on clean audio, where it is effectively
  "did Basic Pitch find every note of the chord". It collapses once
  distortion hides chord tones.

## Part 3: completing today's tab with chord voicings

For each tab step with >=2 notes, the process is:

1. Name the chord at the step's onset.
2. Pick a voicing of that chord from generated playable shapes: 4-6
   contiguous strings up to the high e (2-3 for power chords), bass on the
   lowest string, every chord tone present (only the fifth may be left
   out, in 4+ tone chords), a fretted span of 3 frets or less, fingering
   possible for one hand, and open strings only in open position. This
   covers the standard open and barre forms.
3. Choose the voicing that contains the most detected pitches (at least
   2), nearest the current position.
4. Move the detected notes to the voicing's strings and add its missing
   notes as **inferred**. A detected note outside the voicing keeps its
   string.

There are two guards. **any** completes whenever 2 or more notes fit.
**agree** also requires every detected note to be a tone of the named
chord. **oracle** uses the ground-truth chord name, so it's an upper bound.

Tab recall / precision [position agreement], all segments:

| tone | system | recall | precision | position | F1 | inferred notes (precision) |
|---|---|---|---|---|---|---|
| clean | **today** | 71.9% | **85.2%** | 57.0% | **78.0** | - |
| | Chordino/agree | 75.0% | 65.0% | 59.9% | 69.7 | 489 (10.2%) |
| | BTC/agree | 73.9% | 69.3% | 61.1% | 71.5 | 351 (9.1%) |
| | crema/agree | 72.8% | 76.2% | 60.0% | 74.5 | 176 (8.0%) |
| | crema/any | 74.9% | 70.7% | 64.2% | 72.7 | 364 (17.6%) |
| | oracle/agree | 76.6% | 79.3% | 62.9% | 77.9 | 203 (40.9%) |
| moderate | **today** | 55.8% | 80.7% | 54.5% | 66.0 | - |
| | crema/agree | 58.3% | 69.3% | 57.1% | 63.3 | 235 (16.6%) |
| | oracle/agree | 60.4% | 76.9% | 59.9% | 67.7 | 149 (49.0%) |
| heavy | **today** | 38.5% | 71.7% | 54.1% | 50.1 | - |
| | crema/agree | 39.4% | 60.1% | 55.3% | 47.6 | 186 (7.5%) |
| | oracle/agree | 39.6% | 70.1% | 54.6% | 50.6 | 44 (38.6%) |

The chord segments on their own show the same pattern (clean: today 61.9 /
83.5, best model 66-68 / 60-75, oracle 69.3 / 74.9). All systems are in
`data/_chords/results.json`.

Findings:

- **Completion lowers F1 with every real namer, in every tone.** Only
  5-18% of the inferred notes are right, so each point of recall gained
  costs 5-10 points of precision.
- **The failure mode is real.** Wrong names drove 13-67 of the completed
  steps. Completion also lands on 2-note steps that aren't chords at all.
  The "agree" guard reduces both but doesn't fix them.
- **Even perfect names don't help.** With ground-truth names, only 41% of
  inferred notes are right on clean audio, and F1 ties today's (77.9 vs
  78.0). The player's real voicings (inversions, partial and jazz shapes)
  often aren't the "common voicing" that contains the detected notes.
  The bottleneck is choosing the voicing, not only naming the chord.
- **Side effect worth noting:** moving the detected notes into a voicing
  raises position agreement by 3-7 points (crema/any, clean: 57.0 to
  64.2%; chords 56.1 to 69.1%). A variant that only repositions, without
  adding notes, wasn't measured.

## Firefire windows (no ground truth, for listening)

| window | Chordino | crema | BTC | bp-notes (>=3-note steps) |
|---|---|---|---|---|
| 28-31 mix | E, F#m@30.28 | A, E@28.98, F#m@30.28 | A, E@29.07, F#m@30.37 | (Ab B)@29.65, E/B@30.0, C#sus4@30.66 |
| 39.5-42.5 mix | C#m7, F#7@40.82, E/B@42.07 | C#m, F#m7@40.87, B@41.98 | C#m, F#m@40.83, B@42.04 | F#sus4@40.71 |
| 60-63 mix | Eb@60.0, C#m7@60.19, E@61.63 | E, C#m@60.09, A@61.49, B@62.79 | C#m, A@61.57, B@62.78 | (C# Ab B)@60.78, (E Ab)@61.17, A5@62.43 |
| 28-31 stem | A, E@28.98, F@30.28, Ab7@30.7 | A, E@28.98 | A, E@28.89, A@30.46, G#@30.74 | (A B), (E Ab)@29.36, E@30.01 |
| 39.5-42.5 stem | E6, A@41.98 | C#m, E@41.89 | C#m, E@40.0, A@41.85 | power chords/dyads |
| 60-63 stem | E | E | E | B5@60.47 |

On the full mix the three models broadly agree on a progression in E major
(A-E-F#m, C#m-F#m-B, C#m-A-B). Our own note-based naming mostly sees
dyads and power chords.

## Reproduce

```sh
# repo root, Git Bash: prefix with MSYS_NO_PATHCONV=1
docker build -t chords-bench experiments/chord_recognition
# data/_chords/manifest.json: 36 EGSet12 files + data/_bakeoff/firefire/{mix,stem}.wav
RUN="docker run --rm -v $(pwd -W)/data:/app/data -v $(pwd -W)/experiments/chord_recognition:/chords --entrypoint python chords-bench"
$RUN /chords/recognize.py /app/data/_chords/manifest.json
$RUN /chords/evaluate.py
```
