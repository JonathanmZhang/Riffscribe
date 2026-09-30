# Rhythm: beat and downbeat ground truth, and beat trackers (2026-09-30)

This is measurement only; nothing in the pipeline changed. Notes are not
quantized yet (next step).

## What the app knows today

`transcribe._estimate_tempo_bpm` loads the transcription audio (the
normalized 22.05kHz mono mix, or the guitar stem), runs
`librosa.beat.beat_track`, and keeps only the tempo, rounded to a whole
BPM. The beat frames it also returns are thrown away. `tempo_bpm` is
stored on the job, copied into the TabResult, and shown as "Tempo (est.)".
Nothing else uses it: there are no beat times, downbeats, bars or time
signature anywhere in the app.

## Ground truth: EGSet12's Guitar Pro files (`worker/scripts/rhythm_benchmark.py truth`)

Each `.gp` (Guitar Pro 7/8: a zip with `Content/score.gpif` XML) has a
time signature per bar, a tempo automation, and every note's rhythm (note
value, dots, tuplets, ties). The script:

1. Parses the score into note onsets in quarter notes.
2. Matches them to the JAMS onsets by pitch.
3. Fits `time = first-bar offset + quarter index × seconds per quarter`.

| perf | GP tempo | JAMS tempo | fitted tempo | time sig. | score onsets matched | strict grid vs real onsets: median / p90 |
|---|---|---|---|---|---|---|
| **01** | 80 | 80 | **80.02** | 4/4 | 92/92 | **3.7 / 10.8 ms** |
| 02 | 140 | 140 | 139.95 | 4/4 + 5/4 | 242/242 | 15.2 / 38.0 |
| 03 | 60 | 60 | 59.99 | 4/4 | 78/78 | 20.3 / 39.9 |
| 04 | 110 | 110 | 110.01 | 4/4 | 170/170 | 5.7 / 13.0 |
| 05 | 72 | 70 | 72.06 | 4/4 | 102/102 | 11.3 / 37.3 |
| **06** | 103 | 100 | **102.88** | 4/4 | 390/390 | **14.9 / 36.0 ms** |
| 07 | 150 | 150 | 149.80 | 4/4 | 64/64 | 7.9 / 20.4 |
| 08 | 98 | 100 | 98.09 | 4/4 | 84/84 | 13.8 / 32.0 |
| 09 | 95 | 100 | 95.01 | 4/4 | 89/90 | 3.7 / 10.6 |
| 10 | 95 | 100 | 95.05 | 4/4 | 70/71 | 14.4 / 39.5 |
| 11 | 176 | 180 | 176.04 | 4/4 | 87/104* | 3.8 / 21.1 |
| 12 | 165 | 160 | 165.25 | 4/4 | 82/82 | 7.9 / 32.7 |

- **Every performance was played to the GP tempo.** The fitted tempo is
  within 0.3 bpm of the GP value, and the first bar starts at the start of
  the audio (01: +0.004s, 06: −0.028s). JAMS's tempo is only a rounded
  label: it's wrong by 2-5 bpm in 6 of 12.
- **A strict grid is accurate enough to serve as ground truth.** Its error
  against the real onsets is 4-20ms median and at most 40ms at p90, so it's
  well inside the 70ms scoring tolerance.
- **Following the player doesn't help.** A local map that bends through
  the matched notes isn't more accurate (leave-one-out 3-21ms median). It
  follows strum spread and expressive timing, which is what a performance
  played to a click looks like.
- **Durations** come straight from the score: note value × dots × tuplet
  ratio, with tied notes as one note.
- *In 11, the first 11 notes are written a whole tone higher in the GP
  file than in the JAMS. Their timing matches (within ~30ms), so the grid
  isn't affected.

Beats are placed at every quarter note of the fitted grid, and downbeats
at every bar start (02's 5/4 bars included). Output:
`data/egset12/rhythm_truth.json`.

## Beat trackers (`experiments/rhythm/track.py`, scored by `rhythm_benchmark.py eval`)

- **librosa:** `beat_track` exactly as the app runs it, but keeping the
  beat times. librosa has no downbeat tracker, so downbeats use a
  heuristic: assume 4/4 and pick the phase (every 4th beat) with the
  strongest onsets.
- **beat_this** (CPJKU, ISMIR 2024; MIT code and weights): beats and
  downbeats, without DBN postprocessing (that needs madmom, whose models
  are non-commercial). It installs cleanly on the worker image, adding
  `beat-this` and `rotary-embedding-torch`. Its checkpoint is baked into
  the `rhythm-bench` image (`Dockerfile`).

Scores are means over the 12 performances, with a ±70ms tolerance, over
the whole 30s clip. The "MIREX" column skips the first 5s, as the standard
convention does. **Any-octave** is the best beat F among the tracker's
beats, the beats doubled, and the beats halved: it takes the half/double
choice out.

| tracker | tone | beat F | (MIREX) | any-octave | downbeat F | tempo from beats: right / half / double / other | s per s audio |
|---|---|---|---|---|---|---|---|
| librosa (today) | clean | 55.3% | 55.0% | 72.6% | 31.3% | 6 / 0 / 3 / 3 | 0.011 |
| | moderate | 59.6% | 59.8% | 76.2% | 39.0% | 7 / 0 / 3 / 2 | 0.006 |
| | heavy | 62.1% | 62.2% | 76.8% | 46.6% | 8 / 0 / 2 / 2 | 0.006 |
| **beat_this** | clean | **70.1%** | 67.2% | 76.2% | **53.8%** | **8** / 4 / 0 / 0 | 0.095 |
| | moderate | 68.7% | 66.4% | 72.7% | 55.1% | 8 / 3 / 0 / 1 | 0.087 |
| | heavy | 64.6% | 62.4% | 72.3% | 52.6% | 8 / 3 / 0 / 1 | 0.088 |

- **The app's shown tempo** (librosa's global BPM) is right for only
  6/12 clean performances. It's double for 3 (the slow ones: 01 80 → 161,
  05 72 → 144) and "other" for 3 (for example 03, 60 → 144; 10, 95 →
  185).
- **beat_this** gets the tempo right for 8/12 in every tone. Its misses
  are all *half* tempo, on the four fast songs (02, 07, 11, 12 at 140-176
  bpm come out at 70-88). A listener would often tap that too. It doesn't
  suffer librosa's double-tempo and off-beat failures: librosa's 06 has
  the right tempo but beat F 0.02, because it locked onto the off-beats.
- **Timing at the right metrical level is similar** for both (any-octave
  72-77%). The gap in beat F is mostly the octave choice. About a quarter
  of beats still land more than 70ms off even at the best octave: solo
  guitar with no drums is hard for both.
- **Downbeats are the weak point.** beat_this gets 53-55%, with 06 and 08
  at 0.96 and several below 0.45. librosa's phase heuristic gets 31-47%.
  Bar lines from a tracker alone will be wrong about half the time.
- **Speed:** beat_this costs about 0.09 s per second of audio (about 9s
  for a 100s song); librosa about 0.01.

## Reproduce

```sh
docker compose build worker            # the benchmark runs under the stale-image guard
docker build -t rhythm-bench experiments/rhythm
M="-v $(pwd -W)/data:/app/data -v $(pwd -W):/repo:ro"   # Git Bash: MSYS_NO_PATHCONV=1
docker run --rm $M --entrypoint python stratotab-worker -m scripts.rhythm_benchmark truth --show 01 06
docker run --rm $M -v $(pwd -W)/experiments/rhythm:/rhythm --entrypoint python rhythm-bench /rhythm/track.py
docker run --rm $M --entrypoint python stratotab-worker -m scripts.rhythm_benchmark eval
```
