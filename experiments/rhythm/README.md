# Rhythm: beat and downbeat ground truth, beat trackers, bar lines, quantization (2026-09-30)

Part 1 (ground truth, trackers) was measurement only. Part 2 put beat_this
in the pipeline, tried chord changes for bar lines (rejected), and measured
quantization. Notes are still not quantized in the app.

## What the app knew before this branch

`transcribe._estimate_tempo_bpm` loaded the transcription audio (the
normalized 22.05kHz mono mix, or the guitar stem), ran
`librosa.beat.beat_track`, and kept only the tempo, rounded to a whole
BPM. The beat frames it also returned were thrown away. `tempo_bpm` was
stored on the job, copied into the TabResult, and shown as "Tempo (est.)".
Nothing else used it: there were no beat times, downbeats, bars or time
signature anywhere in the app.

## What it knows now (`tasks/beats.py`)

transcribe runs beat_this on the same audio as Basic Pitch and stores
`beats`, `downbeats` (seconds) and `tempo_bpm` = 60 / median inter-beat
interval. map_fretboard copies all three into the TabResult. The beat
lists are empty for older jobs. If beat_this raises, the error is logged
and the job gets librosa's tempo with empty beat lists; it doesn't fail.
"Tempo (est.)" shows the new tempo.

- **Cost:** on a 102s song (firefire, full mix), 5.0-5.8s warm, vs 0.25s
  for librosa's tempo: **about +5.5s per song**. The first call in a
  worker process adds ~3s for the model load (7.9s in all). The checkpoint
  is baked into the image. On the 30s EGSet12 clips it's 0.10-0.13 s per
  second of audio, because a fixed per-call overhead matters more there.
- **Pitch output is unchanged:** regression_check before and after is
  identical in every section (synthetic, firefire, solo clip, EGSet12).
- On firefire, beat_this says 187 bpm where librosa said 92. That song has
  no ground truth, so it's unknown which is right.

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
  ratio, with tied notes as one note. (Before part 2, the parser gave a
  tied note only its first beat. Ties occur only in 09, 10 and 12, and
  the fix changed no onset, beat or downbeat.) `rhythm_truth.json` also
  stores the grid (offset, seconds per quarter) and every score note's
  position and length, linked to its JAMS onset.
- *In 11, the first 11 notes are written a whole tone higher in the GP
  file than in the JAMS. Their timing matches (within ~30ms), so the grid
  isn't affected.

Beats are placed at every quarter note of the fitted grid, and downbeats
at every bar start (02's 5/4 bars included). Output:
`data/egset12/rhythm_truth.json`.

## Beat trackers (`experiments/rhythm/track.py`, scored by `rhythm_benchmark.py eval`)

- **librosa:** `beat_track` exactly as the app ran it before, but keeping
  the beat times. librosa has no downbeat tracker, so downbeats use a
  heuristic: assume 4/4 and pick the phase (every 4th beat) with the
  strongest onsets.
- **beat_this** (CPJKU, ISMIR 2024; MIT code and weights): beats and
  downbeats, without DBN postprocessing (that needs madmom, whose models
  are non-commercial). It installs cleanly on the worker image, adding
  `beat-this` and `rotary-embedding-torch` (`worker/requirements-beats.txt`),
  with its checkpoint baked in. The separate `rhythm-bench` image from
  part 1 is gone.

Scores are means over the 12 performances, with a ±70ms tolerance, over
the whole 30s clip. The "MIREX" column skips the first 5s, as the standard
convention does. **Any-octave** is the best beat F among the tracker's
beats, the beats doubled, and the beats halved: it takes the half/double
choice out.

| tracker | tone | beat F | (MIREX) | any-octave | downbeat F | tempo from beats: right / half / double / other | s per s audio |
|---|---|---|---|---|---|---|---|
| librosa (before) | clean | 55.3% | 55.0% | 72.6% | 31.3% | 6 / 0 / 3 / 3 | 0.004 |
| | moderate | 59.6% | 59.8% | 76.2% | 39.0% | 7 / 0 / 3 / 2 | 0.004 |
| | heavy | 62.1% | 62.2% | 76.8% | 46.6% | 8 / 0 / 2 / 2 | 0.005 |
| **beat_this (now)** | clean | **70.1%** | 67.1% | 76.3% | **53.9%** | **8** / 4 / 0 / 0 | 0.130 |
| | moderate | 68.7% | 66.4% | 72.8% | 55.1% | 8 / 3 / 0 / 1 | 0.119 |
| | heavy | 64.6% | 62.4% | 72.3% | 52.6% | 8 / 3 / 0 / 1 | 0.104 |

- **The tempo the app showed before** (librosa's global BPM) is right for only
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
- **Speed:** see "What it knows now" above; the per-second figures in the
  table are from the 30s clips.

In part 2, both trackers were re-run on the pipeline's input: 22.05kHz
mono, normalized (part 1 used the original files). The scores are the same
within 0.1 point. The table shows the part 2 numbers.

## Bar lines from chord changes (`bars.py`): rejected

This tried choosing the bar phase (which beat is the "1") from BTC's
chord-change times, alone or with low-note onsets (≤ E3, weighted by
amplitude), and optionally beat_this's own downbeats as a prior. All
weights are 1, untuned. Each variant keeps beat_this's beats. The phase
is one choice for the whole clip, or "local": re-chosen per beat from
±2 bars. The bar length M is either beat_this's (its most common count
between downbeats) or assumed to be 4. Downbeat F at ±70ms, mean of 12:

| variant | clean | moderate | heavy |
|---|---|---|---|
| **beat_this's own downbeats** | **53.9%** | **55.1%** | **52.6%** |
| M from beat_this: chords | 43.9 | 30.7 | 40.9 |
| M from beat_this: chords + bass + beat_this | 49.9 | 33.0 | 44.1 |
| M from beat_this: *oracle phase* | 49.9 | 43.4 | 45.3 |
| M = 4: chords | 40.3 | 29.7 | 39.8 |
| M = 4: chords + bass | 44.3 | 25.7 | 39.6 |
| M = 4: chords + bass + beat_this | 41.6 | 29.6 | 41.6 |
| M = 4: *oracle phase* | 54.0 | 44.3 | 52.2 |
| M = 4, local: beat_this made regular | 50.7 | 48.4 | 49.7 |
| M = 4, local: chords | 44.2 | 40.8 | 39.7 |
| M = 4, local: chords + bass + beat_this | 48.8 | 48.0 | 42.6 |

- **Nothing beats beat_this's own downbeats.** Even the oracle, the best
  single phase chosen by looking at the truth, only ties it on clean and
  heavy and is 11 points worse on moderate. So the phase choice is not
  what limits bar lines. The beats underneath are:
  - **Beat slips.** On 01, beat F is 0.98, yet the best M = 4 phase gets
    only 0.70: one missed or extra beat moves the phase for the rest of
    the clip. beat_this's own downbeats re-sync after a slip; a fixed
    phase can't. Local phase recovers part of this, but not all.
  - **Wrong bar length.** beat_this's beats per bar is often 2 (or 1) on
    4/4 songs, even at the right tempo (01, 03, 05, 09, 10).
  - **Half tempo.** On the fast songs, a 4/4 bar is 2 tracker beats, so
    assuming M = 4 marks every other bar.
- **Chord changes are the weakest evidence** in every combination. Chord
  changes alone are 10-25 points below beat_this. BTC's changes are
  noisy: single-note passages give it spurious changes (04: 0.00-0.08,
  vs 0.36-0.55 for beat_this), and its 93ms frames blur timing.
- **Not integrated.** Better bar lines would need better beats or meter
  (for example beat_this with its DBN, which needs madmom's
  non-commercial models, or a meter model), not a phase heuristic.

## Quantization (`quantize.py`): measurement only

Each note start is snapped to the nearest 16th within its beat, and each
length ((end − start) in beats) to the nearest of 16th, 8th, dotted 8th,
quarter, dotted quarter, half and whole, by ratio. Scoring uses the score
note each tab note matched (same pitch, within 50ms of its JAMS onset):

- **start right:** the quantized time falls at the note's true position on
  the GP grid.
- **length right:** the snapped value, converted to true quarters, is the
  notated length within 10%, with tied notes counted as one note.

With half-tempo beats, a tracker "quarter" is a true half note. The first
row quantizes the JAMS notes themselves, which shows how far a 16th grid
and 7 note values can represent this playing at all.

% right: start / length from note end / length from next onset [notes]:

| notes / beats | tone | all | chords | single-note | fast |
|---|---|---|---|---|---|
| truth notes / true beats | - | 99.4 / 79.9 / 77.4 [1550] | 100 / 88.4 / 78.3 | 98.1 / 62.9 / 78.5 | 100 / 88.0 / 60.2 |
| tab / true beats | clean | 99.0 / 60.4 / 69.6 [1127] | 98.8 / 59.0 / 66.2 | 99.2 / 61.7 / 75.9 | 100 / 63.9 / 52.5 |
| tab / beat_this | clean | **96.2 / 61.9** / 67.0 | 97.1 / 62.6 / 66.2 | 95.8 / 62.1 / 70.9 | 90.2 / 54.1 / 44.3 |
| tab / true beats | moderate | 99.2 / 55.6 / 63.5 [874] | 99.2 / 52.3 / 54.2 | 99.1 / 57.2 / 73.5 | 100 / 64.4 / 47.5 |
| tab / beat_this | moderate | **95.3 / 55.0** / 61.0 | 93.3 / 49.9 / 51.5 | 97.3 / 59.5 / 71.7 | 93.2 / 54.2 / 40.7 |
| tab / true beats | heavy | 99.7 / 56.0 / 67.2 [604] | 100 / 46.3 / 51.2 | 99.5 / 58.1 / 74.7 | 100 / 72.1 / 58.1 |
| tab / beat_this | heavy | **97.7 / 64.1** / 66.9 | 98.1 / 56.8 / 51.2 | 98.5 / 68.7 / 76.2 | 88.4 / 48.8 / 39.5 |

The sets are only the notes the pipeline found: 1127 / 874 / 604 of the
1550 score notes with a JAMS onset. Fast segments have 43-61 notes, so
treat them loosely.

- **Start positions are nearly solved.** With the true beats, 99-100% of
  starts land on the right 16th: the quantizer and Basic Pitch's onset
  timing are not the problem. With beat_this's beats it's 95-98%, and
  the misses are mostly one 16th early. Fast passages drop most (88-93%).
  On half-tempo performances it's 86-97%, because a 16th slot of a
  half-tempo beat is a true 8th, and most notes are 8ths anyway. Half
  tempo still doubles every written value (8ths are shown as 16ths).
- **Lengths are the hard part, and not because of beat tracking.** Using
  sounding length (Basic Pitch's end − start), only 55-64% come out
  right, with either beat grid. Even the JAMS durations give only 80%,
  and 63% on single notes: players let notes ring past, or cut them
  short of, the written value. The tab's lengths are mostly **too short**
  (27-38% too short vs 6-11% too long), since Basic Pitch ends notes as
  they decay.
- **Time to the next onset is a better length for single-note lines**
  (72-76% vs 57-69%). It is worse for chords (sustained under the next
  note) and for fast passages. A quantizer probably needs both, per
  segment type.
- **16ths are enough.** Of 1569 score onsets, 24 are in triplets (1.5%),
  all in 01 and 03 (12 each), and 16 onsets (1.0%) are off the 16th grid.
  Notated lengths not among the 7 values: 3.0% (triplet 8ths ×24, dotted
  half ×17, 2.5 and 4.5 quarters ×3 each). Adding the dotted half would
  cover the largest group.

## Reproduce

```sh
docker compose build worker            # everything runs under the stale-image guard
M="-v $(pwd -W)/data:/app/data -v $(pwd -W):/repo:ro -v $(pwd -W)/experiments/rhythm:/rhythm"   # Git Bash: MSYS_NO_PATHCONV=1
docker run --rm $M --entrypoint python stratotab-worker -m scripts.rhythm_benchmark truth --show 01 06
docker run --rm $M --entrypoint python stratotab-worker /rhythm/track.py
docker run --rm $M --entrypoint python stratotab-worker -m scripts.rhythm_benchmark eval
docker run --rm $M --entrypoint python stratotab-worker /rhythm/bars.py      # BTC + notes cached in data/_rhythm/_evidence
docker run --rm $M --entrypoint python stratotab-worker /rhythm/quantize.py
```
