# Technique cleanup: vibrato merge and glide merge (2026-10-02)

Measurement. Both merges exist in the pipeline's code
(`worker/tasks/techniques.py`, applied by `transcribe.clean_notes` after
`select_notes`) but are **switched off** (`transcribe.VIBRATO_MERGE`,
`GLIDE_MERGE`), so the pipeline's output is unchanged: `regression_check`
with both off is identical to the last baseline in every section.

All numbers are from the worker image at commit `3052db8`; the stale-image
guard matched on every run. Reports: `data/_technique_cleanup/`.

**Why:** experiments/expression found that on real guitar (IDMT-SMT-Guitar)
vibrato splits 55% of notes into same-fret repeats, and a bend or slide
usually leaves an extra note on a neighbouring pitch.

## The two merges

Both read Basic Pitch's per-note `pitch_bends` (kept on the note events
now; they used to be discarded). Detector settings are the ones fixed in
experiments/expression, unchanged.

- **Vibrato merge:** a run of same-pitch notes that follow each other
  without a gap becomes one note when the run's pitch wobbles at 4-8 Hz
  (12 cents or more). The note is flagged `vibrato`.
- **Glide merge:** a note the pitch glides into (1-2 semitones away, no
  gap, pitch moving across the join) is absorbed by the note it left. The
  result keeps the starting pitch. Chains collapse into their first note.

## Rule for keeping one

Keep a merge only if it reduces wrong notes on IDMT without hurting EGSet12
or the regression check. EGSet12 has no vibrato, bends or slides marked,
so every merge there is a false alarm.

## Results

A tab note is right if a truth note of the same pitch starts within 50 ms
of it, one to one; otherwise it is wrong.

### IDMT-SMT-Guitar, datasets 1 + 2 (661 files, 5,677 truth notes)

| configuration | tab notes | right | wrong | recall | precision | wrong notes | right notes |
|---|---|---|---|---|---|---|---|
| off | 7,528 | 4,580 | 2,948 | 80.7% | 60.8% | | |
| vibrato merge | 7,395 | 4,577 | 2,818 | 80.6% | 61.9% | **-130** | **-3** |
| glide merge | 7,346 | 4,536 | 2,810 | 79.9% | 61.7% | **-138** | **-44** |
| both | 7,213 | 4,533 | 2,680 | 79.8% | 62.8% | -268 | -47 |

Dataset 1 (796 plain notes and chords) is untouched by either merge.

Where the removed notes were (labelled style of the truth note they lay on):

| merge | on vibrato | on bends | on slides | on normal notes | elsewhere |
|---|---|---|---|---|---|
| vibrato | 130 extra, 3 real | | | | |
| glide | 7 extra, 3 real | 64 extra, 5 real | 34 extra, 0 real | 26 extra, **36 real** | 7 extra |

What the tab shows per truth note (dataset 2):

| technique | configuration | one right note | right pitch, split | right + another pitch | another pitch only | nothing |
|---|---|---|---|---|---|---|
| vibrato (162) | off | 42% | **55%** | 2% | 0% | 1% |
| | vibrato merge | 63% | **34%** | 2% | 0% | 1% |
| bend (171) | off | 9% | 1% | **28%** | 48% | 14% |
| | glide merge | 32% | 1% | **3%** | 50% | 15% |
| slide (111) | off | 18% | 0% | **58%** | 18% | 6% |
| | glide merge | 49% | 0% | **27%** | 18% | 6% |
| normal (4,023) | off | 84% | 6% | 4% | 0% | 6% |
| | glide merge | 83% | 6% | 4% | 1% | 6% |

- **The split-note problem shrinks but doesn't go away:** 55% to 34% of
  vibrato notes. The rest are runs where the vibrato detector doesn't fire
  (its recall was 19-21%).
- **The glide merge removes the extra note of a bend almost every time it
  is there** (28% to 3%), and half of the slides'. It does nothing for the
  48% of bends where the tab has only the bent pitch.
- **Its cost is on plain notes:** 36 real notes merged away where no
  technique was played.
- A slide's truth in IDMT is one note at the starting pitch, so merging
  its end note counts as removing an extra. Musically that note is where
  the slide arrives; a tab would want it kept, with a slide marking.

The finer contour as the source of pitch offsets (in place of the integer
`pitch_bends`) gives the same picture: vibrato -121 wrong / -1 right, glide
-148 wrong / -54 right. The pipeline's own `pitch_bends` are enough.

### EGSet12 (no techniques: every change is a false alarm)

Benchmark recall / precision / position agreement, all segments:

| configuration | clean | moderate | heavy |
|---|---|---|---|
| off | 71.9 / 85.2 / 53.2 | 55.8 / 80.7 / 48.3 | 38.5 / 71.7 / 45.0 |
| vibrato merge | 71.8 / 85.2 / 53.3 | 55.8 / 80.9 / 48.3 | 38.2 / 72.1 / 45.4 |
| glide merge | 71.2 / 85.3 / 53.2 | 55.2 / 80.9 / 48.2 | 38.1 / 71.8 / 45.6 |

Notes a merge removed, before the mapper:

| tone (kept notes) | vibrato merge | glide merge |
|---|---|---|
| clean (1,333) | **2 real notes lost**, 0 wrong removed | **11 real lost**, 3 wrong removed |
| moderate (1,094) | 0 real lost, 3 wrong removed | 9 real lost, 5 wrong removed |
| heavy (853) | 5 real lost, 6 wrong removed | 7 real lost, 3 wrong removed |

- The glide merge costs most on fast passages: recall 73.5 to 69.9% clean,
  71.1 to 67.5% moderate.
- The vibrato merge's losses are 6 runs, 5 of them in performance 07: a
  sustained note next to re-struck notes of the same pitch. pYIN sees a
  small real wobble on those sustained notes (about 5 cents at 4-8 Hz),
  which Basic Pitch's coarse offsets read as 13-18 cents. The merge then
  swallows the re-struck notes.

### Regression check (against the saved baseline)

| section | off | vibrato merge | glide merge |
|---|---|---|---|
| synthetic chord set | same | same | **clean program: recall 80.0 to 76.8%, complete chords 40 to 35%** (fast 37.5 to 25%) |
| Fire Force windows | same | same | same |
| solo clip vs its known tab | same | same | same |
| EGSet12 | same | as above | as above |

## Verdict, by the rule

| merge | IDMT wrong notes | EGSet12 | regression check | keep? |
|---|---|---|---|---|
| vibrato | -130 (and -3 right) | loses 2 real notes on clean and 5 on heavy (recall -0.1 / -0.3 points); precision +0.2 / +0.4 on moderate / heavy | unchanged outside EGSet12 | **No, narrowly.** It hurts EGSet12, by 2 notes in 1,333 on the clean tone |
| glide | -138 (and -44 right) | loses 7-11 real notes per tone; fast passages -2.4 to -3.6 recall points | synthetic chords get worse | **No** |

Neither is switched on, and the vibrato marking in the MusicXML export
was not added, since it would only exist with the vibrato merge.

The vibrato merge is close: about 43 wrong notes removed on IDMT for each
real note it loses there, and its EGSet12 cost is 7 notes over three tones
against 9 wrong notes removed. If losing 2 notes on EGSet12's clean tone
is acceptable, it is one line to turn on (`transcribe.VIBRATO_MERGE`). The
obvious refinement, not measured, is to merge only across a join that lies
inside the wobble, so a re-struck note before the vibrato starts stays
separate.

## Reproduce

```sh
R=<repo>
W="docker run --rm -v $R/data:/app/data -v $R:/repo:ro -v $R/experiments/technique_cleanup:/x \
   -v $R/experiments/expression:/expression stratotab-worker"
$W python /x/cleanup.py idmt
$W python /x/cleanup.py egset12
$W python -m scripts.regression_check --compare /app/data/_regression/hq_separation_after.json
$W python -m scripts.regression_check --vibrato-merge 1 --compare /app/data/_regression/hq_separation_after.json
$W python -m scripts.regression_check --glide-merge 1 --compare /app/data/_regression/hq_separation_after.json
```

IDMT-SMT-Guitar V2 (CC BY-NC-ND 4.0) must be unpacked under
`data/_expression/idmt/`; it is not committed.
