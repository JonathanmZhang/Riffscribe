# Technique cleanup: vibrato merge and glide merge (2026-10-02)

**Outcome:** the vibrato merge is **on** (`transcribe.VIBRATO_MERGE = 1`),
with a vibrato marking in the MusicXML export. The glide merge is **off**
(rejected). Sections "Results" and "Verdict" below are the first round,
measured with both off; "Refinement and decision" at the end is the
second round and the final state.

Both merges are in `worker/tasks/techniques.py`, applied by
`transcribe.clean_notes` after `select_notes`. All numbers are from worker
images whose stale-image guard matched (first round `3052db8`, second
round `713924f`). Reports: `data/_technique_cleanup/`.

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

## Verdict of the first round, by the rule

| merge | IDMT wrong notes | EGSet12 | regression check | keep? |
|---|---|---|---|---|
| vibrato | -130 (and -3 right) | loses 2 real notes on clean and 5 on heavy (recall -0.1 / -0.3 points); precision +0.2 / +0.4 on moderate / heavy | unchanged outside EGSet12 | **No, narrowly.** It hurts EGSet12, by 2 notes in 1,333 on the clean tone |
| glide | -138 (and -44 right) | loses 7-11 real notes per tone; fast passages -2.4 to -3.6 recall points | synthetic chords get worse | **No** |

So after the first round neither was switched on.

The vibrato merge is close: about 43 wrong notes removed on IDMT for each
real note it loses there, and its EGSet12 cost is 7 notes over three tones
against 9 wrong notes removed. If losing 2 notes on EGSet12's clean tone
is acceptable, it is one line to turn on (`transcribe.VIBRATO_MERGE`). The
obvious refinement, not measured, is to merge only across a join that lies
inside the wobble, so a re-struck note before the vibrato starts stays
separate.

## Refinement and decision (second round)

The rule was relaxed for the vibrato merge: a few real notes lost for 130
wrong notes removed is an acceptable trade. First the refinement was
tried.

**"Wobble" variant** (`VIBRATO_MERGE = 2`): inside a run that has vibrato,
two pieces are merged only if the wobble is there on both sides of their
join (the strongest 4-8 Hz component of the 0.5 s before and of the 0.5 s
after is 12 cents or more; a side shorter than 0.3 s can't show one, so
that join is left alone). A note struck again before the vibrato starts,
or after it stops, stays separate.

| vibrato merge | IDMT wrong notes | IDMT right notes | vibrato notes that are one right note (of 162) | still split | EGSet12 real notes lost (clean / moderate / heavy) | EGSet12 wrong notes removed |
|---|---|---|---|---|---|---|
| off | | | 68 (42%) | 89 (55%) | | |
| whole runs (plain) | **-130** | -3 | **102 (63%)** | 55 (34%) | 2 / 0 / 5 | 0 / 3 / 6 |
| wobble, 12 cents | -80 | **0** | 73 (45%) | 84 (52%) | **1 / 0 / 1** | 0 / 2 / 1 |
| wobble, 9 cents | -88 | 0 | 74 (46%) | 83 (51%) | 1 / 0 / 1 | 0 / 2 / 1 |
| wobble, 6 cents | -92 | 0 | 76 (47%) | 81 (50%) | 1 / 0 / 1 | 0 / 2 / 2 |

EGSet12 benchmark, recall / precision / position:

| | clean | moderate | heavy |
|---|---|---|---|
| off | 71.9 / 85.2 / 53.2 | 55.8 / 80.7 / 48.3 | 38.5 / 71.7 / 45.0 |
| whole runs | 71.8 / 85.2 / 53.3 | 55.8 / 80.9 / 48.3 | 38.2 / 72.1 / 45.4 |
| wobble | 71.9 / 85.2 / 53.3 | 55.8 / 80.9 / 48.3 | 38.5 / 71.8 / 45.1 |

- **The refinement does what it was meant to:** it loses no right notes on
  IDMT and 2 on EGSet12 (against 3 and 7).
- **But it repairs far less.** It removes 80 wrong notes where the plain
  merge removes 130, and only 5 more vibrato notes come out as one right
  note (the plain merge: 34 more). Vibrato usually splits a note into short
  pieces, and a short piece at the edge of the run has no 0.3 s of wobble
  on its outer side, so its join is left alone, which is the same shape as
  the re-struck note the rule is there to protect. Lowering the threshold
  doesn't change that (92 at 6 cents).

**Decision: the plain merge is on.** Going from the wobble variant to the
plain one removes 50 more wrong notes on IDMT (and 6 more on EGSet12) for 8
more real notes lost across IDMT and the three EGSet12 tones, and it is
the only one that visibly reduces split vibrato notes (55% to 34%). The
wobble variant stays in the code as `VIBRATO_MERGE = 2`.

**Glide merge: off, rejected** (first round: it loses real notes on plain
playing and worsens the synthetic chord set).

**Vibrato marking:** a merged or detected note carries `"vibrato": true`
through the mapper into the result's notes. The MusicXML export writes a
`<wavy-line>` on both staves (on the first piece of a tied note). Checked
with a real vibrato recording from IDMT submitted as a job: 2 of its 32
notes are flagged, the export passes the MusicXML 4.0 XSD, and alphaTab
holds both notes as vibrato on both staves and draws the wavy lines
(`data/_technique_cleanup/vibrato_sheet.png`).

**Regression check with the merge on** (`data/_regression/
vibrato_merge_after.*`, the new baseline): the synthetic chord set, the
Fire Force windows and the solo clip are identical to the previous
baseline. EGSet12 changes as in the table above.

## Reproduce

```sh
R=<repo>
W="docker run --rm -v $R/data:/app/data -v $R:/repo:ro -v $R/experiments/technique_cleanup:/x \
   -v $R/experiments/expression:/expression stratotab-worker"
$W python /x/cleanup.py idmt
$W python /x/cleanup.py egset12
$W python -m scripts.regression_check --compare /app/data/_regression/hq_separation_after.json
$W python -m scripts.regression_check --vibrato-merge 0 --compare /app/data/_regression/hq_separation_after.json
$W python -m scripts.regression_check --vibrato-merge 2 --compare /app/data/_regression/hq_separation_after.json
$W python -m scripts.regression_check --glide-merge 1 --compare /app/data/_regression/hq_separation_after.json
```

IDMT-SMT-Guitar V2 (CC BY-NC-ND 4.0) must be unpacked under
`data/_expression/idmt/`; it is not committed.
