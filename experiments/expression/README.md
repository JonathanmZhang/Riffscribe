# Expression: synth sound and playing techniques (2026-10-02)

Two parts. Part 1 adds one small feature (a playback-tone choice) and
compares SoundFonts for listening; the default SoundFont is unchanged.
Part 2 is measurement only: nothing in the pipeline changed.

The measurements ran on the worker image built from this branch (commit
`1e4d1b3`; the stale-image guard matched on every
run). Data and audio are under `data/_expression/` (git-ignored).

## 1. Synth sound

### What plays today

- The MusicXML export (`worker/tasks/musicxml.py`) writes one part with
  `<midi-program>28</midi-program>` and instrument name "Electric Guitar".
  MusicXML counts programs from 1, so that is General MIDI **Electric
  Guitar (clean)**.
- alphaTab reads it as program 27 (it counts from 0, as MIDI does). Checked
  in the browser: `score.tracks[0].playbackInfo.program === 27`.
- The sound comes from alphaTab's bundled **SONiVOX** SoundFont
  (`sonivox.sf2`, 1.35 MB, Apache-2.0), loaded from `public/alphatab/`.

### Playback tone (added)

`GET /jobs/{id}/musicxml?tone=` sets the program and instrument name; the
sheet-music view's Synth mode has a "Tone" selector that reloads the export
with it. The notes are identical for every tone, and it isn't stored on
the job.

| tone | General MIDI instrument | `midi-program` in the export | program alphaTab holds |
|---|---|---|---|
| `clean` (default) | Electric Guitar (clean) | 28 | 27 |
| `overdriven` | Overdriven Guitar | 30 | 29 |
| `distorted` | Distortion Guitar | 31 | 30 |
| `acoustic` | Acoustic Guitar (steel) | 26 | 25 |

Checked: each selection gives alphaTab the program in the last column
(`render_soundfonts.py` waits for exactly that); the default export is
byte-identical to `tone=clean`; all four exports pass
`experiments/musicxml/validate.py` (MusicXML 4.0 XSD and music21); an
unknown tone returns 422.

### SoundFont candidates and licenses

"Usable" means: may be shipped in this public MIT repository's app.
Licenses were read at the source.

| SoundFont | License, as stated by its author | Size | Guitar presets | Usable? |
|---|---|---|---|---|
| **SONiVOX** (alphaTab's default) | Apache-2.0 | 1.35 MB | all four | in use |
| **FluidR3_GM** (Frank Wen) | **MIT**: "I hereby release Fluid under the MIT license, as described in COPYING." Taken from Debian's `fluid-soundfont-gm` 3.1-5.3 (already in the worker image for the chord test set) | 141.5 MiB | Steel String 25, Clean 27, Overdrive 29, Distortion 30 | **Yes**, with its copyright notice. Too large to ship whole: it would have to be cut down to the guitar presets |
| **GeneralUser GS 2.0.3** (S. Christian Collins) | Its own license (v2.0), not an OSI one: "You may use GeneralUser GS without restriction for your own music creation, private or commercial ... Please feel free to use it in your software projects, and to modify the SoundFont bank or its packaging to suit your needs." | 30.8 MiB | Steel 25, Clean 27, Overdrive 29, Distortion 30 | **Permitted, with a caveat** the author states himself: "I cannot be 100% sure where all of the samples originated", though none came from commercial packages as far as he knows |
| MuseScore_General 0.2 | MIT (FluidR3's license, plus S. Christian Collins' changes) | 215.6 MB (.sf3: 39.9 MB) | its sample-source list names no other source for presets 24-31, so the guitars are FluidR3's samples | Yes, but not rendered: same guitar samples as FluidR3_GM |
| FreePats FSBS electric guitar, clean and distorted | **CC0 1.0** (the cleanest of all) | clean "small" 10.4 MiB; distorted #2 374.5 MiB | one preset each, at program 0 | License yes. **Doesn't work in alphaTab**: see below |
| FreePats FSS steel-string acoustic | GPL-3.0-or-later with FreePats' special exception | 12 MiB | one preset | Not tried: a GPL file in an MIT repo needs its own notice and terms |

I didn't check other well-known free banks (Arachno, SGM, Timbres of
Heaven); their licenses are generally less clear than the ones above.

**FreePats in alphaTab:** both FreePats fonts render as silence in
alphaTab's synth. Its console says why: "Skipping load of unsupported
sample ... sample type 4 is not supported" (and type 2). Those are the left
and right halves of stereo samples, which alphaSynth 1.8.4 skips, and
these fonts hold nothing else. They would need converting to mono samples
first. Not done.

Pinned files used (SHA-256): `FluidR3_GM.sf2` `74594e8f...bc1cb0`,
`GeneralUser-GS.sf2` `9575028c...688cfe` (repository commit `684543d`).

### Comparison renders (`render_soundfonts.py`)

The same passage, rendered by **alphaTab's own synth** (its `exportAudio`,
driven in headless Chromium through the app's page and Tone selector), so
each file is what the Synth button would play with that SoundFont. The
passage is EGSet12 performance 12 as the pipeline transcribed it: 85
notes, 9 bars at 83 bpm, 26 s, chords and single-note lines.

Files, in `data/_expression/soundfont_compare/` (12 WAVs, 44.1 kHz
stereo), named `<tone>__<soundfont>.wav`:

| tone | default | candidate 1 | candidate 2 |
|---|---|---|---|
| clean | `clean__sonivox-default.wav` | `clean__FluidR3_GM.wav` | `clean__GeneralUser-GS.wav` |
| overdriven | `overdriven__sonivox-default.wav` | `overdriven__FluidR3_GM.wav` | `overdriven__GeneralUser-GS.wav` |
| distorted | `distorted__sonivox-default.wav` | `distorted__FluidR3_GM.wav` | `distorted__GeneralUser-GS.wav` |
| acoustic | `acoustic__sonivox-default.wav` | `acoustic__FluidR3_GM.wav` | `acoustic__GeneralUser-GS.wav` |

Every file is scaled to the same RMS level (-20 dBFS), so loudness doesn't
decide the comparison. The synth's own output levels differ a lot, which
matters if a font is adopted (peak of the unscaled output, at alphaTab's
default volume):

| tone | SONiVOX | FluidR3_GM | GeneralUser GS |
|---|---|---|---|
| clean | +3.9 dBFS | +6.3 | -1.5 |
| overdriven | +2.4 | +5.2 | -5.9 |
| distorted | +2.3 | +9.8 | -8.1 |
| acoustic | +1.6 | +7.9 | -13.3 |

Anything above 0 dBFS clips when played through the sound card. So the
default already exceeds full scale on this passage's chords, FluidR3_GM
would need the volume turned down a lot, and GeneralUser GS has headroom.
This is from the exported audio; the live player's output wasn't measured.

**Fixed since (`synth_level.py`):** the synth's master volume is now 0.4
(-8 dB) in the sheet-music view. With the default font at volume 1.0,
three transcriptions (EGSet12 12 and 06, and a 99 s full-band song) x four
tones peaked between +1.0 and +5.9 dBFS; at 0.4 the highest is -2.1 dBFS.
The live player reports the same volume. A denser song could still go
higher; this is measured on those three.

No judgement on which sounds better is given here: that is for listening.
**The SoundFont is not switched.**

## 2. Techniques (measurement only)

### How often techniques appear in EGSet12 (`gp_techniques.py`)

**Never.** The 12 Guitar Pro files hold 755 notated notes (433 distinct
note objects) and not one bend, slide, hammer-on, pull-off, vibrato or
palm mute.

| technique | GPIF marking counted | marked |
|---|---|---|
| bends | note property `Bended` | 0 |
| slides | note property `Slide` | 0 |
| hammer-ons / pull-offs | note properties `HopoOrigin`, `HopoDestination` | 0 |
| vibrato | note `<Vibrato>`, beat `<Whammy>` | 0 |
| palm mutes | note property `PalmMuted` | 0 |

Everything else the files mark, beyond pitch and position: 31 ties, 4
accents, 80 explicit velocities (all in 08), 10 chord diagrams, 5 ottava
marks, 1 arpeggio.

The JAMS files do have a `pitch_contour` track per string, but it is
generated from the score: all 1,567 points sit exactly on an equal-tempered
pitch (largest deviation 0.000 cents). So EGSet12 has **no ground truth for
pitch movement of any kind**, and bends can't be scored against it.

### Substitute ground truth

| set | what it is | notes | license |
|---|---|---|---|
| synthetic (`make_bend_testset.py`) | single notes, FluidSynth + FluidR3_GM, pitch moved with the MIDI pitch wheel; clean, overdriven and distortion programs | 216: 144 with a bend (half step, whole step, slow, bend-release, pre-bend-release, bend + vibrato), 72 with vibrato (48 vibrato only), 24 plain | MIT soundfont |
| **IDMT-SMT-Guitar V2**, dataset 2 | real electric guitars (3), note sequences and 12 licks, with expression and plucking style labelled per note by Fraunhofer IDMT | 4,881: 171 bends, 111 slides, 162 vibrato, 153 harmonics, 261 dead notes, 4,023 normal (1,263 of them plucked "muted") | CC BY-NC-ND 4.0 ([Zenodo 7544110](https://zenodo.org/records/7544110)). Used locally for measurement, not committed or redistributed |
| IDMT-SMT-Guitar V2, dataset 1 | real plain single notes and chords, 4 guitar settings | 796, all normal | same |
| EGSet12 clean | the usual 12 performances | 1,333 tab notes | no technique truth: false alarms only |

The synthetic set is the easy case: a wheel-bent sample glides perfectly
and doesn't change timbre. IDMT is the real test.

### What Basic Pitch's `pitch_bends` holds

`get_pitch_bends` gives each note one integer per frame (11.6 ms): the
offset of the strongest contour bin from the note's pitch, in thirds of a
semitone. Two things limit it:

- **A bent note doesn't stay one note.** The offsets only reach about -1
  to +3 thirds of a semitone. Beyond that, Basic Pitch ends the note and
  starts a new one on the next semitone. A whole-step bend from A3 comes
  out as A3 (0.17 s), then B3: two notes, no gap.
- An unbent note reads +1, not 0 (the contour bins sit a third of a
  semitone off), so offsets have to be read against that resting value.

So the bend detector (`bends.py`) looks at the **joins** between notes: two
notes 1 or 2 semitones apart with no gap are a glide if the pitch is
moving across the join (25 cents or more towards the other note, over 3
frames on either side). Vibrato is the strongest 4-8 Hz component of a
note's pitch (12 cents or more), taken over a run of same-pitch notes.
Both were run from the integer `pitch_bends` alone and from a finer
contour (centroid of the contour bins around each frame's peak). The
settings were fixed after looking at one example of each synthetic type,
before any scoring.

### Results: detection

Recall = share of the truth notes with the technique that were detected.
Precision = share of the detector's firings that were right.

| set | detector | recall | precision | fired on notes without it |
|---|---|---|---|---|
| synthetic | bend (either source) | 95.1% (137/144) | 100% (137/137) | 0 of 72 |
| synthetic | vibrato (either source) | 100% (72/72) | 100% (72/72) | 0 of 144 |
| **IDMT dataset 2 (real)** | bend, fine contour | **26.9%** (46/171) | **41.8%** (46/110) | 1.4% (64/4,710) |
| | bend, `pitch_bends` only | 26.9% (46/171) | 46.5% (46/99) | 1.1% (53/4,710) |
| | vibrato, fine contour | **18.5%** (30/162) | **96.8%** (30/31) | 1 of 4,719 |
| | vibrato, `pitch_bends` only | 21.0% (34/162) | 91.9% (34/37) | 3 of 4,719 |
| IDMT dataset 1 (real, plain notes) | bend / vibrato | - | - | 0 of 796 / 0 of 796 |
| EGSet12 clean | bend | - | - | 19 joins (1.4 per 100 tab notes) |
| | vibrato | - | - | 1 note |

- **On real bends it finds about a quarter.** 42 of its 64 false alarms
  on IDMT are **slides**, which glide the same way: counted as "bend or
  slide", precision is 80% (88/110) and recall 31% (88/282). It can't tell
  the two apart.
- **Why recall is low on real bends:** the detector needs the fretted
  pitch and the bent pitch as two notes. In real licks most bends are
  quick (median 0.25 s in Lick 11) and the tab holds only the bent pitch,
  or nothing (next table). There is no join to find.
- **The integer `pitch_bends` are as good as the finer contour** for
  bends, and slightly better for vibrato recall. Nothing is gained by
  going back to the contour matrix.
- **Vibrato detection is precise but misses most of it** (19-21% recall
  at 92-97% precision). Lowering the amplitude threshold to 6 cents only
  reaches 23%.
- **Thresholds don't rescue it** (`--sweep`): on IDMT, bend recall stays
  at 26-27% from 15 to 40 cents; requiring movement on both sides of the
  join raises precision to 61-67% at 13-18% recall.
- **A variant added after the first IDMT run** ("joins + drift": also
  accept a single note whose pitch holds two levels 40 cents apart)
  changes nothing: 27.5% recall, 41.2% precision.
- **EGSet12:** 19 of its 168 joins between notes 1-2 semitones apart
  triggered the bend detector, with no bends annotated. Ten of them could
  be checked with pYIN (both notes sounding alone): in 8 the pitch spends
  under 25 ms between the two notes, and in 2 it spends 35 ms (a synthetic
  150 ms whole-step bend spends 104-116 ms, a plain note 0). So these are
  mostly false alarms, about 1.4 per 100 notes, not unmarked bends.

### Results: what the tab shows today for these notes

This answers "does a bend show up as an extra wrong note". For each truth
note: the tab notes within 1 semitone below to 3 above its pitch that
overlap it.

IDMT dataset 2 (real guitars):

| technique | notes | one right note | right pitch, split in 2+ | right + another pitch | another pitch only | nothing |
|---|---|---|---|---|---|---|
| normal | 4,023 | 84% | 6% | 4% | 0% | 6% |
| **bend** | 171 | **9%** | 1% | **28%** | **48%** | 14% |
| slide | 111 | 18% | 0% | 58% | 18% | 6% |
| **vibrato** | 162 | 42% | **55%** | 2% | 0% | 1% |
| harmonic | 153 | 86% | 7% | 0% | 0% | 8% |
| dead note | 261 | 11% | 0% | 0% | 7% | 82% |

- **Yes: a real bend is almost never one right note (9%).** In 28% the
  tab has the fretted note plus an extra note at the bent pitch, and in
  48% it has only the bent pitch, one or two semitones above the fret that
  was played. On the synthetic set every bend (144/144) is the fretted
  note plus extra notes.
- **Vibrato splits a note into repeats** in 55% of cases (62-71% on the
  synthetic set): the tab shows the same fret struck two to six times.
- Slides mostly come out as the two end notes, which is roughly right as
  notes, with no slide marking.
- Dead notes are mostly not transcribed (82%), and muted plucking loses
  more plain notes than picking (15% nothing for "muted", 2% picked, 3%
  finger-style).

### Does alphaTab draw techniques from MusicXML? (`alphatab_techniques.py`)

A 3-bar test score built with the app's exporter, with standard MusicXML
elements added on both staves (the file is valid MusicXML 4.0), loaded
into the app's alphaTab 1.8.4:

| technique | MusicXML used | alphaTab's model | drawn |
|---|---|---|---|
| bend | `<technical><bend><bend-alter>2</bend-alter></bend>` | Bend, points (0,0) to (60,4) | **Yes**: arrow and "full" on TAB, bend line on notation |
| bend and release | two `<bend>`, the second with `<release/>` | BendRelease | **Yes** |
| pre-bend and release | `<bend>` with `<pre-bend/>`, then `<release/>` | PrebendRelease | **Yes** |
| slide | `<slide type="start"/>` ... `stop` | slide out: Shift | **Yes**, on both staves |
| vibrato | `<ornaments><wavy-line/>` | vibrato: Slight | **Yes**, wavy line |
| palm mute | `<play><mute>palm</mute></play>` on the note | palm mute on that note | **Yes**, "P.M." |
| palm mute | `<words>P.M.</words>` + `<dashes>` | palm mute, but it **never ends**: the `stop` isn't recognised and every later note is muted too | wrongly |
| hammer-on / pull-off | `<hammer-on>` / `<pull-off>`, with or without a slur | **not kept** (4 variants tried); only the slur survives | slur on the notation staff only; nothing on TAB |
| dead note | `<notehead>x</notehead>` | not marked dead | X notehead on notation; the fret number stays on TAB |

Screenshot: `data/_expression/alphatab_techniques/techniques.png`. Whether
alphaTab's synth plays the bends wasn't tested.

## What this says

1. **EGSet12 can't measure techniques.** It has none, in the Guitar Pro
   files or the JAMS contours. Any work on expression needs IDMT-SMT-Guitar
   (non-commercial license, local use) or new annotations.
2. **`pitch_bends` alone is not a usable bend detector on real playing:**
   27% recall, 42% precision on IDMT, and it can't separate bends from
   slides. On synthetic bends it is near perfect, which shows how
   misleading the easy case is.
3. **The larger effect is on the notes themselves.** Real bends give a
   wrong or extra pitch 76% of the time, and vibrato splits 55% of notes
   into repeats. Marking techniques matters less than what they do to the
   tab today. Merging a detected glide back into one note, and a vibrato
   run into one note, is the obvious next thing to measure. It isn't
   measured here: on EGSet12 it would also merge about 1.4 wrongly per 100
   notes.
4. **alphaTab can show bends, slides, vibrato and per-note palm mutes**
   from MusicXML if the export writes them. It can't show hammer-ons,
   pull-offs or dead notes that arrive that way.
5. Slides, hammer-ons, pull-offs and palm mutes weren't detected at all
   here; `pitch_bends` carries nothing about how a note was plucked.

## Reproduce

```sh
R=<repo>
python experiments/expression/gp_techniques.py
# SoundFont renders (stack up; SoundFonts in data/_expression/soundfonts/):
docker run --rm --shm-size=4g --network stratotab_default -v $R/experiments/expression:/x -v $R/data/_expression:/d \
    mcr.microsoft.com/playwright/python:v1.63.0-noble sh -c \
    "pip install -q playwright==1.63.0 numpy && python /x/render_soundfonts.py <job id>"
# Bends and vibrato (IDMT-SMT-GUITAR_V2.zip unpacked under data/_expression/idmt/):
W="docker run --rm -v $R/data:/app/data -v $R:/repo:ro -v $R/experiments/expression:/x stratotab-worker"
$W python /x/make_bend_testset.py
$W python /x/bends.py synthetic --sweep
$W python /x/bends.py idmt --sweep
$W python /x/bends.py egset12
# alphaTab and MusicXML technique elements:
docker run --rm --network stratotab_default -v $R/experiments/expression:/x -v $R/worker:/w:ro -v $R/data/_expression:/d \
    mcr.microsoft.com/playwright/python:v1.63.0-noble sh -c \
    "pip install -q playwright==1.63.0 && python /x/alphatab_techniques.py"
```

The saved reports are in `data/_expression/results/`.
