# MusicXML export (2026-09-30)

`GET /jobs/{id}/musicxml` returns the finished tab as a MusicXML 4.0 file
(`riffscribe-<id>.musicxml`). The file has one guitar part with two staves:

- **Notation staff:** a treble clef that sounds an octave lower, as guitar
  music is written.
- **TAB staff:** 6 lines in standard tuning, with each note's string and
  fret.

The music is in 4/4. Note starts and lengths are quantized to 16ths, and
BTC's chord names are chord symbols above the notation staff. The backend
builds the file on each request from the stored TabResult and the job's
overrides (`tempo_factor`, `bar_offset_beats`, set with
`PATCH /jobs/{id}`), so nothing is re-transcribed.

The code is `worker/tasks/rhythm.py` (bars and quantization) and
`worker/tasks/musicxml.py` (the XML). Both are stdlib-only. The backend
image copies them in through the `worker` build context, so rebuild the
backend when they change. The measurements behind the choices are in
`experiments/rhythm/README.md` (section "Notation").

## What's in the file

- **Bars:** 4/4. beat_this's beats are kept at one metrical level (stray
  double-time beats dropped, half-time gaps filled), halved or doubled by
  `tempo_factor`, then grouped in 4s. The bar phase is where most of
  beat_this's downbeats fall, moved later by `bar_offset_beats`. Bars are
  extended before the first note and after the last one, so they are
  always full: leading and trailing gaps are rests.
- **Starts:** each note goes to its nearest 16th. Notes on the same 16th
  form one chord. If two notes land on the same 16th and the same string,
  the later one is dropped (1 of 477 notes on firefire, none on
  performance 06).
- **Lengths:** chords and fast single notes use Basic Pitch's length; other
  single notes use the gap to the next note. The length is snapped to a 16th,
  8th, dotted 8th, quarter, dotted quarter, half or whole, then cut at the
  next note, because the notation is one voice. A length that crosses a bar
  line is tied. Lengths that aren't one written value (for example 5
  sixteenths) are tied pieces, longest first.
- **Chord symbols:** from BTC's segments, snapped to a 16th. A symbol is
  kept if it lasts at least a beat and differs from the previous one, which
  hides BTC's short flickers. A symbol that falls in the middle of a
  sustained note gets a MusicXML `<offset>`.
- **Tempo:** a metronome mark and `<sound tempo>` from the grid's median
  beat.
- **Jobs without beats** (older jobs, or when beat tracking failed) use a
  constant grid at the stored tempo, starting at the first note.

## Validation (`validate.py`)

Each file is checked two ways:

1. **Against the official MusicXML 4.0 XSD** (W3C, tag v4.0).
2. **Parsed with music21 9.3**, with these checks:
   - every bar is 4/4 and filled to 4 quarters on both staves;
   - every TAB note has a string and a fret;
   - every tie start has a matching stop.

| export | XSD | bars | notes (notation / TAB) | rests | chord symbols |
|---|---|---|---|---|---|
| EGSet12 06 (clean, 103 bpm) | valid | 13 | 232 / 232 | 5 | 10 |
| EGSet12 06, `bar_offset_beats` 2 | valid | 14 | 267 / 267 | 7 | 10 |
| firefire (102s, full mix, 187 bpm) | valid | 77 | 515 / 515 | 97 | 90 |
| firefire, `tempo_factor` 0.5 (94 bpm) | valid | 39 | 490 / 490 | 47 | 67 |
| an older job with no beats (constant 92 bpm grid) | valid | 39 | 489 / 489 | 58 | 70 |

Note counts include tied pieces.

```sh
# Git Bash: prefix MSYS_NO_PATHCONV=1. Put exports in experiments/musicxml/out/ (git-ignored).
curl -o experiments/musicxml/out/job.musicxml http://localhost:8000/jobs/<job_id>/musicxml
docker run --rm -v "$(pwd -W)/experiments/musicxml:/v" python:3.11-slim \
    sh -c "pip install -q music21==9.3.0 xmlschema==3.4.3 && python /v/validate.py /v/out/*.musicxml"
```

## Checking by eye in MuseScore (4.x)

1. **Open the file:** File → Open, choose the `.musicxml`, or drag it onto
   the MuseScore window. If MuseScore offers import options, keep the
   defaults. It opens as one guitar part with the notation staff and the
   TAB staff joined by a brace, with the tempo mark and chord symbols above.
2. **Compare it with the tab in the app.** Open the same job in Riffscribe
   and check that the TAB staff's frets and strings match the tab view,
   column by column.
3. **Check the rhythm by ear.** Press Space to play the score (MuseScore
   uses its own guitar sound at the exported tempo), and play the original
   audio alongside. The things to judge are:
   - where the bar lines fall (does beat 1 feel like the "1"?);
   - whether chords land on the beat;
   - whether note lengths sound right.
4. **If the bar lines feel wrong,** re-export with an override and reopen:
   ```sh
   curl -X PATCH -H "Content-Type: application/json" -d '{"bar_offset_beats": 1}' http://localhost:8000/jobs/<job_id>
   curl -X PATCH -H "Content-Type: application/json" -d '{"tempo_factor": 0.5}' http://localhost:8000/jobs/<job_id>   # or 2
   ```
5. **If a note looks odd,** click it: the status bar shows its pitch and
   duration, and on the TAB staff its string and fret.

Known limits you will see:

- **One voice.** A chord that rings under the next note is cut where the
  next note starts.
- **Key and spelling.** There is no key signature (C major), and pitches
  are spelled with sharps.
- **No beam information.** MuseScore beams with its defaults.
- **No triplets.** Triplets are 1.5% of EGSet12's Guitar Pro onsets.
- **Bar lines.** With the default 4/4 grouping, the bar lines are right
  about half the time (downbeat F 49-50%). That's why the overrides exist.
