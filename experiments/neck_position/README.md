# Neck position: a user-chosen fret window for the tab (2026-10-03)

The mapper (`worker/tasks/fretmap.py`, spec 3.5) only minimizes hand
movement, so it doesn't know where on the neck the player is. On EGSet12
about 90% of its position errors land ~5 frets toward the nut
(feature/hand-position). A cost model couldn't fix that
(feature/hand-position, rejected). This branch lets the user say where the
player is, for example after watching the video, and re-places the same
notes there without re-transcribing.

## What it does

`PATCH /jobs/{id}` `{"neck_position": "auto" | "open" | N}` (N = 1-17)
stores the choice on the job. On every read, `GET /jobs/{id}` and
`GET /jobs/{id}/musicxml` re-map the job's stored `raw_note_events` with
the window (backend, `routes/jobs._result_data`). That covers the tab, the
sheet-music TAB staff and the MusicXML download. Nothing is stored except
the choice.

- **auto**: the mapper unchanged. Its output is bit-identical to the old
  `tasks/fretboard.py` on 71 inputs: 36 EGSet12 runs plus every finished
  job in the local Redis, compared against a dump from the image built
  before the change.
- **open**: frets 0-4, open strings allowed.
- **around fret N**: frets max(1, N-3) to N+3. Open strings are not
  allowed, unless a note exists only as an open string.

Inside the window:

- A single note's candidates are its positions inside the window. If it
  has none, it gets its nearest positions (fewest frets outside).
- A chord gets every voicing on distinct strings, ranked by total frets
  outside the window, then by a fretted span over 4 frets, then by fret
  sum. The best ones go to the DP (at most 12).
- The DP, its costs and its tie-break are unchanged.
- Which notes are kept is decided before the window is applied
  (`_make_voiceable` uses the window-free voicing check). So every setting
  places the same notes; only strings and frets differ.

Cost per read is 55-90ms on a 641-note, 120s job in the backend container.
One PATCH to "open" took 430ms.

## EGSet12, best case: window = the truth's median fret +- 3

`python /x/measure.py` (worker image, repo at /repo, this folder at /x).
For each performance the window is the median fret of all its annotated
notes, +- 3 (it includes fret 0 when the median is 3 or less). That is the
best a user could do by watching the video and choosing one setting for
the whole song. The script checks that every run's notes are identical to
Auto's, for every setting, and stops if not. Pitch recall and precision
are identical in every row.

Position agreement (same string and fret, among pitch-matched notes):

| tone     | segment     | Auto | median +- 3 | median +- 2 | best UI setting (oracle) |
|----------|-------------|-----:|------------:|------------:|-------------------------:|
| clean    | all         | 53.3 | **69.3**    | 69.2        | 73.1 |
| clean    | chords      | 59.4 | 71.6        | 73.6        | 75.0 |
| clean    | single-note | 49.6 | 67.0        | 65.3        | 71.6 |
| clean    | fast        | 24.6 | 65.6        | 55.7        | 65.6 |
| moderate | all         | 48.3 | **65.8**    | 65.6        | 70.9 |
| moderate | chords      | 54.7 | 66.2        | 70.0        | 72.4 |
| moderate | single-note | 45.9 | 65.2        | 63.1        | 70.1 |
| moderate | fast        | 25.4 | 67.8        | 55.9        | 67.8 |
| heavy    | all         | 45.4 | **66.1**    | 65.8        | 71.8 |
| heavy    | chords      | 50.6 | 66.0        | 71.6        | 77.8 |
| heavy    | single-note | 44.7 | 67.0        | 64.7        | 70.8 |
| heavy    | fast        | 32.6 | 58.1        | 53.5        | 58.1 |

Pitch recall / precision (identical for every setting): clean 71.8 / 85.2,
moderate 55.8 / 80.9, heavy 38.2 / 72.1. Matched notes: 1125 / 874 / 599.

"Best UI setting" is an oracle: for each performance and tone, whichever
of the control's 18 choices agrees most with the truth. It is the ceiling
of the control as built (one setting per song), not something a user can
reliably reach.

Per performance (clean, moderate, heavy; Auto -> median +- 3):

| perf | median | window | truth notes in window | clean | moderate | heavy |
|------|-------:|-------:|----------------------:|------:|---------:|------:|
| 01 | 5  | 2-8  | 45.7%  | 60.0 -> 46.7 | 54.3 -> 44.4 | 44.0 -> 54.0 |
| 02 | 5  | 2-8  | 68.2%  | 57.6 -> 74.3 | 34.1 -> 67.1 | 35.4 -> 64.6 |
| 03 | 6  | 3-9  | 93.6%  | 96.7 -> 100.0 | 76.6 -> 91.5 | 65.5 -> 89.7 |
| 04 | 8  | 5-11 | 89.4%  | 29.8 -> 53.7 | 32.1 -> 58.9 | 30.2 -> 57.1 |
| 05 | 5  | 2-8  | 55.9%  | 41.1 -> 55.8 | 35.6 -> 55.6 | 41.0 -> 61.5 |
| 06 | 0  | 0-3  | 83.8%  | 79.7 -> 79.1 | 78.4 -> 77.0 | 58.8 -> 58.8 |
| 07 | 5  | 2-8  | 70.3%  | 44.3 -> 67.2 | 41.0 -> 59.0 | 44.6 -> 62.5 |
| 08 | 8  | 5-11 | 63.1%  | 46.3 -> 82.9 | 55.6 -> 77.8 | 57.7 -> 84.6 |
| 09 | 12 | 9-15 | 52.8%  | 32.5 -> 50.0 | 37.3 -> 50.7 | 33.3 -> 54.2 |
| 10 | 0  | 0-3  | 100.0% | 95.0 -> 95.0 | 100.0 -> 100.0 | 100.0 -> 100.0 |
| 11 | 7  | 4-10 | 55.3%  | 33.8 -> 52.5 | 32.0 -> 49.3 | 26.8 -> 46.5 |
| 12 | 6  | 3-9  | 97.6%  | 21.5 -> 86.1 | 30.5 -> 84.7 | 36.8 -> 86.8 |

Reading:

- The window helps most where the player stays in one place and the
  mapper picks a lower position: 12 (21.5 -> 86.1), 04 and 08. Performance
  12 is the up-the-neck passage that the hand-position cost model pulled
  down to the nut.
- It hurts where the player moves around. Performance 01 has only 45.7% of
  its notes inside its median window, and clean drops 60.0 -> 46.7. A
  single window per song can't describe a part like that.
- Open-position songs (06, 10) don't change: Auto already plays them
  there.
- +-2 vs +-3: about the same overall. A narrower window helps chords and
  hurts fast single-note runs. The UI keeps +-3.

## In the app

The tab card has a "Neck position" select: Auto, Open position
(frets 0-4), and "Around fret N (frets a-b)" for N = 1-17. It applies to
both views. The sheet-music view reloads the export when the setting
changes. The setting is stored on the job and is still there after a
reload.

Browser test (Playwright, `data/_screenshots/neck_position/neck_test.py`;
screenshots and `compare_*.png` side by side in the same folder):

- the YouTube Short `-_XJDksMn98` (solo guitar, 18s, 88 notes, no
  isolation)
- the Fire Force cover `r7eLva7OBdI` (100s, 625 notes, Isolate guitar with
  Mega 53)

Each was checked at Auto, around fret 7 and around fret 8:

- The stored setting and the notes match Auto's.
- The tab grid shows the API's frets.
- alphaTab's TAB staff uses only the API's positions.
- The notation staff is unchanged.

Notes inside the window: Short 82/88 at fret 7, 78/88 at fret 8; Fire
Force 550/625 and 572/625. The notes left outside are mostly columns
that hold the same pitch twice (a re-trigger within 150ms). Only one copy
fits on the in-window string, so the other goes to the nearest string
outside.

## Known side effects (not from the mapper)

- **MusicXML export:** `rhythm.quantize_notes` drops a note that lands on
  the same 16th and the same string as an earlier one. Because string
  choice changes with the neck position, the export can gain or lose a
  note. Measured: 0-2 notes out of 87-641 on six jobs, against Auto's
  export. The tab JSON's notes are identical.
- **Tab grid:** TabViewer shows one note per string per column. Its
  grouping (anchored on the first shown note) can differ from the
  mapper's (anchored on the first note including dropped ones). On the
  Short, 2 of 88 notes share a cell with another note at every setting,
  Auto included. This predates this branch.
