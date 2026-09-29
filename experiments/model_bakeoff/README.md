# Transcription model bake-off (feasibility, 2026-09-29)

Question: does a stronger transcription model beat Basic Pitch on the EGSet12
real-guitar benchmark? This is a **feasibility pass only**. It covers three
clean-tone segments (200 annotated notes), and nothing here is wired into the
pipeline. Nothing under `worker/` changed.

## Segments (clean tone, from `data/egset12/benchmark.json`)

| type        | segment | notes | notes                                  |
|-------------|---------|-------|----------------------------------------|
| chords      | 02-2    | 106   | chord-note share 0.96                  |
| single-note | 05-2    | 42    | no chords                              |
| fast        | 04-2    | 52    | 9 single-note gaps at 16ths of 110bpm+ |

## Scoring

Pitch recall and precision use `scripts/egset12_benchmark._match`: same MIDI
pitch, onset within 50ms, one-to-one. Speed is steady-state seconds of
processing per second of audio. Each model runs once as a warm-up, and the
second run is timed. That leaves out one-time costs such as numba JIT and JAX
compilation. Model load is reported separately. Peak RAM is the whole
process's `ru_maxrss` (one process per segment), including framework imports.
Everything ran on CPU in Docker Desktop (WSL2, 7.7 GiB RAM visible), with all
cores available to each framework.

For multi-instrument models, **guitar** means GM programs 24-31 only, as
specified. **all-inst** keeps every pitched note. **dedup** is all-inst with
same-pitch notes within 50ms of each other merged across programs.

## Results

| candidate | license (code / weights) | runs on CPU? | s per s audio | peak RAM | recall | precision |
|---|---|---|---|---|---|---|
| Basic Pitch, raw events (baseline) | Apache-2.0 / Apache-2.0 | yes (worker) | 0.17 | 1.15 GB | 71.5% | 92.3% |
| Basic Pitch, pipeline tab (baseline) | ″ | ″ | 0.17 | ″ | 62.0% | 95.4% |
| YourMT3+ (YPTF.MoE+Multi noPS), guitar | GPL-3.0 on GitHub, Apache-2.0 on the HF Space that holds the code and weights | yes, as a layer on the worker image (+protobuf pin) | 2.93 | 2.20 GB | 47.0% | 100% |
| YourMT3+, all-inst | ″ | ″ | ″ | ″ | 51.5% | 95.4% |
| MT3 (multi-instrument ckpt), guitar | Apache-2.0 / no separate license (public GCS bucket, official repo) | yes, **separate container only** | 1.4-1.7 | 2.03 GB | 26.5% | 100% |
| MT3, all-inst | ″ | ″ | ″ | ″ | 88.5% | 75.3% |
| MT3, dedup | ″ | ″ | ″ | ″ | **88.5%** | **87.2%** |
| TabCNN + GuitarProFX (DAFx-24) | MIT (amt-tools) / CC BY 4.0 (EGSet12 Zenodo record) | yes, as a layer on the worker image (+protobuf pin) | 0.11 | 0.97 GB | 59.5% | 53.8% |

Per segment (recall / precision):

| model | chords 02-2 | single-note 05-2 | fast 04-2 |
|---|---|---|---|
| Basic Pitch raw | 63.2 / 90.5 | 97.6 / 91.1 | 67.3 / 97.2 |
| Basic Pitch tab | 48.1 / 94.4 | 97.6 / 95.3 | 61.5 / 97.0 |
| YourMT3+ guitar | 35.8 / 100 | 88.1 / 100 | 36.5 / 100 |
| MT3 guitar | 36.8 / 100 | 14.3 / 100 | 15.4 / 100 |
| MT3 all-inst | 91.5 / 73.5 | 90.5 / 88.4 | 80.8 / 70.0 |
| TabCNN-GPFX | 33.0 / 36.5 | 88.1 / 62.7 | 90.4 / 71.2 |

## Findings

- **MT3** does not recognize this Telecaster as a guitar. It labels 183 of
  its 236 notes (78%) as electric piano (GM 4) or other programs, so its guitar-only
  output is useless. Taken as a pitch detector, though, it has the best recall
  of any candidate, including chords at 91.5% against 48-63% for Basic Pitch.
  Of its false positives, 32 of 58 are the same note emitted under two
  programs, and a simple dedup removes them. Most of the rest are octave
  errors (15). It is 8-10x slower than Basic Pitch; at 1.4-1.7 s/s (two runs; 0.7-2.7 per segment), a 120s
  song takes about 3 minutes. Its stack (Python 3.12, JAX 0.11, TF 2.21,
  numpy 2.5, protobuf 7) can't share the worker image (Python 3.11, TF 2.15,
  numpy 1.26). The official install is unpinned git HEAD and was already
  missing `tensorboard`, so this setup is fragile. `mt3-requirements.lock.txt`
  pins what worked; a rebuild from it reproduced the MT3 notes exactly.
- **YourMT3+** is worse than Basic Pitch on every segment, and about 17x
  slower. It is very precise but misses chord tones and fast notes.
- **TabCNN-GPFX** is fast, and it has the best fast-passage recall (90%). It
  also outputs strings directly. Its precision is poor, and 60 of its 102
  false positives are re-onsets inside real sustained notes (frame-to-note
  decoding). Chord recall is only 33%. Caveat: the authors introduced EGSet12
  as this model's test set and published it as their "best" model, so its
  EGSet12 numbers may be optimistic.

## Dependency notes

- Adding YourMT3+ or amt-tools to the worker image unpinned would upgrade
  **protobuf to 7** (through wandb, and through mirdata → smart_open →
  google-cloud-storage). That breaks TensorFlow 2.15. Pinning `protobuf<5`
  fixes both, and pip then picks wandb 0.28.0. With the pin, torch 2.5.1,
  numpy 1.26.4 and TF 2.15 stay unchanged, and `pip check` is clean for
  YourMT3+.
- amt-tools is installed `--no-deps`. Its `pynput` dependency needs `evdev`
  built against kernel headers, and it's only used for the live-input demo.
  `sounddevice` plus `libportaudio2` are still needed because
  `amt_tools.features` imports them.
- Image sizes: worker 5.64 GB, YourMT3+ 7.2 GB, TabCNN 6.0 GB, MT3 7.0 GB
  (a separate image).

## Reproduce

```sh
# from the repo root, Git Bash: prefix with MSYS_NO_PATHCONV=1
cd experiments/model_bakeoff
docker build -f yourmt3.Dockerfile -t bakeoff-yourmt3 .
docker build -f tabcnn.Dockerfile  -t bakeoff-tabcnn .
docker build -f mt3.Dockerfile     -t bakeoff-mt3 .
cd ../..
RUN="docker run --rm -v $(pwd -W)/data:/app/data -v $(pwd -W)/experiments/model_bakeoff:/bakeoff --entrypoint python"
$RUN stratotab-worker /bakeoff/run_basic_pitch.py cut
for s in 02-2 05-2 04-2; do
  $RUN stratotab-worker /bakeoff/run_basic_pitch.py $s
  $RUN bakeoff-yourmt3  /bakeoff/run_yourmt3.py $s
  $RUN bakeoff-tabcnn   /bakeoff/run_tabcnn.py $s
  $RUN bakeoff-mt3      /bakeoff/run_mt3.py $s
done
$RUN stratotab-worker /bakeoff/score.py
```

Outputs go to `data/_bakeoff/` (gitignored).

## Full MT3 benchmark (2026-09-29)

Scripts: `run_mt3_files.py` runs MT3 (bakeoff-mt3 image) and keeps every
note with its program. `prep_firefire.py` (worker image) produces the
firefire mix and Demucs stem and times Basic Pitch and separation.
`bench_mt3.py` (worker image) does the scoring and writes
`data/_bakeoff/bench_mt3.json`.

MT3 mode: all instruments, drums excluded, same-pitch notes within 50ms
merged across programs. MT3's **tab** goes through the pipeline's own
`run_stages` (select_notes, grouping, fretboard mapper) with amplitude 1.0,
because MT3 gives no per-note confidence. Basic Pitch's tab numbers are
checked in the script to reproduce `egset12_benchmark.evaluate` exactly.

### EGSet12, all 36 segments (1567 notes per tone): recall / precision [position agreement]

| tone | type | BP raw | MT3 raw | BP tab (today) | MT3 tab |
|---|---|---|---|---|---|
| clean | all | 76.6 / 79.2 | 75.7 / 74.2 | **71.9 / 85.2** [57.0] | 74.5 / 77.4 [48.5] |
| clean | chords (950) | 67.6 / 77.2 | 70.8 / 72.4 | 61.9 / 83.5 [56.1] | 69.0 / 76.4 [49.7] |
| clean | single-note (534) | 92.5 / 80.1 | 83.2 / 76.7 | 89.5 / 86.1 [58.0] | 83.0 / 79.0 [49.7] |
| clean | fast (83) | 77.1 / 97.0 | 83.1 / 77.5 | 73.5 / 96.8 [57.4] | 81.9 / 77.3 [29.4] |
| moderate | all | 66.9 / 72.9 | 71.8 / 60.1 | 55.8 / 80.7 [54.5] | 68.2 / 64.5 [52.6] |
| moderate | chords | 54.1 / 69.0 | 64.6 / 58.8 | 39.3 / 78.4 [54.2] | 60.3 / 61.6 [54.1] |
| moderate | single-note | 88.4 / 74.9 | 84.6 / 69.0 | 82.8 / 80.8 [54.3] | 82.8 / 70.3 [53.8] |
| moderate | fast | 75.9 / 96.9 | 71.1 / 34.1 | 71.1 / 98.3 [57.6] | 65.1 / 55.1 [25.9] |
| heavy | all | 51.9 / 61.4 | 62.9 / 48.2 | 38.5 / 71.7 [54.1] | **60.6 / 55.1** [44.9] |
| heavy | chords | 34.5 / 51.9 | 52.0 / 43.6 | 17.1 / 60.5 [53.7] | 48.3 / 51.8 [43.6] |
| heavy | single-note | 80.5 / 67.7 | 78.8 / 52.2 | 74.7 / 75.4 [54.4] | 78.8 / 57.3 [49.4] |
| heavy | fast | 67.5 / 94.9 | 85.5 / 67.0 | 51.8 / 95.6 [53.5] | 83.1 / 68.3 [26.1] |

Tab F1: clean BP 78.0 vs MT3 75.9, moderate 66.0 vs 66.3, heavy 50.1 vs 57.7.
The moderate and heavy tones are processed distortion, not real amps. The
three feasibility segments overstated MT3: clean chords are 70.8% raw recall
here, against 91.5% on 02-2 alone. MT3's position agreement is 2-9 points
lower, and on fast passages it's about half of Basic Pitch's. That wasn't
investigated further.

### Octave errors

MT3's false notes that are exactly one octave off a true note, onset within
50ms: clean 181/412 (44%), moderate 339/748 (45%), heavy 434/1059 (41%).
Counting two-octave errors too raises these to about 50-54%. So one-octave
errors are the largest single category, but not a majority. Most are "ghosts",
where the true note was also found: 56% clean, 73% moderate, 82% heavy. The
rest are substitutions, and 89% of the clean ones are upward. Candidate fixes, clean raw recall / precision:

| fix | clean raw | clean tab | heavy tab |
|---|---|---|---|
| none | 75.7 / 74.2 | 74.5 / 77.4 | 60.6 / 55.1 |
| drop a note with the same note an octave below within 50ms | 69.1 / 78.6 | 68.9 / 81.3 | 49.7 / 66.5 |
| raise notes below E2 an octave | 75.7 / 74.6 | 74.5 / 77.4 | 60.8 / 54.7 |

Dropping upper octaves costs more recall than it gains in precision, because
9.8% of true notes (153/1567) are real octave doublings. Substitutions can't
be fixed by dropping notes. No simple fix is worth it.

### Firefire (full mix, no ground truth), 3 windows

MT3's emitted notes by GM family, `drums` flagged by MT3 itself:

| window | full mix | Demucs stem |
|---|---|---|
| 28-31 | guitar 77, drums 21, piano 2, synth lead 2 | guitar 54, piano 3, strings 1 |
| 39.5-42.5 | drums 37, piano 13, strings 10, guitar 10, synth lead 10 | organ 44, guitar 34, strings 18 |
| 60-63 | piano 47, drums 36, guitar 12, choir 11, synth lead 8, bass 1 | piano 63, organ 30, guitar 10, strings 5 |

- **Drums:** 94 notes across the windows on the full mix. All are removed,
  because MT3 flags them as drums.
- **Bass:** 1 note. MT3 barely reports bass on this mix.
- **Vocals:** MT3 has no voice class, and its labels are unreliable (the
  stem, which should be guitar, comes out as mostly piano and organ).
  Vocals can't be counted directly. The families that appear only in the
  full mix, synth lead and choir, are the likely vocal and other-instrument
  notes. There are 31 of them, and 22 reach the tab: 15% of the full-mix
  tab's 144 notes.

Tab through the pipeline mapper, MT3 vs Basic Pitch (full mix / stem): tab
notes 55/49, 34/50, 55/68 vs 16/20, 12/28, 22/18. MT3 produces 2-4x the
notes and 4.8-5.3-note chord columns, against Basic Pitch's ~3. But on the
stem the mapper drops 32 MT3 notes per window in 2 of 3 windows (0 for Basic
Pitch), which means unplayable note sets. Without ground truth, it's unknown
how much of the extra density is real.

### Processing time per song (measured on firefire, 102.5s, steady state)

| route | seconds | per minute of audio |
|---|---|---|
| Basic Pitch, full mix | 5.2 | 3s |
| Basic Pitch + Isolate guitar (Demucs 88.5s warm, 107s cold) | ~94 | ~55s |
| MT3, full mix | 262 | 2.6 min |
| MT3 + Isolate guitar | 88.5 + 356 = ~445 | ~4.3 min |

MT3's speed depends on note density: 0.7-3.8 s/s per EGSet12 file, with a
mean of 1.8-2.0 s/s per tone. At the stem's 3.5 s/s, MT3 alone would hit the
worker's 900s soft limit at about 4.3 minutes of audio. Peak RAM is 2.2-2.5
GB, against about 1.15 GB for Basic Pitch.
