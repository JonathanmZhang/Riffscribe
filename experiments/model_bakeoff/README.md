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
