# Riffscribe: setup and reference

Everything a reader of the [README](../README.md) might need in detail: running the stack, the guitar-isolation settings and their measured cost, configuration, the HTTP API and the measurement tools. The design spec is in [`spec.md`](spec.md).

## Running locally

Everything runs in Docker. You don't need Python, Node.js or any ML libraries on your machine.

### Prerequisites

- **[Docker Desktop](https://www.docker.com/products/docker-desktop/), installed and running.** On Linux, Docker Engine with the Compose plugin works too. Check that Docker is up with `docker compose version`, which should print a version and not an error.
- **About 12 GB of free disk space.** The built images total about 6 GB, and the build needs working space on top of that. Nearly all of it is the worker image (5.3 GB): Basic Pitch depends on TensorFlow, and guitar separation adds PyTorch and the Demucs model. The two worker services share the same image.
- **About 4 GB of memory available to Docker** if you use **Isolate guitar**. A separation peaked at about 2 GB in testing, on top of the other services.
- **Optional: an NVIDIA GPU**, for the High quality setting of **Isolate guitar**. It needs about 3 GB of free GPU memory and about 8 GB more disk space. Everything else works without one. See [Guitar isolation quality](#guitar-isolation-quality).
- **Ports 3000, 8000 and 6379 free.** 6379 is Redis, so stop any local Redis first.
- **Git**, to clone the repo.

Tested on Windows 11 with Docker Desktop (WSL 2).

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/JonathanmZhang/Riffscribe.git
   ```
2. Move into it:
   ```bash
   cd Riffscribe
   ```
3. Build and start all five services (frontend, API, the main worker, the separation worker, Redis):
   ```bash
   docker compose up -d --build
   ```
   The first build downloads and installs every dependency. **Expect roughly 5–15 minutes, longer on a slow connection.** Most of that is the worker image's TensorFlow and PyTorch installs, during which the output can sit on one step for several minutes. That's normal. Later starts reuse the built images and take seconds.
4. Check that all five containers are running:
   ```bash
   docker compose ps
   ```
   You should see `backend`, `frontend`, `redis`, `worker` and `worker-separation`, each with status `Up`. The API takes a second or two to start accepting requests after its container starts.
5. Open **http://localhost:3000** in your browser.

| Service  | Address |
|----------|---------|
| Frontend | http://localhost:3000 |
| API      | http://localhost:8000 (interactive docs at http://localhost:8000/docs) |
| Redis (Valkey) | localhost:6379 |

To stop everything, run `docker compose down`. Uploaded and normalized audio is kept in `./data/{job_id}/` inside the repo folder.

### What to expect on the first job

**The first job after the stack starts takes about 30–60 seconds, even for a few seconds of audio.** Later jobs on the same audio are much faster. This is a one-time warm-up, not a hang:

- librosa compiles its audio code the first time it's used.
- Basic Pitch loads its model into memory.

Both happen once per worker container. In testing, an 8-second clip took 47 s on a fresh worker and 8 s on the next run. The status badge shows `processing` during the wait. To watch the worker live, run `docker compose logs -f worker`.

### Try it

For a first run, use this 8-second public-domain guitar clip from Wikimedia Commons ([file page](https://commons.wikimedia.org/wiki/File:Guitar_tabulature_sample.ogg)):

1. On http://localhost:3000, choose **Paste URL**.
2. Paste the clip's direct link:
   ```
   https://upload.wikimedia.org/wikipedia/commons/0/08/Guitar_tabulature_sample.ogg
   ```
3. Click **Transcribe**. When the status turns `done`, a tab of about three dozen notes appears. Press play, and the notes being played highlight as the audio plays.

The link has to go through **Paste URL**, because file uploads only accept `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm` and `.mov`, and this clip is `.ogg`.

After that, try your own audio:

- **Upload file:** audio (`.mp3`, `.wav`, `.m4a`) or video (`.mp4`, `.webm`, `.mov`), up to 200 MB and 5 minutes long. For video, only the audio track is used.
- **Paste URL:** a YouTube or SoundCloud link, or any direct link to an audio file.

Clean, solo guitar gives the best results. A short instrumental clip without drums or vocals is a good choice. In a full band mix, other instruments show up as extra notes. See [Known limitations](../README.md#limitations).

### Guitar isolation quality

**Isolate guitar** separates the guitar from the rest of the mix before transcribing. When it's ticked, the upload form offers two qualities:

| | Standard | High quality |
|---|---|---|
| Separator | [Demucs](https://github.com/facebookresearch/demucs) `htdemucs_6s` | [MVSep Mega 53 Stems](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21) (BS-RoFormer), its `guitar` stem |
| Runs on | CPU, any computer | NVIDIA GPU only |
| Started with | `docker compose up -d --build` | the same command with `-f docker-compose.yml -f docker-compose.gpu.yml` |
| Time for a 102 s song (measured, see below) | about 2 to 2.5 minutes | about 3 to 3.5 minutes |
| Memory | about 2 GB of RAM | about 2.9 GB of GPU memory |

Both are limited to 120 seconds of audio (`MAX_SEPARATION_DURATION_SECONDS`).

**Turning High quality on.** It needs an NVIDIA GPU, a current NVIDIA driver, and Docker's GPU support. Docker Desktop on Windows with the WSL 2 backend has that built in; on Linux, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then start the stack with the GPU override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The override rebuilds only the `worker-separation` image, with PyTorch's CUDA build and the Mega 53 weights, and gives that service the GPU. This first build downloads about 4 GB more (the CUDA libraries, and a 1.37 GB checkpoint that is cut down to its two guitar stems during the build) and the image takes about 8 GB more disk space. The GPU settings are in a separate file because a GPU reservation stops the whole stack from starting on a computer without NVIDIA's Docker support.

**Without a GPU.** The separation worker checks for a GPU when it starts and reports it at `GET /capabilities`. If High quality can't run, the form shows it disabled, with the reason. Standard still works.

**If High quality fails during a job.** The job isn't failed. If Mega 53 runs out of GPU memory, is killed, or takes longer than `HQ_SEPARATION_TIMEOUT_SECONDS`, the guitar is separated with Demucs instead, and the job's status card and its `separation_note` say so.

**Measured accuracy.** These numbers come from the full-mix benchmark on the `experiment/full-mix-benchmark` branch (`experiments/full_mix/README.md`). It mixes the 12 real guitar performances of [EGSet12](https://zenodo.org/records/11406378) with a backing band and scores the tab against EGSet12's note annotations: recall / precision / F1 of the notes, in percent, with the right pitch within 50 ms.

| Mix | No separation | Standard (Demucs) | High quality (Mega 53) | The guitar alone, no band |
|---|---|---|---|---|
| Clean guitar, louder than the band | 62.5 / 43.7 / 51.4 | 57.8 / 70.1 / 63.4 | 69.8 / 82.2 / 75.5 | 71.9 / 85.2 / 78.0 |
| Clean guitar, quieter than the band | 55.1 / 31.6 / 40.2 | 49.3 / 62.5 / 55.2 | 67.1 / 80.4 / 73.1 | 71.9 / 85.2 / 78.0 |
| Distorted guitar, louder than the band | 51.2 / 40.5 / 45.2 | 48.7 / 67.0 / 56.4 | 59.5 / 79.7 / 68.1 | 55.8 / 80.7 / 66.0 |
| Distorted guitar, quieter than the band | 47.5 / 29.5 / 36.4 | 43.1 / 58.5 / 49.7 | 58.9 / 79.2 / 67.5 | 55.8 / 80.7 / 66.0 |

Read these with their limits in mind:

- **The band is synthesized.** It is General MIDI drums, bass, pad, electric piano and a "voice oohs" stand-in for a singer, written to play in time and in key with each guitar performance. It is not a real band, and real recordings are probably harder for both separators. There is no ground truth yet for a real full mix.
- The distorted tone is processed distortion applied to the clean recordings, not a real amplifier.
- Because the band plays the guitar's own chord tones, a band note that leaks into the stem can count as a correct guitar note. The band alone, with no guitar, matches 9% of the guitar's notes this way. That is why the High quality figures on the distorted tone are slightly above the guitar alone. The effect applies to every column except the last.
- The stem this pipeline produces was compared with the benchmark's stem for the same mix and is identical, sample for sample.

**Measured time and memory**, on a laptop with a 4 GB NVIDIA GeForce GTX 1650 Max-Q and an 8-thread Intel i7-10510U, in Docker Desktop on Windows:

| Job | Standard | High quality |
|---|---|---|
| A 102 s full-band song, whole job (two runs each) | 127 s, 154 s | 183 s, 199 s |
| of which separation | 108 s, 128 s | 164 s, 169 s |
| A 120 s song (the longest allowed), whole job (one run) | not measured | 222 s |
| of which separation | not measured | 194 s |
| Peak GPU memory on the 120 s song | none | 2,873 MiB of the card's 4,096 MiB |

All of these are on a warm stack. The first job after the stack starts is slower (323 s for the 102 s song at High quality). Demucs' speed varies a lot from run to run with other load on the computer. Mega 53 runs in its own process for each job, so the GPU memory is free again when the job's separation ends. Cards with less than 4 GB haven't been tried.

### Configuration

You don't need to change anything to run locally: `docker-compose.yml` already sets these to the local values. They matter when you deploy the services separately. The frontend container serves a production build of the app, not a dev server.

| Variable | Service | Default | Purpose |
|----------|---------|---------|---------|
| `NEXT_PUBLIC_API_URL` | frontend | `http://localhost:8000` | Base URL of the API |
| `NEXT_PUBLIC_ENABLE_URL_INGESTION` | frontend | `true` | Set to `false` to hide link submission in the UI (file upload only). The API still accepts URLs. |
| `CORS_ALLOWED_ORIGINS` | backend | `http://localhost:3000` | Comma-separated list of allowed frontend origins. Wildcards are rejected. |
| `MAX_AUDIO_DURATION_SECONDS` | worker | `300` | Longest audio accepted |
| `CHORD_TONE_CONFIDENCE_FLOOR` | worker | `0.45` | Keeps a lower-confidence note (at or above this value) when it starts together with a confident note, so the quieter tones of a strummed chord aren't dropped. Set it to `0.5` or higher to turn this off. |
| `BASIC_PITCH_ONSET_THRESHOLD` | worker | `0.5` | Basic Pitch's onset threshold. Lower values mostly add false notes. |
| `BASIC_PITCH_FRAME_THRESHOLD` | worker | `0.4` | Basic Pitch's frame threshold. The library default is 0.3; 0.4 was chosen on the real-guitar benchmark. |
| `BASIC_PITCH_MIN_NOTE_LENGTH_MS` | worker | `80` | The shortest note Basic Pitch reports. The library default is 128 ms, which is longer than a 16th note at ~120 bpm, so fast passages lost notes. |
| `DEMUCS_MODEL` | worker-separation | `htdemucs_6s` | Separation model. Must have a guitar stem. Only the default's weights are built into the image. |
| `DEMUCS_SHIFTS` | worker-separation | `1` | Demucs shift passes. Higher is slightly better and proportionally slower. |
| `MAX_SEPARATION_DURATION_SECONDS` | worker, worker-separation | `120` | Longest audio accepted with **Isolate guitar**. Longer audio fails during ingest, within seconds and before any processing, with a clear message. |
| `SEPARATION_SOFT_TIME_LIMIT_SECONDS` | worker-separation | `900` | Time limit for the separation task. The other tasks keep 120 s. |
| `SEPARATION_TORCH_THREADS` | worker-separation | half the CPUs | CPU threads Demucs may use, so separation doesn't compete with the main worker for every core. |
| `HQ_SEPARATION_HEAD` | worker-separation (`docker-compose.gpu.yml`) | `guitar` | Which Mega 53 stem is transcribed: `guitar` or `electric-guitar`. They scored the same on the benchmark. |
| `HQ_SEPARATION_TIMEOUT_SECONDS` | worker-separation (`docker-compose.gpu.yml`) | `420` | A Mega 53 run longer than this is stopped, and the job falls back to Demucs. |

The `NEXT_PUBLIC_*` values are built into the frontend at build time, so they're Docker build arguments rather than runtime environment variables. After changing one, rebuild with `docker compose up -d --build frontend`, or redeploy on a host such as Vercel.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit a job. Send either `multipart/form-data` with a `file` field (`.mp3`/`.wav`/`.m4a`/`.mp4`/`.webm`/`.mov`, max 200 MB; video files are reduced to their audio track) or JSON `{"url": "..."}`. Add `isolate_guitar` to separate the guitar first: a form field set to `true`, `1` or `on`, or `"isolate_guitar": true` in the JSON. It defaults to off. With it, `separation_quality` chooses the separator: `standard` (the default) or `high`. Returns `202` with `{"job_id": "...", "status": "queued"}`. Invalid input returns `422`. |
| `GET` | `/jobs/{job_id}` | Returns `{job_id, status, error, result, isolate_guitar, stage, stem_available, separation_quality, separator, separation_note}`. `status` is always one of `queued`, `processing`, `done` or `failed`. `stage` names the running step (`ingesting`, `separating`, `transcribing`, `mapping`). `stem_available` turns true once a guitar stem exists. `separation_quality` is what was asked for, `separator` (`demucs` or `mega53`) is what produced the stem, and `separation_note` explains a fallback to Demucs. `result` is filled in only when `done`, and `error` only when `failed`. Unknown IDs return `404`. |
| `POST` | `/jobs/{job_id}/rerun` | Starts a new job from the same source audio, by default with `{"isolate_guitar": true, "separation_quality": "standard"}`. Returns `202` like `POST /jobs`; `404` for an unknown job, `409` if its source audio is gone. |
| `GET` | `/capabilities` | Returns `{"separation": {high_quality_available, high_quality_unavailable_reason, gpu, gpu_memory_mib}}`: whether the separation worker found an NVIDIA GPU and the Mega 53 weights when it started. |
| `GET` | `/jobs/{job_id}/audio` | Streams the job's normalized WAV (`audio/wav`) and supports Range requests for seeking. Returns `404` if the job doesn't exist or its audio hasn't been produced yet. |
| `GET` | `/jobs/{job_id}/stem` | Streams the separated guitar stem (44.1 kHz stereo WAV), with Range support. Returns `404` unless the job ran with `isolate_guitar` and separation has finished. |
| `PATCH` | `/jobs/{job_id}` | Display overrides for a finished job, applied on every read without re-transcribing: `tempo_factor` (`0.5`, `1`, `2`), `bar_offset_beats` (`0`-`3`) and `neck_position` (`"auto"`, `"open"` or a fret `1`-`17`; the notes stay the same, only strings and frets change). `409` unless the job is done. |
| `GET` | `/jobs/{job_id}/musicxml` | The tab as MusicXML 4.0 (notation + TAB staves, 4/4, quantized, chord symbols), built per request with the job's overrides. `?tone=clean\|overdriven\|distorted\|acoustic` sets the General MIDI program; the notes are identical for every tone. `404` unless done. |

`GET /jobs/{job_id}` also returns the three overrides (`tempo_factor`, `bar_offset_beats`, `neck_position`). A finished `result` looks like this:

```json
{
  "job_id": "054e2a96-65ec-47cb-bf8d-5105a93355f2",
  "duration_seconds": 30.08,
  "tempo_bpm": 103.0,
  "notes": [
    { "string": 6, "fret": 0, "start_time": 0.01, "end_time": 3.11, "pitch": "E2", "vibrato": false, "column": 0 }
  ],
  "chords": [{ "start": 0.0, "end": 1.852, "name": "C" }],
  "beats": [0.0, 0.58],
  "downbeats": [0.0, 2.28],
  "bars": [0.0, 2.28]
}
```

Strings are numbered 1 (high e) to 6 (low E). Times are raw seconds. `column` is the tab column (the mapper's chord group) and `bars` the start of each MusicXML measure; both are derived on every read, not stored.

Example:

```bash
curl -F "file=@riff.wav" http://localhost:8000/jobs
curl -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=..."}' http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>
```

## Chord accuracy tools

Developer scripts for measuring where chord notes get lost. They run inside the worker container and don't touch the pipeline, Celery or Redis. Worker code is built into the image, so rebuild with `docker compose up -d --build worker` after changing them.

- `python -m scripts.make_chord_testset` renders a synthetic test set: 12 strummed standard-tuning voicings plus a fast progression at 4 chords per second, in clean, overdriven and distorted General MIDI guitar. Each program gets a WAV and a ground-truth JSON in `data/_testaudio/chord_testset/`.
- `python -m scripts.eval_chords [--separate]` runs the test set through each pipeline stage. It reports where every expected note was lost (not detected, below the confidence threshold, grouping, or mapper) along with extra notes, and saves the results as JSON.
- `python -m scripts.inspect_chords <file> <start_s> <end_s> [--separate] [--expect "G,B,D"]` shows every note event, the grouping and the fretboard mapping for one window of any recording.

The test set is rendered with [FluidSynth](https://www.fluidsynth.org/) and the **FluidR3_GM** General MIDI soundfont by Frank Wen, released under the **MIT license**. Both are installed from Debian packages (`fluidsynth`, `fluid-soundfont-gm`) when the worker image is built, and neither is committed to this repository.

### Real-guitar benchmark (EGSet12)

`python -m scripts.download_egset12` fetches the audio and annotations, then `python -m scripts.egset12_benchmark build` prepares the benchmark in `data/egset12/` (not committed). `regression_check` includes it once it's built. The benchmark:

- cuts the 12 performances into 36 segments, labelled chords, single-note or fast;
- names every chord in the annotations;
- renders two **processed-distortion** versions of each performance with [pedalboard](https://github.com/spotify/pedalboard)'s Distortion plugin, at moderate and heavy drive. These are a stand-in for driven tones, not real amplifier recordings.

It reports **pitch** recall and precision (the right note within 50 ms, on any string) and **position** agreement (the same string and fret, among correctly found notes), broken down by tone and segment type.

EGSet12 is by Hegel Pedroza, Wallace Abreu, Ryan Corey and Iran R. Roman, available at [zenodo.org/records/11406378](https://zenodo.org/records/11406378) under the **CC BY 4.0** license. It was introduced in their DAFx 2024 paper, *"Leveraging real electric guitar tones and effects to improve robustness in guitar tablature transcription modeling"*.
