# Riffscribe

**Audio in, guitar tab out.** Riffscribe turns a guitar recording (an uploaded file or a YouTube/SoundCloud link) into playable tablature. It detects the notes in the recording, works out where on the fretboard each one is most naturally played, and shows the result as a tab that highlights along with the audio as it plays.

Transcription is slow (ML inference plus a search over fretboard positions), so the whole pipeline runs asynchronously. The API accepts a job immediately, a Celery worker processes it in the background, and the frontend polls until the tab is ready.

## Architecture

```
 Next.js UI ──POST /jobs──▶ FastAPI ──enqueue──▶ Redis (broker + job state)
     ▲                         │                        │
     │                         │                        ▼
     └──GET /jobs/{id} (poll)──┘            Celery workers: chained tasks
        GET /jobs/{id}/audio                  1. ingest_audio
        GET /jobs/{id}/stem                  (2. separate_guitar  (Demucs or Mega 53), optional)
                                              3. transcribe     (Basic Pitch)
                                              4. map_fretboard  (DP search)
```

1. **FastAPI** (`backend/`) validates the upload or URL, creates a `job:{job_id}` record in Redis with status `queued`, enqueues the pipeline, and returns `202 Accepted` right away.
2. **Celery + Redis** (`worker/`) runs separate chained tasks. Tasks don't pass return values to each other: each one reads and writes shared job state in Redis by `job_id`. Each task sets `processing` and its pipeline `stage` when it starts. Any exception marks the job `failed` with an error message naming the stage, so a job never fails silently.
   - **`ingest_audio`** downloads URLs with yt-dlp, then normalizes every input to a mono, 22.05 kHz WAV with librosa.
   - **`separate_guitar`** runs only when the job was submitted with **Isolate guitar**. It separates the guitar from the original (full-rate, stereo) source, and the guitar stem is then transcribed instead of the full mix. There are two separators, chosen per job: **Standard** is Demucs' `htdemucs_6s` model on the CPU, and **High quality** is MVSep Mega 53 on an NVIDIA GPU. See [Guitar isolation quality](#guitar-isolation-quality). Separation needs a lot of memory, so it runs on its own `separation` queue, served one job at a time by a dedicated `worker-separation` service. It's also slow: Demucs took about 1–4 seconds per second of audio on an 8-core machine in testing.
   - **`transcribe`** runs Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) for polyphonic pitch detection. Notes below the confidence threshold (0.5 by default) are discarded. A note that vibrato split into several same-pitch pieces is joined back into one and flagged `vibrato`. It also estimates the tempo with librosa's beat tracker.
   - **`map_fretboard`** assigns every note a (string, fret) position. Notes whose onsets fall within 150 ms of each other are grouped into one chord. A Viterbi-style dynamic program then picks the lowest-cost path through all candidate positions for the whole piece, rather than choosing each note greedily. The cost of a move is fret-hand travel, plus a penalty for stretches wider than 4 frets and for staying on the same string, minus a bonus for open strings. When two paths tie, the one lower on the neck wins. Chords are voiced onto distinct strings.
3. **Next.js** (`frontend/`) submits the job, polls its status, and renders the result two ways: a string-by-fret tab grid and a per-note detail table. Both highlight the notes sounding at the current playback position of an `<audio>` element that streams the job's normalized audio. Seeking works because the audio endpoint supports HTTP Range requests. For practice, playback can be slowed to 0.75x or 0.5x, and the highlighting stays in sync because it follows the audio's own playback position.

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

Clean, solo guitar gives the best results. A short instrumental clip without drums or vocals is a good choice. In a full band mix, other instruments show up as extra notes. See [Known limitations](#known-limitations).

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

A finished `result` looks like this:

```json
{
  "job_id": "6ea38006-1774-4e5a-aba8-fc8ea2adbf81",
  "duration_seconds": 97.45,
  "tempo_bpm": 161,
  "notes": [
    { "string": 1, "fret": 5, "start_time": 1.78, "end_time": 2.52, "pitch": "A4", "vibrato": false }
  ]
}
```

Strings are numbered 1 (high e) to 6 (low E). Times are raw seconds.

Example:

```bash
curl -F "file=@riff.wav" http://localhost:8000/jobs
curl -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=..."}' http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>
```

## Known limitations

Riffscribe v1 has been tested against synthetic test tones, real guitar recordings from Wikimedia Commons, and full songs pulled from YouTube. The limitations below all came up in that testing.

- **Standard tuning only (E A D G B E).** The fretboard model assumes standard tuning. In a song in an alternate tuning such as drop D, the low notes (D2, C2 and so on) don't exist on a standard-tuned fretboard, so they are dropped. On one real drop-tuned song, 37 of 409 detected notes (about 9%) were dropped this way, mostly from the low riff. Detecting or selecting the tuning is not implemented.
- **Basic Pitch detection artifacts.** On real, messy audio the model occasionally reports phantom harmonic notes (overtones detected as separate notes) and flips sustained notes by an octave. These come from the model's output. Riffscribe maps what Basic Pitch detects and does not try to correct it.
- **Chord grouping uses a fixed 150 ms onset window.** Notes starting within 150 ms of each other are treated as one chord. The value was tuned empirically against real strums. Very fast runs with sub-150 ms note spacing can be merged into a single chord.
- **Unplayable notes are dropped, not fatal.** A standalone note outside frets 0–20 is dropped. For a chord that can't be placed on distinct strings, the smallest set of notes that resolves the conflict is dropped. Either way the job still completes and a warning naming the pitch, time and amplitude is logged. The dropped notes are not surfaced in the API response.
- **Tempo is an estimate.** `tempo_bpm` comes from librosa's beat tracker and is rounded to a whole BPM. It was within 1–4 BPM on plucked-guitar test clips with known tempos of 100 and 140. The tracker relies on percussive onsets, so it is less reliable on solo recordings without drums or a clear pulse, and it can lock onto half or double the tempo you'd tap along to. The UI labels it as approximate. The tab itself is laid out in seconds, not beats or measures.
- **Audio is capped at 5 minutes.** Recordings longer than `MAX_AUDIO_DURATION_SECONDS` (default 300, set in `docker-compose.yml`) are rejected before any decoding or inference. Because duration isn't known until the audio has been uploaded or downloaded, the check runs in the worker rather than at submission. `POST /jobs` still returns `202`, and the job then moves to `failed` with an error such as `audio is 412 seconds long, which exceeds the 300 second limit`. File uploads are also capped at 200 MB.
- **The tab grid doesn't auto-scroll** to follow playback. On long songs, the highlighted column can be off-screen until you scroll to it.
- **Link submission may be turned off on hosted deployments.** A public deployment can hide the "Paste URL" option by setting `NEXT_PUBLIC_ENABLE_URL_INGESTION=false`. The page then says link transcription is available when running locally. Everything works when you run it yourself with `docker compose`.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Pydantic
- **Task queue:** Celery, with Valkey 8 (a Redis-compatible fork) as broker and job store
- **Audio / ML:** Spotify Basic Pitch, librosa, pretty_midi, yt-dlp
- **Fretboard mapping:** custom dynamic-programming (Viterbi-style) algorithm in pure Python
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Infrastructure:** Docker Compose

The full design spec is in [`docs/spec.md`](docs/spec.md).

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

## License

Riffscribe's own code is released under the [MIT license](LICENSE). Third-party components keep their own licenses. None of them is vendored into this repository: they're installed from PyPI, npm, Debian or Docker Hub when the images are built, or downloaded at build time.

**Pipeline (used on every job)**

| Component | Used for | License |
|---|---|---|
| [Basic Pitch](https://github.com/spotify/basic-pitch) (code and model) | note detection | Apache-2.0 |
| TensorFlow | runs Basic Pitch | Apache-2.0 |
| librosa, resampy | audio loading and resampling | ISC |
| soundfile | WAV I/O | BSD-3-Clause |
| pretty_midi | MIDI note names | MIT |
| NumPy, scikit-learn | numerics (librosa/Basic Pitch dependencies) | BSD-3-Clause |
| [Demucs](https://github.com/facebookresearch/demucs) 4.0.1 (code) | "Isolate guitar", Standard quality | MIT |
| Demucs `htdemucs_6s` weights | same | **Not MIT.** See [Separator weights](#separator-weights) |
| [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (code, pinned commit; only with `docker-compose.gpu.yml`) | "Isolate guitar", High quality | MIT |
| MVSep Mega 53 Stems weights (only with `docker-compose.gpu.yml`) | same | **No license stated.** See [Separator weights](#separator-weights) |
| PyTorch, torchaudio | runs Demucs, Mega 53, BTC and the beat tracker | BSD-3-Clause |
| NVIDIA CUDA libraries (bundled in PyTorch's CUDA wheels; only with `docker-compose.gpu.yml`) | GPU inference for Mega 53 | NVIDIA's proprietary license terms |
| yt-dlp | link ingestion | Unlicense |
| FFmpeg (Debian package) | decoding video/compressed audio, run as a separate program | GPL (Debian builds with `--enable-gpl`) |
| Celery | task queue | BSD-3-Clause |
| [BTC](https://github.com/jayg996/BTC-ISMIR19) (code and large-vocabulary model) | chord names | MIT |
| redis-py | Redis client | MIT |
| [Valkey](https://valkey.io/) server (`valkey/valkey:8-alpine` image; a Redis-compatible fork, used in place of Redis 7.4+, which is RSALv2/SSPLv1) | queue broker and job state, run unmodified as a separate service | BSD-3-Clause |
| FastAPI, Pydantic | API | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| python-multipart | uploads | Apache-2.0 |
| Next.js, React | frontend | MIT |
| Tailwind CSS, TypeScript | frontend build | MIT / Apache-2.0 |

### Separator weights

The trained weights of both guitar separators are **not covered by this repository's MIT license, and this repository does not redistribute them**. Neither file is committed here. Each is downloaded from its publisher when you build the worker image on your own machine, and what you may do with it is governed by its publisher's terms, not by this repository's license.

| Weights | Downloaded from | Stated terms |
|---|---|---|
| Demucs `htdemucs_6s` (Standard) | Meta's servers, by the `demucs` package, during the image build | The Demucs code is MIT, but its maintainer has said the weights are not: "The model weights are not covered by the MIT license, and are provided only for scientific purposes" ([demucs#327](https://github.com/facebookresearch/demucs/issues/327)). |
| MVSep Mega 53 Stems (High quality) | The author's [v1.0.21 release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21) of Music-Source-Separation-Training, at a pinned URL with a pinned SHA-256 (`worker/Dockerfile`) | The repository's code is MIT. The weights are a release asset with no license statement of their own, and their training data isn't stated. Whether the repository's MIT license covers them hasn't been confirmed by the author. |

This is a reading of the published terms, not legal advice. If you deploy Riffscribe publicly or use it commercially, check both before enabling **Isolate guitar**.

**Evaluation tooling only (not used by the pipeline)**

| Component | Used for | License |
|---|---|---|
| FluidSynth (Debian package) | rendering the synthetic chord test set | LGPL-2.1 |
| FluidR3_GM soundfont | same | MIT |
| [pedalboard](https://github.com/spotify/pedalboard) | EGSet12's processed-distortion tones (installed in the worker image) | GPL-3.0 |
| [EGSet12](https://zenodo.org/records/11406378) | real-guitar benchmark audio and annotations | CC BY 4.0 |

The experiment branches (`experiment/*`) tried further third-party models. Each branch's `experiments/*/README.md` lists their licenses.
