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
        GET /jobs/{id}/stem                  (2. separate_guitar  (Demucs), optional)
                                              3. transcribe     (Basic Pitch)
                                              4. map_fretboard  (DP search)
```

1. **FastAPI** (`backend/`) validates the upload or URL, creates a `job:{job_id}` record in Redis with status `queued`, enqueues the pipeline, and returns `202 Accepted` right away.
2. **Celery + Redis** (`worker/`) runs separate chained tasks. Tasks don't pass return values to each other: each one reads and writes shared job state in Redis by `job_id`. Each task sets `processing` and its pipeline `stage` when it starts. Any exception marks the job `failed` with an error message naming the stage, so a job never fails silently.
   - **`ingest_audio`** downloads URLs with yt-dlp, then normalizes every input to a mono, 22.05 kHz WAV with librosa.
   - **`separate_guitar`** runs only when the job was submitted with **Isolate guitar**. It separates the guitar from the original (full-rate, stereo) source with Demucs' `htdemucs_6s` model on the CPU, and the guitar stem is then transcribed instead of the full mix. Separation needs a lot of memory, so it runs on its own `separation` queue, served one job at a time by a dedicated `worker-separation` service. It's also slow: in testing it took about 2–4 seconds per second of audio on an 8-core machine.
   - **`transcribe`** runs Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) for polyphonic pitch detection. Notes below the confidence threshold (0.5 by default) are discarded. It also estimates the tempo with librosa's beat tracker.
   - **`map_fretboard`** assigns every note a (string, fret) position. Notes whose onsets fall within 150 ms of each other are grouped into one chord. A Viterbi-style dynamic program then picks the lowest-cost path through all candidate positions for the whole piece, rather than choosing each note greedily. The cost of a move is fret-hand travel, plus a penalty for stretches wider than 4 frets and for staying on the same string, minus a bonus for open strings. When two paths tie, the one lower on the neck wins. Chords are voiced onto distinct strings.
3. **Next.js** (`frontend/`) submits the job, polls its status, and renders the result two ways: a string-by-fret tab grid and a per-note detail table. Both highlight the notes sounding at the current playback position of an `<audio>` element that streams the job's normalized audio. Seeking works because the audio endpoint supports HTTP Range requests. For practice, playback can be slowed to 0.75x or 0.5x, and the highlighting stays in sync because it follows the audio's own playback position.

## Running locally

Everything runs in Docker. You don't need Python, Node.js or any ML libraries on your machine.

### Prerequisites

- **[Docker Desktop](https://www.docker.com/products/docker-desktop/), installed and running.** On Linux, Docker Engine with the Compose plugin works too. Check that Docker is up with `docker compose version`, which should print a version and not an error.
- **About 12 GB of free disk space.** The built images total about 6 GB, and the build needs working space on top of that. Nearly all of it is the worker image (5.3 GB): Basic Pitch depends on TensorFlow, and guitar separation adds PyTorch and the Demucs model. The two worker services share the same image.
- **About 4 GB of memory available to Docker** if you use **Isolate guitar**. A separation peaked at about 2 GB in testing, on top of the other services.
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
| Redis    | localhost:6379 |

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

### Configuration

You don't need to change anything to run locally: `docker-compose.yml` already sets these to the local values. They matter when you deploy the services separately. The frontend container serves a production build of the app, not a dev server.

| Variable | Service | Default | Purpose |
|----------|---------|---------|---------|
| `NEXT_PUBLIC_API_URL` | frontend | `http://localhost:8000` | Base URL of the API |
| `NEXT_PUBLIC_ENABLE_URL_INGESTION` | frontend | `true` | Set to `false` to hide link submission in the UI (file upload only). The API still accepts URLs. |
| `CORS_ALLOWED_ORIGINS` | backend | `http://localhost:3000` | Comma-separated list of allowed frontend origins. Wildcards are rejected. |
| `MAX_AUDIO_DURATION_SECONDS` | worker | `300` | Longest audio accepted |
| `DEMUCS_MODEL` | worker-separation | `htdemucs_6s` | Separation model. Must have a guitar stem. Only the default's weights are built into the image. |
| `DEMUCS_SHIFTS` | worker-separation | `1` | Demucs shift passes. Higher is slightly better and proportionally slower. |
| `MAX_SEPARATION_DURATION_SECONDS` | worker, worker-separation | `120` | Longest audio accepted with **Isolate guitar**. Longer audio fails during ingest, within seconds and before any processing, with a clear message. |
| `SEPARATION_SOFT_TIME_LIMIT_SECONDS` | worker-separation | `900` | Time limit for the separation task. The other tasks keep 120 s. |
| `SEPARATION_TORCH_THREADS` | worker-separation | half the CPUs | CPU threads Demucs may use, so separation doesn't compete with the main worker for every core. |

The `NEXT_PUBLIC_*` values are built into the frontend at build time, so they're Docker build arguments rather than runtime environment variables. After changing one, rebuild with `docker compose up -d --build frontend`, or redeploy on a host such as Vercel.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit a job. Send either `multipart/form-data` with a `file` field (`.mp3`/`.wav`/`.m4a`/`.mp4`/`.webm`/`.mov`, max 200 MB; video files are reduced to their audio track) or JSON `{"url": "..."}`. Add `isolate_guitar` to separate the guitar first: a form field set to `true`, `1` or `on`, or `"isolate_guitar": true` in the JSON. It defaults to off. Returns `202` with `{"job_id": "...", "status": "queued"}`. Invalid input returns `422`. |
| `GET` | `/jobs/{job_id}` | Returns `{job_id, status, error, result, isolate_guitar, stage, stem_available}`. `status` is always one of `queued`, `processing`, `done` or `failed`. `stage` names the running step (`ingesting`, `separating`, `transcribing`, `mapping`). `stem_available` turns true once a guitar stem exists. `result` is filled in only when `done`, and `error` only when `failed`. Unknown IDs return `404`. |
| `GET` | `/jobs/{job_id}/audio` | Streams the job's normalized WAV (`audio/wav`) and supports Range requests for seeking. Returns `404` if the job doesn't exist or its audio hasn't been produced yet. |
| `GET` | `/jobs/{job_id}/stem` | Streams the separated guitar stem (44.1 kHz stereo WAV), with Range support. Returns `404` unless the job ran with `isolate_guitar` and separation has finished. |

A finished `result` looks like this:

```json
{
  "job_id": "6ea38006-1774-4e5a-aba8-fc8ea2adbf81",
  "duration_seconds": 97.45,
  "tempo_bpm": 161,
  "notes": [
    { "string": 1, "fret": 5, "start_time": 1.78, "end_time": 2.52, "pitch": "A4" }
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
- **Task queue:** Celery, Redis 7
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
