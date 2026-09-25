# Riffscribe

**Audio in, guitar tab out.** Riffscribe turns a guitar recording (an uploaded file or a YouTube/SoundCloud link) into playable tablature. It detects the notes in the recording, works out where on the fretboard each one is most naturally played, and shows the result as a tab that highlights along with the audio as it plays.

Transcription is slow (ML inference plus a search over fretboard positions), so the whole pipeline runs asynchronously. The API accepts a job immediately, a Celery worker processes it in the background, and the frontend polls until the tab is ready.

## Architecture

```
 Next.js UI ──POST /jobs──▶ FastAPI ──enqueue──▶ Redis (broker + job state)
     ▲                         │                        │
     │                         │                        ▼
     └──GET /jobs/{id} (poll)──┘            Celery worker: 3 chained tasks
        GET /jobs/{id}/audio                  1. ingest_audio
                                              2. transcribe     (Basic Pitch)
                                              3. map_fretboard  (DP search)
```

1. **FastAPI** (`backend/`) validates the upload or URL, creates a `job:{job_id}` record in Redis with status `queued`, enqueues the pipeline, and returns `202 Accepted` right away.
2. **Celery + Redis** (`worker/`) runs three separate chained tasks. Tasks don't pass return values to each other: each one reads and writes shared job state in Redis by `job_id`. Each task sets `processing` when it starts. Any exception marks the job `failed` with an error message naming the stage, so a job never fails silently.
   - **`ingest_audio`** downloads URLs with yt-dlp, then normalizes every input to a mono, 22.05 kHz WAV with librosa.
   - **`transcribe`** runs Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) for polyphonic pitch detection. Notes below the confidence threshold (0.5 by default) are discarded. It also estimates the tempo with librosa's beat tracker.
   - **`map_fretboard`** assigns every note a (string, fret) position. Notes whose onsets fall within 150 ms of each other are grouped into one chord. A Viterbi-style dynamic program then picks the lowest-cost path through all candidate positions for the whole piece, rather than choosing each note greedily. The cost of a move is fret-hand travel, plus a penalty for stretches wider than 4 frets and for staying on the same string, minus a bonus for open strings. When two paths tie, the one lower on the neck wins. Chords are voiced onto distinct strings.
3. **Next.js** (`frontend/`) submits the job, polls its status, and renders the result two ways: a string-by-fret tab grid and a per-note detail table. Both highlight the notes sounding at the current playback position of an `<audio>` element that streams the job's normalized audio. Seeking works because the audio endpoint supports HTTP Range requests. For practice, playback can be slowed to 0.75x or 0.5x, and the highlighting stays in sync because it follows the audio's own playback position.

## Running locally

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine with the Compose plugin). Nothing else needs to be installed on the host.

```bash
git clone https://github.com/JonathanmZhang/Riffscribe.git
cd Riffscribe
docker compose up -d --build
```

| Service  | URL / port              |
|----------|-------------------------|
| Frontend | http://localhost:3000   |
| API      | http://localhost:8000 (interactive docs at `/docs`) |
| Redis    | localhost:6379          |

Open http://localhost:3000, then upload a `.mp3`, `.wav` or `.m4a` file or paste a link.

The first job after a fresh start takes noticeably longer. That's the one-time cost of loading the Basic Pitch model (~20 s) and warming up librosa's numba JIT compiler (~30 s), not a hang. Later jobs skip it.

Uploaded and normalized audio is stored in `./data/{job_id}/` on the host.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit a job. Send either `multipart/form-data` with a `file` field (`.mp3`/`.wav`/`.m4a`, max 15 MB) or JSON `{"url": "..."}`. Returns `202` with `{"job_id": "...", "status": "queued"}`. Invalid input returns `422`. |
| `GET` | `/jobs/{job_id}` | Returns `{job_id, status, error, result}`. `status` is always one of `queued`, `processing`, `done` or `failed`. `result` is filled in only when `done`, and `error` only when `failed`. Unknown IDs return `404`. |
| `GET` | `/jobs/{job_id}/audio` | Streams the job's normalized WAV (`audio/wav`) and supports Range requests for seeking. Returns `404` if the job doesn't exist or its audio hasn't been produced yet. |

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
- **Audio is capped at 5 minutes.** Recordings longer than `MAX_AUDIO_DURATION_SECONDS` (default 300, set in `docker-compose.yml`) are rejected before any decoding or inference. Because duration isn't known until the audio has been uploaded or downloaded, the check runs in the worker rather than at submission. `POST /jobs` still returns `202`, and the job then moves to `failed` with an error such as `audio is 412 seconds long, which exceeds the 300 second limit`. File uploads are also capped at 15 MB.
- **The tab grid doesn't auto-scroll** to follow playback. On long songs, the highlighted column can be off-screen until you scroll to it.
- **Local development setup.** The frontend's API base URL is hardcoded to `http://localhost:8000`, and CORS allows only `http://localhost:3000`.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Pydantic
- **Task queue:** Celery, Redis 7
- **Audio / ML:** Spotify Basic Pitch, librosa, pretty_midi, yt-dlp
- **Fretboard mapping:** custom dynamic-programming (Viterbi-style) algorithm in pure Python
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Infrastructure:** Docker Compose

The full design spec is in [`docs/spec.md`](docs/spec.md).
