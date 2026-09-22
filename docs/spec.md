# StratoTab: Asynchronous Audio-to-Tablature Transcription Platform

## 1. Project Overview & Core Value Proposition

StratoTab converts raw instrumental guitar audio — an uploaded file or a link
(YouTube, SoundCloud, etc.) — into interactive, playable guitar tablature.
A user submits audio, the system asynchronously performs polyphonic pitch
detection and onset/offset estimation, maps the detected notes onto a
physical fretboard using a cost-minimizing playability algorithm, and
returns a scrollable, audio-synced tab.

## 2. Architecture & Data Flow

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React / Next.js | Upload UI, job status polling, interactive tab renderer, audio-sync playback |
| API Layer | FastAPI | Job submission, job status/result retrieval |
| Task Queue | Celery + Redis | Decouples slow inference from the request/response cycle |
| ML Inference | Spotify's Basic Pitch | Polyphonic pitch detection → note events |
| Fretboard Engine | Custom Python algorithm | Maps MIDI note events to guitar string/fret positions |
| Storage | Redis (job state) + local storage (audio + result JSON) | Persists intermediate and final artifacts |

Data flow: Submission → Ingestion (validate/resolve link) → Enqueue (202
Accepted, job_id returned immediately) → Async processing (preprocess →
Basic Pitch inference → fretboard mapping) → State updates in Redis →
Frontend polls until done → Renders interactive tab.

## 3. Detailed Technical Specification

### 3.1 Repository Structure

stratotab/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   └── app/
│       ├── main.py
│       ├── routes/jobs.py
│       ├── schemas/job.py, tab.py
│       ├── services/ingestion.py, storage.py
│       └── core/config.py
├── worker/
│   ├── Dockerfile
│   └── tasks/celery_app.py, pipeline.py, transcribe.py, fretboard.py
└── frontend/

### 3.2 Tech Stack — Pinned Choices

FastAPI (Python 3.11), Celery 5.x, Redis 7.x, basic-pitch (pip),
librosa + ffmpeg-python, yt-dlp, Next.js 14 (App Router) + TypeScript.
Audio sync via native <audio> element + requestAnimationFrame polling.

### 3.3 API Contract

POST /jobs — multipart (file: UploadFile, max 15MB, .mp3/.wav/.m4a)
OR JSON ({"url": "..."}). Returns 202: {job_id, status: "queued"}.
Validation errors → 422.

GET /jobs/{job_id} — Returns 200:
{job_id, status: "queued"|"processing"|"done"|"failed", error, result}.
result only populated when status is "done". Unknown job_id → 404.
Status is a FIXED enum — never a fifth value.

### 3.4 Celery Task Chain

Three chained tasks, not one monolithic task:

1. ingest_audio(job_id, source) — resolves file/link to normalized WAV
   (mono, 22.05kHz). Writes artifact path to Redis.
2. transcribe(job_id) — runs Basic Pitch, produces note events
   (pitch, start_time, end_time, velocity, confidence). Filters notes
   below confidence threshold (0.5 default).
3. map_fretboard(job_id) — runs DP mapping algorithm, produces final
   TabResult, sets status to "done".

Each task sets status: "processing" at start, catches exceptions to set
status: "failed" with a clear error message. soft_time_limit: 120s per task.

### 3.5 Fretboard Mapping Algorithm

Input: ordered note events, standard tuning (E2 A2 D3 G3 B3 E4).
Candidate generation: every valid (string, fret) pair per pitch, fret 0-20.

Cost function between consecutive positions (s1,f1) → (s2,f2):

cost = |f1 - f2|                      # fret-hand travel distance
     + (2 if f2 == 0 else 0)          # open-string bonus
     + (3 if |f1 - f2| > 4 else 0)    # uncomfortable stretch penalty
     + (1 if s1 == s2 else 0)         # same-string penalty

(Constants are placeholders — tune by ear against known-correct tabs.)

Algorithm: dynamic programming — dp[i][pos] = min cumulative cost to reach
candidate pos for note i. Viterbi-style backward pointer reconstruction.
Chords: treat as one combined position; v1 picks lowest-fret valid
combination (hand-shape realism is a stretch goal).

### 3.6 Tab JSON Schema

{
  "job_id": "uuid",
  "duration_seconds": 184.2,
  "tempo_bpm": 120,
  "notes": [
    {"string": 5, "fret": 3, "start_time": 1.24, "end_time": 1.58, "pitch": "C4"}
  ]
}

Time is stored in raw seconds, not beats/measures, for v1.

### 3.7 Known Edge Cases

- Audio > 5 min → reject at ingestion with 422
- Silent audio → status "done" with empty notes: [], not an error
- Broken link → ingest_audio catches yt-dlp exception → "failed"
- Note stretch exceeds hand span → still produce a valid position
  (accept the stretch penalty), don't fail

### 3.8 Environment Variables

REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
MAX_UPLOAD_MB=15
MAX_AUDIO_DURATION_SECONDS=300
BASIC_PITCH_CONFIDENCE_THRESHOLD=0.5

## 4. Current Build Status

Repo scaffolding, Dockerfiles, docker-compose.yml complete. Minimal
FastAPI /health endpoint and minimal Celery app confirmed working via
`docker compose up`. Next: real /jobs endpoint + full task chain.