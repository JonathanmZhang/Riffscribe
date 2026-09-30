# Riffscribe: Asynchronous Audio-to-Tablature Transcription Platform

## 1. Project Overview & Core Value Proposition

Riffscribe converts raw instrumental guitar audio — an uploaded file or a link
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

riffscribe/
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

FastAPI (Python 3.11), Celery 5.x, Valkey 8 (BSD-licensed Redis fork, used
as the Redis server; the compose service is still named "redis"),
basic-pitch (pip), BTC chord recognizer (cloned into the worker image),
librosa + ffmpeg-python, yt-dlp, Next.js 14 (App Router) + TypeScript.
Audio sync via native <audio> element + requestAnimationFrame polling.

### 3.3 API Contract

POST /jobs — multipart (file: UploadFile, max 200MB, .mp3/.wav/.m4a or
video .mp4/.webm/.mov — ingest_audio extracts the audio track with ffmpeg)
OR JSON ({"url": "..."}). Returns 202: {job_id, status: "queued"}.
Validation errors → 422.

GET /jobs/{job_id} — Returns 200:
{job_id, status: "queued"|"processing"|"done"|"failed", error, result}.
result only populated when status is "done". Unknown job_id → 404.
Status is a FIXED enum — never a fifth value. Also returns the job's
notation overrides, tempo_factor (0.5 | 1 | 2, default 1) and
bar_offset_beats (0-3, default 0).

PATCH /jobs/{job_id} — JSON {tempo_factor?, bar_offset_beats?}; fields
left out keep their value. Finished jobs only (409 otherwise); invalid
values → 422. Stores the overrides; nothing is re-transcribed. Returns
the same body as GET /jobs/{job_id}, whose result.bars reflects the new
overrides.

GET /jobs/{job_id}/musicxml — the finished tab as MusicXML 4.0
(application/vnd.recordare.musicxml+xml, attachment
riffscribe-<id>.musicxml): one guitar part, notation + TAB staves, 4/4,
quantized starts and lengths, BTC chord symbols. Built on each request by
the backend (tasks/musicxml.py, shared from the worker) with the job's
current overrides. 404 if the job is unknown or not done.

### 3.4 Celery Task Chain

Three chained tasks, not one monolithic task:

1. ingest_audio(job_id, source) — resolves file/link to normalized WAV
   (mono, 22.05kHz). Writes artifact path to Redis.
2. transcribe(job_id) — runs Basic Pitch, produces note events
   (pitch, start_time, end_time, velocity, confidence). Filters notes
   below confidence threshold (0.5 default). Also tracks beats and
   downbeats with beat_this (tasks/beats.py; MIT, "final0" checkpoint, no
   DBN) on the same audio, and derives tempo_bpm from the beats (60 /
   median inter-beat interval, rounded to an integer). All three go in the
   job's Redis record for map_fretboard. If beat_this fails, the error is
   logged and the job continues with librosa.beat.beat_track()'s tempo and
   empty beat lists. The tempo is an estimate: fast songs can come out at
   half tempo, and downbeats are only about half right.
   Also names chords from the same audio with BTC (tasks/chords.py; large
   vocabulary, 12 roots x 14 qualities) and stores the timed segments as
   the internal chord_segments field. Chord names are display-only: if
   recognition fails, the error is logged and the job continues with none.
3. map_fretboard(job_id) — runs DP mapping algorithm, produces final
   TabResult (copying chord_segments into its "chords"), sets status to
   "done".

Each task sets status: "processing" at start, catches exceptions to set
status: "failed" with a clear error message. soft_time_limit: 120s per task.

### 3.5 Fretboard Mapping Algorithm

Input: ordered note events, standard tuning (E2 A2 D3 G3 B3 E4).
Candidate generation: every valid (string, fret) pair per pitch, fret 0-20.

Cost function between consecutive positions (s1,f1) → (s2,f2):

cost = |f1 - f2|                      # fret-hand travel distance
     - (2 if f2 == 0 else 0)          # open-string bonus (reduces cost)
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
  ],
  "chords": [
    {"start": 0.0, "end": 1.3, "name": "E"},
    {"start": 1.76, "end": 2.87, "name": "Am7"}
  ],
  "beats": [0.012, 0.512, 1.013],
  "downbeats": [0.012],
  "bars": [0.012]
}

Time is stored in raw seconds, not beats/measures, for v1. "chords" are
BTC's segments with no-chord stretches left out and repeats merged;
names are root + suffix ("", m, dim, aug, 6, m6, 7, m7, maj7, m(maj7),
dim7, m7b5, sus2, sus4), roots spelled with sharps. It's an empty list
for jobs finished before chord names existed. TabViewer prints a name
above the first column of each chord change. "beats" and "downbeats" are
beat and bar-start times in seconds (downbeats a subset of beats); both
are empty for jobs finished before beat tracking existed or when it
failed. "bars" is the start time in seconds of each measure of the
MusicXML export (tasks/rhythm.measure_starts): 4/4, the beats (kept at one
metrical level, then halved/doubled by the job's tempo_factor) in 4s,
phased where most downbeats fall and moved by bar_offset_beats, extended
to cover every note (a constant grid at tempo_bpm without beats). It isn't
stored: the backend derives it on every read, so it always matches the
export for the current overrides. The sheet-music view uses it as
alphaTab sync points. The JSON's notes stay in raw seconds; quantization
to 16ths happens only in the MusicXML export.

### 3.7 Known Edge Cases

- Audio longer than MAX_AUDIO_DURATION_SECONDS (default 300s / 5 min) →
  rejected inside the async ingest_audio task, not at POST /jobs:
  duration isn't known until the file is uploaded or downloaded, so
  POST /jobs still returns 202. ingest_audio reads the duration from the
  file header (ffprobe) before decoding, sets status "failed" with an
  error like "ingest_audio failed: audio is 412 seconds long, which
  exceeds the 300 second limit", and the chain stops there (transcribe
  and map_fretboard never run).
- Silent audio → status "done" with empty notes: [], not an error
- Broken link → ingest_audio catches yt-dlp exception → "failed"
- Note stretch exceeds hand span → still produce a valid position
  (accept the stretch penalty), don't fail

### 3.8 Environment Variables

REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
MAX_UPLOAD_MB=200   # backend; local-only app, sized for music videos
MAX_AUDIO_DURATION_SECONDS=300   # read by the worker (ingest_audio); set in docker-compose.yml
BASIC_PITCH_CONFIDENCE_THRESHOLD=0.5
CHORD_TONE_CONFIDENCE_FLOOR=0.45   # worker (transcribe): a note below the
  threshold is still kept if it's >= this AND a note >= the threshold starts
  within the chord onset window (150ms) of it, i.e. a chord tone. Values >=
  the threshold disable it. 0.45 chosen from the chord eval: full recall on
  the real tab-sample clip; lower values mostly add octave errors there.
  Re-checked on the EGSet12 benchmark with the settings below: still the
  best balance (0.40 adds ~2-4pt recall but loses precision on real audio).
BASIC_PITCH_ONSET_THRESHOLD=0.5      # worker (transcribe): Basic Pitch's own
BASIC_PITCH_FRAME_THRESHOLD=0.4      #   note-creation settings, passed to
BASIC_PITCH_MIN_NOTE_LENGTH_MS=80    #   model_output_to_notes. predict()'s
  defaults are 0.5 / 0.3 / 127.7ms; 0.4 / 80ms chosen on the EGSet12
  real-guitar benchmark (recall 66.6->71.9% clean, 49.5->55.8% moderate,
  34.5->38.5% heavy distortion; precision -2.4/-3.5/-6.8pt; fast passages
  +12 to +17pt). The old 128ms minimum is longer than a 16th note at
  ~120bpm. Lowering the onset threshold mainly adds false notes.

# Backend: comma-separated explicit allowlist of frontend origins for CORS.
# A "*" wildcard is rejected at startup.
CORS_ALLOWED_ORIGINS=http://localhost:3000

# Frontend (NEXT_PUBLIC_* are inlined into the client bundle when Next
# compiles - at build time in production, so changing them on a host like
# Vercel requires a redeploy):
NEXT_PUBLIC_API_URL=http://localhost:8000    # backend base URL
NEXT_PUBLIC_ENABLE_URL_INGESTION=true        # "false" hides the Paste URL
  option in the UI (file upload only, plus a note pointing to the README).
  Frontend-only: POST /jobs still accepts {"url": ...} regardless.

# Guitar separation (optional separate_guitar task; read by the
# worker-separation service, set in docker-compose.yml):
DEMUCS_MODEL=htdemucs_6s                   # must have a "guitar" source; only
  the default's weights are baked into the worker image
DEMUCS_SHIFTS=1                            # Demucs random-shift passes;
  higher is slightly better quality and proportionally slower
MAX_SEPARATION_DURATION_SECONDS=120        # isolate_guitar jobs with longer
  audio fail in ingest_audio right after the ffprobe duration check, with
  a readable message (separate_guitar re-checks as a safety net); read by
  both worker services
SEPARATION_SOFT_TIME_LIMIT_SECONDS=900     # separate_guitar's soft time
  limit (the other tasks keep 120s)
SEPARATION_TORCH_THREADS=                  # torch intra-op threads for
  Demucs; empty/unset = half the container's CPUs

## 4. Current Build Status

Repo scaffolding, Dockerfiles, docker-compose.yml complete. Minimal
FastAPI /health endpoint and minimal Celery app confirmed working via
`docker compose up`. Next: real /jobs endpoint + full task chain.