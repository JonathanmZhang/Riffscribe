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
Validation errors → 422. Optional in both forms: isolate_guitar (form
"true"/"1"/"on", or a JSON boolean; default off) and, with it,
separation_quality: "standard" (default; Demucs) or "high" (Mega 53, needs
an NVIDIA GPU). Any other value → 422. "high" is accepted even where it
can't run: the job is then separated with Demucs (see 3.4).

POST /jobs/{job_id}/rerun — JSON {isolate_guitar (default true),
separation_quality (default "standard")}. A new job from a copy of the
existing job's source audio (the file ingest_audio kept, so a link isn't
downloaded again), returned like POST /jobs (202 {job_id, status:
"queued"}). 404 if the job is unknown, 409 if its source audio is gone.

GET /capabilities — Returns 200: {separation: {high_quality_available,
high_quality_unavailable_reason, gpu, gpu_memory_mib}}, as reported by the
separation worker when it started (Redis key separation:capabilities).
Unavailable, with a reason, when the worker has no NVIDIA GPU, was built
without Mega 53, or hasn't started. The upload form uses it to enable or
disable "High quality".

GET /jobs/{job_id} — Returns 200:
{job_id, status: "queued"|"processing"|"done"|"failed", error, result}.
result only populated when status is "done". Unknown job_id → 404.
Status is a FIXED enum — never a fifth value. Also returns the job's
notation overrides, tempo_factor (0.5 | 1 | 2, default 1) and
bar_offset_beats (0-3, default 0), its neck_position ("auto" (default) |
"open" | an integer 1-17, see 3.5), and for isolate_guitar jobs:
separation_quality (what was asked for), separator ("demucs" | "mega53",
what produced the stem, set once it exists) and separation_note (why a
"high" job was separated with Demucs; null otherwise). These are fields
of the job, not of result.

PATCH /jobs/{job_id} — JSON {tempo_factor?, bar_offset_beats?,
neck_position?}; fields left out keep their value. Finished jobs only (409
otherwise; also 409 for a neck position on a job with no stored
raw_note_events); invalid values → 422. Stores the overrides; nothing is
re-transcribed. Returns the same body as GET /jobs/{job_id}, whose
result.bars reflects the new overrides. A neck_position other than "auto"
re-maps the job's stored raw_note_events inside that fret window on every
read of GET /jobs/{job_id} and the MusicXML export, so result.notes' (and
the TAB staff's) strings and frets change. Which notes there are doesn't.

GET /jobs/{job_id}/musicxml — the finished tab as MusicXML 4.0
(application/vnd.recordare.musicxml+xml, attachment
riffscribe-<id>.musicxml): one guitar part, notation + TAB staves, 4/4,
quantized starts and lengths, BTC chord symbols. Optional query parameter
tone = clean (default) | overdriven | distorted | acoustic: the part's
General MIDI program (28 / 30 / 31 / 26) and instrument name, which is what
a synth plays the file with; other values → 422. The sheet-music view's
Tone selector uses it for alphaTab's synth. Built on each request by
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
   Then the vibrato merge (tasks/techniques.py): a run of same-pitch
   notes with no gap between them whose pitch wobbles at 4-8 Hz is one
   note that vibrato split, so it becomes one note again, flagged
   "vibrato". It uses Basic Pitch's per-note pitch offsets.
   Also names chords from the same audio with BTC (tasks/chords.py; large
   vocabulary, 12 roots x 14 qualities) and stores the timed segments as
   the internal chord_segments field. Chord names are display-only: if
   recognition fails, the error is logged and the job continues with none.
3. map_fretboard(job_id) — runs DP mapping algorithm, produces final
   TabResult (copying chord_segments into its "chords"), sets status to
   "done".

Each task sets status: "processing" at start, catches exceptions to set
status: "failed" with a clear error message. soft_time_limit: 120s per task.

Optional separate_guitar(job_id), between 1 and 2, only for isolate_guitar
jobs, on the "separation" queue (worker-separation service, concurrency 1,
soft_time_limit 900s). It writes the guitar stem that transcribe then
reads. The separator follows the job's separation_quality:
- "standard": Demucs htdemucs_6s on the CPU, in the worker process.
- "high": MVSep Mega 53 Stems (BS-RoFormer) on an NVIDIA GPU
  (tasks/mega53.py), with only its two guitar heads loaded and the release's
  inference settings (fp32, 20s chunks, 2 overlaps). It runs in a
  subprocess per job, so GPU memory is freed afterwards and a crash can't
  take the worker down. If it can't run or fails (no GPU, out of GPU
  memory, killed, over HQ_SEPARATION_TIMEOUT_SECONDS), the task separates
  with Demucs instead and sets separation_note; the job doesn't fail.
Both are capped at MAX_SEPARATION_DURATION_SECONDS of audio. Mega 53 exists
only in the worker-separation image built by docker-compose.gpu.yml
(HQ_SEPARATION=1), which also reserves the GPU; the default stack has
neither, so it starts on any computer.

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

Neck position (a user choice, PATCH /jobs/{id}; worker/tasks/fretmap.py,
pure stdlib so the backend runs it too): a fret window. "auto" = none,
the algorithm above exactly. "open" = frets 0-4. N = frets max(1, N-3) to
N+3, without open strings. With a window, a single note's candidates are
its positions inside the window, or its nearest ones (fewest frets
outside) if it has none. A chord's candidates are its voicings on distinct
strings with the fewest total frets outside the window, preferring a
fretted span of at most 4 frets, at most 12 of them, lowest fret sum
first. The DP then chooses as above. Which notes are dropped as unplayable
is decided without the window, so every setting places the same notes.
Measured in experiments/neck_position/README.md.

### 3.6 Tab JSON Schema

{
  "job_id": "uuid",
  "duration_seconds": 184.2,
  "tempo_bpm": 120,
  "notes": [
    {"string": 5, "fret": 3, "start_time": 1.24, "end_time": 1.58, "pitch": "C4", "vibrato": false, "column": 0}
  ],
  "chords": [
    {"start": 0.0, "end": 1.3, "name": "E"},
    {"start": 1.76, "end": 2.87, "name": "Am7"}
  ],
  "beats": [0.012, 0.512, 1.013],
  "downbeats": [0.012],
  "bars": [0.012]
}

Time is stored in raw seconds, not beats/measures, for v1. A note's
"vibrato" is true when the vibrato merge detected vibrato on it (false on
jobs finished before it existed); the MusicXML export draws it as a wavy
line. A note's "column" is the tab column (0, 1, ...) the mapper grouped
it into (3.5's 150ms chord grouping), so notes with the same column are one
chord on distinct strings; the tab shows one column per value. Like "bars",
the backend derives it on every read from the stored note events and never
stores it (null only for a job without them). "chords" are
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

# High-quality separation (Mega 53). Read by worker-separation and set in
# docker-compose.gpu.yml, the override that enables it:
HQ_SEPARATION_HEAD=guitar                  # which Mega 53 stem is
  transcribed: guitar | electric-guitar
HQ_SEPARATION_TIMEOUT_SECONDS=420          # a Mega 53 run longer than this is
  stopped and the job falls back to Demucs; keep it far enough under
  SEPARATION_SOFT_TIME_LIMIT_SECONDS for Demucs to finish
# Build arg of worker/Dockerfile, not an environment variable:
HQ_SEPARATION=0                            # 1 (docker-compose.gpu.yml, for
  worker-separation only) adds PyTorch's CUDA 12.1 build, MSST and the
  Mega 53 weights, downloaded from a pinned URL with a pinned SHA-256

## 4. Current Build Status

Repo scaffolding, Dockerfiles, docker-compose.yml complete. Minimal
FastAPI /health endpoint and minimal Celery app confirmed working via
`docker compose up`. Next: real /jobs endpoint + full task chain.