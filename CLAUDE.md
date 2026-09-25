# Riffscribe — Project Context

## What this is
Async audio-to-tablature transcription app. Audio in → guitar tab out.
Full architecture and rationale: see docs/spec.md. This file is the
condensed, enforceable rules Claude Code should follow every session.

## Stack (do not substitute without asking)
- Backend: FastAPI (Python 3.11), in backend/app/
- Task queue: Celery + Redis, worker code in worker/tasks/
- ML: Spotify's basic-pitch for pitch detection, librosa for audio prep
- Frontend: Next.js 14 (App Router) + TypeScript, in frontend/
- Everything runs via docker-compose.yml — don't suggest running services
  outside Docker

## API contract (fixed — see docs/spec.md 3.3 for full detail)
- POST /jobs — accepts file upload OR {"url": "..."} JSON, returns 202
  with {job_id, status: "queued"}
- GET /jobs/{job_id} — returns {job_id, status, error, result}
- GET /jobs/{job_id}/audio — returns FileResponse over the job's stored
  audio (normalized_audio_path), media_type audio/wav. 404 if the job or
  file doesn't exist. Range requests work out of the box via Starlette's
  FileResponse.
- status is ALWAYS one of: queued | processing | done | failed — never
  invent a new status value
- result is ONLY the final TabResult (spec 3.6 shape) — intermediate
  pipeline data (raw_note_events etc.) is never exposed via the /jobs/{id}
  response body, it's internal Redis state only

## Celery task chain (see docs/spec.md 3.4)
Three separate chained tasks, not one monolithic task:
ingest_audio → transcribe → map_fretboard
Each task must update Redis status at start, and catch exceptions to set
status: failed with a clear error message — never let a task fail silently.
Chained via Celery's chain() with immutable=True on each signature —
tasks read/write job state via job_id in Redis, they do NOT pass return
values to each other. Don't remove immutable=True.

## Conventions
- Pydantic models for all request/response schemas, in app/schemas/
- Time in the tab JSON output is stored in raw seconds, not beats/measures
- Don't add new environment variables without updating docker-compose.yml
  and docs/spec.md 3.8 to match
- Shared job state lives in Redis as job:{job_id}, via
  worker/tasks/storage.py's get_job/update_job helpers — use these
  rather than writing raw Redis calls inline
- Shared audio/data files live under /app/data/{job_id}/ inside containers
  (mounted from ./data on the host via docker-compose volumes)
- Internal intermediate pipeline data (e.g. raw_note_events) stays out
  of the public TabResult schema
- scripts/ is NOT volume-mounted — rebuild the worker image after
  editing anything under worker/scripts/
- Frontend has a light Tailwind polish pass done (colors, layout, status
  badges, sticky grid headers) but no real design system yet — a Figma
  pass is planned later. Don't over-invest further in visual redesign
  until that exists.
- When reading a job's stored data server-side for anything beyond the
  public API shape, read the raw Redis dict directly — don't go through
  the JobRecord/response Pydantic models, which intentionally drop
  internal-only fields like normalized_audio_path.
- Deployment config is env-driven (see docs/spec.md 3.8), with local-dev
  fallbacks and docker-compose.yml setting the local values:
  - NEXT_PUBLIC_API_URL (frontend, default http://localhost:8000) — the
    backend base URL, read in frontend/app/lib/api.ts.
  - CORS_ALLOWED_ORIGINS (backend, default http://localhost:3000) —
    comma-separated explicit allowlist; "*" is rejected, keep it that way.
  - NEXT_PUBLIC_ENABLE_URL_INGESTION (frontend, default true) — "false"
    hides the Paste URL option in UploadForm and shows a note pointing to
    the README. Frontend-only by design: don't gate POST /jobs on it.
  NEXT_PUBLIC_* values are inlined into the JS bundle by `next build`, so
  they're Docker BUILD ARGS (docker-compose.yml frontend.build.args), not
  runtime env — setting them as container environment does nothing.
  Changing one needs `docker compose up -d --build frontend` locally, or
  a redeploy on Vercel.
- The frontend image is a production build (multi-stage Dockerfile, Next
  standalone output via next.config.mjs, `node server.js` as non-root),
  not `next dev`. The source was never volume-mounted, so there's no
  hot reload either way — rebuild the frontend image after any change.

## Known gotchas (don't rediscover these)
- basic_pitch.inference.predict()'s note_events returns UNNAMED TUPLES:
  (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends).
  pitch_midi is an int — convert via pretty_midi.note_number_to_name().
- basic-pitch's model load (~20s) and librosa's numba JIT warmup (~30s)
  are one-time per-container costs on first use, not bugs.
- Open-string cost term is -2 (a genuine bonus, reducing cost) — this
  was originally written as +2, a sign error caught during real-audio
  testing and fixed in both fretboard.py and spec 3.5. Don't reintroduce
  the +2 version.
- Chord-onset grouping tolerance is 150ms (raised from an initial 50ms
  after real strum testing showed wider onset spreads), used consistently
  in both fretboard.py and TabViewer.tsx. This is empirical, not proven
  optimal — fast riffs with sub-150ms onsets may still misgroup.
- Unplayable notes are dropped, not fatal, in two places:
  (1) within a chord, the minimum-conflict-resolving subset of notes is
  dropped (not a greedy lowest-amplitude drop — that was tried first and
  gave wrong results); (2) a single note outside standard tuning's fret
  0-20 range is dropped individually. Both log a warning naming the
  dropped pitch/time/amplitude and continue the job rather than failing it.
- Riffscribe assumes STANDARD TUNING ONLY (E A D G B E). Real-song testing
  found songs in alternate tunings (e.g. drop D) have their low notes
  (e.g. D2, C2) dropped, since those pitches don't exist in standard
  tuning's fretboard. This is a known v1 scope limitation, not a bug —
  alternate-tuning support/detection is unimplemented future work.
- On Windows/Git Bash specifically: `docker compose exec` container
  paths can get mangled by Git Bash's POSIX-path conversion — prefix
  with MSYS_NO_PATHCONV=1.

## Current build status
CORE PIPELINE + FULL FRONTEND ARE COMPLETE AND VERIFIED AGAINST BOTH
SYNTHETIC TEST TONES AND REAL, LICENSED GUITAR AUDIO (including a real
YouTube URL via the yt-dlp ingestion path).

- Full pipeline (ingest → transcribe → fretboard mapping): done, real,
  verified against synthetic tones AND real audio (Wikimedia Commons
  clips + a real YouTube song)
- Two real bugs found via real-audio testing and fixed: (1) unplayable
  chords/notes used to fail the whole job, now gracefully drop the
  minimum necessary notes; (2) open-string cost sign error
- Frontend: full flow (upload/URL → poll → render → audio-synced
  playback highlighting) done and verified, plus a Tailwind visual
  polish pass (colors, card layout, status badges, sticky grid/table
  headers for large real-song note counts)
- Known, documented limitations (not yet fixed, intentionally scoped
  out of v1): standard tuning only; some Basic Pitch detection artifacts
  on real audio (phantom harmonics, occasional octave flips) are
  inherent to the model, not addressed by this codebase; tab grid
  doesn't auto-scroll to follow playback

NEXT: README, then deployment (Render/Railway for backend+worker,
Vercel for frontend), then resume bullets locked once a real live link
exists.