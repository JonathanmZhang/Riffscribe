# StratoTab — Project Context

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
  FileResponse — don't hand-roll range handling, it's already supported.
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
values to each other. Don't remove immutable=True; without it, Celery
appends each task's return value as an extra positional arg to the next
task, which breaks the (job_id) signature.

## Conventions
- Pydantic models for all request/response schemas, in app/schemas/
- Time in the tab JSON output is stored in raw seconds, not beats/measures
  (v1 decision — don't add beat/measure grouping unless asked)
- Don't add new environment variables without updating docker-compose.yml
  and docs/spec.md 3.8 to match
- Shared job state lives in Redis as job:{job_id}, read/written by both
  backend and worker via the same JSON shape — see worker/tasks/storage.py
  for the get_job/update_job helpers; use these rather than writing raw
  Redis calls inline
- Shared audio/data files live under /app/data/{job_id}/ inside containers
  (mounted from ./data on the host via docker-compose volumes)
- Internal intermediate pipeline data (e.g. raw_note_events from
  transcribe) is stored in the job's Redis record but is NOT part of the
  public TabResult schema — keep internal state and the public API shape
  separate, don't conflate them
- scripts/ is NOT volume-mounted — after editing anything under
  worker/scripts/, the worker image must be rebuilt
  (docker compose up -d --build worker) before changes take effect
- Frontend has no design system yet — a Figma design is planned for
  later, not now. For now, prioritize clean, functional, reasonably
  presentable UI (Tailwind defaults are fine) over custom visual design.
  Don't over-invest time in styling polish until a real design exists.
- When reading a job's stored data server-side for anything beyond the
  public API shape (e.g. serving its audio file), read the raw Redis
  dict directly — don't go through the JobRecord/response Pydantic
  models, since those intentionally drop internal-only fields like
  normalized_audio_path.
- Frontend API base URL is hardcoded to http://localhost:8000 for now
  (not yet an env var) — CORS is enabled on the backend for
  http://localhost:3000 specifically, not wildcarded.

## Known gotchas (don't rediscover these)
- basic_pitch.inference.predict()'s note_events returns UNNAMED TUPLES:
  (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends) — not a
  dict, not named fields. pitch_midi is an int (MIDI note number) —
  convert via pretty_midi.note_number_to_name(), already done in
  transcribe.py, don't hand-roll this conversion elsewhere.
- basic-pitch's model load is slow on first use in a fresh container
  (~20s) and librosa's numba JIT warmup adds ~30s to the first
  ingest_audio call too — both are one-time per-container costs, not bugs.
- The fretboard cost function (spec 3.5) has no built-in preference for
  absolute neck position — only relative movement, same-string reuse, and
  open-string avoidance are penalized/rewarded. Resolved via a secondary
  lexicographic tie-break: (cost, total_fret_sum) — among equal-cost
  paths, prefer the lowest total fret sum. Don't remove this tie-break.
- Chord grouping (both in fretboard.py and in the frontend's TabViewer)
  uses a 50ms onset tolerance — notes starting within 50ms of each other
  are treated as one simultaneous chord/column. Keep this tolerance
  consistent between backend and frontend if either changes.
- Next.js is pinned to v14 — `create-next-app` with no version pinned
  defaults to a newer major version; always use `create-next-app@14`
  explicitly if the frontend ever needs to be re-scaffolded.
- On Windows/Git Bash specifically (not relevant inside containers or on
  other OSes): `docker compose exec` container paths can get mangled by
  Git Bash's automatic POSIX-path conversion — fix is prefixing the host
  command with MSYS_NO_PATHCONV=1. This is a host-shell issue, not a
  container or code issue.

## Current build status
CORE BACKEND PIPELINE + FULL FRONTEND FLOW (UPLOAD -> POLL -> RENDER ->
AUDIO-SYNCED PLAYBACK) ARE COMPLETE AND VERIFIED END TO END IN A REAL
BROWSER. Effectively Thursday AND most of Friday's roadmap items are
done a day ahead of schedule.

- Repo scaffolding, Dockerfiles, docker-compose.yml: done
- POST /jobs, GET /jobs/{job_id}, GET /jobs/{job_id}/audio: done and
  verified (file upload + URL, status polling, audio streaming with
  working Range/seek support)
- Celery chain (ingest_audio -> transcribe -> map_fretboard), all three
  stages fully real and verified — see prior entries in this file's
  history for the detailed verification of each stage (known-frequency
  test WAVs correctly transcribed, DP fretboard mapping producing
  playable positions including chords, tie-break for neck position)
- Frontend: Next.js 14 app scaffolded and fully wired —
  UploadForm (file or URL) -> JobStatus (live polling + status display)
  -> TabViewer (string/fret grid + detail table) -> real <audio controls>
  element with timeupdate-driven highlight sync between playback position
  and the currently-sounding note(s) in the grid and table.
- Verified via actual Playwright browser automation against the live
  Docker stack at each step, not just described: real file upload,
  real status transitions, real rendered DOM content, real audio
  playback with duration/readyState checks, real seeking with confirmed
  206 Range responses, and confirmed highlight correctly moves between
  notes on seek.

NEXT: deployment (Render/Railway for backend+worker, Vercel for
frontend), README (architecture, setup, demo), and a stress-test pass
against real (non-synthetic) songs — this is genuinely new: everything
verified so far has used known-frequency test tones, not real messy
audio, so real-song behavior is still an open question worth checking
before Friday's resume-lock.