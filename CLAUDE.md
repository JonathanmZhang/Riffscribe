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
- status is ALWAYS one of: queued | processing | done | failed — never
  invent a new status value

## Celery task chain (see docs/spec.md 3.4)
Three separate chained tasks, not one monolithic task:
ingest_audio → transcribe → map_fretboard
Each task must update Redis status at start, and catch exceptions to set
status: failed with a clear error message — never let a task fail silently.

## Conventions
- Pydantic models for all request/response schemas, in app/schemas/
- Time in the tab JSON output is stored in raw seconds, not beats/measures
  (v1 decision — don't add beat/measure grouping unless asked)
- Don't add new environment variables without updating docker-compose.yml
  and docs/spec.md 3.8 to match

## Current build status
Repo scaffolding, Dockerfiles, docker-compose.yml, and a minimal
FastAPI /health endpoint + minimal Celery app are done and confirmed
working via `docker compose up`. Next: build the real /jobs endpoint
and wire the Celery task chain.