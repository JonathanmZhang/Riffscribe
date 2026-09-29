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

Optional 4th task, separate_guitar (tasks/separate.py), runs between
ingest_audio and transcribe ONLY when the job has isolate_guitar=true;
with it off the chain is exactly the three tasks above. It's routed to
the "separation" queue (set explicitly on the backend's signature and in
celery_app task_routes), served by the worker-separation compose service
at --concurrency 1. The main worker serves only the default "celery"
queue at --concurrency 2. Keep separation off the main worker.

Each task sets a `stage` field at start (ingesting | separating |
transcribing | mapping); map_fretboard clears it on done, and a failed
job keeps the stage it failed in. `stage` is NOT a status value — status
stays the fixed four-value enum.

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
  of the public TabResult schema. Other internal job fields:
  source_audio_path / source_duration_seconds (ingest), stem_audio_path /
  transcription_audio_path / separation_seconds (separate_guitar).
  transcribe reads transcription_audio_path if set, else
  normalized_audio_path. The API exposes only stage, isolate_guitar and
  stem_available (plus GET /jobs/{id}/stem), read from the raw dict.
- Pure, Redis/Celery-free helpers for scripts: tasks/audio_io.py
  (job_dir, probe_duration_seconds, normalize_to_wav),
  transcribe.extract_notes, separate.separate_guitar_stem,
  fretboard.map_notes_to_positions. scripts/ab_separation.py uses them.
- Worker code is NOT volume-mounted: tasks/ and scripts/ are both
  COPY'd into the image, so ANY change under worker/ needs a rebuild
  (docker compose up -d --build worker worker-separation — both services
  use the same image). Only ./data is mounted.
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
- Demucs (guitar separation) is called through its Python API
  (get_model + apply_model) with librosa/soundfile I/O — never the demucs
  CLI or torchaudio I/O (Demucs 4.0.1's torchaudio I/O path breaks on
  recent torchaudio). torch/torchaudio are pinned to 2.5.1 CPU wheels
  from the PyTorch index in their own Dockerfile layer, and demucs 4.0.1
  is in worker/requirements-separation.txt — keep them out of
  requirements.txt so the big TensorFlow layer stays cached. The
  htdemucs_6s weights are baked into the image (TORCH_HOME=/app/models).
- Separation runs on the ORIGINAL source (44.1/48kHz stereo), not
  normalized.wav. It's slow on CPU: real-time factor ~1.3–3.7 measured
  (varies run to run with host load, not with torch thread count), so
  120s audio can take ~3-7.5 min against the 900s soft limit. The first
  separation after a worker restart is ~30-50% slower than later ones. Peak memory ~1.9 GiB for 98s of audio.
  DEMUCS_SHIFTS>=1 applies a random shift, so the stem, and therefore
  the note count, varies slightly between runs of the same file (the
  pipeline doesn't seed it; scripts/ab_separation.py --seed does).
- Chord accuracy (measured with scripts/eval_chords.py and
  scripts/regression_check.py: synthetic GM chord set + 3 fixed firefire
  windows + the tab_sample clip, which has a real known tab). The biggest
  loss is Basic Pitch giving the quieter tones of a strum 0.35-0.5
  confidence. transcribe.select_notes keeps a note >=
  CHORD_TONE_CONFIDENCE_FLOOR (0.45) when a >=0.5 note starts within
  150ms of it. Lower floors look better on synthetic audio but only add
  false notes (mostly octave errors) on real guitar - keep 0.45 unless
  real-audio numbers say otherwise.
- Tried and reverted (see git history for numbers - don't retry as-is):
  (1) merging same-pitch "re-triggers": Basic Pitch leaves a 0 gap
  between consecutive same-pitch notes ~90% of the time for BOTH
  re-triggers and real repeats, and on real rhythm guitar re-strums look
  identical to re-triggers on onset activation and amplitude, so merging
  erases real strums. (2) splitting/gap-based grouping for fast chord
  changes: no completeness gain, because the affected chords also lose
  notes at detection/threshold. Backend grouping must stay identical to
  TabViewer.tsx's (anchor on first note, 150ms) unless the frontend is
  changed to use a backend-provided step index.
- Tested and rejected (never merged; branch experiment/time-stretch):
  time-stretching audio to 0.75x/0.5x (Rubber Band R3, pitch kept) before
  Basic Pitch, rescaling note times. Synthetic recall/complete dropped
  (clean 76%/35% -> 50%/15% at 0.75x, 53%/10% at 0.5x), no fast-
  progression gain (complete 25/25/0% -> 12/25/0% and 0/25/0%), known-tab
  clip neutral at 0.75x and worse at 0.5x (97% recall, 14/15), and
  detection ~3x slower. Extra notes it adds on real audio are mostly false.
- Tested and rejected (never merged; branch feature/hand-position): a
  hand-position fretboard mapper (DP over voicing + index-finger fret,
  span/stretch/shift costs, open-position bonus, constants in HandCosts,
  tuned with 2-fold cross-validation by performance on EGSet12). The
  current mapper's position errors are real (~90% land ~5 frets toward
  the nut), but the fix didn't hold up. Cross-validated held-out position
  agreement vs the current mapper: clean 53.2 -> 55.8% (chords 59.4 ->
  60.2), moderate 48.3 -> 49.7, heavy 45.0 -> 44.7 (chords 50.6 -> 45.1);
  it shifted hand position ~half as often as players. Rejected because
  this cost model can't tell an open-position shape from an up-the-neck
  shape for the same notes: a real open-string bonus fixed open chords
  (performance 06) but pulled up-neck passages (12) down to the nut, and
  with the final all-12 constants both showcase passages broke (12: 2/12,
  06: 8/23 positions right) and clean chords lost even in-sample (59.4 ->
  56.1). Don't retry by retuning constants; it needs more context than
  this cost model has.
- Tested and rejected (never merged; branch experiment/model-bakeoff,
  experiments/model_bakeoff/README.md): replacing Basic Pitch with a
  stronger transcription model. YourMT3+ (47% recall vs Basic Pitch's
  62% tab, ~17x slower) and TabCNN-GuitarProFX (54% precision, 33% chord
  recall) lost at feasibility; FretNet has no downloadable weights. MT3
  went to a full run (all instruments, drums dropped, same-pitch notes
  within 50ms merged across programs - its instrument labels are
  useless, 78% of a Telecaster came
  out as piano/other - then through our run_stages/mapper). EGSet12 tab
  recall/precision [position] vs Basic Pitch: clean 74.5/77.4 [48.5] vs
  71.9/85.2 [57.0] (F1 75.9 vs 78.0), moderate F1 66.3 vs 66.0, heavy
  60.6/55.1 vs 38.5/71.7 (F1 57.7 vs 50.1, but that's processed
  distortion, not a real amp). Fast-passage position agreement is ~half
  Basic Pitch's. One-octave errors are 41-45% of its false notes and
  neither simple fix helps (dropping upper octaves costs more recall than
  it gains precision; ~10% of true notes are real octave doublings). On
  the firefire stem it makes denser chords but the mapper drops 32
  notes/window as unplayable. Per 102s song: Basic Pitch 5s, +Demucs ~94s,
  MT3 262s, MT3+Demucs ~445s (MT3 is 0.7-3.8 s/s, scaling with note
  density; ~2.3 GB RAM). It also needs its own container (Python 3.12,
  JAX 0.11, TF 2.21, numpy 2.5). The 3-segment feasibility check was
  misleading (88.5% recall; clean chords 91.5% on 02-2 vs 70.8% raw over
  all 36 segments) - never decide on a few segments. Revisit only with
  ground truth for real distorted / full-mix recordings. Adding YourMT3+
  or amt-tools to the worker unpinned pulls protobuf 7 and breaks TF
  2.15 - pin protobuf<5.
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