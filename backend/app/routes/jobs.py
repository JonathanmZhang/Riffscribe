import json
import logging
import os
import shutil
from uuid import uuid4

import redis.asyncio as redis
from celery import Celery, chain, signature
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from app.schemas.job import (
    JobCreateResponse,
    JobOverrides,
    JobRecord,
    JobRerun,
    JobStatus,
    JobStatusResponse,
    PlaybackTone,
)
from app.schemas.tab import TabResult
# Pure code shared with the worker (worker/tasks/rhythm.py, musicxml.py and
# fretmap.py, copied in by backend/Dockerfile from the "worker" context).
from tasks import fretmap, musicxml, rhythm

# The mapper's "dropped an unplayable note" warnings: the worker logged them
# when the job was first mapped, so re-mapping on every read stays quiet.
logging.getLogger("tasks.fretboard").setLevel(logging.ERROR)

router = APIRouter(prefix="/jobs", tags=["jobs"])

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
# Local-only app, so large enough for music videos.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
# Audio files, plus video containers: ingest_audio extracts the audio track
# with ffmpeg and ignores the video.
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "webm", "mov"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
DATA_DIR = "/app/data"

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Client-only Celery app: sends tasks by name to the worker, doesn't execute them itself.
celery_client = Celery("riffscribe-client", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


def _enqueue_pipeline(job_id: str, source: dict, isolate_guitar: bool) -> None:
    steps = [signature("ingest_audio", args=(job_id, source), app=celery_client, immutable=True)]
    if isolate_guitar:
        # Its own queue, served one job at a time by the worker-separation
        # service (Demucs needs ~3GB RAM per run). The task reads the job's
        # separation_quality to pick the separator.
        steps.append(
            signature("separate_guitar", args=(job_id,), app=celery_client, immutable=True, queue="separation")
        )
    steps += [
        signature("transcribe", args=(job_id,), app=celery_client, immutable=True),
        signature("map_fretboard", args=(job_id,), app=celery_client, immutable=True),
    ]
    chain(*steps).apply_async()


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


TRUTHY_FORM_VALUES = {"true", "1", "on"}
SEPARATION_QUALITIES = ("standard", "high")


def _separation_quality(value, isolate_guitar: bool) -> str | None:
    """The job's separation_quality from the request's value (form field or
    JSON; missing = standard). None when the guitar isn't isolated. "high"
    is accepted even where it can't run: the worker then falls back to
    Demucs and says why in separation_note."""
    if value is None or value == "":
        value = "standard"
    if value not in SEPARATION_QUALITIES:
        raise HTTPException(status_code=422, detail=f"'separation_quality' must be one of {list(SEPARATION_QUALITIES)}")
    return value if isolate_guitar else None


def _existing_file(path: str | None) -> str | None:
    return path if path and os.path.isfile(path) else None


@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request) -> JobCreateResponse:
    content_type = request.headers.get("content-type", "")
    job_id = str(uuid4())

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", None):
            raise HTTPException(status_code=422, detail="A 'file' field is required for multipart uploads")

        extension = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '.{extension}'; allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        job_dir = os.path.join(DATA_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        saved_path = os.path.join(job_dir, f"original.{extension}")
        # Copied in chunks rather than read() into memory whole, now that
        # uploads can be hundreds of MB; oversized files are rejected (and
        # the partial copy removed) as soon as they cross the limit.
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        written = 0
        with open(saved_path, "wb") as f:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    break
                f.write(chunk)
        if written > max_bytes:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit")

        source = {"type": "file", "path": saved_path}
        # Checkbox-style form field: "true"/"1"/"on" (any case) enable it.
        isolate_guitar = str(form.get("isolate_guitar", "")).strip().lower() in TRUTHY_FORM_VALUES
        try:
            separation_quality = _separation_quality(form.get("separation_quality"), isolate_guitar)
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    elif content_type.startswith("application/json"):
        body = await request.json()
        url = body.get("url") if isinstance(body, dict) else None
        if not url:
            raise HTTPException(status_code=422, detail="A 'url' field is required")

        source = {"type": "url", "url": url}
        isolate_guitar = body.get("isolate_guitar", False)
        if not isinstance(isolate_guitar, bool):
            raise HTTPException(status_code=422, detail="'isolate_guitar' must be a boolean")
        separation_quality = _separation_quality(body.get("separation_quality"), isolate_guitar)

    else:
        raise HTTPException(
            status_code=422,
            detail="Content-Type must be multipart/form-data (file upload) or application/json ({'url': ...})",
        )

    job = JobRecord(
        job_id=job_id,
        status=JobStatus.queued,
        error=None,
        result=None,
        isolate_guitar=isolate_guitar,
        separation_quality=separation_quality,
    )
    await redis_client.set(_job_key(job_id), job.model_dump_json())

    _enqueue_pipeline(job_id, source, isolate_guitar)

    return JobCreateResponse(job_id=job_id, status=JobStatus.queued)


@router.post("/{job_id}/rerun", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def rerun_job(job_id: str, options: JobRerun) -> JobCreateResponse:
    """A new job from an existing job's source audio (the file ingest_audio
    kept, so a link isn't downloaded again), with other isolation settings.
    The audio is copied, so the two jobs stay independent."""
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # source_audio_path is internal (written by ingest_audio): raw-dict read.
    source_path = _existing_file(json.loads(raw).get("source_audio_path"))
    if source_path is None:
        raise HTTPException(status_code=409, detail="This job's source audio is not available to run again")

    new_id = str(uuid4())
    new_dir = os.path.join(DATA_DIR, new_id)
    os.makedirs(new_dir, exist_ok=True)
    saved_path = os.path.join(new_dir, f"original{os.path.splitext(source_path)[1].lower()}")
    await run_in_threadpool(shutil.copyfile, source_path, saved_path)

    job = JobRecord(
        job_id=new_id,
        status=JobStatus.queued,
        error=None,
        result=None,
        isolate_guitar=options.isolate_guitar,
        separation_quality=options.separation_quality if options.isolate_guitar else None,
    )
    await redis_client.set(_job_key(new_id), job.model_dump_json())
    _enqueue_pipeline(new_id, {"type": "file", "path": saved_path}, options.isolate_guitar)
    return JobCreateResponse(job_id=new_id, status=JobStatus.queued)


def _result_data(job_data: dict) -> dict:
    """The stored TabResult dict with the job's neck position applied: for
    anything but "auto", the notes are re-mapped from the stored
    raw_note_events (internal, written by transcribe) inside that fret
    window. Same notes, only strings and frets differ. Done per read, like
    the bars (~10-100ms for a few hundred notes)."""
    result = job_data["result"]
    window = fretmap.neck_window(job_data.get("neck_position", "auto"))
    if window is not None and job_data.get("raw_note_events"):
        result = {**result, "notes": fretmap.map_notes_to_positions(job_data["raw_note_events"], window)}
    return result


def _status_response(raw: str) -> JobStatusResponse:
    job = JobRecord.model_validate_json(raw)
    # stage, stem_audio_path, separator and the overrides aren't part of JobRecord, so
    # read them from the raw stored dict (as get_job_audio does).
    job_data = json.loads(raw)
    result = None
    if job.result is not None:
        result_data = _result_data(job_data)
        # Derived on every read, not stored, so it always matches the
        # MusicXML export for the job's current overrides (older jobs too).
        bars = rhythm.measure_starts(
            result_data, job_data.get("tempo_factor", 1.0), job_data.get("bar_offset_beats", 0))
        result = TabResult.model_validate({**result_data, "bars": bars})
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        result=result,
        isolate_guitar=job.isolate_guitar,
        stage=job_data.get("stage"),
        stem_available=_existing_file(job_data.get("stem_audio_path")) is not None,
        separation_quality=job.separation_quality,
        separator=job_data.get("separator"),
        separation_note=job_data.get("separation_note"),
        tempo_factor=job_data.get("tempo_factor", 1.0),
        bar_offset_beats=job_data.get("bar_offset_beats", 0),
        neck_position=job_data.get("neck_position", "auto"),
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _status_response(raw)


@router.patch("/{job_id}", response_model=JobStatusResponse)
async def update_job_overrides(job_id: str, overrides: JobOverrides) -> JobStatusResponse:
    """Sets the job's notation overrides. Nothing is re-transcribed: the
    response's result.bars, its notes' positions (neck_position) and the
    MusicXML export are derived from the stored notes and beats with the
    new overrides on every read. Only for finished jobs, so this never
    races the worker's writes to the same record."""
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = json.loads(raw)
    if job_data.get("status") != JobStatus.done.value or not job_data.get("result"):
        raise HTTPException(status_code=409, detail="Overrides can only be set on a finished job")
    if overrides.neck_position not in (None, "auto") and not job_data.get("raw_note_events"):
        raise HTTPException(status_code=409, detail="This job has no stored notes to place on the neck again")

    for field in ("tempo_factor", "bar_offset_beats", "neck_position"):
        value = getattr(overrides, field)
        if value is not None:
            job_data[field] = value
    raw = json.dumps(job_data)
    await redis_client.set(_job_key(job_id), raw)
    return _status_response(raw)


@router.get("/{job_id}/musicxml")
async def get_job_musicxml(job_id: str, tone: PlaybackTone = musicxml.DEFAULT_TONE) -> Response:
    """The finished tab as MusicXML (notation + TAB staff, 4/4, quantized,
    chord symbols), built on request with the job's current overrides.
    tone sets the part's General MIDI program (what a synth plays it with);
    the notes are the same for every tone."""
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = json.loads(raw)
    if job_data.get("status") != JobStatus.done.value or not job_data.get("result"):
        raise HTTPException(status_code=404, detail="MusicXML not available for this job yet")

    content = musicxml.to_musicxml(_result_data(job_data), job_data.get("tempo_factor", 1.0),
                                   job_data.get("bar_offset_beats", 0), tone=tone)
    return Response(
        content=content,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="riffscribe-{job_id[:8]}.musicxml"'},
    )


@router.get("/{job_id}/audio")
async def get_job_audio(job_id: str) -> FileResponse:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # normalized_audio_path isn't part of the public JobRecord schema (it's
    # internal, written by ingest_audio), so read the raw stored dict rather
    # than the pydantic model, which would silently drop it.
    job_data = json.loads(raw)
    audio_path = job_data.get("normalized_audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        raise HTTPException(status_code=404, detail="Audio not available for this job yet")

    return FileResponse(audio_path, media_type="audio/wav")


@router.get("/{job_id}/stem")
async def get_job_stem(job_id: str) -> FileResponse:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # stem_audio_path is internal (written by separate_guitar); same raw-dict
    # read as get_job_audio. Serves the 44.1kHz stereo stem, for playback.
    stem_path = _existing_file(json.loads(raw).get("stem_audio_path"))
    if stem_path is None:
        raise HTTPException(status_code=404, detail="Guitar stem not available for this job")

    return FileResponse(stem_path, media_type="audio/wav")
