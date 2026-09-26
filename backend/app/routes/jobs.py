import json
import os
import shutil
from uuid import uuid4

import redis.asyncio as redis
from celery import Celery, chain, signature
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse

from app.schemas.job import JobCreateResponse, JobRecord, JobStatus, JobStatusResponse

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
        # service (Demucs needs ~3GB RAM per run).
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

    elif content_type.startswith("application/json"):
        body = await request.json()
        url = body.get("url") if isinstance(body, dict) else None
        if not url:
            raise HTTPException(status_code=422, detail="A 'url' field is required")

        source = {"type": "url", "url": url}
        isolate_guitar = body.get("isolate_guitar", False)
        if not isinstance(isolate_guitar, bool):
            raise HTTPException(status_code=422, detail="'isolate_guitar' must be a boolean")

    else:
        raise HTTPException(
            status_code=422,
            detail="Content-Type must be multipart/form-data (file upload) or application/json ({'url': ...})",
        )

    job = JobRecord(job_id=job_id, status=JobStatus.queued, error=None, result=None, isolate_guitar=isolate_guitar)
    await redis_client.set(_job_key(job_id), job.model_dump_json())

    _enqueue_pipeline(job_id, source, isolate_guitar)

    return JobCreateResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JobRecord.model_validate_json(raw)
    # stage and stem_audio_path are written by the worker and aren't part of
    # JobRecord, so read them from the raw stored dict (as get_job_audio does).
    job_data = json.loads(raw)
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        result=job.result,
        isolate_guitar=job.isolate_guitar,
        stage=job_data.get("stage"),
        stem_available=_existing_file(job_data.get("stem_audio_path")) is not None,
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
