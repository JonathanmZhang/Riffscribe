import json
import os
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
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a"}
DATA_DIR = "/app/data"

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Client-only Celery app: sends tasks by name to the worker, doesn't execute them itself.
celery_client = Celery("riffscribe-client", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


def _enqueue_pipeline(job_id: str, source: dict) -> None:
    workflow = chain(
        signature("ingest_audio", args=(job_id, source), app=celery_client, immutable=True),
        signature("transcribe", args=(job_id,), app=celery_client, immutable=True),
        signature("map_fretboard", args=(job_id,), app=celery_client, immutable=True),
    )
    workflow.apply_async()


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


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

        contents = await upload.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=422, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit")

        job_dir = os.path.join(DATA_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        saved_path = os.path.join(job_dir, f"original.{extension}")
        with open(saved_path, "wb") as f:
            f.write(contents)

        source = {"type": "file", "path": saved_path}

    elif content_type.startswith("application/json"):
        body = await request.json()
        url = body.get("url") if isinstance(body, dict) else None
        if not url:
            raise HTTPException(status_code=422, detail="A 'url' field is required")

        source = {"type": "url", "url": url}

    else:
        raise HTTPException(
            status_code=422,
            detail="Content-Type must be multipart/form-data (file upload) or application/json ({'url': ...})",
        )

    job = JobRecord(job_id=job_id, status=JobStatus.queued, error=None, result=None)
    await redis_client.set(_job_key(job_id), job.model_dump_json())

    _enqueue_pipeline(job_id, source)

    return JobCreateResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    raw = await redis_client.get(_job_key(job_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JobRecord.model_validate_json(raw)
    return JobStatusResponse(job_id=job.job_id, status=job.status, error=job.error, result=job.result)


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
