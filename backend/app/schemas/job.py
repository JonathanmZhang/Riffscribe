from enum import Enum
from typing import Optional

from pydantic import BaseModel

from app.schemas.tab import TabResult


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None


class JobRecord(BaseModel):
    """Internal representation persisted in Redis."""

    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None
