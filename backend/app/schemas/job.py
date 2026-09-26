from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

from app.schemas.tab import TabResult


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


# Which pipeline task is running (set by each worker task when it starts).
# Separate from status, which stays the fixed four-value enum. None when
# queued or done; left at the failing stage when a job fails.
JobStage = Literal["ingesting", "separating", "transcribing", "mapping"]


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None
    isolate_guitar: bool = False
    stage: Optional[JobStage] = None
    # True once the separated guitar stem exists; served by GET /jobs/{id}/stem.
    stem_available: bool = False


class JobRecord(BaseModel):
    """Internal representation persisted in Redis."""

    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None
    isolate_guitar: bool = False
