import logging
import time

from tasks.celery_app import app
from tasks.storage import update_job

logger = logging.getLogger(__name__)


@app.task(name="transcribe", soft_time_limit=120)
def transcribe(job_id: str) -> str:
    try:
        update_job(job_id, status="processing")
        logger.info("transcribe: ran for job %s", job_id)
        time.sleep(1)
    except Exception as exc:
        logger.exception("transcribe failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"transcribe failed: {exc}")
        raise
    return job_id
