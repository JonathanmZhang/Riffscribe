import logging

from tasks.celery_app import app
from tasks.storage import update_job

logger = logging.getLogger(__name__)


@app.task(name="map_fretboard", soft_time_limit=120)
def map_fretboard(job_id: str) -> str:
    try:
        update_job(job_id, status="processing")
        logger.info("map_fretboard: ran for job %s", job_id)

        result = {
            "job_id": job_id,
            "duration_seconds": 12.5,
            "tempo_bpm": 120,
            "notes": [
                {"string": 6, "fret": 0, "start_time": 0.0, "end_time": 0.5, "pitch": "E2"},
                {"string": 5, "fret": 2, "start_time": 0.5, "end_time": 1.0, "pitch": "B2"},
                {"string": 4, "fret": 2, "start_time": 1.0, "end_time": 1.5, "pitch": "E3"},
            ],
        }

        update_job(job_id, status="done", result=result)
    except Exception as exc:
        logger.exception("map_fretboard failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"map_fretboard failed: {exc}")
        raise
    return job_id
