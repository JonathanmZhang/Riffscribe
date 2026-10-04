import logging

from tasks.celery_app import app
# The mapper itself is pure stdlib in tasks/fretmap.py (the backend re-maps
# finished jobs with it for the neck position); re-exported here for the
# scripts that import it from this module.
from tasks.fretmap import (  # noqa: F401
    CHORD_ONSET_TOLERANCE_SECONDS,
    MAX_FRET,
    _candidates_for_pitch,
    map_notes_to_positions,
    map_notes_with_steps,
)
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)


@app.task(name="map_fretboard", soft_time_limit=120)
def map_fretboard(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="mapping")

        job = get_job(job_id)
        raw_note_events = job.get("raw_note_events")
        if raw_note_events is None:
            raise ValueError(f"job {job_id} has no raw_note_events; transcribe must run first")
        tempo_bpm = job.get("tempo_bpm")
        if tempo_bpm is None:
            raise ValueError(f"job {job_id} has no tempo_bpm; transcribe must run first")

        logger.info("map_fretboard: job %s mapping %d note(s)", job_id, len(raw_note_events))

        mapped_notes = map_notes_to_positions(raw_note_events)
        duration_seconds = max((note["end_time"] for note in mapped_notes), default=0.0)

        result = {
            "job_id": job_id,
            "duration_seconds": duration_seconds,
            # Estimated by transcribe from beat_this's beats (librosa's
            # tempo if beat_this failed); see tasks/beats.py for accuracy.
            "tempo_bpm": tempo_bpm,
            "notes": mapped_notes,
            # Timed chord names from transcribe (tasks/chords.py); display-only.
            "chords": job.get("chord_segments") or [],
            # Beat and bar-start times in seconds (tasks/beats.py); empty if
            # beat tracking failed.
            "beats": job.get("beats") or [],
            "downbeats": job.get("downbeats") or [],
            # "bars" isn't stored: the backend derives it on every read from
            # the notes and the job's overrides (tasks/rhythm.measure_starts).
        }

        # stage only describes in-progress work, so it's cleared once done. On
        # failure it's left as-is so the status shows where the job stopped.
        update_job(job_id, status="done", stage=None, result=result)
        logger.info("map_fretboard: job %s done with %d mapped note(s)", job_id, len(mapped_notes))
    except Exception as exc:
        logger.exception("map_fretboard failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"map_fretboard failed: {exc}")
        raise
    return job_id
