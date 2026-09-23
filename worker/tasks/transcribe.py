import logging
import os

import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from tasks.celery_app import app
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.environ.get("BASIC_PITCH_CONFIDENCE_THRESHOLD", "0.5"))


@app.task(name="transcribe", soft_time_limit=120)
def transcribe(job_id: str) -> str:
    try:
        update_job(job_id, status="processing")

        job = get_job(job_id)
        normalized_audio_path = job.get("normalized_audio_path")
        if not normalized_audio_path:
            raise ValueError(f"job {job_id} has no normalized_audio_path; ingest_audio must run first")

        logger.info("transcribe: job %s running basic-pitch on %s", job_id, normalized_audio_path)

        _, _, note_events = predict(normalized_audio_path, ICASSP_2022_MODEL_PATH)

        raw_note_events = []
        for start_time, end_time, pitch_midi, amplitude, _pitch_bends in note_events:
            if amplitude < CONFIDENCE_THRESHOLD:
                continue
            raw_note_events.append(
                {
                    "pitch": pretty_midi.note_number_to_name(pitch_midi),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

        logger.info(
            "transcribe: job %s kept %d/%d notes above confidence threshold %.2f",
            job_id,
            len(raw_note_events),
            len(note_events),
            CONFIDENCE_THRESHOLD,
        )

        update_job(job_id, raw_note_events=raw_note_events)
    except Exception as exc:
        logger.exception("transcribe failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"transcribe failed: {exc}")
        raise
    return job_id
