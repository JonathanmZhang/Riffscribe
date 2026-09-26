import logging
import os

import librosa
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from tasks.celery_app import app
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.environ.get("BASIC_PITCH_CONFIDENCE_THRESHOLD", "0.5"))


def extract_notes(audio_path: str) -> tuple[int, list[dict]]:
    """Runs Basic Pitch on audio_path and applies the confidence filter.
    Returns (raw note count, kept notes in raw_note_events shape). Pure: no
    Redis/Celery, so scripts/ab_separation.py runs exactly the same
    extraction as the pipeline.
    """
    _, _, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)

    kept = []
    for start_time, end_time, pitch_midi, amplitude, _pitch_bends in note_events:
        if amplitude < CONFIDENCE_THRESHOLD:
            continue
        kept.append(
            {
                "pitch": pretty_midi.note_number_to_name(pitch_midi),
                "start_time": start_time,
                "end_time": end_time,
                "amplitude": float(amplitude),
            }
        )
    return len(note_events), kept


def _estimate_tempo_bpm(audio_path: str) -> int:
    """Global tempo estimate via librosa's onset-based beat tracker, rounded
    to the nearest integer BPM. This is an ESTIMATE, not a measured tempo:
    beat_track keys on percussive onsets, so it is noticeably less reliable
    on solo instrument recordings with no drums or clear pulse (legato
    playing, rubato, sparse picking), where it can lock onto half/double the
    felt tempo or onto note density instead of the beat. Returns 0 when no
    beat is found (e.g. silent audio).
    """
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    tempo, _beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
    # librosa >= 0.10 returns tempo as a 1-element array, older versions a scalar.
    return int(round(float(np.atleast_1d(tempo)[0])))


@app.task(name="transcribe", soft_time_limit=120)
def transcribe(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="transcribing")

        job = get_job(job_id)
        # separate_guitar sets transcription_audio_path to the normalized
        # guitar stem; without separation, transcribe the normalized full mix.
        audio_path = job.get("transcription_audio_path") or job.get("normalized_audio_path")
        if not audio_path:
            raise ValueError(f"job {job_id} has no normalized_audio_path; ingest_audio must run first")

        source = "separated guitar stem" if job.get("transcription_audio_path") else "full mix"
        logger.info("transcribe: job %s running basic-pitch on %s (%s)", job_id, audio_path, source)

        raw_count, raw_note_events = extract_notes(audio_path)

        logger.info(
            "transcribe: job %s kept %d/%d notes above confidence threshold %.2f",
            job_id,
            len(raw_note_events),
            raw_count,
            CONFIDENCE_THRESHOLD,
        )

        # Same file Basic Pitch used, so tempo and notes describe the same audio.
        tempo_bpm = _estimate_tempo_bpm(audio_path)
        logger.info("transcribe: job %s estimated tempo ~%d bpm", job_id, tempo_bpm)

        update_job(job_id, raw_note_events=raw_note_events, tempo_bpm=tempo_bpm)
    except Exception as exc:
        logger.exception("transcribe failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"transcribe failed: {exc}")
        raise
    return job_id
