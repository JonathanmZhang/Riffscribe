import logging
import os
import subprocess

import librosa
import soundfile as sf
import yt_dlp

from tasks.celery_app import app
from tasks.storage import update_job

logger = logging.getLogger(__name__)

DATA_DIR = "/app/data"
TARGET_SAMPLE_RATE = 22050
MAX_AUDIO_DURATION_SECONDS = float(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "300"))


def _job_dir(job_id: str) -> str:
    job_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def _download_from_url(url: str, job_dir: str) -> str:
    outtmpl = os.path.join(job_dir, "source.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    if not os.path.isfile(downloaded_path):
        raise FileNotFoundError(f"yt-dlp reported {downloaded_path} but it does not exist")

    return downloaded_path


def _probe_duration_seconds(input_path: str) -> float:
    """Reads the duration from the container header via ffprobe, without
    decoding the audio, so over-long inputs are rejected before the
    expensive librosa load/resample. Works for every format ingest accepts
    (mp3/wav/m4a uploads and whatever yt-dlp downloads, e.g. webm), unlike
    librosa.get_duration(path=), which depends on soundfile's format support.
    """
    output = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    return float(output)


def _normalize_to_wav(input_path: str, job_dir: str) -> str:
    output_path = os.path.join(job_dir, "normalized.wav")
    audio, _ = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)
    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return output_path


@app.task(name="ingest_audio", soft_time_limit=120)
def ingest_audio(job_id: str, source: dict) -> str:
    try:
        update_job(job_id, status="processing")
        logger.info("ingest_audio: job %s received source: %s", job_id, source)

        job_dir = _job_dir(job_id)
        source_type = source.get("type")

        if source_type == "url":
            input_path = _download_from_url(source["url"], job_dir)
        elif source_type == "file":
            input_path = source["path"]
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Uploaded file not found at {input_path}")
        else:
            raise ValueError(f"Unknown source type: {source_type!r}")

        duration = _probe_duration_seconds(input_path)
        if duration > MAX_AUDIO_DURATION_SECONDS:
            # Raising (rather than returning) also stops the chain, so
            # transcribe/map_fretboard never run on a rejected job.
            raise ValueError(
                f"audio is {duration:.0f} seconds long, which exceeds the "
                f"{MAX_AUDIO_DURATION_SECONDS:.0f} second limit"
            )

        normalized_path = _normalize_to_wav(input_path, job_dir)
        logger.info("ingest_audio: job %s normalized audio at %s", job_id, normalized_path)

        update_job(job_id, normalized_audio_path=normalized_path)
    except Exception as exc:
        logger.exception("ingest_audio failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"ingest_audio failed: {exc}")
        raise
    return job_id
