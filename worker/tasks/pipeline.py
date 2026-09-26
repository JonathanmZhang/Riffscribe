import logging
import os

import yt_dlp

from tasks.audio_io import ensure_decodable_audio, job_dir, normalize_to_wav, probe_duration_seconds
from tasks.celery_app import app
from tasks.separation_limits import separation_duration_error
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

MAX_AUDIO_DURATION_SECONDS = float(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "300"))


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


@app.task(name="ingest_audio", soft_time_limit=120)
def ingest_audio(job_id: str, source: dict) -> str:
    try:
        update_job(job_id, status="processing", stage="ingesting")
        logger.info("ingest_audio: job %s received source: %s", job_id, source)

        directory = job_dir(job_id)
        source_type = source.get("type")

        if source_type == "url":
            input_path = _download_from_url(source["url"], directory)
        elif source_type == "file":
            input_path = source["path"]
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Uploaded file not found at {input_path}")
        else:
            raise ValueError(f"Unknown source type: {source_type!r}")

        duration = probe_duration_seconds(input_path)
        if duration > MAX_AUDIO_DURATION_SECONDS:
            # Raising (rather than returning) also stops the chain, so
            # transcribe/map_fretboard never run on a rejected job.
            raise ValueError(
                f"audio is {duration:.0f} seconds long, which exceeds the "
                f"{MAX_AUDIO_DURATION_SECONDS:.0f} second limit"
            )

        # Early separation cap: isolate_guitar is on the job record (set by the
        # API), so reject audio separate_guitar would refuse anyway, before
        # spending time extracting/normalizing it.
        if get_job(job_id).get("isolate_guitar"):
            too_long = separation_duration_error(duration)
            if too_long:
                raise ValueError(too_long)

        # Video containers (mp4/mov/webm) and m4a are extracted to a WAV with
        # ffmpeg first (audio track only); that WAV is also what
        # separate_guitar runs on.
        input_path = ensure_decodable_audio(input_path, directory)

        normalized_path = normalize_to_wav(input_path, os.path.join(directory, "normalized.wav"))
        logger.info("ingest_audio: job %s normalized audio at %s", job_id, normalized_path)

        # source_audio_path/source_duration_seconds are internal: separate_guitar
        # runs on the original (full-rate, stereo) source, not normalized.wav.
        update_job(
            job_id,
            normalized_audio_path=normalized_path,
            source_audio_path=input_path,
            source_duration_seconds=duration,
        )
    except Exception as exc:
        logger.exception("ingest_audio failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"ingest_audio failed: {exc}")
        raise
    return job_id
