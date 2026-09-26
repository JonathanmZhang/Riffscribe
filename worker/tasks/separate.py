"""Optional guitar source separation (Demucs htdemucs_6s), run between
ingest_audio and transcribe when a job has isolate_guitar set.

Uses Demucs' Python API directly rather than the demucs CLI, and does file
I/O with librosa/soundfile rather than torchaudio: Demucs 4.0.1 reads and
writes through torchaudio, whose I/O layer has changed in recent releases.
"""

import functools
import logging
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from tasks.audio_io import job_dir, normalize_to_wav
from tasks.celery_app import app
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

# htdemucs_6s is the only standard Demucs model with a guitar stem. Its
# weights are downloaded into the image at build time (see worker/Dockerfile);
# any other model is downloaded on first use and not persisted.
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs_6s")
DEMUCS_SHIFTS = int(os.environ.get("DEMUCS_SHIFTS", "1"))
# Separation is far slower than the rest of the pipeline on CPU, so it has
# its own, tighter duration cap and a longer time limit than the 120s tasks.
MAX_SEPARATION_DURATION_SECONDS = float(os.environ.get("MAX_SEPARATION_DURATION_SECONDS", "120"))
SEPARATION_SOFT_TIME_LIMIT_SECONDS = int(os.environ.get("SEPARATION_SOFT_TIME_LIMIT_SECONDS", "900"))

GUITAR_SOURCE = "guitar"


@functools.lru_cache(maxsize=None)
def _load_model(name: str):
    """Loaded once per worker process; repeat jobs reuse it."""
    model = get_model(name)
    model.eval()
    return model


def separate_guitar_stem(input_path: str) -> tuple[np.ndarray, int]:
    """Separates the guitar from input_path. Returns (stem, sample_rate) with
    stem shaped (channels, samples), clipped to [-1, 1]. Pure: no Redis or
    Celery, so scripts can call it directly.
    """
    model = _load_model(DEMUCS_MODEL)
    if GUITAR_SOURCE not in model.sources:
        raise ValueError(f"Demucs model {DEMUCS_MODEL!r} has no guitar stem (sources: {model.sources})")

    # Load at the model's native rate (44.1kHz for htdemucs_6s) and keep
    # channels, then match the model's channel count the way Demucs' own
    # loader does: duplicate mono, keep the first N of a wider layout.
    audio, _ = librosa.load(input_path, sr=model.samplerate, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, model.audio_channels, axis=0)
    elif audio.shape[0] > model.audio_channels:
        audio = audio[: model.audio_channels]

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    # Same normalization demucs.separate applies around apply_model.
    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std()
    if std == 0:  # silent input: nothing to scale, avoid dividing by zero
        std = torch.tensor(1.0)
    wav = (wav - mean) / std

    with torch.no_grad():
        sources = apply_model(
            model, wav[None], shifts=DEMUCS_SHIFTS, split=True, overlap=0.25, progress=False
        )[0]
    sources = sources * std + mean

    stem = sources[model.sources.index(GUITAR_SOURCE)].cpu().numpy()
    return np.clip(stem, -1.0, 1.0), model.samplerate


@app.task(name="separate_guitar", soft_time_limit=SEPARATION_SOFT_TIME_LIMIT_SECONDS)
def separate_guitar(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="separating")

        job = get_job(job_id)
        source_path = job.get("source_audio_path")
        if not source_path:
            raise ValueError(f"job {job_id} has no source_audio_path; ingest_audio must run first")

        duration = job.get("source_duration_seconds")
        if duration is not None and duration > MAX_SEPARATION_DURATION_SECONDS:
            raise ValueError(
                f"guitar isolation is limited to {MAX_SEPARATION_DURATION_SECONDS:.0f} seconds of "
                f"audio, and this file is {duration:.0f} seconds long. Use a shorter clip, or "
                "turn off \"Isolate guitar\"."
            )

        load_start = time.perf_counter()
        _load_model(DEMUCS_MODEL)
        load_seconds = time.perf_counter() - load_start

        # Runs on the original source (e.g. 44.1/48kHz stereo), not
        # normalized.wav: htdemucs_6s is trained on 44.1kHz stereo.
        start = time.perf_counter()
        stem, sample_rate = separate_guitar_stem(source_path)
        separation_seconds = time.perf_counter() - start

        directory = job_dir(job_id)
        stem_path = os.path.join(directory, "guitar_stem.wav")
        sf.write(stem_path, stem.T, sample_rate, subtype="PCM_16")
        # Same conversion ingest applies, so Basic Pitch gets its usual format.
        transcription_path = normalize_to_wav(stem_path, os.path.join(directory, "guitar_stem_normalized.wav"))

        audio_seconds = stem.shape[1] / sample_rate
        logger.info(
            "separate_guitar: job %s separated %.1fs of audio in %.1fs (real-time factor %.2f, "
            "model %s, shifts %d, model load %.1fs)",
            job_id,
            audio_seconds,
            separation_seconds,
            separation_seconds / audio_seconds if audio_seconds else 0.0,
            DEMUCS_MODEL,
            DEMUCS_SHIFTS,
            load_seconds,
        )

        update_job(
            job_id,
            stem_audio_path=stem_path,
            transcription_audio_path=transcription_path,
            separation_seconds=round(separation_seconds, 2),
        )
    except Exception as exc:
        logger.exception("separate_guitar failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"separate_guitar failed: {exc}")
        raise
    return job_id
